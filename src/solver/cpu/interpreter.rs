//! A tree-walking interpreter over the backend-agnostic kernel IR
//! (`cfd2_ir::ast`).
//!
//! This is the Phase-1 CPU "reference" executor: it runs the *exact same* typed
//! `Stmt`/`Expr` AST that the WGSL emitter consumes
//! (`lower_kernel_program_to_wgsl`), one dispatch index (cell/face) at a time,
//! against CPU-side `Vec<f32>` / `Vec<u32>` buffers. Because it consumes the same
//! AST, it tracks the codegen automatically — there is no second emitter to keep
//! in sync (that trade-off is what Phase 3's transpiler buys speed with).
//!
//! ## WGSL → CPU semantics
//! - Floating-point math is `f32` (matching the GPU); indices are `u32`.
//! - Vector types `vecN<f32>` are small fixed arrays.
//! - GPU-only constructs map to CPU equivalents:
//!   - `arrayLength(&b)` → `b.len()`
//!   - `workgroupBarrier()` / `storageBarrier()` → no-ops (per-index granularity)
//!   - `atomicAdd`/`atomicStore`/… → plain serial ops (Phase 1 is single-threaded)
//!
//! The interpreter here is intentionally decoupled from `KernelProgram`/buffer
//! allocation/scheduling: it operates on `&[Stmt]` + a [`Buffers`] store + a
//! per-invocation [`Ctx`]. Wiring it to real generated `KernelProgram`s and the
//! recipe-driven schedule is the next step.

use std::collections::HashMap;

use cfd2_ir::ast::{
    AssignOp, BinaryOp, Block, Expr, ExprNode, ForInit, ForStep, Literal, Stmt, Type, UnaryOp,
};

/// A runtime value. WGSL is f32-typed for reals and u32-typed for indices; small
/// vectors are fixed arrays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    F32(f32),
    I32(i32),
    U32(u32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
}

impl Value {
    pub fn as_f32(self) -> f32 {
        match self {
            Value::F32(v) => v,
            Value::I32(v) => v as f32,
            Value::U32(v) => v as f32,
            Value::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
            other => panic!("expected scalar, got vector {other:?}"),
        }
    }

    pub fn as_i32(self) -> i32 {
        match self {
            Value::I32(v) => v,
            Value::U32(v) => v as i32,
            Value::F32(v) => v as i32,
            Value::Bool(b) => b as i32,
            other => panic!("expected integer, got {other:?}"),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Value::U32(v) => v,
            Value::I32(v) => v as u32,
            Value::F32(v) => v as u32,
            Value::Bool(b) => b as u32,
            other => panic!("expected integer, got {other:?}"),
        }
    }

    /// Coerce to an array index. Negative `i32` is a bug (generated code guards
    /// sentinels like `neighbor != -1` before indexing).
    pub fn as_index(self) -> usize {
        match self {
            Value::U32(v) => v as usize,
            Value::I32(v) if v >= 0 => v as usize,
            other => panic!("invalid array index {other:?}"),
        }
    }

    pub fn as_bool(self) -> bool {
        match self {
            Value::Bool(b) => b,
            Value::U32(v) => v != 0,
            Value::I32(v) => v != 0,
            other => panic!("expected bool, got {other:?}"),
        }
    }

    /// Component `axis` (0=x,1=y,2=z,3=w) of a vector value.
    pub fn component(self, axis: usize) -> f32 {
        match self {
            Value::Vec2(a) => a[axis],
            Value::Vec3(a) => a[axis],
            Value::Vec4(a) => a[axis],
            // A 1-component "vector" access on a scalar is the scalar itself in
            // some swizzle patterns; accept axis 0.
            Value::F32(v) if axis == 0 => v,
            other => panic!("component {axis} of non-vector {other:?}"),
        }
    }

    fn set_component(&mut self, axis: usize, v: f32) {
        match self {
            Value::Vec2(a) => a[axis] = v,
            Value::Vec3(a) => a[axis] = v,
            Value::Vec4(a) => a[axis] = v,
            other => panic!("set component {axis} of non-vector {other:?}"),
        }
    }

    fn zero_of(ty: &Type) -> Value {
        match ty {
            Type::F32 => Value::F32(0.0),
            Type::U32 => Value::U32(0),
            Type::I32 => Value::I32(0),
            Type::Bool => Value::Bool(false),
            Type::Vec2(_) => Value::Vec2([0.0; 2]),
            Type::Vec3(_) => Value::Vec3([0.0; 3]),
            Type::Vec4(_) => Value::Vec4([0.0; 4]),
            // Codegen emits the named structs `VectorN { x, y, ... }` for vector locals.
            Type::Custom(name) if name == "Vector2" => Value::Vec2([0.0; 2]),
            Type::Custom(name) if name == "Vector3" => Value::Vec3([0.0; 3]),
            Type::Custom(name) if name == "Vector4" => Value::Vec4([0.0; 4]),
            other => panic!("no default value for type {other}"),
        }
    }
}

/// Backing store for one named buffer. WGSL storage buffers are either flat
/// scalar arrays (`array<f32>`/`array<u32>`/`array<i32>`) or arrays of small
/// structs (`array<Vector2>`), the latter stored interleaved with a component
/// stride. `arrayLength(&b)` returns the *element* count (floats/stride).
#[derive(Debug, Clone)]
enum Store {
    /// `comps == 1` → `array<f32>`; `comps == 2` → `array<Vector2>`, etc.
    F32 { data: Vec<f32>, comps: usize },
    U32(Vec<u32>),
    I32(Vec<i32>),
}

/// Named CPU buffers, the runtime stand-in for wgpu storage buffers.
#[derive(Debug, Default, Clone)]
pub struct Buffers {
    map: HashMap<String, Store>,
}

impl Buffers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a flat `array<f32>` buffer.
    pub fn insert_f32(&mut self, name: impl Into<String>, data: Vec<f32>) {
        self.map
            .insert(name.into(), Store::F32 { data, comps: 1 });
    }

    /// Insert an `array<Vector2>` buffer (interleaved x,y,x,y,...).
    pub fn insert_vec2(&mut self, name: impl Into<String>, data: Vec<f32>) {
        debug_assert!(data.len() % 2 == 0, "Vector2 buffer must have even length");
        self.map
            .insert(name.into(), Store::F32 { data, comps: 2 });
    }

    pub fn insert_u32(&mut self, name: impl Into<String>, data: Vec<u32>) {
        self.map.insert(name.into(), Store::U32(data));
    }

    pub fn insert_i32(&mut self, name: impl Into<String>, data: Vec<i32>) {
        self.map.insert(name.into(), Store::I32(data));
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// Read back a flat `f32` buffer (also works for Vector2 as interleaved data).
    pub fn f32_slice(&self, name: &str) -> &[f32] {
        match self.map.get(name) {
            Some(Store::F32 { data, .. }) => data,
            _ => panic!("`{name}` is not an f32 buffer"),
        }
    }

    pub fn f32_slice_mut(&mut self, name: &str) -> &mut [f32] {
        match self.map.get_mut(name) {
            Some(Store::F32 { data, .. }) => data,
            _ => panic!("`{name}` is not an f32 buffer"),
        }
    }

    pub fn u32_slice(&self, name: &str) -> &[u32] {
        match self.map.get(name) {
            Some(Store::U32(data)) => data,
            _ => panic!("`{name}` is not a u32 buffer"),
        }
    }

    /// Element count — the value of `arrayLength(&name)`.
    fn len(&self, name: &str) -> usize {
        match self.map.get(name) {
            Some(Store::F32 { data, comps }) => data.len() / comps,
            Some(Store::U32(d)) => d.len(),
            Some(Store::I32(d)) => d.len(),
            None => panic!("arrayLength of unknown buffer `{name}`"),
        }
    }

    fn load(&self, name: &str, idx: usize) -> Value {
        match self.map.get(name) {
            Some(Store::F32 { data, comps }) => match comps {
                1 => Value::F32(data[idx]),
                2 => Value::Vec2([data[2 * idx], data[2 * idx + 1]]),
                3 => Value::Vec3([data[3 * idx], data[3 * idx + 1], data[3 * idx + 2]]),
                n => panic!("unsupported buffer comps {n}"),
            },
            Some(Store::U32(d)) => Value::U32(d[idx]),
            Some(Store::I32(d)) => Value::I32(d[idx]),
            None => panic!("read of unknown buffer `{name}`"),
        }
    }

    fn store(&mut self, name: &str, idx: usize, v: Value) {
        match self.map.get_mut(name) {
            Some(Store::F32 { data, comps }) => match comps {
                1 => data[idx] = v.as_f32(),
                2 => {
                    data[2 * idx] = v.component(0);
                    data[2 * idx + 1] = v.component(1);
                }
                3 => {
                    data[3 * idx] = v.component(0);
                    data[3 * idx + 1] = v.component(1);
                    data[3 * idx + 2] = v.component(2);
                }
                n => panic!("unsupported buffer comps {n}"),
            },
            Some(Store::U32(d)) => d[idx] = v.as_u32(),
            Some(Store::I32(d)) => d[idx] = v.as_i32(),
            None => panic!("write of unknown buffer `{name}`"),
        }
    }
}

/// Per-invocation immutable context: uniform/struct values (e.g. `constants`)
/// and builtins (e.g. the launch's invocation-index variable).
#[derive(Debug, Default, Clone)]
pub struct Ctx {
    /// Struct-typed uniforms accessed as `name.field` (e.g. `constants.dt`).
    pub structs: HashMap<String, HashMap<String, Value>>,
    /// Builtin identifiers available to every invocation.
    pub builtins: HashMap<String, Value>,
}

impl Ctx {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_constant(mut self, struct_name: &str, field: &str, v: Value) -> Self {
        self.structs
            .entry(struct_name.to_string())
            .or_default()
            .insert(field.to_string(), v);
        self
    }

    pub fn with_builtin(mut self, name: &str, v: Value) -> Self {
        self.builtins.insert(name.to_string(), v);
        self
    }
}

/// Control-flow signal threaded out of statement execution.
#[derive(Debug, Clone, PartialEq)]
enum Flow {
    Normal,
    Break,
    Continue,
    Return(Option<Value>),
}

/// Per-invocation mutable state (local `let`/`var` bindings).
#[derive(Debug, Default)]
pub struct Frame {
    locals: HashMap<String, Value>,
}

impl Frame {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_local(mut self, name: &str, v: Value) -> Self {
        self.locals.insert(name.to_string(), v);
        self
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        self.locals.get(name).copied()
    }
}

/// The interpreter: borrows the shared buffer store and per-invocation context.
pub struct Interpreter<'a> {
    buffers: &'a mut Buffers,
    ctx: &'a Ctx,
}

impl<'a> Interpreter<'a> {
    pub fn new(buffers: &'a mut Buffers, ctx: &'a Ctx) -> Self {
        Self { buffers, ctx }
    }

    /// Execute a kernel body (a slice of statements) for one invocation.
    pub fn run(&mut self, stmts: &[Stmt], frame: &mut Frame) {
        self.exec_block(stmts, frame);
    }

    fn exec_block(&mut self, stmts: &[Stmt], frame: &mut Frame) -> Flow {
        for stmt in stmts {
            let flow = self.exec_stmt(stmt, frame);
            if flow != Flow::Normal {
                return flow;
            }
        }
        Flow::Normal
    }

    fn exec_stmt(&mut self, stmt: &Stmt, frame: &mut Frame) -> Flow {
        match stmt {
            Stmt::Comment(_) => {}
            Stmt::Let { name, expr, .. } => {
                let v = self.eval(expr, frame);
                frame.locals.insert(name.clone(), v);
            }
            Stmt::Var { name, ty, expr } => {
                let v = match expr {
                    Some(e) => self.eval(e, frame),
                    None => Value::zero_of(ty.as_ref().expect("untyped uninitialised var")),
                };
                frame.locals.insert(name.clone(), v);
            }
            Stmt::Assign { target, value } => {
                let v = self.eval(value, frame);
                self.assign(target, v, frame);
            }
            Stmt::AssignOp { target, op, value } => {
                let cur = self.eval(target, frame);
                let rhs = self.eval(value, frame);
                let v = apply_assign_op(cur, *op, rhs);
                self.assign(target, v, frame);
            }
            Stmt::If {
                cond,
                then_block,
                else_block,
            } => {
                if self.eval(cond, frame).as_bool() {
                    return self.exec_block(&then_block.stmts, frame);
                } else if let Some(eb) = else_block {
                    return self.exec_block(&eb.stmts, frame);
                }
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
            } => return self.exec_for(init, cond, step, body, frame),
            Stmt::Loop { body } => loop {
                match self.exec_block(&body.stmts, frame) {
                    Flow::Break => break,
                    Flow::Return(v) => return Flow::Return(v),
                    _ => {}
                }
            },
            Stmt::While { cond, body } => {
                while self.eval(cond, frame).as_bool() {
                    match self.exec_block(&body.stmts, frame) {
                        Flow::Break => break,
                        Flow::Return(v) => return Flow::Return(v),
                        _ => {}
                    }
                }
            }
            Stmt::Break => return Flow::Break,
            Stmt::Continue => return Flow::Continue,
            Stmt::Return(e) => {
                let v = e.as_ref().map(|e| self.eval(e, frame));
                return Flow::Return(v);
            }
            Stmt::Call(expr) => {
                // Evaluated for side effects (atomics, barriers).
                let _ = self.eval(expr, frame);
            }
            Stmt::Increment(target) => {
                let cur = self.eval(target, frame);
                let v = apply_assign_op(cur, AssignOp::Add, Value::U32(1));
                self.assign(target, v, frame);
            }
            Stmt::Decrement(target) => {
                let cur = self.eval(target, frame);
                let v = apply_assign_op(cur, AssignOp::Sub, Value::U32(1));
                self.assign(target, v, frame);
            }
        }
        Flow::Normal
    }

    fn exec_for(
        &mut self,
        init: &ForInit,
        cond: &Expr,
        step: &ForStep,
        body: &Block,
        frame: &mut Frame,
    ) -> Flow {
        match init {
            ForInit::Let { name, expr, .. } | ForInit::Var { name, expr, .. } => {
                let v = self.eval(expr, frame);
                frame.locals.insert(name.clone(), v);
            }
            ForInit::Assign { target, value } => {
                let v = self.eval(value, frame);
                self.assign(target, v, frame);
            }
        }
        while self.eval(cond, frame).as_bool() {
            match self.exec_block(&body.stmts, frame) {
                Flow::Break => break,
                Flow::Return(v) => return Flow::Return(v),
                _ => {}
            }
            match step {
                ForStep::Increment(t) => {
                    let v = apply_assign_op(self.eval(t, frame), AssignOp::Add, Value::U32(1));
                    self.assign(t, v, frame);
                }
                ForStep::Decrement(t) => {
                    let v = apply_assign_op(self.eval(t, frame), AssignOp::Sub, Value::U32(1));
                    self.assign(t, v, frame);
                }
                ForStep::Assign { target, value } => {
                    let v = self.eval(value, frame);
                    self.assign(target, v, frame);
                }
                ForStep::AssignOp { target, op, value } => {
                    let cur = self.eval(target, frame);
                    let rhs = self.eval(value, frame);
                    let v = apply_assign_op(cur, *op, rhs);
                    self.assign(target, v, frame);
                }
            }
        }
        Flow::Normal
    }

    /// Store `val` into an lvalue: a local, a buffer element, or a vector
    /// component of a local.
    fn assign(&mut self, target: &Expr, val: Value, frame: &mut Frame) {
        match target.node() {
            ExprNode::Ident(name) => {
                frame.locals.insert(name.clone(), val);
            }
            ExprNode::Index { base, index } => {
                let buf = ident_name(base)
                    .unwrap_or_else(|| panic!("indexed store into non-buffer base"));
                let idx = self.eval(index, frame).as_index();
                self.buffers.store(buf, idx, val);
            }
            ExprNode::Field { base, field } => {
                let axis = axis_of(field).expect("struct-field store unsupported");
                match base.node() {
                    // Local vector component: `n.x = ...`.
                    ExprNode::Ident(name) => {
                        let mut cur = frame.get(name).unwrap_or_else(|| {
                            panic!("component store on undefined local `{name}`")
                        });
                        cur.set_component(axis, val.as_f32());
                        frame.locals.insert(name.to_string(), cur);
                    }
                    // Buffer element component: `grad_state[idx].x = ...`.
                    ExprNode::Index { base: bbase, index } => {
                        let buf = ident_name(bbase)
                            .expect("component store into non-buffer element");
                        let i = self.eval(index, frame).as_index();
                        let mut cur = self.buffers.load(buf, i);
                        cur.set_component(axis, val.as_f32());
                        self.buffers.store(buf, i, cur);
                    }
                    other => panic!("unsupported component-store base {other:?}"),
                }
            }
            other => panic!("unsupported assignment target {other:?}"),
        }
    }

    fn eval(&mut self, expr: &Expr, frame: &mut Frame) -> Value {
        match expr.node() {
            ExprNode::Literal(lit) => eval_literal(lit),
            ExprNode::Ident(name) => frame
                .get(name)
                .or_else(|| self.ctx.builtins.get(name).copied())
                .unwrap_or_else(|| panic!("undefined identifier `{name}`")),
            ExprNode::Field { base, field } => {
                // `struct.field` (uniform) vs `vec.x` (component).
                if let Some(sname) = ident_name(base) {
                    if let Some(s) = self.ctx.structs.get(sname) {
                        return *s
                            .get(field)
                            .unwrap_or_else(|| panic!("unknown field `{sname}.{field}`"));
                    }
                }
                let axis = axis_of(field)
                    .unwrap_or_else(|| panic!("field `.{field}` on non-struct value"));
                Value::F32(self.eval(base, frame).component(axis))
            }
            ExprNode::Index { base, index } => {
                let buf = ident_name(base).unwrap_or_else(|| panic!("index of non-buffer base"));
                let idx = self.eval(index, frame).as_index();
                self.buffers.load(buf, idx)
            }
            ExprNode::Unary { op, expr: inner } => {
                let v = self.eval(inner, frame);
                match op {
                    UnaryOp::Negate => match v {
                        Value::F32(x) => Value::F32(-x),
                        Value::I32(x) => Value::I32(-x),
                        Value::Vec2(a) => Value::Vec2([-a[0], -a[1]]),
                        Value::Vec3(a) => Value::Vec3([-a[0], -a[1], -a[2]]),
                        Value::Vec4(a) => Value::Vec4([-a[0], -a[1], -a[2], -a[3]]),
                        other => panic!("cannot negate {other:?}"),
                    },
                    UnaryOp::Not => Value::Bool(!v.as_bool()),
                    UnaryOp::AddressOf | UnaryOp::Deref => {
                        panic!("pointer op `{op}` only valid inside intrinsics")
                    }
                }
            }
            ExprNode::Binary { left, op, right } => {
                // Short-circuit logical operators.
                match op {
                    BinaryOp::And => {
                        return Value::Bool(
                            self.eval(left, frame).as_bool() && self.eval(right, frame).as_bool(),
                        )
                    }
                    BinaryOp::Or => {
                        return Value::Bool(
                            self.eval(left, frame).as_bool() || self.eval(right, frame).as_bool(),
                        )
                    }
                    _ => {}
                }
                let l = self.eval(left, frame);
                let r = self.eval(right, frame);
                apply_binop(l, *op, r)
            }
            ExprNode::Call { callee, args } => {
                let name = ident_name(callee).unwrap_or_else(|| panic!("call of non-ident callee"));
                self.call_intrinsic(name, args, frame)
            }
        }
    }

    fn call_intrinsic(&mut self, name: &str, args: &[Expr], frame: &mut Frame) -> Value {
        // Intrinsics whose first argument is a *pointer* (`&buffer[idx]`) or buffer,
        // not a value — these must inspect the AST, so handle them before the eager
        // argument evaluation below.
        match name {
            "arrayLength" => {
                let buf = addr_of_ident(&args[0]).expect("arrayLength(&buffer)");
                return Value::U32(self.buffers.len(buf) as u32);
            }
            // Barriers are meaningless at per-index (single-invocation) granularity.
            "workgroupBarrier" | "storageBarrier" => return Value::U32(0),
            "atomicLoad" => {
                let (buf, idx) = self.atomic_target(&args[0], frame);
                return self.buffers.load(&buf, idx);
            }
            "atomicStore" => {
                let (buf, idx) = self.atomic_target(&args[0], frame);
                let v = self.eval(&args[1], frame);
                self.buffers.store(&buf, idx, v);
                return Value::U32(0);
            }
            "atomicAdd" => return self.atomic_rmw(&args[0], &args[1], frame, |old, v| old + v),
            "atomicMin" => return self.atomic_rmw(&args[0], &args[1], frame, |old, v| old.min(v)),
            "atomicMax" => return self.atomic_rmw(&args[0], &args[1], frame, |old, v| old.max(v)),
            _ => {}
        }

        // All remaining intrinsics take value arguments — evaluate eagerly.
        let a: Vec<Value> = args.iter().map(|e| self.eval(e, frame)).collect();
        match name {
            // ── elementwise scalar math ──
            "sqrt" => Value::F32(a[0].as_f32().sqrt()),
            "abs" => Value::F32(a[0].as_f32().abs()),
            "floor" => Value::F32(a[0].as_f32().floor()),
            "ceil" => Value::F32(a[0].as_f32().ceil()),
            "round" => Value::F32(round_half_even(a[0].as_f32())),
            "sign" => Value::F32(sign_f32(a[0].as_f32())),
            "exp2" => Value::F32(a[0].as_f32().exp2()),
            "log2" => Value::F32(a[0].as_f32().log2()),
            "pow" => Value::F32(a[0].as_f32().powf(a[1].as_f32())),
            "fma" => Value::F32(a[0].as_f32().mul_add(a[1].as_f32(), a[2].as_f32())),
            "min" => Value::F32(a[0].as_f32().min(a[1].as_f32())),
            "max" => Value::F32(a[0].as_f32().max(a[1].as_f32())),
            "clamp" => Value::F32(a[0].as_f32().max(a[1].as_f32()).min(a[2].as_f32())),
            "mix" => {
                let (x, y, t) = (a[0].as_f32(), a[1].as_f32(), a[2].as_f32());
                Value::F32(x * (1.0 - t) + y * t)
            }
            "smoothstep" => {
                let (e0, e1, x) = (a[0].as_f32(), a[1].as_f32(), a[2].as_f32());
                let t = (((x - e0) / (e1 - e0)).max(0.0)).min(1.0);
                Value::F32(t * t * (3.0 - 2.0 * t))
            }
            // select(false_value, true_value, condition)
            "select" => {
                if a[2].as_bool() {
                    a[1]
                } else {
                    a[0]
                }
            }
            // ── vector ops ──
            "dot" => Value::F32(dot(a[0], a[1])),
            "length" => Value::F32(dot(a[0], a[0]).sqrt()),
            "distance" => {
                let d = sub_vec(a[0], a[1]);
                Value::F32(dot(d, d).sqrt())
            }
            // ── vector constructors (splat when 1 arg) ──
            "vec2<f32>" => {
                if a.len() == 1 {
                    let s = a[0].as_f32();
                    Value::Vec2([s, s])
                } else {
                    Value::Vec2([a[0].as_f32(), a[1].as_f32()])
                }
            }
            "vec3<f32>" => {
                if a.len() == 1 {
                    let s = a[0].as_f32();
                    Value::Vec3([s, s, s])
                } else {
                    Value::Vec3([a[0].as_f32(), a[1].as_f32(), a[2].as_f32()])
                }
            }
            "vec4<f32>" => {
                if a.len() == 1 {
                    let s = a[0].as_f32();
                    Value::Vec4([s, s, s, s])
                } else {
                    Value::Vec4([a[0].as_f32(), a[1].as_f32(), a[2].as_f32(), a[3].as_f32()])
                }
            }
            // ── type constructors / casts ──
            "f32" => Value::F32(a[0].as_f32()),
            "u32" => Value::U32(a[0].as_u32()),
            "i32" => Value::I32(a[0].as_i32()),
            "bool" => Value::Bool(a[0].as_bool()),
            other if other.starts_with("bitcast<") => {
                let target = &other["bitcast<".len()..other.len() - 1];
                bitcast(target, a[0])
            }
            other => panic!("unsupported intrinsic `{other}` (args: {})", a.len()),
        }
    }

    /// Resolve an `atomic*` first argument of the form `&buffer[idx]` to a
    /// `(buffer, index)` pair.
    fn atomic_target(&mut self, arg: &Expr, frame: &mut Frame) -> (String, usize) {
        let inner = match arg.node() {
            ExprNode::Unary {
                op: UnaryOp::AddressOf,
                expr,
            } => expr,
            _ => panic!("atomic pointer must be `&buffer[idx]`"),
        };
        match inner.node() {
            ExprNode::Index { base, index } => {
                let buf = ident_name(base).expect("atomic on non-buffer").to_string();
                let idx = self.eval(index, frame).as_index();
                (buf, idx)
            }
            other => panic!("unsupported atomic target {other:?}"),
        }
    }

    fn atomic_rmw(
        &mut self,
        ptr: &Expr,
        value: &Expr,
        frame: &mut Frame,
        f: impl Fn(f32, f32) -> f32,
    ) -> Value {
        let (buf, idx) = self.atomic_target(ptr, frame);
        let v = self.eval(value, frame);
        let old = self.buffers.load(&buf, idx);
        // Phase 1 is single-threaded: a plain read-modify-write is correct.
        let new = match old {
            Value::F32(o) => Value::F32(f(o, v.as_f32())),
            Value::U32(o) => Value::U32(f(o as f32, v.as_f32()) as u32),
            other => panic!("atomic rmw on {other:?}"),
        };
        self.buffers.store(&buf, idx, new);
        old
    }
}

// ── free helpers ────────────────────────────────────────────────────────────

fn ident_name(expr: &Expr) -> Option<&str> {
    match expr.node() {
        ExprNode::Ident(name) => Some(name),
        _ => None,
    }
}

fn addr_of_ident(expr: &Expr) -> Option<&str> {
    match expr.node() {
        ExprNode::Unary {
            op: UnaryOp::AddressOf,
            expr,
        } => ident_name(expr),
        _ => None,
    }
}

fn axis_of(field: &str) -> Option<usize> {
    match field {
        "x" => Some(0),
        "y" => Some(1),
        "z" => Some(2),
        "w" => Some(3),
        _ => None,
    }
}

fn eval_literal(lit: &Literal) -> Value {
    match lit {
        Literal::Bool(b) => Value::Bool(*b),
        Literal::Int(v) => Value::I32(*v),
        Literal::Uint(v) => Value::U32(*v),
        Literal::Float(s) => Value::F32(parse_float_literal(s)),
    }
}

/// Parse the float literal *string* the AST stores (`format_f32_literal`):
/// normal decimals like `"1.0"`, scientific `"1e-3"`, and the sentinels
/// `"nan()"`, `"inf()"`, `"-inf()"`.
fn parse_float_literal(s: &str) -> f32 {
    match s {
        "nan()" => f32::NAN,
        "inf()" => f32::INFINITY,
        "-inf()" => f32::NEG_INFINITY,
        _ => s
            .parse::<f32>()
            .unwrap_or_else(|_| panic!("unparseable float literal `{s}`")),
    }
}

fn dot(a: Value, b: Value) -> f32 {
    match (a, b) {
        (Value::F32(x), Value::F32(y)) => x * y,
        (Value::Vec2(x), Value::Vec2(y)) => x[0] * y[0] + x[1] * y[1],
        (Value::Vec3(x), Value::Vec3(y)) => x[0] * y[0] + x[1] * y[1] + x[2] * y[2],
        (Value::Vec4(x), Value::Vec4(y)) => {
            x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
        }
        _ => panic!("dot of mismatched/non-vector operands {a:?}, {b:?}"),
    }
}

fn sub_vec(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Vec2(x), Value::Vec2(y)) => Value::Vec2([x[0] - y[0], x[1] - y[1]]),
        (Value::Vec3(x), Value::Vec3(y)) => {
            Value::Vec3([x[0] - y[0], x[1] - y[1], x[2] - y[2]])
        }
        (Value::Vec4(x), Value::Vec4(y)) => {
            Value::Vec4([x[0] - y[0], x[1] - y[1], x[2] - y[2], x[3] - y[3]])
        }
        (Value::F32(x), Value::F32(y)) => Value::F32(x - y),
        _ => panic!("vector sub of mismatched operands"),
    }
}

fn bitcast(target: &str, v: Value) -> Value {
    match target {
        "f32" => Value::F32(f32::from_bits(v.as_u32())),
        "u32" => Value::U32(v.as_f32().to_bits()),
        "i32" => Value::I32(v.as_f32().to_bits() as i32),
        other => panic!("unsupported bitcast target `{other}`"),
    }
}

fn sign_f32(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x // preserves +0/-0/NaN like WGSL
    }
}

/// WGSL `round` is round-half-to-even.
fn round_half_even(x: f32) -> f32 {
    let r = x.round(); // round-half-away-from-zero
    if (x - x.floor() - 0.5).abs() < f32::EPSILON {
        // exactly .5 → round to even
        let down = x.floor();
        if (down as i64) % 2 == 0 {
            down
        } else {
            down + 1.0
        }
    } else {
        r
    }
}

fn apply_binop(l: Value, op: BinaryOp, r: Value) -> Value {
    use BinaryOp::*;
    // Integer-preserving ops when both operands are integers.
    let both_uint = matches!((l, r), (Value::U32(_), Value::U32(_)));
    let both_int = matches!(
        (l, r),
        (Value::I32(_), Value::I32(_)) | (Value::U32(_), Value::U32(_))
    );

    match op {
        Add | Sub | Mul | Div | Modulo => {
            // Vector arithmetic (component-wise) when either side is a vector.
            if is_vector(l) || is_vector(r) {
                return vector_arith(l, op, r);
            }
            if both_uint {
                let a = l.as_u32();
                let b = r.as_u32();
                return Value::U32(match op {
                    Add => a.wrapping_add(b),
                    Sub => a.wrapping_sub(b),
                    Mul => a.wrapping_mul(b),
                    Div => a / b,
                    Modulo => a % b,
                    _ => unreachable!(),
                });
            }
            if both_int {
                let a = l.as_i32();
                let b = r.as_i32();
                return Value::I32(match op {
                    Add => a.wrapping_add(b),
                    Sub => a.wrapping_sub(b),
                    Mul => a.wrapping_mul(b),
                    Div => a / b,
                    Modulo => a % b,
                    _ => unreachable!(),
                });
            }
            let a = l.as_f32();
            let b = r.as_f32();
            Value::F32(match op {
                Add => a + b,
                Sub => a - b,
                Mul => a * b,
                Div => a / b,
                Modulo => a % b,
                _ => unreachable!(),
            })
        }
        Less | LessEq | Greater | GreaterEq => {
            let a = l.as_f32();
            let b = r.as_f32();
            Value::Bool(match op {
                Less => a < b,
                LessEq => a <= b,
                Greater => a > b,
                GreaterEq => a >= b,
                _ => unreachable!(),
            })
        }
        Equal | NotEqual => {
            let eq = if both_int {
                l.as_i32() == r.as_i32()
            } else {
                l.as_f32() == r.as_f32()
            };
            Value::Bool(if matches!(op, Equal) { eq } else { !eq })
        }
        ShiftLeft => Value::U32(l.as_u32() << r.as_u32()),
        ShiftRight => Value::U32(l.as_u32() >> r.as_u32()),
        BitwiseAnd => Value::U32(l.as_u32() & r.as_u32()),
        BitwiseOr => Value::U32(l.as_u32() | r.as_u32()),
        And | Or => unreachable!("logical ops are short-circuited in eval"),
    }
}

fn is_vector(v: Value) -> bool {
    matches!(v, Value::Vec2(_) | Value::Vec3(_) | Value::Vec4(_))
}

/// Component-wise vector arithmetic, including vector·scalar broadcast.
fn vector_arith(l: Value, op: BinaryOp, r: Value) -> Value {
    let f = |a: f32, b: f32| -> f32 {
        match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Modulo => a % b,
            _ => unreachable!(),
        }
    };
    // Broadcast a scalar against a vector of the matching width.
    let width = match (l, r) {
        (Value::Vec2(_), _) | (_, Value::Vec2(_)) => 2,
        (Value::Vec3(_), _) | (_, Value::Vec3(_)) => 3,
        (Value::Vec4(_), _) | (_, Value::Vec4(_)) => 4,
        _ => unreachable!(),
    };
    let comp = |v: Value, i: usize| -> f32 {
        match v {
            Value::F32(s) => s,
            _ => v.component(i),
        }
    };
    let mut out = [0.0f32; 4];
    for i in 0..width {
        out[i] = f(comp(l, i), comp(r, i));
    }
    match width {
        2 => Value::Vec2([out[0], out[1]]),
        3 => Value::Vec3([out[0], out[1], out[2]]),
        _ => Value::Vec4([out[0], out[1], out[2], out[3]]),
    }
}

fn apply_assign_op(cur: Value, op: AssignOp, rhs: Value) -> Value {
    let binop = match op {
        AssignOp::Add => BinaryOp::Add,
        AssignOp::Sub => BinaryOp::Sub,
        AssignOp::Mul => BinaryOp::Mul,
        AssignOp::Div => BinaryOp::Div,
        AssignOp::Modulo => BinaryOp::Modulo,
        AssignOp::ShiftRight => BinaryOp::ShiftRight,
        AssignOp::ShiftLeft => BinaryOp::ShiftLeft,
        AssignOp::BitwiseAnd => BinaryOp::BitwiseAnd,
        AssignOp::BitwiseOr => BinaryOp::BitwiseOr,
    };
    apply_binop(cur, binop, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd2_ir::ast::stmt::Block;
    use cfd2_ir::ast::Expr;

    fn run(stmts: &[Stmt], buffers: &mut Buffers, ctx: &Ctx, frame: &mut Frame) {
        Interpreter::new(buffers, ctx).run(stmts, frame);
    }

    fn eval_expr(e: &Expr, frame: &mut Frame) -> Value {
        let mut buffers = Buffers::new();
        let ctx = Ctx::new();
        Interpreter::new(&mut buffers, &ctx).eval(e, frame)
    }

    #[test]
    fn arithmetic_precedence_and_floats() {
        // a + b * c with a=1, b=2, c=3 → 7
        let e = Expr::ident("a") + Expr::ident("b") * Expr::ident("c");
        let mut frame = Frame::new()
            .with_local("a", Value::F32(1.0))
            .with_local("b", Value::F32(2.0))
            .with_local("c", Value::F32(3.0));
        assert_eq!(eval_expr(&e, &mut frame), Value::F32(7.0));
    }

    #[test]
    fn integer_division_stays_integer() {
        // 7u / 2u == 3u (not 3.5)
        let e = Expr::from(7u32) / Expr::from(2u32);
        let mut frame = Frame::new();
        assert_eq!(eval_expr(&e, &mut frame), Value::U32(3));
    }

    #[test]
    fn vec_construct_component_and_dot() {
        // dot(vec2(a,b), vec2(1,0)) == a
        let v = Expr::call_named("vec2<f32>", vec![Expr::ident("a"), Expr::ident("b")]);
        let ex = Expr::call_named("vec2<f32>", vec![1.0.into(), 0.0.into()]);
        let e = Expr::call_named("dot", vec![v.clone(), ex]);
        let mut frame = Frame::new()
            .with_local("a", Value::F32(4.0))
            .with_local("b", Value::F32(2.0));
        assert_eq!(eval_expr(&e, &mut frame), Value::F32(4.0));
        // Component access: vec2(a,b).y == b
        let cy = v.field("y");
        assert_eq!(eval_expr(&cy, &mut frame), Value::F32(2.0));
    }

    #[test]
    fn builtin_math_and_select() {
        let mut frame = Frame::new().with_local("x", Value::F32(-9.0));
        let e = Expr::call_named("sqrt", vec![Expr::call_named("abs", vec![Expr::ident("x")])]);
        assert_eq!(eval_expr(&e, &mut frame), Value::F32(3.0));
        // select(false_val=10, true_val=20, cond = x < 0) → 20
        let cond = Expr::ident("x").lt(0.0);
        let sel = Expr::call_named("select", vec![10.0.into(), 20.0.into(), cond]);
        assert_eq!(eval_expr(&sel, &mut frame), Value::F32(20.0));
    }

    #[test]
    fn clamp_min_max() {
        let mut frame = Frame::new();
        let e = Expr::call_named("clamp", vec![5.0.into(), 0.0.into(), 3.0.into()]);
        assert_eq!(eval_expr(&e, &mut frame), Value::F32(3.0));
    }

    #[test]
    fn struct_uniform_field_access() {
        let ctx = Ctx::new().with_constant("constants", "dt", Value::F32(0.25));
        let mut buffers = Buffers::new();
        let mut frame = Frame::new();
        let e = Expr::ident("constants").field("dt");
        let v = Interpreter::new(&mut buffers, &ctx).eval(&e, &mut frame);
        assert_eq!(v, Value::F32(0.25));
    }

    #[test]
    fn buffer_read_write_and_array_length() {
        let mut buffers = Buffers::new();
        buffers.insert_f32("a", vec![10.0, 20.0, 30.0]);
        buffers.insert_f32("out", vec![0.0; 3]);
        let ctx = Ctx::new();
        // out[i] = a[i] * 2 + 1   for i in 0..arrayLength(&a)
        let i = Expr::ident("i");
        let body = Block::new(vec![Stmt::Assign {
            target: Expr::ident("out").index(i.clone()),
            value: Expr::ident("a").index(i.clone()) * 2.0 + 1.0,
        }]);
        let loop_stmt = Stmt::For {
            init: ForInit::Var {
                name: "i".to_string(),
                ty: None,
                expr: 0u32.into(),
            },
            cond: Expr::ident("i").lt(Expr::call_named(
                "arrayLength",
                vec![Expr::ident("a").addr_of()],
            )),
            step: ForStep::Increment(Expr::ident("i")),
            body,
        };
        let mut frame = Frame::new();
        run(&[loop_stmt], &mut buffers, &ctx, &mut frame);
        assert_eq!(buffers.f32_slice("out"), &[21.0, 41.0, 61.0]);
    }

    #[test]
    fn if_else_and_accumulation_loop() {
        // sum of even numbers in [0, 10) using if + for + accumulation
        let mut buffers = Buffers::new();
        let ctx = Ctx::new();
        let stmts = vec![
            Stmt::Var {
                name: "sum".to_string(),
                ty: None,
                expr: Some(0.0.into()),
            },
            Stmt::For {
                init: ForInit::Var {
                    name: "k".to_string(),
                    ty: None,
                    expr: 0u32.into(),
                },
                cond: Expr::ident("k").lt(10u32),
                step: ForStep::Increment(Expr::ident("k")),
                body: Block::new(vec![Stmt::If {
                    cond: Expr::ident("k").modulo(2u32).eq(0u32),
                    then_block: Block::new(vec![Stmt::AssignOp {
                        target: Expr::ident("sum"),
                        op: AssignOp::Add,
                        value: Expr::call_named("f32", vec![Expr::ident("k")]),
                    }]),
                    else_block: None,
                }]),
            },
        ];
        let mut frame = Frame::new();
        run(&stmts, &mut buffers, &ctx, &mut frame);
        // 0+2+4+6+8 = 20
        assert_eq!(frame.get("sum"), Some(Value::F32(20.0)));
    }

    #[test]
    fn atomic_add_serial() {
        let mut buffers = Buffers::new();
        buffers.insert_f32("acc", vec![5.0]);
        let ctx = Ctx::new();
        // old = atomicAdd(&acc[0], 3.0); store old into a local for inspection.
        let stmts = vec![Stmt::Let {
            name: "old".to_string(),
            ty: None,
            expr: Expr::call_named(
                "atomicAdd",
                vec![Expr::ident("acc").index(0u32).addr_of(), 3.0.into()],
            ),
        }];
        let mut frame = Frame::new();
        run(&stmts, &mut buffers, &ctx, &mut frame);
        assert_eq!(frame.get("old"), Some(Value::F32(5.0)));
        assert_eq!(buffers.f32_slice("acc"), &[8.0]);
    }

    #[test]
    fn vector_scalar_broadcast() {
        // vec2(2,4) * 0.5 == vec2(1,2)
        let v = Expr::call_named("vec2<f32>", vec![2.0.into(), 4.0.into()]);
        let e = v * 0.5;
        let mut frame = Frame::new();
        assert_eq!(eval_expr(&e, &mut frame), Value::Vec2([1.0, 2.0]));
    }

    #[test]
    fn component_store_into_local_vector() {
        let mut buffers = Buffers::new();
        let ctx = Ctx::new();
        let stmts = vec![
            Stmt::Var {
                name: "n".to_string(),
                ty: Some(Type::vec2_f32()),
                expr: Some(Expr::call_named("vec2<f32>", vec![0.0.into(), 0.0.into()])),
            },
            Stmt::Assign {
                target: Expr::ident("n").field("x"),
                value: 7.0.into(),
            },
            Stmt::Assign {
                target: Expr::ident("n").field("y"),
                value: 9.0.into(),
            },
        ];
        let mut frame = Frame::new();
        run(&stmts, &mut buffers, &ctx, &mut frame);
        assert_eq!(frame.get("n"), Some(Value::Vec2([7.0, 9.0])));
    }
}
