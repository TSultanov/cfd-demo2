use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    F32,
    U32,
    I32,
    Bool,
    Vec2(Box<Type>),
    Vec3(Box<Type>),
    Vec4(Box<Type>),
    Array(Box<Type>),
    /// Fixed-size array, e.g. `array<f32, 64>`.
    SizedArray(Box<Type>, u32),
    /// Pointer type, e.g. `ptr<function, f32>`.
    Ptr(Box<Type>, AddressSpace),
    /// Atomic type, e.g. `atomic<u32>`.
    Atomic(Box<Type>),
    Custom(String),
}

impl Type {
    pub fn vec2_f32() -> Self {
        Type::Vec2(Box::new(Type::F32))
    }

    pub fn vec3_u32() -> Self {
        Type::Vec3(Box::new(Type::U32))
    }

    pub fn array(inner: Type) -> Self {
        Type::Array(Box::new(inner))
    }

    pub fn sized_array(inner: Type, size: u32) -> Self {
        Type::SizedArray(Box::new(inner), size)
    }

    pub fn ptr(inner: Type, address_space: AddressSpace) -> Self {
        Type::Ptr(Box::new(inner), address_space)
    }

    pub fn atomic(inner: Type) -> Self {
        Type::Atomic(Box::new(inner))
    }
}

/// WGSL address space for pointer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressSpace {
    Function,
    Private,
    Workgroup,
    Storage,
    Uniform,
}

impl fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AddressSpace::Function => write!(f, "function"),
            AddressSpace::Private => write!(f, "private"),
            AddressSpace::Workgroup => write!(f, "workgroup"),
            AddressSpace::Storage => write!(f, "storage"),
            AddressSpace::Uniform => write!(f, "uniform"),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::F32 => write!(f, "f32"),
            Type::U32 => write!(f, "u32"),
            Type::I32 => write!(f, "i32"),
            Type::Bool => write!(f, "bool"),
            Type::Vec2(inner) => write!(f, "vec2<{}>", inner),
            Type::Vec3(inner) => write!(f, "vec3<{}>", inner),
            Type::Vec4(inner) => write!(f, "vec4<{}>", inner),
            Type::Array(inner) => write!(f, "array<{}>", inner),
            Type::SizedArray(inner, size) => write!(f, "array<{}, {}>", inner, size),
            Type::Ptr(inner, space) => write!(f, "ptr<{}, {}>", space, inner),
            Type::Atomic(inner) => write!(f, "atomic<{}>", inner),
            Type::Custom(name) => write!(f, "{}", name),
        }
    }
}

// ── Render context for WGSL emission ───────────────────────────────────

/// A simple indentation-aware WGSL line writer.
///
/// This is the common rendering infrastructure used by both `Stmt::render()`
/// (in `cfd2_ir::ast`) and the module/function-level renderer in `cfd2_codegen`.
pub struct RenderContext<'a> {
    pub(crate) output: &'a mut String,
    pub(crate) indent: usize,
    pub(crate) pending_line: String,
}

impl<'a> RenderContext<'a> {
    pub fn new(output: &'a mut String) -> Self {
        Self {
            output,
            indent: 0,
            pending_line: String::new(),
        }
    }

    pub fn indent(&mut self) {
        self.indent += 1;
    }

    pub fn dedent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    pub fn line(&mut self, text: &str) {
        if !self.pending_line.is_empty() {
            self.flush_line();
        }
        for _ in 0..self.indent {
            self.output.push_str("    ");
        }
        self.output.push_str(text);
        self.output.push('\n');
    }

    pub fn blank_line(&mut self) {
        if !self.pending_line.is_empty() {
            self.flush_line();
        }
        self.output.push('\n');
    }

    pub fn inline_attr(&mut self, text: &str) {
        if !self.pending_line.is_empty() {
            self.pending_line.push(' ');
        }
        self.pending_line.push_str(text);
    }

    pub fn space(&mut self) {
        if !self.pending_line.is_empty() {
            self.pending_line.push(' ');
        }
    }

    pub fn flush_line(&mut self) {
        if self.pending_line.is_empty() {
            return;
        }
        for _ in 0..self.indent {
            self.output.push_str("    ");
        }
        self.output.push_str(&self.pending_line);
        self.output.push('\n');
        self.pending_line.clear();
    }
}
