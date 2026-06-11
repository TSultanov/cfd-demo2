//! GMRES Classical Gram-Schmidt (CGS) kernel generator implementation.
//!
//! This module contains the implementation of generate_gmres_cgs() which was
//! ported from the handwritten src/solver/gpu/shaders/gmres_cgs.wgsl file.

use crate::solver::codegen::kernel_wgsl::KernelWgsl;
use crate::solver::codegen::wgsl_ast::*;
use crate::solver::codegen::wgsl_dsl::*;

/// Generate the gmres_cgs.wgsl kernel with 3 entry points:
/// - calc_dots_cgs: vec4 vectorized dot product computation
/// - reduce_dots_cgs: reduction of partial dot products
/// - update_w_cgs: update w vector
pub fn generate_gmres_cgs() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Comment("GMRES Classical Gram-Schmidt (CGS)".into()));

    // ── Structs ─────────────────────────────────────────────────────────────

    m.push(Item::Struct(StructDef::new(
        "Params",
        vec![
            StructField::new("n", Type::U32),
            StructField::new("num_cells", Type::U32),
            StructField::new("num_iters", Type::U32),
            StructField::new("omega", Type::F32),
            StructField::new("dispatch_x", Type::U32),
            StructField::new("max_restart", Type::U32),
            StructField::new("column_offset", Type::U32),
            StructField::new("pad3", Type::U32),
        ],
    )));

    // ── Constants ───────────────────────────────────────────────────────────

    m.push(Item::Const {
        name: "WORKGROUP_SIZE".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(64),
    });

    m.push(Item::Const {
        name: "SCALAR_STOP".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(8),
    });

    // ── Global variables ────────────────────────────────────────────────────

    // @group(0) @binding(0) var<uniform> params: Params;
    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("Params".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));

    // @group(0) @binding(1) var<storage, read> b_basis: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "b_basis",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));

    // @group(0) @binding(2) var<storage, read_write> b_w: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "b_w",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));

    // @group(0) @binding(3) var<storage, read_write> b_dot_partial: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "b_dot_partial",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));

    // @group(0) @binding(4) var<storage, read_write> b_hessenberg: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "b_hessenberg",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(4)],
    )));

    // @group(0) @binding(5) var<storage, read> scalars: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "scalars",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(5)],
    )));

    // var<workgroup> sdata: array<f32, 64>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "sdata",
        Type::sized_array(Type::F32, 64),
        StorageClass::Workgroup,
        None,
        vec![],
    )));

    // var<workgroup> sdata_vec4: array<vec4<f32>, 64>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "sdata_vec4",
        Type::sized_array(Type::Vec4(Box::new(Type::F32)), 64),
        StorageClass::Workgroup,
        None,
        vec![],
    )));

    // ── Entry point: calc_dots_cgs ──────────────────────────────────────────
    {
        let body = block(vec![
            // if (scalars[SCALAR_STOP] > 0.5) { return; }
            if_block_expr(
                Expr::ident("scalars")
                    .index(Expr::ident("SCALAR_STOP"))
                    .gt(Expr::lit_f32(0.5)),
                block(vec![return_void()]),
                None,
            ),
            // let j = params.num_iters;
            let_expr("j", Expr::ident("params").field("num_iters")),
            // let n = params.n;
            let_expr("n", Expr::ident("params").field("n")),
            // let num_groups_n = (n + (WORKGROUP_SIZE - 1u)) / WORKGROUP_SIZE;
            let_expr(
                "num_groups_n",
                (Expr::ident("n") + (Expr::ident("WORKGROUP_SIZE") - Expr::lit_u32(1)))
                    / Expr::ident("WORKGROUP_SIZE"),
            ),
            // let stride_x = num_workgroups.x * WORKGROUP_SIZE;
            let_expr(
                "stride_x",
                Expr::ident("num_workgroups").field("x") * Expr::ident("WORKGROUP_SIZE"),
            ),
            // let idx = global_id.y * stride_x + global_id.x;
            let_expr(
                "idx",
                Expr::ident("global_id").field("y") * Expr::ident("stride_x")
                    + Expr::ident("global_id").field("x"),
            ),
            // let group_flat = group_id.y * num_workgroups.x + group_id.x;
            let_expr(
                "group_flat",
                Expr::ident("group_id").field("y") * Expr::ident("num_workgroups").field("x")
                    + Expr::ident("group_id").field("x"),
            ),
            // if (group_flat >= num_groups_n) { return; }
            if_block_expr(
                Expr::ident("group_flat").ge(Expr::ident("num_groups_n")),
                block(vec![return_void()]),
                None,
            ),
            // let stride_bytes = (n * 4u + 255u) & 4294967040u;
            let_expr(
                "stride_bytes",
                (Expr::ident("n") * Expr::lit_u32(4) + Expr::lit_u32(255))
                    .bitwise_and(Expr::lit_u32(4294967040)),
            ),
            // let stride_words = stride_bytes / 4u;
            let_expr(
                "stride_words",
                Expr::ident("stride_bytes") / Expr::lit_u32(4),
            ),
            // var w_val = 0.0;
            var_expr("w_val", Expr::lit_f32(0.0)),
            // if (idx < n) { w_val = b_w[idx]; }
            if_block_expr(
                Expr::ident("idx").lt(Expr::ident("n")),
                block(vec![assign_expr(
                    Expr::ident("w_val"),
                    Expr::ident("b_w").index(Expr::ident("idx")),
                )]),
                None,
            ),
            // for (var i = 0u; i <= j; i += 4u) { ... }
            for_loop_expr(
                ForInit::Var {
                    name: "i".into(),
                    ty: Some(Type::U32),
                    expr: Expr::lit_u32(0),
                },
                Expr::ident("i").le(Expr::ident("j")),
                for_step_assign_op_expr(AssignOp::Add, Expr::ident("i"), Expr::lit_u32(4)),
                block(vec![
                    // var v = vec4<f32>(0.0);
                    var_expr("v", vec4_f32_splat(0.0)),
                    // if (idx < n) { ... }
                    if_block_expr(
                        Expr::ident("idx").lt(Expr::ident("n")),
                        block(vec![
                            // if (i <= j) { v.x = b_basis[i * stride_words + idx]; }
                            if_block_expr(
                                Expr::ident("i").le(Expr::ident("j")),
                                block(vec![assign_expr(
                                    Expr::ident("v").field("x"),
                                    Expr::ident("b_basis").index(
                                        Expr::ident("i") * Expr::ident("stride_words")
                                            + Expr::ident("idx"),
                                    ),
                                )]),
                                None,
                            ),
                            // if (i + 1u <= j) { v.y = b_basis[(i + 1u) * stride_words + idx]; }
                            if_block_expr(
                                (Expr::ident("i") + Expr::lit_u32(1)).le(Expr::ident("j")),
                                block(vec![assign_expr(
                                    Expr::ident("v").field("y"),
                                    Expr::ident("b_basis").index(
                                        (Expr::ident("i") + Expr::lit_u32(1))
                                            * Expr::ident("stride_words")
                                            + Expr::ident("idx"),
                                    ),
                                )]),
                                None,
                            ),
                            // if (i + 2u <= j) { v.z = b_basis[(i + 2u) * stride_words + idx]; }
                            if_block_expr(
                                (Expr::ident("i") + Expr::lit_u32(2)).le(Expr::ident("j")),
                                block(vec![assign_expr(
                                    Expr::ident("v").field("z"),
                                    Expr::ident("b_basis").index(
                                        (Expr::ident("i") + Expr::lit_u32(2))
                                            * Expr::ident("stride_words")
                                            + Expr::ident("idx"),
                                    ),
                                )]),
                                None,
                            ),
                            // if (i + 3u <= j) { v.w = b_basis[(i + 3u) * stride_words + idx]; }
                            if_block_expr(
                                (Expr::ident("i") + Expr::lit_u32(3)).le(Expr::ident("j")),
                                block(vec![assign_expr(
                                    Expr::ident("v").field("w"),
                                    Expr::ident("b_basis").index(
                                        (Expr::ident("i") + Expr::lit_u32(3))
                                            * Expr::ident("stride_words")
                                            + Expr::ident("idx"),
                                    ),
                                )]),
                                None,
                            ),
                        ]),
                        None,
                    ),
                    // let prod = v * w_val;
                    let_expr("prod", Expr::ident("v") * Expr::ident("w_val")),
                    // sdata_vec4[local_id.x] = prod;
                    assign_expr(
                        Expr::ident("sdata_vec4").index(Expr::ident("local_id").field("x")),
                        Expr::ident("prod"),
                    ),
                    // workgroupBarrier();
                    workgroup_barrier(),
                    // Reduction: stride 32
                    if_block_expr(
                        Expr::ident("local_id").field("x").lt(Expr::lit_u32(32)),
                        block(vec![assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("sdata_vec4").index(Expr::ident("local_id").field("x")),
                            Expr::ident("sdata_vec4")
                                .index(Expr::ident("local_id").field("x") + Expr::lit_u32(32)),
                        )]),
                        None,
                    ),
                    workgroup_barrier(),
                    // Reduction: stride 16
                    if_block_expr(
                        Expr::ident("local_id").field("x").lt(Expr::lit_u32(16)),
                        block(vec![assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("sdata_vec4").index(Expr::ident("local_id").field("x")),
                            Expr::ident("sdata_vec4")
                                .index(Expr::ident("local_id").field("x") + Expr::lit_u32(16)),
                        )]),
                        None,
                    ),
                    workgroup_barrier(),
                    // Reduction: stride 8
                    if_block_expr(
                        Expr::ident("local_id").field("x").lt(Expr::lit_u32(8)),
                        block(vec![assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("sdata_vec4").index(Expr::ident("local_id").field("x")),
                            Expr::ident("sdata_vec4")
                                .index(Expr::ident("local_id").field("x") + Expr::lit_u32(8)),
                        )]),
                        None,
                    ),
                    workgroup_barrier(),
                    // Reduction: stride 4
                    if_block_expr(
                        Expr::ident("local_id").field("x").lt(Expr::lit_u32(4)),
                        block(vec![assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("sdata_vec4").index(Expr::ident("local_id").field("x")),
                            Expr::ident("sdata_vec4")
                                .index(Expr::ident("local_id").field("x") + Expr::lit_u32(4)),
                        )]),
                        None,
                    ),
                    workgroup_barrier(),
                    // Reduction: stride 2
                    if_block_expr(
                        Expr::ident("local_id").field("x").lt(Expr::lit_u32(2)),
                        block(vec![assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("sdata_vec4").index(Expr::ident("local_id").field("x")),
                            Expr::ident("sdata_vec4")
                                .index(Expr::ident("local_id").field("x") + Expr::lit_u32(2)),
                        )]),
                        None,
                    ),
                    workgroup_barrier(),
                    // Reduction: stride 1 (write results)
                    if_block_expr(
                        Expr::ident("local_id").field("x").lt(Expr::lit_u32(1)),
                        block(vec![
                            assign_op_expr(
                                AssignOp::Add,
                                Expr::ident("sdata_vec4").index(Expr::ident("local_id").field("x")),
                                Expr::ident("sdata_vec4")
                                    .index(Expr::ident("local_id").field("x") + Expr::lit_u32(1)),
                            ),
                            // let sum = sdata_vec4[0];
                            let_expr("sum", Expr::ident("sdata_vec4").index(Expr::lit_u32(0))),
                            // Write partial sums
                            if_block_expr(
                                Expr::ident("i").le(Expr::ident("j")),
                                block(vec![assign_expr(
                                    Expr::ident("b_dot_partial").index(
                                        Expr::ident("i") * Expr::ident("num_groups_n")
                                            + Expr::ident("group_flat"),
                                    ),
                                    Expr::ident("sum").field("x"),
                                )]),
                                None,
                            ),
                            if_block_expr(
                                (Expr::ident("i") + Expr::lit_u32(1)).le(Expr::ident("j")),
                                block(vec![assign_expr(
                                    Expr::ident("b_dot_partial").index(
                                        (Expr::ident("i") + Expr::lit_u32(1))
                                            * Expr::ident("num_groups_n")
                                            + Expr::ident("group_flat"),
                                    ),
                                    Expr::ident("sum").field("y"),
                                )]),
                                None,
                            ),
                            if_block_expr(
                                (Expr::ident("i") + Expr::lit_u32(2)).le(Expr::ident("j")),
                                block(vec![assign_expr(
                                    Expr::ident("b_dot_partial").index(
                                        (Expr::ident("i") + Expr::lit_u32(2))
                                            * Expr::ident("num_groups_n")
                                            + Expr::ident("group_flat"),
                                    ),
                                    Expr::ident("sum").field("z"),
                                )]),
                                None,
                            ),
                            if_block_expr(
                                (Expr::ident("i") + Expr::lit_u32(3)).le(Expr::ident("j")),
                                block(vec![assign_expr(
                                    Expr::ident("b_dot_partial").index(
                                        (Expr::ident("i") + Expr::lit_u32(3))
                                            * Expr::ident("num_groups_n")
                                            + Expr::ident("group_flat"),
                                    ),
                                    Expr::ident("sum").field("w"),
                                )]),
                                None,
                            ),
                        ]),
                        None,
                    ),
                    workgroup_barrier(),
                ]),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "calc_dots_cgs",
            vec![
                Param::new(
                    "global_id",
                    Type::vec3_u32(),
                    vec![Attribute::Builtin("global_invocation_id".into())],
                ),
                Param::new(
                    "local_id",
                    Type::vec3_u32(),
                    vec![Attribute::Builtin("local_invocation_id".into())],
                ),
                Param::new(
                    "group_id",
                    Type::vec3_u32(),
                    vec![Attribute::Builtin("workgroup_id".into())],
                ),
                Param::new(
                    "num_workgroups",
                    Type::vec3_u32(),
                    vec![Attribute::Builtin("num_workgroups".into())],
                ),
            ],
            None,
            vec![Attribute::Compute, Attribute::WorkgroupSize(64)],
            body,
        )));
    }

    // ── Entry point: reduce_dots_cgs ────────────────────────────────────────
    push_reduce_dots_cgs(&mut m, "reduce_dots_cgs", false);

    // ── Entry point: update_w_cgs ───────────────────────────────────────────
    push_update_w_cgs(&mut m, "update_w_cgs", false);

    // ── CGS2 pass-2 entry points ─────────────────────────────────────────────
    //
    // Classical Gram-Schmidt re-orthogonalization ("twice is enough"): after
    // the first projection pass, re-dispatch calc_dots_cgs on the corrected
    // w, then these two variants. They differ from pass 1 in exactly two
    // ways:
    // - reduce_dots_cgs_reortho ACCUMULATES into the Hessenberg entry
    //   (H must hold d1 + d2 for the Givens least-squares) and stashes the
    //   pass-2 coefficient d2 into b_dot_partial[i * num_groups_n] (race
    //   free: workgroup i owns row i and all reads of the row happen before
    //   the barriers that precede the write);
    // - update_w_cgs_reortho sources its projection coefficients from that
    //   stash instead of b_hessenberg (which now holds d1 + d2; subtracting
    //   that from the already-corrected w would double-project).
    push_reduce_dots_cgs(&mut m, "reduce_dots_cgs_reortho", true);
    push_update_w_cgs(&mut m, "update_w_cgs_reortho", true);

    KernelWgsl::new(m)
}

/// Push the partial-dot reduction entry point. `pass2` switches the final
/// Hessenberg write from overwrite to accumulate-and-stash (see the CGS2
/// comment in `generate_gmres_cgs`).
fn push_reduce_dots_cgs(m: &mut Module, name: &str, pass2: bool) {
    {
        let final_write_stmts = if pass2 {
            vec![
                // let d = sdata[0];
                let_expr("d", Expr::ident("sdata").index(Expr::lit_u32(0))),
                // b_hessenberg[h_idx] += d;
                assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("b_hessenberg").index(Expr::ident("h_idx")),
                    Expr::ident("d"),
                ),
                // b_dot_partial[i * num_groups_n] = d;
                assign_expr(
                    Expr::ident("b_dot_partial")
                        .index(Expr::ident("i") * Expr::ident("num_groups_n")),
                    Expr::ident("d"),
                ),
            ]
        } else {
            vec![
                // b_hessenberg[h_idx] = sdata[0];
                assign_expr(
                    Expr::ident("b_hessenberg").index(Expr::ident("h_idx")),
                    Expr::ident("sdata").index(Expr::lit_u32(0)),
                ),
            ]
        };
        let body = block(vec![
            // if (scalars[SCALAR_STOP] > 0.5) { return; }
            if_block_expr(
                Expr::ident("scalars")
                    .index(Expr::ident("SCALAR_STOP"))
                    .gt(Expr::lit_f32(0.5)),
                block(vec![return_void()]),
                None,
            ),
            // let i = group_id.x;
            let_expr("i", Expr::ident("group_id").field("x")),
            // let j = params.num_iters;
            let_expr("j", Expr::ident("params").field("num_iters")),
            // if (i > j || i >= params.max_restart) { return; }
            if_block_expr(
                Expr::ident("i").gt(Expr::ident("j"))
                    | Expr::ident("i").ge(Expr::ident("params").field("max_restart")),
                block(vec![return_void()]),
                None,
            ),
            // let num_groups_n = (params.n + 63u) / 64u;
            let_expr(
                "num_groups_n",
                (Expr::ident("params").field("n") + Expr::lit_u32(63)) / Expr::lit_u32(64),
            ),
            // var sum = 0.0;
            var_expr("sum", Expr::lit_f32(0.0)),
            // for (var k = local_id.x; k < num_groups_n; k += 64u) { ... }
            for_loop_expr(
                ForInit::Var {
                    name: "k".into(),
                    ty: Some(Type::U32),
                    expr: Expr::ident("local_id").field("x"),
                },
                Expr::ident("k").lt(Expr::ident("num_groups_n")),
                for_step_assign_op_expr(AssignOp::Add, Expr::ident("k"), Expr::lit_u32(64)),
                block(vec![assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("sum"),
                    Expr::ident("b_dot_partial")
                        .index(Expr::ident("i") * Expr::ident("num_groups_n") + Expr::ident("k")),
                )]),
            ),
            // sdata[local_id.x] = sum;
            assign_expr(
                Expr::ident("sdata").index(Expr::ident("local_id").field("x")),
                Expr::ident("sum"),
            ),
            // workgroupBarrier();
            workgroup_barrier(),
            // Reduction: stride 32
            if_block_expr(
                Expr::ident("local_id").field("x").lt(Expr::lit_u32(32)),
                block(vec![assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("sdata").index(Expr::ident("local_id").field("x")),
                    Expr::ident("sdata")
                        .index(Expr::ident("local_id").field("x") + Expr::lit_u32(32)),
                )]),
                None,
            ),
            workgroup_barrier(),
            // Reduction: stride 16
            if_block_expr(
                Expr::ident("local_id").field("x").lt(Expr::lit_u32(16)),
                block(vec![assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("sdata").index(Expr::ident("local_id").field("x")),
                    Expr::ident("sdata")
                        .index(Expr::ident("local_id").field("x") + Expr::lit_u32(16)),
                )]),
                None,
            ),
            workgroup_barrier(),
            // Reduction: stride 8
            if_block_expr(
                Expr::ident("local_id").field("x").lt(Expr::lit_u32(8)),
                block(vec![assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("sdata").index(Expr::ident("local_id").field("x")),
                    Expr::ident("sdata")
                        .index(Expr::ident("local_id").field("x") + Expr::lit_u32(8)),
                )]),
                None,
            ),
            workgroup_barrier(),
            // Reduction: stride 4
            if_block_expr(
                Expr::ident("local_id").field("x").lt(Expr::lit_u32(4)),
                block(vec![assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("sdata").index(Expr::ident("local_id").field("x")),
                    Expr::ident("sdata")
                        .index(Expr::ident("local_id").field("x") + Expr::lit_u32(4)),
                )]),
                None,
            ),
            workgroup_barrier(),
            // Reduction: stride 2
            if_block_expr(
                Expr::ident("local_id").field("x").lt(Expr::lit_u32(2)),
                block(vec![assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("sdata").index(Expr::ident("local_id").field("x")),
                    Expr::ident("sdata")
                        .index(Expr::ident("local_id").field("x") + Expr::lit_u32(2)),
                )]),
                None,
            ),
            workgroup_barrier(),
            // Reduction: stride 1 (write result)
            if_block_expr(
                Expr::ident("local_id").field("x").lt(Expr::lit_u32(1)),
                block(
                    vec![
                        assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("sdata").index(Expr::ident("local_id").field("x")),
                            Expr::ident("sdata")
                                .index(Expr::ident("local_id").field("x") + Expr::lit_u32(1)),
                        ),
                        // let max_restart = params.max_restart;
                        let_expr("max_restart", Expr::ident("params").field("max_restart")),
                        // let h_idx = j * (max_restart + 1u) + i;
                        let_expr(
                            "h_idx",
                            Expr::ident("j") * (Expr::ident("max_restart") + Expr::lit_u32(1))
                                + Expr::ident("i"),
                        ),
                    ]
                    .into_iter()
                    .chain(final_write_stmts)
                    .collect(),
                ),
                None,
            ),
        ]);

        m.push(Item::Function(Function::new(
            name,
            vec![
                Param::new(
                    "global_id",
                    Type::vec3_u32(),
                    vec![Attribute::Builtin("global_invocation_id".into())],
                ),
                Param::new(
                    "local_id",
                    Type::vec3_u32(),
                    vec![Attribute::Builtin("local_invocation_id".into())],
                ),
                Param::new(
                    "group_id",
                    Type::vec3_u32(),
                    vec![Attribute::Builtin("workgroup_id".into())],
                ),
            ],
            None,
            vec![Attribute::Compute, Attribute::WorkgroupSize(64)],
            body,
        )));
    }
}

/// Push the w-projection-update entry point. `pass2` switches the
/// projection-coefficient source from `b_hessenberg` (pass 1) to the
/// per-column stash in `b_dot_partial[i * num_groups_n]` written by
/// `reduce_dots_cgs_reortho` (see the CGS2 comment in `generate_gmres_cgs`).
fn push_update_w_cgs(m: &mut Module, name: &str, pass2: bool) {
    // h_val source for column (i + offset).
    let h_val_expr = |offset: u32| -> Expr {
        let col = if offset == 0 {
            Expr::ident("i")
        } else {
            Expr::ident("i") + Expr::lit_u32(offset)
        };
        if pass2 {
            Expr::ident("b_dot_partial").index(col * Expr::ident("num_groups_n"))
        } else {
            Expr::ident("b_hessenberg")
                .index(Expr::ident("j") * (Expr::ident("max_restart") + Expr::lit_u32(1)) + col)
        }
    };
    // Per-offset unrolled projection block: correction += h * v.
    let unrolled_block = |offset: u32| -> Stmt {
        let col = if offset == 0 {
            Expr::ident("i")
        } else {
            Expr::ident("i") + Expr::lit_u32(offset)
        };
        if_block_expr(
            col.clone().le(Expr::ident("j")),
            block(vec![
                let_expr("h_val", h_val_expr(offset)),
                // let v_val = b_basis[(i + offset) * stride_words + idx];
                let_expr(
                    "v_val",
                    Expr::ident("b_basis")
                        .index(col * Expr::ident("stride_words") + Expr::ident("idx")),
                ),
                // correction += h_val * v_val;
                assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("correction"),
                    Expr::ident("h_val") * Expr::ident("v_val"),
                ),
            ]),
            None,
        )
    };

    let mut stmts = vec![
        // if (scalars[SCALAR_STOP] > 0.5) { return; }
        if_block_expr(
            Expr::ident("scalars")
                .index(Expr::ident("SCALAR_STOP"))
                .gt(Expr::lit_f32(0.5)),
            block(vec![return_void()]),
            None,
        ),
        // let stride_x = num_workgroups.x * WORKGROUP_SIZE;
        let_expr(
            "stride_x",
            Expr::ident("num_workgroups").field("x") * Expr::ident("WORKGROUP_SIZE"),
        ),
        // let idx = global_id.y * stride_x + global_id.x;
        let_expr(
            "idx",
            Expr::ident("global_id").field("y") * Expr::ident("stride_x")
                + Expr::ident("global_id").field("x"),
        ),
        // let j = params.num_iters;
        let_expr("j", Expr::ident("params").field("num_iters")),
        // let n = params.n;
        let_expr("n", Expr::ident("params").field("n")),
        // let max_restart = params.max_restart;
        let_expr("max_restart", Expr::ident("params").field("max_restart")),
        // let stride_bytes = (n * 4u + 255u) & 4294967040u;
        let_expr(
            "stride_bytes",
            (Expr::ident("n") * Expr::lit_u32(4) + Expr::lit_u32(255))
                .bitwise_and(Expr::lit_u32(4294967040)),
        ),
        // let stride_words = stride_bytes / 4u;
        let_expr(
            "stride_words",
            Expr::ident("stride_bytes") / Expr::lit_u32(4),
        ),
    ];
    if pass2 {
        // Pass 2 reads its coefficients from the b_dot_partial stash, which
        // is laid out by row stride num_groups_n (matching reduce_dots).
        stmts.push(let_expr(
            "num_groups_n",
            (Expr::ident("n") + Expr::lit_u32(63)) / Expr::lit_u32(64),
        ));
    }
    stmts.extend([
        // if (idx >= n) { return; }
        if_block_expr(
            Expr::ident("idx").ge(Expr::ident("n")),
            block(vec![return_void()]),
            None,
        ),
        // var correction = 0.0;
        var_expr("correction", Expr::lit_f32(0.0)),
        // for (var i = 0u; i <= j; i += 4u) { unrolled x4 }
        for_loop_expr(
            ForInit::Var {
                name: "i".into(),
                ty: Some(Type::U32),
                expr: Expr::lit_u32(0),
            },
            Expr::ident("i").le(Expr::ident("j")),
            for_step_assign_op_expr(AssignOp::Add, Expr::ident("i"), Expr::lit_u32(4)),
            block(vec![
                unrolled_block(0),
                unrolled_block(1),
                unrolled_block(2),
                unrolled_block(3),
            ]),
        ),
        // b_w[idx] = b_w[idx] - correction;
        assign_expr(
            Expr::ident("b_w").index(Expr::ident("idx")),
            Expr::ident("b_w").index(Expr::ident("idx")) - Expr::ident("correction"),
        ),
    ]);
    let body = block(stmts);

    m.push(Item::Function(Function::new(
        name,
        vec![
            Param::new(
                "global_id",
                Type::vec3_u32(),
                vec![Attribute::Builtin("global_invocation_id".into())],
            ),
            Param::new(
                "num_workgroups",
                Type::vec3_u32(),
                vec![Attribute::Builtin("num_workgroups".into())],
            ),
        ],
        None,
        vec![Attribute::Compute, Attribute::WorkgroupSize(64)],
        body,
    )));
}
