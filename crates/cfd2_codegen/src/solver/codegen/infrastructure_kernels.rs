//! DSL generators for infrastructure compute kernels.
//!
//! These are solver-infrastructure kernels (dot product, CG linear solver, AMG,
//! scalars reduction, GMRES helpers, Schur setup, etc.).  Each generator builds
//! a [`Module`] via the structured DSL and returns a [`KernelWgsl`] so the build
//! system can write the result to `shaders/generated/` and let `wgsl_bindgen` +
//! the kernel registry handle the rest.

use super::kernel_wgsl::KernelWgsl;
use super::wgsl_ast::*;
use super::wgsl_dsl::*;

mod gmres_cgs_impl;
pub use gmres_cgs_impl::generate_gmres_cgs;

mod gmres_logic_impl;
pub use gmres_logic_impl::generate_gmres_logic;

mod gmres_ops_impl;
pub use gmres_ops_impl::generate_gmres_ops;

mod schur_precond_generic_impl;
pub use schur_precond_generic_impl::generate_schur_precond_generic;

mod block_precond_impl;
pub use block_precond_impl::generate_block_precond;

// ─── dot_product ────────────────────────────────────────────────────────

/// Workgroup‐reduction dot product.  Produces a single `@compute` entry point
/// named `main` with workgroup_size(64).
pub fn generate_dot_product() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Struct(StructDef::new(
        "SolverParams",
        vec![
            StructField::new("n", Type::U32),
            StructField::new("num_groups", Type::U32),
            StructField::new("padding", Type::Vec2(Box::new(Type::U32))),
        ],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("SolverParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "dot_result",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(1), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "dot_a",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(1), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "dot_b",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(1), Attribute::Binding(2)],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "scratch",
        Type::sized_array(Type::F32, 64),
        StorageClass::Workgroup,
        None,
        vec![],
    )));

    let global_id = Expr::ident("global_id");
    let local_id = Expr::ident("local_id");
    let group_id = Expr::ident("group_id");
    let num_workgroups = Expr::ident("num_workgroups");

    let body = block(vec![
        let_expr("stride_x", num_workgroups.clone().field("x") * 64u32),
        let_expr(
            "idx",
            global_id.clone().field("y") * Expr::ident("stride_x") + global_id.clone().field("x"),
        ),
        let_expr("lid", local_id.clone().field("x")),
        var_expr("val", Expr::lit_f32(0.0)),
        if_block_expr(
            Expr::ident("idx").lt(Expr::ident("params").field("n")),
            block(vec![assign_expr(
                Expr::ident("val"),
                Expr::ident("dot_a").index(Expr::ident("idx"))
                    * Expr::ident("dot_b").index(Expr::ident("idx")),
            )]),
            None,
        ),
        assign_expr(
            Expr::ident("scratch").index(Expr::ident("lid")),
            Expr::ident("val"),
        ),
        workgroup_barrier(),
        comment("Reduction in shared memory"),
        for_loop_expr(
            ForInit::Var {
                name: "i".into(),
                ty: None,
                expr: Expr::lit_u32(32),
            },
            Expr::ident("i").gt(0u32),
            ForStep::AssignOp {
                target: Expr::ident("i"),
                op: AssignOp::ShiftRight,
                value: Expr::lit_u32(1),
            },
            block(vec![
                if_block_expr(
                    Expr::ident("lid").lt(Expr::ident("i")),
                    block(vec![assign_op_expr(
                        AssignOp::Add,
                        Expr::ident("scratch").index(Expr::ident("lid")),
                        Expr::ident("scratch").index(Expr::ident("lid") + Expr::ident("i")),
                    )]),
                    None,
                ),
                workgroup_barrier(),
            ]),
        ),
        if_block_expr(
            Expr::ident("lid").eq(0u32),
            block(vec![
                let_expr(
                    "group_flat",
                    group_id.clone().field("y") * num_workgroups.clone().field("x") + group_id.clone().field("x"),
                ),
                if_block_expr(
                    Expr::ident("group_flat").lt(Expr::ident("params").field("num_groups")),
                    block(vec![assign_expr(
                        Expr::ident("dot_result").index(Expr::ident("group_flat")),
                        Expr::ident("scratch").index(Expr::lit_u32(0)),
                    )]),
                    None,
                ),
            ]),
            None,
        ),
    ]);

    m.push(Item::Function(Function::new(
        "main",
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

    KernelWgsl::new(m)
}

// ─── dot_product_pair ───────────────────────────────────────────────────

/// Dual workgroup‐reduction dot product (two independent dot products in one
/// dispatch).  Single `main` entry point, workgroup_size(64).
pub fn generate_dot_product_pair() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Struct(StructDef::new(
        "SolverParams",
        vec![
            StructField::new("n", Type::U32),
            StructField::new("num_groups", Type::U32),
            StructField::new("padding", Type::Vec2(Box::new(Type::U32))),
        ],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("SolverParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));

    for (i, name) in [
        "dot_result_a",
        "dot_result_b",
        "dot_a0",
        "dot_b0",
        "dot_a1",
        "dot_b1",
    ]
    .iter()
    .enumerate()
    {
        let access = if i < 2 {
            AccessMode::ReadWrite
        } else {
            AccessMode::Read
        };
        m.push(Item::GlobalVar(GlobalVar::new(
            *name,
            Type::array(Type::F32),
            StorageClass::Storage,
            Some(access),
            vec![Attribute::Group(1), Attribute::Binding(i as u32)],
        )));
    }

    m.push(Item::GlobalVar(GlobalVar::new(
        "scratch_a",
        Type::sized_array(Type::F32, 64),
        StorageClass::Workgroup,
        None,
        vec![],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "scratch_b",
        Type::sized_array(Type::F32, 64),
        StorageClass::Workgroup,
        None,
        vec![],
    )));

    let global_id = Expr::ident("global_id");
    let local_id = Expr::ident("local_id");
    let group_id = Expr::ident("group_id");
    let num_wg = Expr::ident("num_workgroups");

    let body = block(vec![
        let_expr("stride_x", num_wg.clone().field("x") * 64u32),
        let_expr(
            "idx",
            global_id.clone().field("y") * Expr::ident("stride_x") + global_id.clone().field("x"),
        ),
        let_expr("lid", local_id.clone().field("x")),
        var_expr("val0", Expr::lit_f32(0.0)),
        var_expr("val1", Expr::lit_f32(0.0)),
        if_block_expr(
            Expr::ident("idx").lt(Expr::ident("params").field("n")),
            block(vec![
                assign_expr(
                    Expr::ident("val0"),
                    Expr::ident("dot_a0").index(Expr::ident("idx"))
                        * Expr::ident("dot_b0").index(Expr::ident("idx")),
                ),
                assign_expr(
                    Expr::ident("val1"),
                    Expr::ident("dot_a1").index(Expr::ident("idx"))
                        * Expr::ident("dot_b1").index(Expr::ident("idx")),
                ),
            ]),
            None,
        ),
        assign_expr(
            Expr::ident("scratch_a").index(Expr::ident("lid")),
            Expr::ident("val0"),
        ),
        assign_expr(
            Expr::ident("scratch_b").index(Expr::ident("lid")),
            Expr::ident("val1"),
        ),
        workgroup_barrier(),
        var_expr("offset", Expr::lit_u32(32)),
        loop_block(block(vec![
            if_block_expr(
                Expr::ident("lid").lt(Expr::ident("offset")),
                block(vec![
                    assign_op_expr(
                        AssignOp::Add,
                        Expr::ident("scratch_a").index(Expr::ident("lid")),
                        Expr::ident("scratch_a").index(Expr::ident("lid") + Expr::ident("offset")),
                    ),
                    assign_op_expr(
                        AssignOp::Add,
                        Expr::ident("scratch_b").index(Expr::ident("lid")),
                        Expr::ident("scratch_b").index(Expr::ident("lid") + Expr::ident("offset")),
                    ),
                ]),
                None,
            ),
            workgroup_barrier(),
            if_block_expr(
                Expr::ident("offset").eq(1u32),
                block(vec![break_stmt()]),
                None,
            ),
            assign_expr(
                Expr::ident("offset"),
                Expr::ident("offset").shr(Expr::lit_u32(1)),
            ),
        ])),
        if_block_expr(
            Expr::ident("lid").eq(0u32),
            block(vec![
                let_expr(
                    "group_flat",
                    group_id.clone().field("y") * num_wg.clone().field("x") + group_id.clone().field("x"),
                ),
                if_block_expr(
                    Expr::ident("group_flat").lt(Expr::ident("params").field("num_groups")),
                    block(vec![
                        assign_expr(
                            Expr::ident("dot_result_a").index(Expr::ident("group_flat")),
                            Expr::ident("scratch_a").index(Expr::lit_u32(0)),
                        ),
                        assign_expr(
                            Expr::ident("dot_result_b").index(Expr::ident("group_flat")),
                            Expr::ident("scratch_b").index(Expr::lit_u32(0)),
                        ),
                    ]),
                    None,
                ),
            ]),
            None,
        ),
    ]);

    m.push(Item::Function(Function::new(
        "main",
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

    KernelWgsl::new(m)
}

// ─── outer_convergence ──────────────────────────────────────────────────

/// Per-cell convergence metric via `atomicMax`.  Single `main` entry point,
/// workgroup_size(256).
pub fn generate_outer_convergence() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Struct(StructDef::new(
        "Params",
        vec![
            StructField::new("num_cells", Type::U32),
            StructField::new("stride", Type::U32),
            StructField::new("num_targets", Type::U32),
            StructField::new("_pad0", Type::U32),
        ],
    )));

    m.push(Item::Struct(StructDef::new(
        "TargetDesc",
        vec![
            StructField::new("offsets", Type::sized_array(Type::U32, 4)),
            StructField::new("num_comps", Type::U32),
            StructField::new("_pad0", Type::sized_array(Type::U32, 3)),
        ],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "input",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "targets",
        Type::array(Type::Custom("TargetDesc".into())),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "out_bits",
        Type::array(Type::atomic(Type::U32)),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("Params".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));

    let gid = Expr::ident("gid");
    let params = Expr::ident("params");

    let body = block(vec![
        let_expr("cell", gid.field("x")),
        if_block_expr(
            Expr::ident("cell").ge(params.clone().field("num_cells")),
            block(vec![return_void()]),
            None,
        ),
        let_expr("base", Expr::ident("cell") * params.clone().field("stride")),
        for_loop_expr(
            ForInit::Var {
                name: "t".into(),
                ty: Some(Type::U32),
                expr: Expr::lit_u32(0),
            },
            Expr::ident("t").lt(params.clone().field("num_targets")),
            ForStep::Assign {
                target: Expr::ident("t"),
                value: Expr::ident("t") + 1u32,
            },
            block(vec![
                let_expr("desc", Expr::ident("targets").index(Expr::ident("t"))),
                var_typed_expr("mag2", Type::F32, Some(Expr::lit_f32(0.0))),
                for_loop_expr(
                    ForInit::Var {
                        name: "c".into(),
                        ty: Some(Type::U32),
                        expr: Expr::lit_u32(0),
                    },
                    Expr::ident("c").lt(Expr::ident("desc").field("num_comps")),
                    ForStep::Assign {
                        target: Expr::ident("c"),
                        value: Expr::ident("c") + 1u32,
                    },
                    block(vec![
                        let_expr(
                            "off",
                            Expr::ident("desc").field("offsets").index(Expr::ident("c")),
                        ),
                        let_expr(
                            "v",
                            Expr::ident("input").index(Expr::ident("base") + Expr::ident("off")),
                        ),
                        assign_expr(
                            Expr::ident("mag2"),
                            Expr::ident("mag2") + Expr::ident("v") * Expr::ident("v"),
                        ),
                    ]),
                ),
                let_expr("mag", sqrt(Expr::ident("mag2"))),
                let_expr("bits", bitcast("u32", Expr::ident("mag"))),
                call_stmt_expr(atomic_max(
                    Expr::ident("out_bits").index(Expr::ident("t")).addr_of(),
                    Expr::ident("bits"),
                )),
            ]),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "main",
        vec![Param::new(
            "gid",
            Type::vec3_u32(),
            vec![Attribute::Builtin("global_invocation_id".into())],
        )],
        None,
        vec![Attribute::Compute, Attribute::WorkgroupSize(256)],
        body,
    )));

    KernelWgsl::new(m)
}

// ─── scalars ────────────────────────────────────────────────────────────

/// CG scalar reduction kernel with 3 entry points:
/// `reduce_rho_new_r_r`, `reduce_r0_v`, `init_cg_scalars`.
pub fn generate_scalars() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Struct(StructDef::new(
        "GpuScalars",
        vec![
            StructField::new("rho_old", Type::F32),
            StructField::new("rho_new", Type::F32),
            StructField::new("alpha", Type::F32),
            StructField::new("beta", Type::F32),
            StructField::new("r0_v", Type::F32),
            StructField::new("r_r", Type::F32),
            StructField::new("stop", Type::F32),
        ],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "scalars",
        Type::Custom("GpuScalars".into()),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "dot_result_1",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "dot_result_2",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));

    m.push(Item::Struct(StructDef::new(
        "ReduceParams",
        vec![
            StructField::new("n", Type::U32),
            StructField::new("num_groups", Type::U32),
        ],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("ReduceParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "scratch1",
        Type::sized_array(Type::F32, 64),
        StorageClass::Workgroup,
        None,
        vec![],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "scratch2",
        Type::sized_array(Type::F32, 64),
        StorageClass::Workgroup,
        None,
        vec![],
    )));

    // Helper: the standard tree‐reduction for scratch1 (and optionally scratch2)
    fn tree_reduce(dual: bool) -> Vec<Stmt> {
        let mut stmts = vec![];
        let inner = if dual {
            vec![
                assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("scratch1").index(Expr::ident("lid")),
                    Expr::ident("scratch1").index(Expr::ident("lid") + Expr::ident("i")),
                ),
                assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("scratch2").index(Expr::ident("lid")),
                    Expr::ident("scratch2").index(Expr::ident("lid") + Expr::ident("i")),
                ),
            ]
        } else {
            vec![assign_op_expr(
                AssignOp::Add,
                Expr::ident("scratch1").index(Expr::ident("lid")),
                Expr::ident("scratch1").index(Expr::ident("lid") + Expr::ident("i")),
            )]
        };

        stmts.push(for_loop_expr(
            ForInit::Var {
                name: "i".into(),
                ty: None,
                expr: Expr::lit_u32(32),
            },
            Expr::ident("i").gt(0u32),
            ForStep::AssignOp {
                target: Expr::ident("i"),
                op: AssignOp::ShiftRight,
                value: Expr::lit_u32(1),
            },
            block(vec![
                if_block_expr(Expr::ident("lid").lt(Expr::ident("i")), block(inner), None),
                workgroup_barrier(),
            ]),
        ));
        stmts
    }

    // Helper: sequential accumulation of dot_result into scratch
    fn accum_loop(var_name: &str, buf: &str) -> Vec<Stmt> {
        vec![
            var_expr(var_name, Expr::lit_f32(0.0)),
            for_loop_expr(
                ForInit::Var {
                    name: "i".into(),
                    ty: None,
                    expr: Expr::ident("lid"),
                },
                Expr::ident("i").lt(Expr::ident("n")),
                ForStep::AssignOp {
                    target: Expr::ident("i"),
                    op: AssignOp::Add,
                    value: Expr::lit_u32(64),
                },
                block(vec![assign_op_expr(
                    AssignOp::Add,
                    Expr::ident(var_name),
                    Expr::ident(buf).index(Expr::ident("i")),
                )]),
            ),
        ]
    }

    // Helper: fused dual accumulation of two buffers in a single loop
    fn accum_loop_dual(var1: &str, buf1: &str, var2: &str, buf2: &str) -> Vec<Stmt> {
        vec![
            var_expr(var1, Expr::lit_f32(0.0)),
            var_expr(var2, Expr::lit_f32(0.0)),
            for_loop_expr(
                ForInit::Var {
                    name: "i".into(),
                    ty: None,
                    expr: Expr::ident("lid"),
                },
                Expr::ident("i").lt(Expr::ident("n")),
                ForStep::AssignOp {
                    target: Expr::ident("i"),
                    op: AssignOp::Add,
                    value: Expr::lit_u32(64),
                },
                block(vec![
                    assign_op_expr(
                        AssignOp::Add,
                        Expr::ident(var1),
                        Expr::ident(buf1).index(Expr::ident("i")),
                    ),
                    assign_op_expr(
                        AssignOp::Add,
                        Expr::ident(var2),
                        Expr::ident(buf2).index(Expr::ident("i")),
                    ),
                ]),
            ),
        ]
    }

    // ── reduce_rho_new_r_r ──
    {
        let mut body_stmts = vec![
            if_block_expr(
                Expr::ident("scalars").field("stop").gt(0.5f32),
                block(vec![return_void()]),
                None,
            ),
            let_expr("n", Expr::ident("params").field("num_groups")),
            let_expr("lid", Expr::ident("local_id").field("x")),
        ];
        body_stmts.extend(accum_loop_dual(
            "sum1",
            "dot_result_1",
            "sum2",
            "dot_result_2",
        ));
        body_stmts.push(assign_expr(
            Expr::ident("scratch1").index(Expr::ident("lid")),
            Expr::ident("sum1"),
        ));
        body_stmts.push(assign_expr(
            Expr::ident("scratch2").index(Expr::ident("lid")),
            Expr::ident("sum2"),
        ));
        body_stmts.push(workgroup_barrier());
        body_stmts.extend(tree_reduce(true));
        body_stmts.push(if_block_expr(
            Expr::ident("lid").eq(0u32),
            block(vec![
                assign_expr(
                    Expr::ident("scalars").field("rho_new"),
                    Expr::ident("scratch1").index(Expr::lit_u32(0)),
                ),
                assign_expr(
                    Expr::ident("scalars").field("r_r"),
                    Expr::ident("scratch2").index(Expr::lit_u32(0)),
                ),
            ]),
            None,
        ));

        m.push(Item::Function(Function::new(
            "reduce_rho_new_r_r",
            vec![Param::new(
                "local_id",
                Type::vec3_u32(),
                vec![Attribute::Builtin("local_invocation_id".into())],
            )],
            None,
            vec![Attribute::Compute, Attribute::WorkgroupSize(64)],
            block(body_stmts),
        )));
    }

    // ── reduce_r0_v ──
    {
        let mut body_stmts = vec![
            if_block_expr(
                Expr::ident("scalars").field("stop").gt(0.5f32),
                block(vec![return_void()]),
                None,
            ),
            let_expr("n", Expr::ident("params").field("num_groups")),
            let_expr("lid", Expr::ident("local_id").field("x")),
        ];
        body_stmts.extend(accum_loop("sum", "dot_result_1"));
        body_stmts.push(assign_expr(
            Expr::ident("scratch1").index(Expr::ident("lid")),
            Expr::ident("sum"),
        ));
        body_stmts.push(workgroup_barrier());
        body_stmts.extend(tree_reduce(false));
        body_stmts.push(if_block_expr(
            Expr::ident("lid").eq(0u32),
            block(vec![assign_expr(
                Expr::ident("scalars").field("r0_v"),
                Expr::ident("scratch1").index(Expr::lit_u32(0)),
            )]),
            None,
        ));

        m.push(Item::Function(Function::new(
            "reduce_r0_v",
            vec![Param::new(
                "local_id",
                Type::vec3_u32(),
                vec![Attribute::Builtin("local_invocation_id".into())],
            )],
            None,
            vec![Attribute::Compute, Attribute::WorkgroupSize(64)],
            block(body_stmts),
        )));
    }

    // ── init_cg_scalars ──
    {
        let mut body_stmts = vec![
            let_expr("n", Expr::ident("params").field("num_groups")),
            let_expr("lid", Expr::ident("local_id").field("x")),
        ];
        body_stmts.extend(accum_loop("sum", "dot_result_1"));
        body_stmts.push(assign_expr(
            Expr::ident("scratch1").index(Expr::ident("lid")),
            Expr::ident("sum"),
        ));
        body_stmts.push(workgroup_barrier());
        body_stmts.extend(tree_reduce(false));
        body_stmts.push(if_block_expr(
            Expr::ident("lid").eq(0u32),
            block(vec![
                assign_expr(
                    Expr::ident("scalars").field("rho_old"),
                    Expr::ident("scratch1").index(Expr::lit_u32(0)),
                ),
                assign_expr(Expr::ident("scalars").field("alpha"), Expr::lit_f32(0.0)),
                assign_expr(Expr::ident("scalars").field("beta"), Expr::lit_f32(0.0)),
                assign_expr(Expr::ident("scalars").field("stop"), Expr::lit_f32(0.0)),
            ]),
            None,
        ));

        m.push(Item::Function(Function::new(
            "init_cg_scalars",
            vec![Param::new(
                "local_id",
                Type::vec3_u32(),
                vec![Attribute::Builtin("local_invocation_id".into())],
            )],
            None,
            vec![Attribute::Compute, Attribute::WorkgroupSize(64)],
            block(body_stmts),
        )));
    }

    KernelWgsl::new(m)
}

// ─── linear_solver ──────────────────────────────────────────────────────

/// CG linear solver kernel with 3 entry points: `spmv_p_v`, `cg_update_x_r`,
/// `cg_update_p`.
pub fn generate_linear_solver() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Comment("Linear Solver (CG)".into()));

    // Group 0: Vectors
    for (i, name) in ["x", "r", "p", "v"].iter().enumerate() {
        m.push(Item::GlobalVar(GlobalVar::new(
            *name,
            Type::array(Type::F32),
            StorageClass::Storage,
            Some(AccessMode::ReadWrite),
            vec![Attribute::Group(0), Attribute::Binding(i as u32)],
        )));
    }

    // Group 1: Matrix & Params
    m.push(Item::GlobalVar(GlobalVar::new(
        "row_offsets",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(1), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "col_indices",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(1), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "matrix_values",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(1), Attribute::Binding(2)],
    )));

    m.push(Item::Struct(StructDef::new(
        "GpuScalars",
        vec![
            StructField::new("rho_old", Type::F32),
            StructField::new("rho_new", Type::F32),
            StructField::new("alpha", Type::F32),
            StructField::new("beta", Type::F32),
            StructField::new("r0_v", Type::F32),
            StructField::new("r_r", Type::F32),
            StructField::new("stop", Type::F32),
        ],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "scalars",
        Type::Custom("GpuScalars".into()),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(1), Attribute::Binding(3)],
    )));

    m.push(Item::Struct(StructDef::new(
        "SolverParams",
        vec![StructField::new("n", Type::U32)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("SolverParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(1), Attribute::Binding(4)],
    )));

    m.push(Item::Function(Function::new(
        "global_index",
        vec![
            Param::new("global_id", Type::vec3_u32(), vec![]),
            Param::new("num_workgroups", Type::vec3_u32(), vec![]),
        ],
        Some(Type::U32),
        vec![],
        block(vec![return_expr(
            Expr::ident("global_id").field("y")
                * (Expr::ident("num_workgroups").field("x") * 64u32)
                + Expr::ident("global_id").field("x"),
        )]),
    )));

    let std_params = vec![
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
    ];
    let std_attrs = vec![Attribute::Compute, Attribute::WorkgroupSize(64)];

    // ── spmv_p_v ──
    {
        let body = block(vec![
            if_block_expr(
                Expr::ident("scalars").field("stop").gt(0.5f32),
                block(vec![return_void()]),
                None,
            ),
            let_expr(
                "row",
                Expr::call_named(
                    "global_index",
                    vec![Expr::ident("global_id"), Expr::ident("num_workgroups")],
                ),
            ),
            if_block_expr(
                Expr::ident("row").ge(Expr::ident("params").field("n")),
                block(vec![return_void()]),
                None,
            ),
            let_expr(
                "start",
                Expr::ident("row_offsets").index(Expr::ident("row")),
            ),
            let_expr(
                "end",
                Expr::ident("row_offsets").index(Expr::ident("row") + 1u32),
            ),
            var_expr("sum", Expr::lit_f32(0.0)),
            for_loop_expr(
                ForInit::Var {
                    name: "k".into(),
                    ty: None,
                    expr: Expr::ident("start"),
                },
                Expr::ident("k").lt(Expr::ident("end")),
                ForStep::Increment(Expr::ident("k")),
                block(vec![
                    let_expr("col", Expr::ident("col_indices").index(Expr::ident("k"))),
                    let_expr("val", Expr::ident("matrix_values").index(Expr::ident("k"))),
                    assign_op_expr(
                        AssignOp::Add,
                        Expr::ident("sum"),
                        Expr::ident("val") * Expr::ident("p").index(Expr::ident("col")),
                    ),
                ]),
            ),
            assign_expr(
                Expr::ident("v").index(Expr::ident("row")),
                Expr::ident("sum"),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "spmv_p_v",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── cg_update_x_r ──
    {
        let body = block(vec![
            if_block_expr(
                Expr::ident("scalars").field("stop").gt(0.5f32),
                block(vec![return_void()]),
                None,
            ),
            let_expr(
                "idx",
                Expr::call_named(
                    "global_index",
                    vec![Expr::ident("global_id"), Expr::ident("num_workgroups")],
                ),
            ),
            var_expr("alpha", Expr::lit_f32(0.0)),
            if_block_expr(
                abs(Expr::ident("scalars").field("r0_v")).ge(Expr::lit_f32(1e-20)),
                block(vec![assign_expr(
                    Expr::ident("alpha"),
                    Expr::ident("scalars").field("rho_old") / Expr::ident("scalars").field("r0_v"),
                )]),
                None,
            ),
            if_block_expr(
                Expr::ident("idx").eq(0u32),
                block(vec![assign_expr(
                    Expr::ident("scalars").field("alpha"),
                    Expr::ident("alpha"),
                )]),
                None,
            ),
            if_block_expr(
                Expr::ident("idx").ge(Expr::ident("params").field("n")),
                block(vec![return_void()]),
                None,
            ),
            assign_op_expr(
                AssignOp::Add,
                Expr::ident("x").index(Expr::ident("idx")),
                Expr::ident("alpha") * Expr::ident("p").index(Expr::ident("idx")),
            ),
            assign_op_expr(
                AssignOp::Sub,
                Expr::ident("r").index(Expr::ident("idx")),
                Expr::ident("alpha") * Expr::ident("v").index(Expr::ident("idx")),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "cg_update_x_r",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── cg_update_p ──
    {
        let body = block(vec![
            if_block_expr(
                Expr::ident("scalars").field("stop").gt(0.5f32),
                block(vec![return_void()]),
                None,
            ),
            let_expr(
                "idx",
                Expr::call_named(
                    "global_index",
                    vec![Expr::ident("global_id"), Expr::ident("num_workgroups")],
                ),
            ),
            var_expr("beta", Expr::lit_f32(0.0)),
            if_block_expr(
                abs(Expr::ident("scalars").field("rho_old")).ge(Expr::lit_f32(1e-20)),
                block(vec![assign_expr(
                    Expr::ident("beta"),
                    Expr::ident("scalars").field("rho_new")
                        / Expr::ident("scalars").field("rho_old"),
                )]),
                None,
            ),
            if_block_expr(
                Expr::ident("idx").eq(0u32),
                block(vec![
                    assign_expr(Expr::ident("scalars").field("beta"), Expr::ident("beta")),
                    comment("Update rho_old for next iteration"),
                    assign_expr(
                        Expr::ident("scalars").field("rho_old"),
                        Expr::ident("scalars").field("rho_new"),
                    ),
                ]),
                None,
            ),
            if_block_expr(
                Expr::ident("idx").ge(Expr::ident("params").field("n")),
                block(vec![return_void()]),
                None,
            ),
            assign_expr(
                Expr::ident("p").index(Expr::ident("idx")),
                Expr::ident("r").index(Expr::ident("idx"))
                    + Expr::ident("beta") * Expr::ident("p").index(Expr::ident("idx")),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "cg_update_p",
            std_params,
            None,
            std_attrs,
            body,
        )));
    }

    KernelWgsl::new(m)
}

// ─── amg ────────────────────────────────────────────────────────────────

/// AMG smoother / prolongation / restriction / clear.  4 entry points.
pub fn generate_amg() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Struct(StructDef::new(
        "AmgParams",
        vec![
            StructField::new("n", Type::U32),
            StructField::new("omega", Type::F32),
            StructField::new("padding", Type::Vec2(Box::new(Type::U32))),
        ],
    )));

    // Group 0: sparse matrix A
    m.push(Item::GlobalVar(GlobalVar::new(
        "row_offsets",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "col_indices",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "values",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));

    // Group 1: vectors + params
    m.push(Item::GlobalVar(GlobalVar::new(
        "x",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(1), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "b",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(1), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("AmgParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(1), Attribute::Binding(2)],
    )));

    // Group 2: operator matrix (restriction/prolongation)
    m.push(Item::GlobalVar(GlobalVar::new(
        "op_row_offsets",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(2), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "op_col_indices",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(2), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "op_values",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(2), Attribute::Binding(2)],
    )));

    // Group 3: cross-level
    m.push(Item::GlobalVar(GlobalVar::new(
        "coarse_vec",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(3), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "scalars",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(3), Attribute::Binding(1)],
    )));

    m.push(Item::Function(Function::new(
        "amg_should_stop",
        vec![],
        Some(Type::Bool),
        vec![],
        block(vec![return_expr(
            Expr::ident("scalars").index(Expr::lit_u32(8)).gt(0.5f32),
        )]),
    )));

    let std_params = vec![
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
    ];
    let std_attrs = vec![Attribute::Compute, Attribute::WorkgroupSize(64)];

    // Standard preamble: stop check + compute flat index
    fn amg_preamble() -> Vec<Stmt> {
        vec![
            if_block_expr(
                Expr::call_named("amg_should_stop", vec![]),
                block(vec![return_void()]),
                None,
            ),
            let_expr("stride_x", Expr::ident("num_workgroups").field("x") * 64u32),
            let_expr(
                "i",
                Expr::ident("global_id").field("y") * Expr::ident("stride_x")
                    + Expr::ident("global_id").field("x"),
            ),
            if_block_expr(
                Expr::ident("i").ge(Expr::ident("params").field("n")),
                block(vec![return_void()]),
                None,
            ),
        ]
    }

    // ── smooth_op ──
    {
        let mut stmts = amg_preamble();
        stmts.extend(vec![
            let_expr("start", Expr::ident("row_offsets").index(Expr::ident("i"))),
            let_expr(
                "end",
                Expr::ident("row_offsets").index(Expr::ident("i") + 1u32),
            ),
            var_expr("sigma", Expr::lit_f32(0.0)),
            var_expr("diag", Expr::lit_f32(1.0)),
            for_loop_expr(
                ForInit::Var {
                    name: "k".into(),
                    ty: None,
                    expr: Expr::ident("start"),
                },
                Expr::ident("k").lt(Expr::ident("end")),
                ForStep::Increment(Expr::ident("k")),
                block(vec![
                    let_expr("col", Expr::ident("col_indices").index(Expr::ident("k"))),
                    let_expr("val", Expr::ident("values").index(Expr::ident("k"))),
                    if_block_expr(
                        Expr::ident("col").eq(Expr::ident("i")),
                        block(vec![assign_expr(Expr::ident("diag"), Expr::ident("val"))]),
                        Some(block(vec![assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("sigma"),
                            Expr::ident("val") * Expr::ident("x").index(Expr::ident("col")),
                        )])),
                    ),
                ]),
            ),
            if_block_expr(
                abs(Expr::ident("diag")).lt(Expr::lit_f32(1e-14)),
                block(vec![assign_expr(Expr::ident("diag"), Expr::lit_f32(1.0))]),
                None,
            ),
            let_expr(
                "x_new",
                (Expr::ident("b").index(Expr::ident("i")) - Expr::ident("sigma"))
                    / Expr::ident("diag"),
            ),
            assign_expr(
                Expr::ident("x").index(Expr::ident("i")),
                mix(
                    Expr::ident("x").index(Expr::ident("i")),
                    Expr::ident("x_new"),
                    Expr::ident("params").field("omega"),
                ),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "smooth_op",
            std_params.clone(),
            None,
            std_attrs.clone(),
            block(stmts),
        )));
    }

    // ── prolongate_op ──
    {
        let mut stmts = amg_preamble();
        stmts.extend(vec![
            let_expr(
                "start",
                Expr::ident("op_row_offsets").index(Expr::ident("i")),
            ),
            let_expr(
                "end",
                Expr::ident("op_row_offsets").index(Expr::ident("i") + 1u32),
            ),
            var_expr("correction", Expr::lit_f32(0.0)),
            for_loop_expr(
                ForInit::Var {
                    name: "k".into(),
                    ty: None,
                    expr: Expr::ident("start"),
                },
                Expr::ident("k").lt(Expr::ident("end")),
                ForStep::Increment(Expr::ident("k")),
                block(vec![
                    let_expr(
                        "coarse_idx",
                        Expr::ident("op_col_indices").index(Expr::ident("k")),
                    ),
                    let_expr("val", Expr::ident("op_values").index(Expr::ident("k"))),
                    assign_op_expr(
                        AssignOp::Add,
                        Expr::ident("correction"),
                        Expr::ident("val")
                            * Expr::ident("coarse_vec").index(Expr::ident("coarse_idx")),
                    ),
                ]),
            ),
            assign_op_expr(
                AssignOp::Add,
                Expr::ident("x").index(Expr::ident("i")),
                Expr::ident("correction"),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "prolongate_op",
            std_params.clone(),
            None,
            std_attrs.clone(),
            block(stmts),
        )));
    }

    // ── restrict_residual ──
    {
        let mut stmts = amg_preamble();
        stmts.extend(vec![
            let_expr(
                "start",
                Expr::ident("op_row_offsets").index(Expr::ident("i")),
            ),
            let_expr(
                "end",
                Expr::ident("op_row_offsets").index(Expr::ident("i") + 1u32),
            ),
            var_expr("sum", Expr::lit_f32(0.0)),
            for_loop_expr(
                ForInit::Var {
                    name: "k".into(),
                    ty: None,
                    expr: Expr::ident("start"),
                },
                Expr::ident("k").lt(Expr::ident("end")),
                ForStep::Increment(Expr::ident("k")),
                block(vec![
                    let_expr(
                        "fine_idx",
                        Expr::ident("op_col_indices").index(Expr::ident("k")),
                    ),
                    let_expr("r_val", Expr::ident("op_values").index(Expr::ident("k"))),
                    let_expr(
                        "a_start",
                        Expr::ident("row_offsets").index(Expr::ident("fine_idx")),
                    ),
                    let_expr(
                        "a_end",
                        Expr::ident("row_offsets").index(Expr::ident("fine_idx") + 1u32),
                    ),
                    var_expr("ax", Expr::lit_f32(0.0)),
                    for_loop_expr(
                        ForInit::Var {
                            name: "j".into(),
                            ty: None,
                            expr: Expr::ident("a_start"),
                        },
                        Expr::ident("j").lt(Expr::ident("a_end")),
                        ForStep::Increment(Expr::ident("j")),
                        block(vec![assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("ax"),
                            Expr::ident("values").index(Expr::ident("j"))
                                * Expr::ident("x")
                                    .index(Expr::ident("col_indices").index(Expr::ident("j"))),
                        )]),
                    ),
                    let_expr(
                        "fine_r",
                        Expr::ident("b").index(Expr::ident("fine_idx")) - Expr::ident("ax"),
                    ),
                    assign_op_expr(
                        AssignOp::Add,
                        Expr::ident("sum"),
                        Expr::ident("r_val") * Expr::ident("fine_r"),
                    ),
                ]),
            ),
            assign_expr(
                Expr::ident("coarse_vec").index(Expr::ident("i")),
                Expr::ident("sum"),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "restrict_residual",
            std_params.clone(),
            None,
            std_attrs.clone(),
            block(stmts),
        )));
    }

    // ── clear ──
    {
        let mut stmts = amg_preamble();
        stmts.push(assign_expr(
            Expr::ident("x").index(Expr::ident("i")),
            Expr::lit_f32(0.0),
        ));

        m.push(Item::Function(Function::new(
            "clear",
            std_params,
            None,
            std_attrs,
            block(stmts),
        )));
    }

    KernelWgsl::new(m)
}

// ─── gmres_update_fused ─────────────────────────────────────────────────

/// GMRES solution accumulator.  Single entry point: `accumulate_solution`.
pub fn generate_gmres_update_fused() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Struct(StructDef::new(
        "GmresParams",
        vec![
            StructField::new("n", Type::U32),
            StructField::new("num_cells", Type::U32),
            StructField::new("num_iters", Type::U32),
            StructField::new("omega", Type::F32),
            StructField::new("dispatch_x", Type::U32),
            StructField::new("max_restart", Type::U32),
            StructField::new("column_offset", Type::U32),
            StructField::new("_pad3", Type::U32),
        ],
    )));

    m.push(Item::Struct(StructDef::new(
        "IterParams",
        vec![
            StructField::new("current_idx", Type::U32),
            StructField::new("max_restart", Type::U32),
            StructField::new("_pad1", Type::U32),
            StructField::new("_pad2", Type::U32),
        ],
    )));

    m.push(Item::Function(Function::new(
        "global_index",
        vec![
            Param::new("global_id", Type::vec3_u32(), vec![]),
            Param::new("num_workgroups", Type::vec3_u32(), vec![]),
        ],
        Some(Type::U32),
        vec![],
        block(vec![return_expr(
            Expr::ident("global_id").field("y")
                * (Expr::ident("num_workgroups").field("x") * 64u32)
                + Expr::ident("global_id").field("x"),
        )]),
    )));

    // Group 0: vectors
    m.push(Item::GlobalVar(GlobalVar::new(
        "vec_x",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "vec_y",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "vec_z",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));

    // Group 1: matrix
    m.push(Item::GlobalVar(GlobalVar::new(
        "row_offsets",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(1), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "col_indices",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(1), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "matrix_values",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(1), Attribute::Binding(2)],
    )));

    // Group 2: diag vectors
    for (i, name) in ["diag_u", "diag_v", "diag_p"].iter().enumerate() {
        m.push(Item::GlobalVar(GlobalVar::new(
            *name,
            Type::array(Type::F32),
            StorageClass::Storage,
            Some(AccessMode::ReadWrite),
            vec![Attribute::Group(2), Attribute::Binding(i as u32)],
        )));
    }

    // Group 3: params + scalars + hessenberg + y_sol
    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("GmresParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(3), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "scalars",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(3), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "iter_params",
        Type::Custom("IterParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(3), Attribute::Binding(2)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "hessenberg",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(3), Attribute::Binding(3)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "y_sol",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(3), Attribute::Binding(4)],
    )));

    let body = block(vec![
        let_expr(
            "idx",
            Expr::call_named(
                "global_index",
                vec![Expr::ident("global_id"), Expr::ident("num_workgroups")],
            ),
        ),
        if_block_expr(
            Expr::ident("idx").ge(Expr::ident("params").field("n")),
            block(vec![return_void()]),
            None,
        ),
        // Skip if prior chunk already converged
        if_block_expr(
            Expr::ident("scalars").index(Expr::lit_u32(15)).gt(0.5f32),
            block(vec![return_void()]),
            None,
        ),
        let_expr(
            "k",
            u32_cast(clamp(
                round(Expr::ident("scalars").index(Expr::lit_u32(10))),
                Expr::lit_f32(1.0),
                f32_cast(Expr::ident("iter_params").field("max_restart")),
            )),
        ),
        let_expr(
            "z_stride",
            max(
                Expr::ident("params").field("column_offset"),
                Expr::ident("params").field("n"),
            ),
        ),
        var_expr("acc", Expr::ident("vec_y").index(Expr::ident("idx"))),
        for_loop_expr(
            ForInit::Var {
                name: "i".into(),
                ty: None,
                expr: Expr::lit_u32(0),
            },
            Expr::ident("i").lt(Expr::ident("k")),
            ForStep::Increment(Expr::ident("i")),
            block(vec![assign_expr(
                Expr::ident("acc"),
                Expr::ident("params").field("omega")
                    * Expr::ident("y_sol").index(Expr::ident("i"))
                    * Expr::ident("vec_x")
                        .index(Expr::ident("i") * Expr::ident("z_stride") + Expr::ident("idx"))
                    + Expr::ident("acc"),
            )]),
        ),
        assign_expr(
            Expr::ident("vec_y").index(Expr::ident("idx")),
            Expr::ident("acc"),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "accumulate_solution",
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

    KernelWgsl::new(m)
}

// ─── generic_coupled_schur_setup ────────────────────────────────────────

/// Builds per‐cell diagonal inverses and extracts pressure block for Schur
/// preconditioner.  Single entry point: `build_diag_and_pressure`.
pub fn generate_generic_coupled_schur_setup() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::GlobalVar(GlobalVar::new(
        "scalar_row_offsets",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "diagonal_indices",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "matrix_values",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "diag_u_inv",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "diag_p_inv",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(4)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "p_matrix_values",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(5)],
    )));

    m.push(Item::Struct(StructDef::new(
        "SetupParams",
        vec![
            StructField::new("num_cells", Type::U32),
            StructField::new("unknowns_per_cell", Type::U32),
            StructField::new("p", Type::U32),
            StructField::new("u_len", Type::U32),
            StructField::new("u0123", Type::Vec4(Box::new(Type::U32))),
            StructField::new("u4567", Type::Vec4(Box::new(Type::U32))),
        ],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("SetupParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(0), Attribute::Binding(6)],
    )));

    m.push(Item::Function(Function::new(
        "u_index",
        vec![Param::new("i", Type::U32, vec![])],
        Some(Type::U32),
        vec![],
        block(vec![
            if_block_expr(
                Expr::ident("i").lt(4u32),
                block(vec![return_expr(
                    Expr::ident("params").field("u0123").index(Expr::ident("i")),
                )]),
                None,
            ),
            return_expr(
                Expr::ident("params")
                    .field("u4567")
                    .index(Expr::ident("i") - 4u32),
            ),
        ]),
    )));

    m.push(Item::Function(Function::new(
        "safe_inverse",
        vec![Param::new("val", Type::F32, vec![])],
        Some(Type::F32),
        vec![],
        block(vec![
            if_block_expr(
                abs(Expr::ident("val")).gt(Expr::lit_f32(1e-14)),
                block(vec![return_expr(Expr::lit_f32(1.0) / Expr::ident("val"))]),
                None,
            ),
            return_expr(Expr::lit_f32(0.0)),
        ]),
    )));

    let params = Expr::ident("params");
    let body = block(vec![
        let_expr(
            "cell",
            Expr::ident("global_id").field("y")
                * (Expr::ident("num_workgroups").field("x") * 64u32)
                + Expr::ident("global_id").field("x"),
        ),
        if_block_expr(
            Expr::ident("cell").ge(params.clone().field("num_cells")),
            block(vec![return_void()]),
            None,
        ),
        let_expr(
            "scalar_offset",
            Expr::ident("scalar_row_offsets").index(Expr::ident("cell")),
        ),
        let_expr(
            "scalar_end",
            Expr::ident("scalar_row_offsets").index(Expr::ident("cell") + 1u32),
        ),
        let_expr(
            "num_neighbors",
            Expr::ident("scalar_end") - Expr::ident("scalar_offset"),
        ),
        let_expr(
            "diag_rank",
            Expr::ident("diagonal_indices").index(Expr::ident("cell"))
                - Expr::ident("scalar_offset"),
        ),
        let_expr(
            "block_stride",
            params.clone().field("unknowns_per_cell") * params.clone().field("unknowns_per_cell"),
        ),
        let_expr(
            "start_row_0",
            Expr::ident("scalar_offset") * Expr::ident("block_stride"),
        ),
        let_expr(
            "row_stride",
            Expr::ident("num_neighbors") * params.clone().field("unknowns_per_cell"),
        ),
        let_expr(
            "start_row_p",
            Expr::ident("start_row_0") + params.clone().field("p") * Expr::ident("row_stride"),
        ),
        let_expr(
            "diag_p",
            Expr::ident("matrix_values").index(
                Expr::ident("start_row_p")
                    + Expr::ident("diag_rank") * params.clone().field("unknowns_per_cell")
                    + params.clone().field("p"),
            ),
        ),
        for_loop_expr(
            ForInit::Var {
                name: "i".into(),
                ty: None,
                expr: Expr::lit_u32(0),
            },
            Expr::ident("i").lt(params.clone().field("u_len")),
            ForStep::Increment(Expr::ident("i")),
            block(vec![
                let_expr("u", Expr::call_named("u_index", vec![Expr::ident("i")])),
                let_expr(
                    "start_row_u",
                    Expr::ident("start_row_0") + Expr::ident("u") * Expr::ident("row_stride"),
                ),
                let_expr(
                    "diag_u",
                    Expr::ident("matrix_values").index(
                        Expr::ident("start_row_u")
                            + Expr::ident("diag_rank") * params.clone().field("unknowns_per_cell")
                            + Expr::ident("u"),
                    ),
                ),
                let_expr(
                    "inv_u",
                    Expr::call_named("safe_inverse", vec![Expr::ident("diag_u")]),
                ),
                assign_expr(
                    Expr::ident("diag_u_inv")
                        .index(Expr::ident("cell") * params.clone().field("u_len") + Expr::ident("i")),
                    Expr::ident("inv_u"),
                ),
            ]),
        ),
        assign_expr(
            Expr::ident("diag_p_inv").index(Expr::ident("cell")),
            Expr::call_named("safe_inverse", vec![Expr::ident("diag_p")]),
        ),
        for_loop_expr(
            ForInit::Var {
                name: "rank".into(),
                ty: None,
                expr: Expr::lit_u32(0),
            },
            Expr::ident("rank").lt(Expr::ident("num_neighbors")),
            ForStep::Increment(Expr::ident("rank")),
            block(vec![assign_expr(
                Expr::ident("p_matrix_values")
                    .index(Expr::ident("scalar_offset") + Expr::ident("rank")),
                Expr::ident("matrix_values").index(
                    Expr::ident("start_row_p")
                        + Expr::ident("rank") * params.clone().field("unknowns_per_cell")
                        + params.clone().field("p"),
                ),
            )]),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "build_diag_and_pressure",
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

    KernelWgsl::new(m)
}

// ─── outer_gate ─────────────────────────────────────────────────────────

/// GPU-side adaptive outer-loop gate.  Single `main` entry point,
/// workgroup_size(1,1,1).
///
/// Reads `break_status[0]`:
/// - If 0 (not converged): copies real dispatch args into indirect args and
///   atomically increments the iteration counter.
/// - If ≠ 0 (converged): zeroes the indirect dispatch args so subsequent
///   dispatches are no-ops.
pub fn generate_outer_gate() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::GlobalVar(GlobalVar::new(
        "break_status",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "real_args_cells",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "indirect_args_cells",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "real_args_faces",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "indirect_args_faces",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(4)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "iter_counter",
        Type::array(Type::atomic(Type::U32)),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(5)],
    )));

    let global_id = Expr::ident("global_id");
    let break_status = Expr::ident("break_status");
    let real_args_cells = Expr::ident("real_args_cells");
    let indirect_args_cells = Expr::ident("indirect_args_cells");
    let real_args_faces = Expr::ident("real_args_faces");
    let indirect_args_faces = Expr::ident("indirect_args_faces");
    let iter_counter = Expr::ident("iter_counter");
    let converged = Expr::ident("converged");

    let body = block(vec![
        if_block_expr(
            global_id.clone().field("x").ne(Expr::lit_u32(0)),
            block(vec![return_void()]),
            None,
        ),
        let_expr("converged", break_status.clone().index(Expr::lit_u32(0))),
        if_block_expr(
            converged.clone().eq(Expr::lit_u32(0)),
            block(vec![
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(0)),
                    real_args_cells.clone().index(Expr::lit_u32(0)),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(1)),
                    real_args_cells.clone().index(Expr::lit_u32(1)),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(2)),
                    real_args_cells.clone().index(Expr::lit_u32(2)),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(0)),
                    real_args_faces.clone().index(Expr::lit_u32(0)),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(1)),
                    real_args_faces.clone().index(Expr::lit_u32(1)),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(2)),
                    real_args_faces.clone().index(Expr::lit_u32(2)),
                ),
                call_stmt_expr(atomic_add(
                    iter_counter.clone().index(Expr::lit_u32(0)).addr_of(),
                    Expr::lit_u32(1),
                )),
            ]),
            Some(block(vec![
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(0)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(1)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_cells.clone().index(Expr::lit_u32(2)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(0)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(1)),
                    Expr::lit_u32(0),
                ),
                assign_expr(
                    indirect_args_faces.clone().index(Expr::lit_u32(2)),
                    Expr::lit_u32(0),
                ),
            ])),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "main",
        vec![Param::new(
            "global_id",
            Type::vec3_u32(),
            vec![Attribute::Builtin("global_invocation_id".into())],
        )],
        None,
        vec![Attribute::Compute, Attribute::WorkgroupSize3(1, 1, 1)],
        body,
    )));

    KernelWgsl::new(m)
}

// ─── outer_stop_inject ──────────────────────────────────────────────────

/// STOP-inject kernel that reads `break_status[0]` and writes it as an f32
/// into a scalars buffer at the given index.  Single `main` entry point,
/// workgroup_size(1,1,1).
fn generate_outer_stop_inject(scalar_stop: usize) -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::GlobalVar(GlobalVar::new(
        "break_status",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "scalars",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));

    let global_id = Expr::ident("global_id");
    let break_status = Expr::ident("break_status");
    let scalars = Expr::ident("scalars");

    let body = block(vec![
        if_block_expr(
            global_id.clone().field("x").ne(Expr::lit_u32(0)),
            block(vec![return_void()]),
            None,
        ),
        assign_expr(
            scalars.clone().index(Expr::lit_u32(scalar_stop as u32)),
            f32_cast(break_status.clone().index(Expr::lit_u32(0))),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "main",
        vec![Param::new(
            "global_id",
            Type::vec3_u32(),
            vec![Attribute::Builtin("global_invocation_id".into())],
        )],
        None,
        vec![Attribute::Compute, Attribute::WorkgroupSize3(1, 1, 1)],
        body,
    )));

    KernelWgsl::new(m)
}

/// STOP-inject for FGMRES (scalar index 8).
pub fn generate_outer_stop_inject_fgmres() -> KernelWgsl {
    generate_outer_stop_inject(8)
}

/// STOP-inject for CG (scalar index 6).
pub fn generate_outer_stop_inject_cg() -> KernelWgsl {
    generate_outer_stop_inject(6)
}

// ─── outer_convergence_break ────────────────────────────────────────────

/// Convergence break kernel: checks per-target delta vs scale tolerances.
/// Single `main` entry point, workgroup_size(1,1,1).
///
/// Legacy mode (`plateau_mode == 0`): reads `delta[i]` and `scale[i]` for
/// `i < params.count`, applies `tol = tol_abs + tol_rel * max(scale, 1.0)`,
/// and writes `status[0] = 1` if all targets are converged, `0` otherwise.
///
/// Plateau mode (`plateau_mode != 0`): GPU port of the host outer-loop
/// plateau detector (`generic_coupled::outer_corrections_plateaued`), so
/// plateau-driven models can run the batched one-submission outer path.
/// Semantics replicated EXACTLY (f32, same inputs — the same delta/scale
/// reductions the host read back):
///   - scaled correction `r = delta[i] / max(scale[i], 1.0)`;
///   - a field is UNDER TOL when `r <= tol_rel || r <= tol_abs`;
///   - TOLERANCE exit: every field under tol, from `min_iters_tol` sweeps;
///   - STALL exit: every field under tol OR adjacent-ratio
///     `r/r_prev ∈ [plateau_factor, plateau_ceiling]`, from
///     `min_iters_stall` sweeps.
/// The kernel keeps its own eval counter and previous-delta copy
/// (`eval_count`, `delta_prev`, cleared by the host at each step's first
/// outer); `delta_prev` starts zeroed so the first ratio is huge and blocks
/// the stall exit, matching the host's missing-prev rule.
pub fn generate_outer_convergence_break() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Struct(StructDef::new(
        "BreakParams",
        vec![
            StructField::new("count", Type::U32),
            StructField::new("tol_rel", Type::F32),
            StructField::new("tol_abs", Type::F32),
            StructField::new("plateau_factor", Type::F32),
            StructField::new("plateau_ceiling", Type::F32),
            StructField::new("min_iters_tol", Type::U32),
            StructField::new("min_iters_stall", Type::U32),
            StructField::new("plateau_mode", Type::U32),
        ],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "delta",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "scale",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "status",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("BreakParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "delta_prev",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(4)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "eval_count",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(5)],
    )));

    let global_id = Expr::ident("global_id");
    let params = Expr::ident("params");
    let delta = Expr::ident("delta");
    let scale = Expr::ident("scale");
    let status = Expr::ident("status");
    let delta_prev = Expr::ident("delta_prev");
    let eval_count = Expr::ident("eval_count");
    let it = Expr::ident("it");
    let i = Expr::ident("i");
    let d = Expr::ident("d");
    let s_raw = Expr::ident("s_raw");
    let bad_d = Expr::ident("bad_d");
    let bad_s = Expr::ident("bad_s");
    let s = Expr::ident("s");
    let tol = Expr::ident("tol");
    let converged = Expr::ident("converged");
    let r_cur = Expr::ident("r_cur");
    let r_prev = Expr::ident("r_prev");
    let ratio = Expr::ident("ratio");
    let all_under = Expr::ident("all_under");
    let all_band = Expr::ident("all_band");
    let st = Expr::ident("st");

    let legacy_loop = for_loop_expr(
        for_init_var_typed_expr("i", Type::U32, Expr::lit_u32(0)),
        i.clone().lt(params.clone().field("count")),
        for_step_increment_expr(i.clone()),
        block(vec![
            let_expr("d", delta.clone().index(i.clone())),
            let_expr("s_raw", scale.clone().index(i.clone())),
            let_expr(
                "bad_d",
                (!d.clone().le(d.clone())) | abs(d.clone()).gt(Expr::lit_f32(1.0e30)),
            ),
            let_expr(
                "bad_s",
                (!s_raw.clone().le(s_raw.clone()))
                    | abs(s_raw.clone()).gt(Expr::lit_f32(1.0e30)),
            ),
            if_block_expr(
                bad_d.clone() | bad_s.clone(),
                block(vec![
                    assign_expr(converged.clone(), Expr::lit_u32(0)),
                    break_stmt(),
                ]),
                None,
            ),
            let_expr("s", max(s_raw.clone(), Expr::lit_f32(1.0))),
            let_expr(
                "tol",
                params.clone().field("tol_abs") + params.clone().field("tol_rel") * s.clone(),
            ),
            if_block_expr(
                d.clone().gt(tol.clone()),
                block(vec![
                    assign_expr(converged.clone(), Expr::lit_u32(0)),
                    break_stmt(),
                ]),
                None,
            ),
        ]),
    );

    // Plateau-mode field loop; host semantics replicated exactly.
    let plateau_loop = for_loop_expr(
        for_init_var_typed_expr("i", Type::U32, Expr::lit_u32(0)),
        i.clone().lt(params.clone().field("count")),
        for_step_increment_expr(i.clone()),
        block(vec![
            let_expr("d", delta.clone().index(i.clone())),
            let_expr("s_raw", scale.clone().index(i.clone())),
            let_expr(
                "bad_d",
                (!d.clone().le(d.clone())) | abs(d.clone()).gt(Expr::lit_f32(1.0e30)),
            ),
            let_expr(
                "bad_s",
                (!s_raw.clone().le(s_raw.clone()))
                    | abs(s_raw.clone()).gt(Expr::lit_f32(1.0e30)),
            ),
            if_block_expr(
                bad_d.clone() | bad_s.clone(),
                block(vec![
                    assign_expr(all_under.clone(), Expr::lit_u32(0)),
                    assign_expr(all_band.clone(), Expr::lit_u32(0)),
                    break_stmt(),
                ]),
                None,
            ),
            let_expr("s", max(s_raw.clone(), Expr::lit_f32(1.0))),
            let_expr("r_cur", d.clone() / s.clone()),
            if_block_expr(
                !(r_cur.clone().le(params.clone().field("tol_rel"))
                    | r_cur.clone().le(params.clone().field("tol_abs"))),
                block(vec![
                    assign_expr(all_under.clone(), Expr::lit_u32(0)),
                    let_expr("r_prev", delta_prev.clone().index(i.clone()) / s.clone()),
                    let_expr(
                        "ratio",
                        r_cur.clone() / max(r_prev.clone(), Expr::lit_f32(1.0e-30)),
                    ),
                    if_block_expr(
                        ratio.clone().lt(params.clone().field("plateau_factor"))
                            | ratio.clone().gt(params.clone().field("plateau_ceiling")),
                        block(vec![assign_expr(all_band.clone(), Expr::lit_u32(0))]),
                        None,
                    ),
                ]),
                None,
            ),
        ]),
    );

    let body = block(vec![
        if_block_expr(
            global_id.clone().field("x").ne(Expr::lit_u32(0)),
            block(vec![return_void()]),
            None,
        ),
        // Sweeps completed including this one; host clears `eval_count` before
        // each step's first outer.
        let_expr("it", eval_count.clone().index(Expr::lit_u32(0)) + Expr::lit_u32(1)),
        assign_expr(eval_count.clone().index(Expr::lit_u32(0)), it.clone()),
        if_block_expr(
            params.clone().field("plateau_mode").eq(Expr::lit_u32(0)),
            block(vec![
                var_typed_expr("converged", Type::U32, Some(Expr::lit_u32(1))),
                legacy_loop,
                assign_expr(status.clone().index(Expr::lit_u32(0)), converged.clone()),
            ]),
            Some(block(vec![
                var_typed_expr("all_under", Type::U32, Some(Expr::lit_u32(1))),
                var_typed_expr("all_band", Type::U32, Some(Expr::lit_u32(1))),
                plateau_loop,
                var_typed_expr("st", Type::U32, Some(Expr::lit_u32(0))),
                if_block_expr(
                    it.clone().ge(params.clone().field("min_iters_tol")),
                    block(vec![if_block_expr(
                        all_under.clone().eq(Expr::lit_u32(1)),
                        block(vec![assign_expr(st.clone(), Expr::lit_u32(1))]),
                        None,
                    )]),
                    None,
                ),
                if_block_expr(
                    it.clone().ge(params.clone().field("min_iters_stall")),
                    block(vec![if_block_expr(
                        all_band.clone().eq(Expr::lit_u32(1)),
                        block(vec![assign_expr(st.clone(), Expr::lit_u32(1))]),
                        None,
                    )]),
                    None,
                ),
                assign_expr(status.clone().index(Expr::lit_u32(0)), st.clone()),
            ])),
        ),
        // Roll current deltas into `delta_prev` for the next sweep's ratio
        // (after all reads).
        for_loop_expr(
            for_init_var_typed_expr("i", Type::U32, Expr::lit_u32(0)),
            i.clone().lt(params.clone().field("count")),
            for_step_increment_expr(i.clone()),
            block(vec![assign_expr(
                delta_prev.clone().index(i.clone()),
                delta.clone().index(i.clone()),
            )]),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "main",
        vec![Param::new(
            "global_id",
            Type::vec3_u32(),
            vec![Attribute::Builtin("global_invocation_id".into())],
        )],
        None,
        vec![Attribute::Compute, Attribute::WorkgroupSize3(1, 1, 1)],
        body,
    )));

    KernelWgsl::new(m)
}

/// Returns all infrastructure kernel generators as `(filename, generator)` pairs.
/// The filename should be used when writing to `shaders/generated/`.
pub fn all_infrastructure_kernels() -> Vec<(&'static str, fn() -> KernelWgsl)> {
    vec![
        ("dot_product.wgsl", generate_dot_product),
        ("dot_product_pair.wgsl", generate_dot_product_pair),
        ("outer_convergence.wgsl", generate_outer_convergence),
        (
            "outer_convergence_break.wgsl",
            generate_outer_convergence_break,
        ),
        ("outer_gate.wgsl", generate_outer_gate),
        (
            "outer_stop_inject_fgmres.wgsl",
            generate_outer_stop_inject_fgmres,
        ),
        (
            "outer_stop_inject_cg.wgsl",
            generate_outer_stop_inject_cg,
        ),
        ("scalars.wgsl", generate_scalars),
        ("linear_solver.wgsl", generate_linear_solver),
        ("amg.wgsl", generate_amg),
        ("gmres_update_fused.wgsl", generate_gmres_update_fused),
        (
            "generic_coupled_schur_setup.wgsl",
            generate_generic_coupled_schur_setup,
        ),
        ("gmres_ops.wgsl", generate_gmres_ops),
        ("gmres_cgs.wgsl", generate_gmres_cgs),
        ("gmres_logic.wgsl", generate_gmres_logic),
        ("schur_precond_generic.wgsl", generate_schur_precond_generic),
        ("block_precond.wgsl", generate_block_precond),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_generates_valid_wgsl() {
        let wgsl = generate_dot_product().to_wgsl();
        assert!(wgsl.contains("fn main("));
        assert!(wgsl.contains("@compute"));
        assert!(wgsl.contains("@workgroup_size(64)"));
        assert!(wgsl.contains("var<workgroup> scratch: array<f32, 64>"));
        assert!(wgsl.contains("dot_a[idx]"));
        assert!(wgsl.contains("dot_b[idx]"));
        assert!(wgsl.contains("workgroupBarrier()"));
    }

    #[test]
    fn dot_product_pair_generates_valid_wgsl() {
        let wgsl = generate_dot_product_pair().to_wgsl();
        assert!(wgsl.contains("fn main("));
        assert!(wgsl.contains("scratch_a"));
        assert!(wgsl.contains("scratch_b"));
        assert!(wgsl.contains("dot_a0"));
        assert!(wgsl.contains("dot_b1"));
        assert!(wgsl.contains("loop {"));
        assert!(wgsl.contains("break;"));
    }

    #[test]
    fn outer_convergence_generates_valid_wgsl() {
        let wgsl = generate_outer_convergence().to_wgsl();
        assert!(wgsl.contains("fn main("));
        assert!(wgsl.contains("@workgroup_size(256)"));
        assert!(wgsl.contains("atomicMax"));
        assert!(wgsl.contains("bitcast<u32>"));
        assert!(wgsl.contains("TargetDesc"));
    }

    #[test]
    fn outer_gate_generates_valid_wgsl() {
        let wgsl = generate_outer_gate().to_wgsl();
        assert!(wgsl.contains("fn main("));
        assert!(wgsl.contains("@workgroup_size(1, 1, 1)"));
        assert!(wgsl.contains("break_status"));
        assert!(wgsl.contains("indirect_args_cells"));
        assert!(wgsl.contains("indirect_args_faces"));
        assert!(wgsl.contains("atomicAdd"));
        assert!(wgsl.contains("iter_counter"));
    }

    #[test]
    fn outer_stop_inject_fgmres_generates_valid_wgsl() {
        let wgsl = generate_outer_stop_inject_fgmres().to_wgsl();
        assert!(wgsl.contains("fn main("));
        assert!(wgsl.contains("@workgroup_size(1, 1, 1)"));
        assert!(wgsl.contains("break_status"));
        assert!(wgsl.contains("scalars[8u]"));
    }

    #[test]
    fn outer_stop_inject_cg_generates_valid_wgsl() {
        let wgsl = generate_outer_stop_inject_cg().to_wgsl();
        assert!(wgsl.contains("fn main("));
        assert!(wgsl.contains("@workgroup_size(1, 1, 1)"));
        assert!(wgsl.contains("break_status"));
        assert!(wgsl.contains("scalars[6u]"));
    }

    #[test]
    fn outer_convergence_break_generates_valid_wgsl() {
        let wgsl = generate_outer_convergence_break().to_wgsl();
        assert!(wgsl.contains("fn main("));
        assert!(wgsl.contains("@workgroup_size(1, 1, 1)"));
        assert!(wgsl.contains("BreakParams"));
        assert!(wgsl.contains("delta"));
        assert!(wgsl.contains("scale"));
        assert!(wgsl.contains("status[0u]"));
        assert!(wgsl.contains("tol_abs"));
        assert!(wgsl.contains("tol_rel"));
    }

    #[test]
    fn scalars_generates_valid_wgsl() {
        let wgsl = generate_scalars().to_wgsl();
        assert!(wgsl.contains("fn reduce_rho_new_r_r("));
        assert!(wgsl.contains("fn reduce_r0_v("));
        assert!(wgsl.contains("fn init_cg_scalars("));
        assert!(wgsl.contains("GpuScalars"));
        assert!(wgsl.contains("scratch1"));
    }

    #[test]
    fn linear_solver_generates_valid_wgsl() {
        let wgsl = generate_linear_solver().to_wgsl();
        assert!(wgsl.contains("fn spmv_p_v("));
        assert!(wgsl.contains("fn cg_update_x_r("));
        assert!(wgsl.contains("fn cg_update_p("));
        assert!(wgsl.contains("fn global_index("));
    }

    #[test]
    fn amg_generates_valid_wgsl() {
        let wgsl = generate_amg().to_wgsl();
        assert!(wgsl.contains("fn smooth_op("));
        assert!(wgsl.contains("fn prolongate_op("));
        assert!(wgsl.contains("fn restrict_residual("));
        assert!(wgsl.contains("fn clear("));
        assert!(wgsl.contains("fn amg_should_stop("));
    }

    #[test]
    fn gmres_update_fused_generates_valid_wgsl() {
        let wgsl = generate_gmres_update_fused().to_wgsl();
        assert!(wgsl.contains("fn accumulate_solution("));
        assert!(wgsl.contains("y_sol"));
        assert!(wgsl.contains("GmresParams"));
        assert!(wgsl.contains("IterParams"));
    }

    #[test]
    fn generic_coupled_schur_setup_generates_valid_wgsl() {
        let wgsl = generate_generic_coupled_schur_setup().to_wgsl();
        assert!(wgsl.contains("fn build_diag_and_pressure("));
        assert!(wgsl.contains("fn u_index("));
        assert!(wgsl.contains("fn safe_inverse("));
        assert!(wgsl.contains("SetupParams"));
    }

    #[test]
    fn gmres_ops_generates_valid_wgsl() {
        let wgsl = generate_gmres_ops().to_wgsl();
        // Check all 14 entry points
        assert!(wgsl.contains("fn spmv("));
        assert!(wgsl.contains("fn axpy("));
        assert!(wgsl.contains("fn axpy_from_y("));
        assert!(wgsl.contains("fn axpby("));
        assert!(wgsl.contains("fn scale("));
        assert!(wgsl.contains("fn scale_in_place("));
        assert!(wgsl.contains("fn copy("));
        assert!(wgsl.contains("fn dot_product_partial("));
        assert!(wgsl.contains("fn norm_sq_partial("));
        assert!(wgsl.contains("fn orthogonalize("));
        assert!(wgsl.contains("fn reduce_final("));
        assert!(wgsl.contains("fn reduce_final_and_finish_norm("));
        assert!(wgsl.contains("fn extract_diag_inv("));
        assert!(wgsl.contains("fn apply_diag_inv("));
        // Check helper functions
        assert!(wgsl.contains("fn global_index("));
        assert!(wgsl.contains("fn workgroup_index("));
        assert!(wgsl.contains("fn safe_inverse("));
        // Check structs and constants
        assert!(wgsl.contains("struct GmresParams"));
        assert!(wgsl.contains("struct IterParams"));
        assert!(wgsl.contains("const WORKGROUP_SIZE: u32 = 64u"));
        assert!(wgsl.contains("const SCALAR_STOP: u32 = 8u"));
        // Check global vars
        assert!(wgsl.contains("var<workgroup> partial_sums: array<f32, 64>"));
        assert!(wgsl.contains("@group(0) @binding(0)"));
        assert!(wgsl.contains("@group(3) @binding(4)"));
    }

    #[test]
    fn gmres_cgs_generates_valid_wgsl() {
        let wgsl = generate_gmres_cgs().to_wgsl();
        // Check all 3 entry points
        assert!(wgsl.contains("fn calc_dots_cgs("));
        assert!(wgsl.contains("fn reduce_dots_cgs("));
        assert!(wgsl.contains("fn update_w_cgs("));
        // Check struct
        assert!(wgsl.contains("struct Params"));
        assert!(wgsl.contains("num_iters"));
        assert!(wgsl.contains("max_restart"));
        // Check constants
        assert!(wgsl.contains("const WORKGROUP_SIZE: u32 = 64u"));
        assert!(wgsl.contains("const SCALAR_STOP: u32 = 8u"));
        // Check workgroup vars
        assert!(wgsl.contains("var<workgroup> sdata: array<f32, 64>"));
        assert!(wgsl.contains("var<workgroup> sdata_vec4: array<vec4<f32>, 64>"));
        // Check global vars
        assert!(wgsl.contains("b_basis"));
        assert!(wgsl.contains("b_w"));
        assert!(wgsl.contains("b_dot_partial"));
        assert!(wgsl.contains("b_hessenberg"));
        assert!(wgsl.contains("scalars"));
        // Check bindings
        assert!(wgsl.contains("@group(0) @binding(0)"));
        assert!(wgsl.contains("@group(0) @binding(5)"));
        // Check compute attributes
        assert!(wgsl.contains("@compute"));
        assert!(wgsl.contains("@workgroup_size(64)"));
        // Check builtin params
        assert!(wgsl.contains("@builtin(global_invocation_id)"));
        assert!(wgsl.contains("@builtin(local_invocation_id)"));
        assert!(wgsl.contains("@builtin(workgroup_id)"));
        assert!(wgsl.contains("@builtin(num_workgroups)"));
        // Check workgroupBarrier
        assert!(wgsl.contains("workgroupBarrier()"));
    }

    #[test]
    fn gmres_logic_generates_valid_wgsl() {
        let wgsl = generate_gmres_logic().to_wgsl();
        // Check all 3 entry points
        assert!(wgsl.contains("fn update_hessenberg_givens("));
        assert!(wgsl.contains("fn solve_triangular("));
        assert!(wgsl.contains("fn finish_norm("));
        // Check struct
        assert!(wgsl.contains("struct IterParams"));
        assert!(wgsl.contains("current_idx"));
        assert!(wgsl.contains("max_restart"));
        // Check constants
        assert!(wgsl.contains("const SCALAR_STOP: u32 = 8u"));
        assert!(wgsl.contains("const SCALAR_CONVERGED: u32 = 9u"));
        assert!(wgsl.contains("const SCALAR_ITERS_USED: u32 = 10u"));
        assert!(wgsl.contains("const SCALAR_RESIDUAL_EST: u32 = 11u"));
        assert!(wgsl.contains("const SCALAR_TOL_REL_RHS: u32 = 12u"));
        assert!(wgsl.contains("const SCALAR_TOL_ABS: u32 = 13u"));
        assert!(wgsl.contains("const SCALAR_RHS_NORM: u32 = 14u"));
        assert!(wgsl.contains("const SCALAR_SKIP_UPDATE: u32 = 15u"));
        // Check helper function
        assert!(wgsl.contains("fn h_idx("));
        // Check global vars
        assert!(wgsl.contains("hessenberg"));
        assert!(wgsl.contains("givens"));
        assert!(wgsl.contains("g_rhs"));
        assert!(wgsl.contains("y_sol"));
        assert!(wgsl.contains("scalars"));
        assert!(wgsl.contains("indirect_args"));
        assert!(wgsl.contains("iter_params"));
        // Check bindings
        assert!(wgsl.contains("@group(0) @binding(0)"));
        assert!(wgsl.contains("@group(1) @binding(2)"));
        // Check compute attributes
        assert!(wgsl.contains("@compute"));
        assert!(wgsl.contains("@workgroup_size(1)"));
        // Check vec2<f32> constructor in Givens storage
        assert!(wgsl.contains("vec2<f32>(c, s)"));
        // Check vec4<u32> for indirect_args zeroing
        assert!(wgsl.contains("vec4<u32>(0u, 0u, 0u, 0u)"));
        // Check Givens rotation logic
        assert!(wgsl.contains("sqrt("));
        assert!(wgsl.contains("abs("));
        // Check backward substitution
        assert!(wgsl.contains("clamp("));
        assert!(wgsl.contains("round("));
    }

    #[test]
    fn schur_precond_generic_generates_valid_wgsl() {
        let wgsl = generate_schur_precond_generic().to_wgsl();
        // Check all 3 entry points
        assert!(wgsl.contains("fn relax_pressure("));
        assert!(wgsl.contains("fn correct_velocity("));
        assert!(wgsl.contains("fn predict_and_form_schur("));
        // Check struct
        assert!(wgsl.contains("struct PrecondParams"));
        assert!(wgsl.contains("unknowns_per_cell"));
        assert!(wgsl.contains("u0123"));
        assert!(wgsl.contains("u4567"));
        // Check helper functions
        assert!(wgsl.contains("fn safe_inverse("));
        assert!(wgsl.contains("fn u_index("));
        assert!(wgsl.contains("fn global_cell("));
        // Check constants
        assert!(wgsl.contains("const WORKGROUP_SIZE: u32 = 64u"));
        // Check global vars across all groups
        assert!(wgsl.contains("r_in"));
        assert!(wgsl.contains("z_out"));
        assert!(wgsl.contains("temp_p"));
        assert!(wgsl.contains("p_sol"));
        assert!(wgsl.contains("p_prev"));
        assert!(wgsl.contains("diag_u_inv"));
        assert!(wgsl.contains("diag_p_inv"));
        assert!(wgsl.contains("p_row_offsets"));
        assert!(wgsl.contains("p_col_indices"));
        assert!(wgsl.contains("p_matrix_values"));
        // Check bindings
        assert!(wgsl.contains("@group(0) @binding(0)"));
        assert!(wgsl.contains("@group(3) @binding(2)"));
        // Check compute attributes
        assert!(wgsl.contains("@compute"));
        assert!(wgsl.contains("@workgroup_size(64)"));
        // Check mix() call for Chebyshev/SOR
        assert!(wgsl.contains("mix("));
        // Check modulo for velocity correction
        assert!(wgsl.contains("%"));
    }

    #[test]
    fn block_precond_generates_valid_wgsl() {
        let wgsl = generate_block_precond().to_wgsl();
        // Check both entry points
        assert!(wgsl.contains("fn build_block_inv("));
        assert!(wgsl.contains("fn apply_block_precond("));
        // Check structs
        assert!(wgsl.contains("struct GmresParams"));
        assert!(wgsl.contains("struct IterParams"));
        // Check const
        assert!(wgsl.contains("const MAX_BLOCK: u32 = 16u"));
        // Check helper functions
        assert!(wgsl.contains("fn safe_inverse("));
        assert!(wgsl.contains("fn swap_rows("));
        // Check pointer params in swap_rows
        assert!(wgsl.contains("ptr<function, array<array<f32, MAX_BLOCK>, MAX_BLOCK>>"));
        // Check pointer dereference
        assert!(wgsl.contains("*a"));
        assert!(wgsl.contains("*b"));
        // Check global vars across all groups
        assert!(wgsl.contains("vec_x"));
        assert!(wgsl.contains("vec_y"));
        assert!(wgsl.contains("vec_z"));
        assert!(wgsl.contains("row_offsets"));
        assert!(wgsl.contains("col_indices"));
        assert!(wgsl.contains("matrix_values"));
        assert!(wgsl.contains("block_inv"));
        assert!(wgsl.contains("scalars"));
        assert!(wgsl.contains("hessenberg"));
        assert!(wgsl.contains("y_sol"));
        // Check bindings
        assert!(wgsl.contains("@group(0) @binding(0)"));
        assert!(wgsl.contains("@group(3) @binding(4)"));
        // Check compute attributes
        assert!(wgsl.contains("@compute"));
        assert!(wgsl.contains("@workgroup_size(64)"));
        // Check safe_inverse with sign() and 3-branch logic
        assert!(wgsl.contains("sign("));
        assert!(wgsl.contains("abs("));
        // Check Gauss-Jordan logic
        assert!(wgsl.contains("singular"));
        assert!(wgsl.contains("pivot"));
        assert!(wgsl.contains("swap_rows("));
        assert!(wgsl.contains("select("));
        // Check address-of for swap_rows call
        assert!(wgsl.contains("&a"));
        assert!(wgsl.contains("&inv"));
    }

    /// Cross-check: every infrastructure kernel must have non-empty
    /// structured bindings when generated through `KernelWgsl::new(Module)`.
    ///
    /// This ensures that `Module::bindings()` extracts the same metadata that
    /// the text-based parser would, and catches regressions where a kernel
    /// accidentally drops all its `@group/@binding` annotations.
    #[test]
    fn all_infrastructure_kernels_have_structured_bindings() {
        for (filename, generator) in all_infrastructure_kernels() {
            let kernel = generator();
            let bindings = kernel.bindings();
            assert!(
                !bindings.is_empty(),
                "infrastructure kernel '{}' has no structured bindings — \
                 did Module::bindings() extraction break?",
                filename
            );
            // Verify that every binding has a non-empty name
            for bd in bindings {
                assert!(
                    !bd.name.is_empty(),
                    "infrastructure kernel '{}': binding @group({}) @binding({}) has empty name",
                    filename,
                    bd.group,
                    bd.binding
                );
            }
        }
    }

    /// Cross-check: structured bindings from Module::bindings() must match
    /// the text-parsed bindings from the WGSL output.
    #[test]
    fn infrastructure_kernel_bindings_match_text_parse() {
        use crate::solver::codegen::kernel_wgsl::parse_wgsl_bindings_from_text;

        for (filename, generator) in all_infrastructure_kernels() {
            let kernel = generator();
            let structured = kernel.bindings();
            let text_parsed = parse_wgsl_bindings_from_text(&kernel.to_wgsl());

            assert_eq!(
                structured.len(),
                text_parsed.len(),
                "infrastructure kernel '{}': structured binding count ({}) != text-parsed count ({})",
                filename,
                structured.len(),
                text_parsed.len()
            );

            for (s, t) in structured.iter().zip(text_parsed.iter()) {
                assert_eq!(
                    (s.group, s.binding, &s.name),
                    (t.group, t.binding, &t.name),
                    "infrastructure kernel '{}': binding mismatch at @group({}) @binding({})",
                    filename,
                    s.group,
                    s.binding
                );
            }
        }
    }
}
