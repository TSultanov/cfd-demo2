//! GMRES operations kernel generator implementation.
//!
//! This module contains the implementation of generate_gmres_ops() which was
//! ported from the handwritten src/solver/gpu/shaders/gmres_ops.wgsl file.

use crate::solver::codegen::kernel_wgsl::KernelWgsl;
use crate::solver::codegen::wgsl_ast::*;
use crate::solver::codegen::wgsl_dsl::*;

/// Generate the gmres_ops.wgsl kernel with 14 entry points.
pub fn generate_gmres_ops() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Comment("GMRES/FGMRES GPU Operations".into()));

    // ── Structs ─────────────────────────────────────────────────────────────

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

    m.push(Item::Const {
        name: "SCALAR_GUARD_FLAG".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(17),
    });

    // ── Helper functions ────────────────────────────────────────────────────

    // global_index
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
                * (Expr::ident("num_workgroups").field("x") * Expr::ident("WORKGROUP_SIZE"))
                + Expr::ident("global_id").field("x"),
        )]),
    )));

    // workgroup_index
    m.push(Item::Function(Function::new(
        "workgroup_index",
        vec![
            Param::new("group_id", Type::vec3_u32(), vec![]),
            Param::new("num_workgroups", Type::vec3_u32(), vec![]),
        ],
        Some(Type::U32),
        vec![],
        block(vec![return_expr(
            Expr::ident("group_id").field("y") * Expr::ident("num_workgroups").field("x")
                + Expr::ident("group_id").field("x"),
        )]),
    )));

    // safe_inverse
    m.push(Item::Function(Function::new(
        "safe_inverse",
        vec![Param::new("val", Type::F32, vec![])],
        Some(Type::F32),
        vec![],
        block(vec![
            let_expr("abs_val", abs(Expr::ident("val"))),
            if_block_expr(
                Expr::ident("abs_val").gt(Expr::lit_f32(1e-12)),
                block(vec![return_expr(Expr::lit_f32(1.0) / Expr::ident("val"))]),
                None,
            ),
            if_block_expr(
                Expr::ident("abs_val").gt(Expr::lit_f32(0.0)),
                block(vec![return_expr(
                    sign(Expr::ident("val")) * Expr::lit_f32(1.0e12),
                )]),
                None,
            ),
            return_expr(Expr::lit_f32(0.0)),
        ]),
    )));

    // ── Global variables ────────────────────────────────────────────────────

    // Group 0: Vectors
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

    // Group 1: Matrix (CSR format)
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

    // Group 2: Preconditioner data
    m.push(Item::GlobalVar(GlobalVar::new(
        "diag_u",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(2), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "diag_v",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(2), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "diag_p",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(2), Attribute::Binding(2)],
    )));

    // Group 3: Parameters and scalars
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

    // Workgroup variable
    m.push(Item::GlobalVar(GlobalVar::new(
        "partial_sums",
        Type::sized_array(Type::F32, 64),
        StorageClass::Workgroup,
        None,
        vec![],
    )));

    // ── Standard parameters for compute shaders ─────────────────────────────

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

    let std_params_with_local = vec![
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
    ];

    // ── Entry point: spmv ───────────────────────────────────────────────────
    {
        let body = block(vec![
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
                for_step_increment_expr(Expr::ident("k")),
                block(vec![
                    let_expr("col", Expr::ident("col_indices").index(Expr::ident("k"))),
                    let_expr("val", Expr::ident("matrix_values").index(Expr::ident("k"))),
                    assign_op_expr(
                        AssignOp::Add,
                        Expr::ident("sum"),
                        Expr::ident("val") * Expr::ident("vec_x").index(Expr::ident("col")),
                    ),
                ]),
            ),
            assign_expr(
                Expr::ident("vec_y").index(Expr::ident("row")),
                Expr::ident("sum"),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "spmv",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: axpy ───────────────────────────────────────────────────
    {
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
            let_expr("alpha", Expr::ident("scalars").index(Expr::lit_u32(0))),
            assign_expr(
                Expr::ident("vec_y").index(Expr::ident("idx")),
                Expr::ident("alpha") * Expr::ident("vec_x").index(Expr::ident("idx"))
                    + Expr::ident("vec_y").index(Expr::ident("idx")),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "axpy",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: axpy_from_y ────────────────────────────────────────────
    {
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
            let_expr(
                "alpha",
                Expr::ident("y_sol").index(Expr::ident("iter_params").field("current_idx")),
            ),
            assign_expr(
                Expr::ident("vec_y").index(Expr::ident("idx")),
                Expr::ident("alpha") * Expr::ident("vec_x").index(Expr::ident("idx"))
                    + Expr::ident("vec_y").index(Expr::ident("idx")),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "axpy_from_y",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: axpby ──────────────────────────────────────────────────
    {
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
            let_expr("alpha", Expr::ident("scalars").index(Expr::lit_u32(0))),
            let_expr("beta", Expr::ident("scalars").index(Expr::lit_u32(1))),
            assign_expr(
                Expr::ident("vec_z").index(Expr::ident("idx")),
                Expr::ident("alpha") * Expr::ident("vec_x").index(Expr::ident("idx"))
                    + Expr::ident("beta") * Expr::ident("vec_y").index(Expr::ident("idx")),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "axpby",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: scale ──────────────────────────────────────────────────
    {
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
            let_expr("alpha", Expr::ident("scalars").index(Expr::lit_u32(0))),
            assign_expr(
                Expr::ident("vec_y").index(Expr::ident("idx")),
                Expr::ident("alpha") * Expr::ident("vec_x").index(Expr::ident("idx")),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "scale",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: scale_in_place ─────────────────────────────────────────
    {
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
            let_expr("alpha", Expr::ident("scalars").index(Expr::lit_u32(0))),
            assign_expr(
                Expr::ident("vec_y").index(Expr::ident("idx")),
                Expr::ident("alpha") * Expr::ident("vec_y").index(Expr::ident("idx")),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "scale_in_place",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: copy ───────────────────────────────────────────────────
    {
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
            assign_expr(
                Expr::ident("vec_y").index(Expr::ident("idx")),
                Expr::ident("vec_x").index(Expr::ident("idx")),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "copy",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: dot_product_partial ────────────────────────────────────
    {
        let body = block(vec![
            let_expr(
                "idx",
                Expr::call_named(
                    "global_index",
                    vec![Expr::ident("global_id"), Expr::ident("num_workgroups")],
                ),
            ),
            let_expr("lid", Expr::ident("local_id").field("x")),
            var_expr("local_sum", Expr::lit_f32(0.0)),
            if_block_expr(
                Expr::ident("idx").lt(Expr::ident("params").field("n")),
                block(vec![assign_expr(
                    Expr::ident("local_sum"),
                    Expr::ident("vec_x").index(Expr::ident("idx"))
                        * Expr::ident("vec_y").index(Expr::ident("idx")),
                )]),
                None,
            ),
            assign_expr(
                Expr::ident("partial_sums").index(Expr::ident("lid")),
                Expr::ident("local_sum"),
            ),
            workgroup_barrier(),
            for_loop_expr(
                ForInit::Var {
                    name: "stride".into(),
                    ty: None,
                    expr: Expr::lit_u32(32),
                },
                Expr::ident("stride").gt(Expr::lit_u32(0)),
                for_step_assign_op_expr(
                    AssignOp::ShiftRight,
                    Expr::ident("stride"),
                    Expr::lit_u32(1),
                ),
                block(vec![
                    if_block_expr(
                        Expr::ident("lid").lt(Expr::ident("stride")),
                        block(vec![assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("partial_sums").index(Expr::ident("lid")),
                            Expr::ident("partial_sums")
                                .index(Expr::ident("lid") + Expr::ident("stride")),
                        )]),
                        None,
                    ),
                    workgroup_barrier(),
                ]),
            ),
            if_block_expr(
                Expr::ident("lid").eq(Expr::lit_u32(0)),
                block(vec![
                    let_expr(
                        "wg_idx",
                        Expr::call_named(
                            "workgroup_index",
                            vec![Expr::ident("group_id"), Expr::ident("num_workgroups")],
                        ),
                    ),
                    let_expr(
                        "num_groups_n",
                        (Expr::ident("params").field("n") + (Expr::ident("WORKGROUP_SIZE") - 1u32))
                            / Expr::ident("WORKGROUP_SIZE"),
                    ),
                    if_block_expr(
                        Expr::ident("wg_idx").lt(Expr::ident("num_groups_n")),
                        block(vec![assign_expr(
                            Expr::ident("vec_z").index(Expr::ident("wg_idx")),
                            Expr::ident("partial_sums").index(Expr::lit_u32(0)),
                        )]),
                        None,
                    ),
                ]),
                None,
            ),
        ]);

        m.push(Item::Function(Function::new(
            "dot_product_partial",
            std_params_with_local.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: norm_sq_partial ────────────────────────────────────────
    {
        let body = block(vec![
            let_expr(
                "idx",
                Expr::call_named(
                    "global_index",
                    vec![Expr::ident("global_id"), Expr::ident("num_workgroups")],
                ),
            ),
            let_expr("lid", Expr::ident("local_id").field("x")),
            var_expr("local_sum", Expr::lit_f32(0.0)),
            if_block_expr(
                Expr::ident("idx").lt(Expr::ident("params").field("n")),
                block(vec![
                    let_expr("val", Expr::ident("vec_x").index(Expr::ident("idx"))),
                    assign_expr(
                        Expr::ident("local_sum"),
                        Expr::ident("val") * Expr::ident("val"),
                    ),
                ]),
                None,
            ),
            assign_expr(
                Expr::ident("partial_sums").index(Expr::ident("lid")),
                Expr::ident("local_sum"),
            ),
            workgroup_barrier(),
            for_loop_expr(
                ForInit::Var {
                    name: "stride".into(),
                    ty: None,
                    expr: Expr::lit_u32(32),
                },
                Expr::ident("stride").gt(Expr::lit_u32(0)),
                for_step_assign_op_expr(
                    AssignOp::ShiftRight,
                    Expr::ident("stride"),
                    Expr::lit_u32(1),
                ),
                block(vec![
                    if_block_expr(
                        Expr::ident("lid").lt(Expr::ident("stride")),
                        block(vec![assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("partial_sums").index(Expr::ident("lid")),
                            Expr::ident("partial_sums")
                                .index(Expr::ident("lid") + Expr::ident("stride")),
                        )]),
                        None,
                    ),
                    workgroup_barrier(),
                ]),
            ),
            if_block_expr(
                Expr::ident("lid").eq(Expr::lit_u32(0)),
                block(vec![
                    let_expr(
                        "wg_idx",
                        Expr::call_named(
                            "workgroup_index",
                            vec![Expr::ident("group_id"), Expr::ident("num_workgroups")],
                        ),
                    ),
                    let_expr(
                        "num_groups_n",
                        (Expr::ident("params").field("n") + (Expr::ident("WORKGROUP_SIZE") - 1u32))
                            / Expr::ident("WORKGROUP_SIZE"),
                    ),
                    if_block_expr(
                        Expr::ident("wg_idx").lt(Expr::ident("num_groups_n")),
                        block(vec![assign_expr(
                            Expr::ident("vec_z").index(Expr::ident("wg_idx")),
                            Expr::ident("partial_sums").index(Expr::lit_u32(0)),
                        )]),
                        None,
                    ),
                ]),
                None,
            ),
        ]);

        m.push(Item::Function(Function::new(
            "norm_sq_partial",
            std_params_with_local.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: orthogonalize ──────────────────────────────────────────
    {
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
            let_expr("h", Expr::ident("scalars").index(Expr::lit_u32(0))),
            assign_expr(
                Expr::ident("vec_y").index(Expr::ident("idx")),
                Expr::ident("vec_y").index(Expr::ident("idx"))
                    - Expr::ident("h") * Expr::ident("vec_x").index(Expr::ident("idx")),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "orthogonalize",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: reduce_final ───────────────────────────────────────────
    {
        let body = block(vec![
            var_expr("total_sum", Expr::lit_f32(0.0)),
            let_expr("num_partials", Expr::ident("params").field("n")),
            for_loop_expr(
                ForInit::Var {
                    name: "i".into(),
                    ty: None,
                    expr: Expr::lit_u32(0),
                },
                Expr::ident("i").lt(Expr::ident("num_partials")),
                for_step_increment_expr(Expr::ident("i")),
                block(vec![assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("total_sum"),
                    Expr::ident("vec_x").index(Expr::ident("i")),
                )]),
            ),
            assign_expr(
                Expr::ident("scalars").index(Expr::lit_u32(0)),
                Expr::ident("total_sum"),
            ),
            assign_expr(
                Expr::ident("hessenberg").index(Expr::ident("iter_params").field("current_idx")),
                Expr::ident("total_sum"),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "reduce_final",
            vec![Param::new(
                "global_id",
                Type::vec3_u32(),
                vec![Attribute::Builtin("global_invocation_id".into())],
            )],
            None,
            vec![Attribute::Compute, Attribute::WorkgroupSize(1)],
            body,
        )));
    }

    // ── Entry point: reduce_final_and_finish_norm ───────────────────────────
    {
        let body = block(vec![
            if_block_expr(
                Expr::ident("scalars")
                    .index(Expr::ident("SCALAR_STOP"))
                    .gt(Expr::lit_f32(0.5)),
                block(vec![return_void()]),
                None,
            ),
            var_expr("total_sum", Expr::lit_f32(0.0)),
            let_expr("num_partials", Expr::ident("params").field("n")),
            for_loop_expr(
                ForInit::Var {
                    name: "i".into(),
                    ty: None,
                    expr: Expr::lit_u32(0),
                },
                Expr::ident("i").lt(Expr::ident("num_partials")),
                for_step_increment_expr(Expr::ident("i")),
                block(vec![assign_op_expr(
                    AssignOp::Add,
                    Expr::ident("total_sum"),
                    Expr::ident("vec_x").index(Expr::ident("i")),
                )]),
            ),
            let_expr("norm", sqrt(Expr::ident("total_sum"))),
            assign_expr(
                Expr::ident("hessenberg").index(Expr::ident("iter_params").field("current_idx")),
                Expr::ident("norm"),
            ),
            if_block_expr(
                Expr::ident("norm").gt(Expr::lit_f32(1e-20)),
                block(vec![assign_expr(
                    Expr::ident("scalars").index(Expr::lit_u32(0)),
                    Expr::lit_f32(1.0) / Expr::ident("norm"),
                )]),
                Some(block(vec![assign_expr(
                    Expr::ident("scalars").index(Expr::lit_u32(0)),
                    Expr::lit_f32(0.0),
                )])),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "reduce_final_and_finish_norm",
            vec![Param::new(
                "global_id",
                Type::vec3_u32(),
                vec![Attribute::Builtin("global_invocation_id".into())],
            )],
            None,
            vec![Attribute::Compute, Attribute::WorkgroupSize(1)],
            body,
        )));
    }

    // ── Entry point: extract_diag_inv ───────────────────────────────────────
    {
        let body = block(vec![
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
            var_expr("diag", Expr::lit_f32(1.0)),
            for_loop_expr(
                ForInit::Var {
                    name: "k".into(),
                    ty: None,
                    expr: Expr::ident("start"),
                },
                Expr::ident("k").lt(Expr::ident("end")),
                for_step_assign_expr(Expr::ident("k"), Expr::ident("k") + 1u32),
                block(vec![if_block_expr(
                    Expr::ident("col_indices")
                        .index(Expr::ident("k"))
                        .eq(Expr::ident("row")),
                    block(vec![
                        assign_expr(
                            Expr::ident("diag"),
                            Expr::ident("matrix_values").index(Expr::ident("k")),
                        ),
                        break_stmt(),
                    ]),
                    None,
                )]),
            ),
            let_expr(
                "inv",
                Expr::call_named("safe_inverse", vec![Expr::ident("diag")]),
            ),
            assign_expr(
                Expr::ident("diag_u").index(Expr::ident("row")),
                Expr::ident("inv"),
            ),
            assign_expr(
                Expr::ident("diag_v").index(Expr::ident("row")),
                Expr::ident("inv"),
            ),
            assign_expr(
                Expr::ident("diag_p").index(Expr::ident("row")),
                Expr::ident("inv"),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "extract_diag_inv",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: apply_diag_inv ─────────────────────────────────────────
    {
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
            assign_expr(
                Expr::ident("vec_y").index(Expr::ident("idx")),
                Expr::ident("diag_u").index(Expr::ident("idx"))
                    * Expr::ident("vec_x").index(Expr::ident("idx")),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "apply_diag_inv",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: guard_copy ─────────────────────────────────────────────
    //
    // Companion to gmres_logic/restart_guard: conditionally snapshots or
    // restores the solution vector based on scalars[SCALAR_GUARD_FLAG].
    // Bind the solution x as vec_y (read_write) and the snapshot buffer as
    // vec_z (read_write); vec_x is unused.
    //   flag == 1.0: vec_z = vec_y (snapshot the improved x)
    //   flag == 2.0: vec_y = vec_z (restore the best x after growth)
    {
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
            let_expr(
                "flag",
                Expr::ident("scalars").index(Expr::ident("SCALAR_GUARD_FLAG")),
            ),
            if_block_expr(
                Expr::ident("flag").eq(Expr::lit_f32(1.0)),
                block(vec![assign_expr(
                    Expr::ident("vec_z").index(Expr::ident("idx")),
                    Expr::ident("vec_y").index(Expr::ident("idx")),
                )]),
                Some(block(vec![if_block_expr(
                    Expr::ident("flag").eq(Expr::lit_f32(2.0)),
                    block(vec![assign_expr(
                        Expr::ident("vec_y").index(Expr::ident("idx")),
                        Expr::ident("vec_z").index(Expr::ident("idx")),
                    )]),
                    None,
                )])),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "guard_copy",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    KernelWgsl::new(m)
}
