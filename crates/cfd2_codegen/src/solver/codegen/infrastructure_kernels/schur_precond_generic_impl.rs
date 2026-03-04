//! Schur Preconditioner Generic kernel generator implementation.
//!
//! This module contains the implementation of generate_schur_precond_generic()
//! which was ported from the handwritten
//! src/solver/gpu/shaders/schur_precond_generic.wgsl file.
//!
//! Implements the SIMPLE-like preconditioner for the coupled solver with
//! 3 entry points: relax_pressure, correct_velocity, predict_and_form_schur.

use crate::solver::codegen::kernel_wgsl::KernelWgsl;
use crate::solver::codegen::wgsl_ast::*;
use crate::solver::codegen::wgsl_dsl::*;

/// Generate the schur_precond_generic.wgsl kernel with 3 entry points:
/// - relax_pressure: Chebyshev/SOR pressure relaxation
/// - correct_velocity: Velocity correction with pressure solution
/// - predict_and_form_schur: Merged predict velocity + form Schur RHS
pub fn generate_schur_precond_generic() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Comment(
        "Generic Schur Complement Preconditioner for Coupled Solver".into(),
    ));

    // ── Struct ──────────────────────────────────────────────────────────────

    m.push(Item::Struct(StructDef::new(
        "PrecondParams",
        vec![
            StructField::new("n", Type::U32),
            StructField::new("num_cells", Type::U32),
            StructField::new("omega", Type::F32),
            StructField::new("unknowns_per_cell", Type::U32),
            StructField::new("p", Type::U32),
            StructField::new("u_len", Type::U32),
            StructField::new("_pad0", Type::U32),
            StructField::new("_pad1", Type::U32),
            StructField::new("u0123", Type::Vec4(Box::new(Type::U32))),
            StructField::new("u4567", Type::Vec4(Box::new(Type::U32))),
        ],
    )));

    // ── Group 0: Vectors ────────────────────────────────────────────────────

    // @group(0) @binding(0) var<storage, read> r_in: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "r_in",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));
    // @group(0) @binding(1) var<storage, read_write> z_out: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "z_out",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));
    // @group(0) @binding(2) var<storage, read_write> temp_p: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "temp_p",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));
    // @group(0) @binding(3) var<storage, read_write> p_sol: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "p_sol",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));
    // @group(0) @binding(4) var<storage, read_write> p_prev: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "p_prev",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(4)],
    )));

    // ── Group 1: Coupled Matrix (CSR) ───────────────────────────────────────

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

    // ── Group 2: Diagonals (Inverse) + Params ───────────────────────────────

    m.push(Item::GlobalVar(GlobalVar::new(
        "diag_u_inv",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(2), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "diag_p_inv",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(2), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "params",
        Type::Custom("PrecondParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(2), Attribute::Binding(2)],
    )));

    // ── Group 3: Pressure Matrix (CSR) ──────────────────────────────────────

    m.push(Item::GlobalVar(GlobalVar::new(
        "p_row_offsets",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(3), Attribute::Binding(0)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "p_col_indices",
        Type::array(Type::U32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(3), Attribute::Binding(1)],
    )));
    m.push(Item::GlobalVar(GlobalVar::new(
        "p_matrix_values",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::Read),
        vec![Attribute::Group(3), Attribute::Binding(2)],
    )));

    // ── Helper functions ────────────────────────────────────────────────────

    // fn safe_inverse(val: f32) -> f32
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

    // fn u_index(i: u32) -> u32
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

    // const WORKGROUP_SIZE: u32 = 64u;
    m.push(Item::Const {
        name: "WORKGROUP_SIZE".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(64),
    });

    // fn global_cell(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32
    m.push(Item::Function(Function::new(
        "global_cell",
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

    let params = Expr::ident("params");

    // ── Entry point: relax_pressure ─────────────────────────────────────────
    {
        let body = block(vec![
            let_expr(
                "cell",
                Expr::call_named(
                    "global_cell",
                    vec![Expr::ident("global_id"), Expr::ident("num_workgroups")],
                ),
            ),
            if_block_expr(
                Expr::ident("cell").ge(params.clone().field("num_cells")),
                block(vec![return_void()]),
                None,
            ),
            let_expr(
                "start",
                Expr::ident("p_row_offsets").index(Expr::ident("cell")),
            ),
            let_expr(
                "end",
                Expr::ident("p_row_offsets").index(Expr::ident("cell") + 1u32),
            ),
            var_expr("sigma", Expr::lit_f32(0.0)),
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
                        "col_cell",
                        Expr::ident("p_col_indices").index(Expr::ident("k")),
                    ),
                    if_block_expr(
                        Expr::ident("col_cell").ne(Expr::ident("cell")),
                        block(vec![assign_op_expr(
                            AssignOp::Add,
                            Expr::ident("sigma"),
                            Expr::ident("p_matrix_values").index(Expr::ident("k"))
                                * Expr::ident("p_sol").index(Expr::ident("col_cell")),
                        )]),
                        None,
                    ),
                ]),
            ),
            let_expr(
                "d_inv",
                Expr::ident("diag_p_inv").index(Expr::ident("cell")),
            ),
            let_expr("rhs", Expr::ident("temp_p").index(Expr::ident("cell"))),
            let_expr(
                "hat_x",
                Expr::ident("d_inv") * (Expr::ident("rhs") - Expr::ident("sigma")),
            ),
            let_expr("x_prev", Expr::ident("p_prev").index(Expr::ident("cell"))),
            let_expr(
                "x_new",
                mix(
                    Expr::ident("x_prev"),
                    Expr::ident("hat_x"),
                    params.clone().field("omega"),
                ),
            ),
            assign_expr(
                Expr::ident("p_prev").index(Expr::ident("cell")),
                Expr::ident("x_new"),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "relax_pressure",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: correct_velocity ───────────────────────────────────────
    {
        let body = block(vec![
            let_expr(
                "cell",
                Expr::call_named(
                    "global_cell",
                    vec![Expr::ident("global_id"), Expr::ident("num_workgroups")],
                ),
            ),
            if_block_expr(
                Expr::ident("cell").ge(params.clone().field("num_cells")),
                block(vec![return_void()]),
                None,
            ),
            let_expr(
                "base",
                Expr::ident("cell") * params.clone().field("unknowns_per_cell"),
            ),
            let_expr("p_val", Expr::ident("p_sol").index(Expr::ident("cell"))),
            for_loop_expr(
                ForInit::Var {
                    name: "i".into(),
                    ty: Some(Type::U32),
                    expr: Expr::lit_u32(0),
                },
                Expr::ident("i").lt(params.clone().field("u_len")),
                ForStep::Increment(Expr::ident("i")),
                block(vec![
                    let_expr("u", Expr::call_named("u_index", vec![Expr::ident("i")])),
                    let_expr("row_u", Expr::ident("base") + Expr::ident("u")),
                    let_expr(
                        "start_u",
                        Expr::ident("row_offsets").index(Expr::ident("row_u")),
                    ),
                    let_expr(
                        "end_u",
                        Expr::ident("row_offsets").index(Expr::ident("row_u") + 1u32),
                    ),
                    var_expr("correction_u", Expr::lit_f32(0.0)),
                    for_loop_expr(
                        ForInit::Var {
                            name: "k".into(),
                            ty: None,
                            expr: Expr::ident("start_u"),
                        },
                        Expr::ident("k").lt(Expr::ident("end_u")),
                        ForStep::Increment(Expr::ident("k")),
                        block(vec![
                            let_expr("col", Expr::ident("col_indices").index(Expr::ident("k"))),
                            if_block_expr(
                                (Expr::ident("col").modulo(params.clone().field("unknowns_per_cell")))
                                    .eq(params.clone().field("p")),
                                block(vec![
                                    let_expr(
                                        "p_cell",
                                        Expr::ident("col") / params.clone().field("unknowns_per_cell"),
                                    ),
                                    assign_op_expr(
                                        AssignOp::Add,
                                        Expr::ident("correction_u"),
                                        Expr::ident("matrix_values").index(Expr::ident("k"))
                                            * Expr::ident("p_sol").index(Expr::ident("p_cell")),
                                    ),
                                ]),
                                None,
                            ),
                        ]),
                    ),
                    assign_op_expr(
                        AssignOp::Sub,
                        Expr::ident("z_out").index(Expr::ident("row_u")),
                        Expr::ident("diag_u_inv")
                            .index(Expr::ident("cell") * params.clone().field("u_len") + Expr::ident("i"))
                            * Expr::ident("correction_u"),
                    ),
                ]),
            ),
            assign_expr(
                Expr::ident("z_out").index(Expr::ident("base") + params.clone().field("p")),
                Expr::ident("p_val"),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "correct_velocity",
            std_params.clone(),
            None,
            std_attrs.clone(),
            body,
        )));
    }

    // ── Entry point: predict_and_form_schur ─────────────────────────────────
    {
        let body = block(vec![
            let_expr(
                "cell",
                Expr::call_named(
                    "global_cell",
                    vec![Expr::ident("global_id"), Expr::ident("num_workgroups")],
                ),
            ),
            if_block_expr(
                Expr::ident("cell").ge(params.clone().field("num_cells")),
                block(vec![return_void()]),
                None,
            ),
            // Part 1: Predict Velocity (Local)
            comment("Part 1: Predict Velocity (Local)"),
            let_expr(
                "base",
                Expr::ident("cell") * params.clone().field("unknowns_per_cell"),
            ),
            let_expr("row_p", Expr::ident("base") + params.clone().field("p")),
            // Default to identity for non-(u,p) components
            for_loop_expr(
                ForInit::Var {
                    name: "c".into(),
                    ty: Some(Type::U32),
                    expr: Expr::lit_u32(0),
                },
                Expr::ident("c").lt(params.clone().field("unknowns_per_cell")),
                ForStep::Increment(Expr::ident("c")),
                block(vec![assign_expr(
                    Expr::ident("z_out").index(Expr::ident("base") + Expr::ident("c")),
                    Expr::ident("r_in").index(Expr::ident("base") + Expr::ident("c")),
                )]),
            ),
            for_loop_expr(
                ForInit::Var {
                    name: "i".into(),
                    ty: Some(Type::U32),
                    expr: Expr::lit_u32(0),
                },
                Expr::ident("i").lt(params.clone().field("u_len")),
                ForStep::Increment(Expr::ident("i")),
                block(vec![
                    let_expr("u", Expr::call_named("u_index", vec![Expr::ident("i")])),
                    let_expr("row_u", Expr::ident("base") + Expr::ident("u")),
                    assign_expr(
                        Expr::ident("z_out").index(Expr::ident("row_u")),
                        Expr::ident("diag_u_inv")
                            .index(Expr::ident("cell") * params.clone().field("u_len") + Expr::ident("i"))
                            * Expr::ident("r_in").index(Expr::ident("row_u")),
                    ),
                ]),
            ),
            assign_expr(
                Expr::ident("z_out").index(Expr::ident("row_p")),
                Expr::lit_f32(0.0),
            ),
            // Part 2: Form Schur RHS
            comment("Part 2: Form Schur RHS"),
            var_expr("rhs_p", Expr::ident("r_in").index(Expr::ident("row_p"))),
            let_expr(
                "start",
                Expr::ident("row_offsets").index(Expr::ident("row_p")),
            ),
            let_expr(
                "end",
                Expr::ident("row_offsets").index(Expr::ident("row_p") + 1u32),
            ),
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
                    let_expr(
                        "rem",
                        Expr::ident("col").modulo(params.clone().field("unknowns_per_cell")),
                    ),
                    var_expr("z_val", Expr::lit_f32(0.0)),
                    for_loop_expr(
                        ForInit::Var {
                            name: "i".into(),
                            ty: Some(Type::U32),
                            expr: Expr::lit_u32(0),
                        },
                        Expr::ident("i").lt(params.clone().field("u_len")),
                        ForStep::Increment(Expr::ident("i")),
                        block(vec![
                            let_expr("u", Expr::call_named("u_index", vec![Expr::ident("i")])),
                            if_block_expr(
                                Expr::ident("rem").eq(Expr::ident("u")),
                                block(vec![
                                    let_expr(
                                        "c",
                                        Expr::ident("col") / params.clone().field("unknowns_per_cell"),
                                    ),
                                    assign_expr(
                                        Expr::ident("z_val"),
                                        Expr::ident("r_in").index(Expr::ident("col"))
                                            * Expr::ident("diag_u_inv").index(
                                                Expr::ident("c") * params.clone().field("u_len")
                                                    + Expr::ident("i"),
                                            ),
                                    ),
                                    break_stmt(),
                                ]),
                                None,
                            ),
                        ]),
                    ),
                    assign_op_expr(
                        AssignOp::Sub,
                        Expr::ident("rhs_p"),
                        Expr::ident("matrix_values").index(Expr::ident("k")) * Expr::ident("z_val"),
                    ),
                ]),
            ),
            assign_expr(
                Expr::ident("temp_p").index(Expr::ident("cell")),
                Expr::ident("rhs_p"),
            ),
            // Initialize p_sol with first Jacobi step
            assign_expr(
                Expr::ident("p_sol").index(Expr::ident("cell")),
                Expr::ident("diag_p_inv").index(Expr::ident("cell")) * Expr::ident("rhs_p"),
            ),
            // Initialize p_prev to 0.0 for Chebyshev start
            assign_expr(
                Expr::ident("p_prev").index(Expr::ident("cell")),
                Expr::lit_f32(0.0),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "predict_and_form_schur",
            std_params,
            None,
            std_attrs,
            body,
        )));
    }

    KernelWgsl::new(m)
}
