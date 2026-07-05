//! GMRES Logic kernel generator: small-system operations (Givens rotations,
//! triangular solve, norm finalisation) — all single-threaded (@workgroup_size(1)).

use crate::solver::codegen::kernel_wgsl::KernelWgsl;
use crate::solver::codegen::wgsl_ast::*;
use crate::solver::codegen::wgsl_dsl::*;

/// Generate the gmres_logic.wgsl kernel (Givens rotation update, triangular
/// solve, norm finalisation, restart guard, relative-scale clamp).
pub fn generate_gmres_logic() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Comment(
        "GMRES Logic Shaders (Small system operations)".into(),
    ));

    m.push(Item::Struct(StructDef::new(
        "IterParams",
        vec![
            StructField::new("current_idx", Type::U32),
            StructField::new("max_restart", Type::U32),
            StructField::new("_pad1", Type::U32),
            StructField::new("_pad2", Type::U32),
        ],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "hessenberg",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "givens",
        Type::array(Type::Vec2(Box::new(Type::F32))),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "g_rhs",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "y_sol",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "iter_params",
        Type::Custom("IterParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(1), Attribute::Binding(0)],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "scalars",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(1), Attribute::Binding(1)],
    )));

    m.push(Item::GlobalVar(GlobalVar::new(
        "indirect_args",
        Type::array(Type::Vec4(Box::new(Type::U32))),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(1), Attribute::Binding(2)],
    )));

    m.push(Item::Const {
        name: "SCALAR_STOP".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(8),
    });
    m.push(Item::Const {
        name: "SCALAR_CONVERGED".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(9),
    });
    m.push(Item::Const {
        name: "SCALAR_ITERS_USED".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(10),
    });
    m.push(Item::Const {
        name: "SCALAR_RESIDUAL_EST".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(11),
    });
    m.push(Item::Const {
        name: "SCALAR_TOL_REL_RHS".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(12),
    });
    m.push(Item::Const {
        name: "SCALAR_TOL_ABS".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(13),
    });
    m.push(Item::Const {
        name: "SCALAR_RHS_NORM".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(14),
    });
    m.push(Item::Const {
        name: "SCALAR_SKIP_UPDATE".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(15),
    });
    m.push(Item::Const {
        name: "SCALAR_BEST_RESID".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(16),
    });
    m.push(Item::Const {
        name: "SCALAR_GUARD_FLAG".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(17),
    });
    m.push(Item::Const {
        name: "SCALAR_PREV_RESID".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(18),
    });
    m.push(Item::Const {
        name: "SCALAR_STALL_REL".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(19),
    });
    m.push(Item::Const {
        name: "SCALAR_STALL_COUNT".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(20),
    });
    m.push(Item::Const {
        name: "SCALAR_PREV_EST".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(21),
    });
    m.push(Item::Const {
        name: "SCALAR_STALL_COUNT_ITER".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(22),
    });
    m.push(Item::Const {
        name: "SCALAR_TOTAL_ITERS".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(23),
    });

    m.push(Item::Function(Function::new(
        "h_idx",
        vec![
            Param::new("row", Type::U32, vec![]),
            Param::new("col", Type::U32, vec![]),
        ],
        Some(Type::U32),
        vec![],
        block(vec![return_expr(
            Expr::ident("col") * (Expr::ident("iter_params").field("max_restart") + 1u32)
                + Expr::ident("row"),
        )]),
    )));

    {
        let body = block(vec![
            if_block_expr(
                Expr::ident("scalars")
                    .index(Expr::ident("SCALAR_STOP"))
                    .gt(Expr::lit_f32(0.5)),
                block(vec![return_void()]),
                None,
            ),
            // Count Arnoldi iterations past the STOP early-out so frozen no-op
            // dispatches don't count; feeds the host adaptive iteration budget.
            assign_expr(
                Expr::ident("scalars").index(Expr::ident("SCALAR_TOTAL_ITERS")),
                Expr::ident("scalars").index(Expr::ident("SCALAR_TOTAL_ITERS"))
                    + Expr::lit_f32(1.0),
            ),
            let_expr("j", Expr::ident("iter_params").field("current_idx")),
            comment("Apply previous Givens rotations to the new column H[:, j]"),
            for_loop_expr(
                ForInit::Var {
                    name: "i".into(),
                    ty: Some(Type::U32),
                    expr: Expr::lit_u32(0),
                },
                Expr::ident("i").lt(Expr::ident("j")),
                ForStep::Increment(Expr::ident("i")),
                block(vec![
                    let_expr(
                        "idx_i",
                        Expr::call_named("h_idx", vec![Expr::ident("i"), Expr::ident("j")]),
                    ),
                    let_expr(
                        "idx_i1",
                        Expr::call_named(
                            "h_idx",
                            vec![Expr::ident("i") + 1u32, Expr::ident("j")],
                        ),
                    ),
                    let_expr(
                        "h_ij",
                        Expr::ident("hessenberg").index(Expr::ident("idx_i")),
                    ),
                    let_expr(
                        "h_i1j",
                        Expr::ident("hessenberg").index(Expr::ident("idx_i1")),
                    ),
                    let_expr("cs", Expr::ident("givens").index(Expr::ident("i"))),
                    let_expr("c", Expr::ident("cs").field("x")),
                    let_expr("s", Expr::ident("cs").field("y")),
                    assign_expr(
                        Expr::ident("hessenberg").index(Expr::ident("idx_i")),
                        Expr::ident("c") * Expr::ident("h_ij")
                            + Expr::ident("s") * Expr::ident("h_i1j"),
                    ),
                    assign_expr(
                        Expr::ident("hessenberg").index(Expr::ident("idx_i1")),
                        -Expr::ident("s") * Expr::ident("h_ij")
                            + Expr::ident("c") * Expr::ident("h_i1j"),
                    ),
                ]),
            ),
            comment("Compute new Givens rotation for H[j, j] and H[j+1, j]"),
            let_expr(
                "idx_jj",
                Expr::call_named("h_idx", vec![Expr::ident("j"), Expr::ident("j")]),
            ),
            let_expr(
                "idx_j1j",
                Expr::call_named("h_idx", vec![Expr::ident("j") + 1u32, Expr::ident("j")]),
            ),
            let_expr(
                "h_jj",
                Expr::ident("hessenberg").index(Expr::ident("idx_jj")),
            ),
            let_expr(
                "h_j1j",
                Expr::ident("hessenberg").index(Expr::ident("idx_j1j")),
            ),
            var_expr("c", Expr::lit_f32(1.0)),
            var_expr("s", Expr::lit_f32(0.0)),
            var_expr(
                "rho",
                sqrt(
                    Expr::ident("h_jj") * Expr::ident("h_jj")
                        + Expr::ident("h_j1j") * Expr::ident("h_j1j"),
                ),
            ),
            if_block_expr(
                abs(Expr::ident("rho")).gt(Expr::lit_f32(1e-20)),
                block(vec![
                    assign_expr(Expr::ident("c"), Expr::ident("h_jj") / Expr::ident("rho")),
                    assign_expr(Expr::ident("s"), Expr::ident("h_j1j") / Expr::ident("rho")),
                ]),
                None,
            ),
            comment("Store rotation"),
            assign_expr(
                Expr::ident("givens").index(Expr::ident("j")),
                vec2_f32(Expr::ident("c"), Expr::ident("s")),
            ),
            comment("Apply rotation to H"),
            assign_expr(
                Expr::ident("hessenberg").index(Expr::ident("idx_jj")),
                Expr::ident("rho"),
            ),
            assign_expr(
                Expr::ident("hessenberg").index(Expr::ident("idx_j1j")),
                Expr::lit_f32(0.0),
            ),
            comment("Apply rotation to RHS vector g"),
            let_expr("g_j", Expr::ident("g_rhs").index(Expr::ident("j"))),
            let_expr(
                "g_j1",
                Expr::ident("g_rhs").index(Expr::ident("j") + 1u32),
            ),
            assign_expr(
                Expr::ident("g_rhs").index(Expr::ident("j")),
                Expr::ident("c") * Expr::ident("g_j") + Expr::ident("s") * Expr::ident("g_j1"),
            ),
            assign_expr(
                Expr::ident("g_rhs").index(Expr::ident("j") + 1u32),
                -Expr::ident("s") * Expr::ident("g_j") + Expr::ident("c") * Expr::ident("g_j1"),
            ),
            let_expr(
                "residual",
                abs(Expr::ident("g_rhs").index(Expr::ident("j") + 1u32)),
            ),
            assign_expr(
                Expr::ident("scalars").index(Expr::ident("SCALAR_RESIDUAL_EST")),
                Expr::ident("residual"),
            ),
            let_expr(
                "tol_rel_rhs",
                Expr::ident("scalars").index(Expr::ident("SCALAR_TOL_REL_RHS"))
                    * Expr::ident("scalars").index(Expr::ident("SCALAR_RHS_NORM")),
            ),
            let_expr(
                "tol_abs",
                Expr::ident("scalars").index(Expr::ident("SCALAR_TOL_ABS")),
            ),
            if_block_expr(
                Expr::ident("residual").le(Expr::ident("tol_rel_rhs"))
                    | Expr::ident("residual").le(Expr::ident("tol_abs")),
                block(vec![
                    assign_expr(
                        Expr::ident("scalars").index(Expr::ident("SCALAR_STOP")),
                        Expr::lit_f32(1.0),
                    ),
                    assign_expr(
                        Expr::ident("scalars").index(Expr::ident("SCALAR_CONVERGED")),
                        Expr::lit_f32(1.0),
                    ),
                    assign_expr(
                        Expr::ident("scalars").index(Expr::ident("SCALAR_ITERS_USED")),
                        f32_cast(Expr::ident("j") + 1u32),
                    ),
                    comment(
                        "Zero indirect dispatch dimensions so subsequent heavy kernels become no-ops.",
                    ),
                    assign_expr(
                        Expr::ident("indirect_args").index(Expr::lit_u32(0)),
                        vec4_u32(
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                        ),
                    ),
                    assign_expr(
                        Expr::ident("indirect_args").index(Expr::lit_u32(1)),
                        vec4_u32(
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                        ),
                    ),
                    assign_expr(
                        Expr::ident("indirect_args").index(Expr::lit_u32(2)),
                        vec4_u32(
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                        ),
                    ),
                ]),
                // Mid-cycle stall: the Givens residual estimate is monotone
                // within a cycle (orthogonality loss only makes it optimistic;
                // the restart-boundary guard verifies the true residual). Stop
                // when the estimate improves <0.5% for 10 consecutive iterations
                // AND is below SCALAR_STALL_REL * scalars[SCALAR_RHS_NORM]
                // (0 disables). Reuses the convergence-break machinery (STOP +
                // ITERS_USED = j+1 + zeroed indirect args); SKIP_UPDATE stays 0
                // so the cycle tail applies the partial update, CONVERGED stays 0.
                Some(block(vec![
                    let_expr(
                        "stall_rel",
                        Expr::ident("scalars").index(Expr::ident("SCALAR_STALL_REL")),
                    ),
                    if_block_expr(
                        Expr::ident("stall_rel").gt(Expr::lit_f32(0.0)),
                        block(vec![
                            let_expr(
                                "prev_est",
                                Expr::ident("scalars").index(Expr::ident("SCALAR_PREV_EST")),
                            ),
                            let_expr(
                                "no_improve",
                                Expr::ident("prev_est").gt(Expr::lit_f32(0.0))
                                    & Expr::ident("residual")
                                        .gt(Expr::ident("prev_est") * Expr::lit_f32(0.995)),
                            ),
                            let_expr(
                                "level_ok",
                                Expr::ident("residual").le(
                                    Expr::ident("stall_rel")
                                        * Expr::ident("scalars")
                                            .index(Expr::ident("SCALAR_RHS_NORM")),
                                ),
                            ),
                            if_block_expr(
                                Expr::ident("no_improve") & Expr::ident("level_ok"),
                                block(vec![
                                    assign_expr(
                                        Expr::ident("scalars")
                                            .index(Expr::ident("SCALAR_STALL_COUNT_ITER")),
                                        Expr::ident("scalars")
                                            .index(Expr::ident("SCALAR_STALL_COUNT_ITER"))
                                            + Expr::lit_f32(1.0),
                                    ),
                                    if_block_expr(
                                        Expr::ident("scalars")
                                            .index(Expr::ident("SCALAR_STALL_COUNT_ITER"))
                                            .gt(Expr::lit_f32(9.5)),
                                        block(vec![
                                            assign_expr(
                                                Expr::ident("scalars")
                                                    .index(Expr::ident("SCALAR_STOP")),
                                                Expr::lit_f32(1.0),
                                            ),
                                            assign_expr(
                                                Expr::ident("scalars")
                                                    .index(Expr::ident("SCALAR_ITERS_USED")),
                                                f32_cast(Expr::ident("j") + 1u32),
                                            ),
                                            comment(
                                                "Zero indirect dispatch dimensions so subsequent heavy kernels become no-ops.",
                                            ),
                                            assign_expr(
                                                Expr::ident("indirect_args")
                                                    .index(Expr::lit_u32(0)),
                                                vec4_u32(
                                                    Expr::lit_u32(0),
                                                    Expr::lit_u32(0),
                                                    Expr::lit_u32(0),
                                                    Expr::lit_u32(0),
                                                ),
                                            ),
                                            assign_expr(
                                                Expr::ident("indirect_args")
                                                    .index(Expr::lit_u32(1)),
                                                vec4_u32(
                                                    Expr::lit_u32(0),
                                                    Expr::lit_u32(0),
                                                    Expr::lit_u32(0),
                                                    Expr::lit_u32(0),
                                                ),
                                            ),
                                            assign_expr(
                                                Expr::ident("indirect_args")
                                                    .index(Expr::lit_u32(2)),
                                                vec4_u32(
                                                    Expr::lit_u32(0),
                                                    Expr::lit_u32(0),
                                                    Expr::lit_u32(0),
                                                    Expr::lit_u32(0),
                                                ),
                                            ),
                                        ]),
                                        None,
                                    ),
                                ]),
                                Some(block(vec![assign_expr(
                                    Expr::ident("scalars")
                                        .index(Expr::ident("SCALAR_STALL_COUNT_ITER")),
                                    Expr::lit_f32(0.0),
                                )])),
                            ),
                            assign_expr(
                                Expr::ident("scalars").index(Expr::ident("SCALAR_PREV_EST")),
                                Expr::ident("residual"),
                            ),
                        ]),
                        None,
                    ),
                ])),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "update_hessenberg_givens",
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

    {
        let body = block(vec![
            if_block_expr(
                Expr::ident("scalars")
                    .index(Expr::ident("SCALAR_SKIP_UPDATE"))
                    .gt(Expr::lit_f32(0.5)),
                block(vec![return_void()]),
                None,
            ),
            let_expr(
                "k",
                u32_cast(clamp(
                    round(Expr::ident("scalars").index(Expr::ident("SCALAR_ITERS_USED"))),
                    Expr::lit_f32(1.0),
                    f32_cast(Expr::ident("iter_params").field("max_restart")),
                )),
            ),
            comment("Backward substitution"),
            for_loop_expr(
                ForInit::Var {
                    name: "loop_i".into(),
                    ty: Some(Type::U32),
                    expr: Expr::lit_u32(0),
                },
                Expr::ident("loop_i").lt(Expr::ident("k")),
                ForStep::Increment(Expr::ident("loop_i")),
                block(vec![
                    let_expr("i", Expr::ident("k") - 1u32 - Expr::ident("loop_i")),
                    var_expr("sum", Expr::ident("g_rhs").index(Expr::ident("i"))),
                    for_loop_expr(
                        ForInit::Var {
                            name: "j".into(),
                            ty: Some(Type::U32),
                            expr: Expr::ident("i") + 1u32,
                        },
                        Expr::ident("j").lt(Expr::ident("k")),
                        ForStep::Increment(Expr::ident("j")),
                        block(vec![assign_op_expr(
                            AssignOp::Sub,
                            Expr::ident("sum"),
                            Expr::ident("hessenberg").index(Expr::call_named(
                                "h_idx",
                                vec![Expr::ident("i"), Expr::ident("j")],
                            )) * Expr::ident("y_sol").index(Expr::ident("j")),
                        )]),
                    ),
                    let_expr(
                        "diag",
                        Expr::ident("hessenberg").index(Expr::call_named(
                            "h_idx",
                            vec![Expr::ident("i"), Expr::ident("i")],
                        )),
                    ),
                    if_block_expr(
                        abs(Expr::ident("diag")).gt(Expr::lit_f32(1e-12)),
                        block(vec![assign_expr(
                            Expr::ident("y_sol").index(Expr::ident("i")),
                            Expr::ident("sum") / Expr::ident("diag"),
                        )]),
                        Some(block(vec![assign_expr(
                            Expr::ident("y_sol").index(Expr::ident("i")),
                            Expr::lit_f32(0.0),
                        )])),
                    ),
                ]),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "solve_triangular",
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

    {
        let body = block(vec![
            let_expr("norm_sq", Expr::ident("scalars").index(Expr::lit_u32(0))),
            let_expr("norm", sqrt(Expr::ident("norm_sq"))),
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
            "finish_norm",
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

    // Restart-boundary monotonicity guard. Runs after the encoded seed has
    // written the TRUE residual norm ||b - A*x|| into hessenberg[0]. f32 Arnoldi
    // can lose orthogonality on hard preconditioned systems and a restart cycle
    // may then apply an update that increases the true residual; unguarded this
    // compounds to NaN. On improvement the guard requests a snapshot of x
    // (GUARD_FLAG=1, executed by gmres_ops/guard_copy); on growth past BEST*1.25
    // (or a non-finite seed) it requests a restore (GUARD_FLAG=2), freezes the
    // remaining work like the convergence break (STOP + SKIP_UPDATE + zeroed
    // indirect args), and reports the best residual.
    {
        let r = Expr::ident("r");
        let best = Expr::ident("best");
        let body = block(vec![
            if_block_expr(
                Expr::ident("scalars")
                    .index(Expr::ident("SCALAR_STOP"))
                    .gt(Expr::lit_f32(0.5)),
                block(vec![
                    assign_expr(
                        Expr::ident("scalars").index(Expr::ident("SCALAR_GUARD_FLAG")),
                        Expr::lit_f32(0.0),
                    ),
                    return_void(),
                ]),
                None,
            ),
            let_expr("r", Expr::ident("hessenberg").index(Expr::lit_u32(0))),
            let_expr(
                "best",
                Expr::ident("scalars").index(Expr::ident("SCALAR_BEST_RESID")),
            ),
            let_expr(
                "grew",
                r.clone().ne(r.clone())
                    | (best.clone().gt(Expr::lit_f32(0.0))
                        & r.clone().gt(best.clone() * Expr::lit_f32(1.25))),
            ),
            if_block_expr(
                Expr::ident("grew"),
                block(vec![
                    assign_expr(
                        Expr::ident("scalars").index(Expr::ident("SCALAR_GUARD_FLAG")),
                        Expr::lit_f32(2.0),
                    ),
                    assign_expr(
                        Expr::ident("scalars").index(Expr::ident("SCALAR_STOP")),
                        Expr::lit_f32(1.0),
                    ),
                    assign_expr(
                        Expr::ident("scalars").index(Expr::ident("SCALAR_SKIP_UPDATE")),
                        Expr::lit_f32(1.0),
                    ),
                    assign_expr(
                        Expr::ident("scalars").index(Expr::ident("SCALAR_RESIDUAL_EST")),
                        best.clone(),
                    ),
                    comment(
                        "Zero indirect dispatch dimensions so subsequent heavy kernels become no-ops.",
                    ),
                    assign_expr(
                        Expr::ident("indirect_args").index(Expr::lit_u32(0)),
                        vec4_u32(
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                        ),
                    ),
                    assign_expr(
                        Expr::ident("indirect_args").index(Expr::lit_u32(1)),
                        vec4_u32(
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                        ),
                    ),
                    assign_expr(
                        Expr::ident("indirect_args").index(Expr::lit_u32(2)),
                        vec4_u32(
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                            Expr::lit_u32(0),
                        ),
                    ),
                ]),
                Some(block(vec![
                    if_block_expr(
                        best.clone().le(Expr::lit_f32(0.0)) | r.clone().lt(best.clone()),
                        block(vec![
                            assign_expr(
                                Expr::ident("scalars").index(Expr::ident("SCALAR_BEST_RESID")),
                                r.clone(),
                            ),
                            assign_expr(
                                Expr::ident("scalars").index(Expr::ident("SCALAR_GUARD_FLAG")),
                                Expr::lit_f32(1.0),
                            ),
                        ]),
                        Some(block(vec![assign_expr(
                            Expr::ident("scalars").index(Expr::ident("SCALAR_GUARD_FLAG")),
                            Expr::lit_f32(0.0),
                        )])),
                    ),
                    // Stall-stop: when the true residual stops improving (<2%
                    // across a checkpoint) for two consecutive checkpoints AND is
                    // small relative to SCALAR_RHS_NORM (min(||b||, ||r0||) on the
                    // fully-encoded path; level factor SCALAR_STALL_REL, 0 disables),
                    // freeze the remaining work like the convergence break, keeping
                    // the best iterate. Mirrors the host loop in solve_fgmres — keep
                    // the two in sync.
                    let_expr(
                        "prev",
                        Expr::ident("scalars").index(Expr::ident("SCALAR_PREV_RESID")),
                    ),
                    let_expr(
                        "stall_rel",
                        Expr::ident("scalars").index(Expr::ident("SCALAR_STALL_REL")),
                    ),
                    let_expr(
                        "no_improve",
                        Expr::ident("prev").gt(Expr::lit_f32(0.0))
                            & r.clone().gt(Expr::ident("prev") * Expr::lit_f32(0.98)),
                    ),
                    let_expr(
                        "level_ok",
                        r.clone().le(
                            Expr::ident("stall_rel")
                                * Expr::ident("scalars").index(Expr::ident("SCALAR_RHS_NORM")),
                        ),
                    ),
                    if_block_expr(
                        Expr::ident("stall_rel").gt(Expr::lit_f32(0.0))
                            & Expr::ident("no_improve")
                            & Expr::ident("level_ok"),
                        block(vec![
                            assign_expr(
                                Expr::ident("scalars").index(Expr::ident("SCALAR_STALL_COUNT")),
                                Expr::ident("scalars").index(Expr::ident("SCALAR_STALL_COUNT"))
                                    + Expr::lit_f32(1.0),
                            ),
                            if_block_expr(
                                Expr::ident("scalars")
                                    .index(Expr::ident("SCALAR_STALL_COUNT"))
                                    .gt(Expr::lit_f32(1.5)),
                                block(vec![
                                    let_expr(
                                        "best_now",
                                        Expr::ident("scalars")
                                            .index(Expr::ident("SCALAR_BEST_RESID")),
                                    ),
                                    assign_expr(
                                        Expr::ident("scalars")
                                            .index(Expr::ident("SCALAR_GUARD_FLAG")),
                                        Expr::call_named(
                                            "select",
                                            vec![
                                                Expr::lit_f32(2.0),
                                                Expr::lit_f32(1.0),
                                                r.clone().le(Expr::ident("best_now")),
                                            ],
                                        ),
                                    ),
                                    assign_expr(
                                        Expr::ident("scalars").index(Expr::ident("SCALAR_STOP")),
                                        Expr::lit_f32(1.0),
                                    ),
                                    assign_expr(
                                        Expr::ident("scalars")
                                            .index(Expr::ident("SCALAR_SKIP_UPDATE")),
                                        Expr::lit_f32(1.0),
                                    ),
                                    assign_expr(
                                        Expr::ident("scalars")
                                            .index(Expr::ident("SCALAR_RESIDUAL_EST")),
                                        Expr::call_named(
                                            "min",
                                            vec![r.clone(), Expr::ident("best_now")],
                                        ),
                                    ),
                                    comment(
                                        "Zero indirect dispatch dimensions so subsequent heavy kernels become no-ops.",
                                    ),
                                    assign_expr(
                                        Expr::ident("indirect_args").index(Expr::lit_u32(0)),
                                        vec4_u32(
                                            Expr::lit_u32(0),
                                            Expr::lit_u32(0),
                                            Expr::lit_u32(0),
                                            Expr::lit_u32(0),
                                        ),
                                    ),
                                    assign_expr(
                                        Expr::ident("indirect_args").index(Expr::lit_u32(1)),
                                        vec4_u32(
                                            Expr::lit_u32(0),
                                            Expr::lit_u32(0),
                                            Expr::lit_u32(0),
                                            Expr::lit_u32(0),
                                        ),
                                    ),
                                    assign_expr(
                                        Expr::ident("indirect_args").index(Expr::lit_u32(2)),
                                        vec4_u32(
                                            Expr::lit_u32(0),
                                            Expr::lit_u32(0),
                                            Expr::lit_u32(0),
                                            Expr::lit_u32(0),
                                        ),
                                    ),
                                ]),
                                None,
                            ),
                        ]),
                        Some(block(vec![assign_expr(
                            Expr::ident("scalars").index(Expr::ident("SCALAR_STALL_COUNT")),
                            Expr::lit_f32(0.0),
                        )])),
                    ),
                    assign_expr(
                        Expr::ident("scalars").index(Expr::ident("SCALAR_PREV_RESID")),
                        r.clone(),
                    ),
                ])),
            ),
        ]);

        m.push(Item::Function(Function::new(
            "restart_guard",
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

    // Relative-tolerance scale = min(||b||, ||r0||), matching the host loop
    // (solve_fgmres). Runs on the first restart chunk only, after ||b|| is in
    // scalars[SCALAR_RHS_NORM] and beta = ||r0|| in hessenberg[0]. Without the
    // clamp, a near-converged warm start (||r0|| << ||b||) would declare
    // convergence against ||b|| alone. NaN-safe: the comparison is false for
    // non-finite beta, keeping ||b||.
    {
        let body = block(vec![
            let_expr("beta", Expr::ident("hessenberg").index(Expr::lit_u32(0))),
            if_block_expr(
                Expr::ident("beta").lt(Expr::ident("scalars").index(Expr::ident("SCALAR_RHS_NORM"))),
                block(vec![assign_expr(
                    Expr::ident("scalars").index(Expr::ident("SCALAR_RHS_NORM")),
                    Expr::ident("beta"),
                )]),
                None,
            ),
        ]);

        m.push(Item::Function(Function::new(
            "clamp_rel_scale",
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

    KernelWgsl::new(m)
}
