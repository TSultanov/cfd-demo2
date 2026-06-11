//! GMRES Logic kernel generator implementation.
//!
//! This module contains the implementation of generate_gmres_logic() which was
//! ported from the handwritten src/solver/gpu/shaders/gmres_logic.wgsl file.
//! It contains small-system operations: Givens rotations, triangular solve, and
//! norm finalisation — all single-threaded (@workgroup_size(1)).

use crate::solver::codegen::kernel_wgsl::KernelWgsl;
use crate::solver::codegen::wgsl_ast::*;
use crate::solver::codegen::wgsl_dsl::*;

/// Generate the gmres_logic.wgsl kernel with 3 entry points:
/// - update_hessenberg_givens: Apply Givens rotations to new Hessenberg column
/// - solve_triangular: Backward substitution for upper triangular system
/// - finish_norm: Finalize norm from reduction result (unused but included)
pub fn generate_gmres_logic() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Comment(
        "GMRES Logic Shaders (Small system operations)".into(),
    ));

    // ── Struct ──────────────────────────────────────────────────────────────

    m.push(Item::Struct(StructDef::new(
        "IterParams",
        vec![
            StructField::new("current_idx", Type::U32),
            StructField::new("max_restart", Type::U32),
            StructField::new("_pad1", Type::U32),
            StructField::new("_pad2", Type::U32),
        ],
    )));

    // ── Group 0: Hessenberg and Givens data ─────────────────────────────────

    // @group(0) @binding(0) var<storage, read_write> hessenberg: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "hessenberg",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(0)],
    )));

    // @group(0) @binding(1) var<storage, read_write> givens: array<vec2<f32>>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "givens",
        Type::array(Type::Vec2(Box::new(Type::F32))),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(1)],
    )));

    // @group(0) @binding(2) var<storage, read_write> g_rhs: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "g_rhs",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(2)],
    )));

    // @group(0) @binding(3) var<storage, read_write> y_sol: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "y_sol",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(0), Attribute::Binding(3)],
    )));

    // ── Group 1: Parameters ─────────────────────────────────────────────────

    // @group(1) @binding(0) var<uniform> iter_params: IterParams;
    m.push(Item::GlobalVar(GlobalVar::new(
        "iter_params",
        Type::Custom("IterParams".into()),
        StorageClass::Uniform,
        None,
        vec![Attribute::Group(1), Attribute::Binding(0)],
    )));

    // @group(1) @binding(1) var<storage, read_write> scalars: array<f32>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "scalars",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(1), Attribute::Binding(1)],
    )));

    // @group(1) @binding(2) var<storage, read_write> indirect_args: array<vec4<u32>>;
    m.push(Item::GlobalVar(GlobalVar::new(
        "indirect_args",
        Type::array(Type::Vec4(Box::new(Type::U32))),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(1), Attribute::Binding(2)],
    )));

    // ── Constants ───────────────────────────────────────────────────────────

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

    // ── Helper function: h_idx ──────────────────────────────────────────────

    // fn h_idx(row: u32, col: u32) -> u32 {
    //     return col * (iter_params.max_restart + 1u) + row;
    // }
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

    // ── Entry point: update_hessenberg_givens ───────────────────────────────
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
            // let j = iter_params.current_idx;
            let_expr("j", Expr::ident("iter_params").field("current_idx")),
            // 1. Apply previous Givens rotations to the new column H[:, j]
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
            // 2. Compute new Givens rotation for H[j, j] and H[j+1, j]
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
            // Store rotation
            comment("Store rotation"),
            assign_expr(
                Expr::ident("givens").index(Expr::ident("j")),
                vec2_f32(Expr::ident("c"), Expr::ident("s")),
            ),
            // Apply rotation to H
            comment("Apply rotation to H"),
            assign_expr(
                Expr::ident("hessenberg").index(Expr::ident("idx_jj")),
                Expr::ident("rho"),
            ),
            assign_expr(
                Expr::ident("hessenberg").index(Expr::ident("idx_j1j")),
                Expr::lit_f32(0.0),
            ),
            // 3. Apply rotation to RHS vector g
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
                None,
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

    // ── Entry point: solve_triangular ───────────────────────────────────────
    {
        let body = block(vec![
            // if (scalars[SCALAR_SKIP_UPDATE] > 0.5) { return; }
            if_block_expr(
                Expr::ident("scalars")
                    .index(Expr::ident("SCALAR_SKIP_UPDATE"))
                    .gt(Expr::lit_f32(0.5)),
                block(vec![return_void()]),
                None,
            ),
            // let k = u32(clamp(round(scalars[SCALAR_ITERS_USED]), 1.0, f32(iter_params.max_restart)));
            let_expr(
                "k",
                u32_cast(clamp(
                    round(Expr::ident("scalars").index(Expr::ident("SCALAR_ITERS_USED"))),
                    Expr::lit_f32(1.0),
                    f32_cast(Expr::ident("iter_params").field("max_restart")),
                )),
            ),
            // Backward substitution
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

    // ── Entry point: finish_norm ────────────────────────────────────────────
    {
        let body = block(vec![
            // let norm_sq = scalars[0];
            let_expr("norm_sq", Expr::ident("scalars").index(Expr::lit_u32(0))),
            // let norm = sqrt(norm_sq);
            let_expr("norm", sqrt(Expr::ident("norm_sq"))),
            // hessenberg[iter_params.current_idx] = norm;
            assign_expr(
                Expr::ident("hessenberg").index(Expr::ident("iter_params").field("current_idx")),
                Expr::ident("norm"),
            ),
            // if (norm > 1e-20) { scalars[0] = 1.0 / norm; } else { scalars[0] = 0.0; }
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

    // ── Entry point: restart_guard ──────────────────────────────────────────
    //
    // Restart-boundary monotonicity guard. Runs after the encoded seed has
    // written the TRUE residual norm ||b - A*x|| into hessenberg[0]. f32
    // Arnoldi can lose orthogonality on hard preconditioned systems and a
    // restart cycle may then APPLY an update that increases the true
    // residual; unguarded this compounds across restarts (observed June
    // 2026 on the coupled incompressible system: residual growth by orders
    // of magnitude, ending in NaN). On improvement the guard requests a
    // snapshot of x (GUARD_FLAG=1, executed by gmres_ops/guard_copy); on
    // growth past BEST*1.25 (or a non-finite seed) it requests a restore
    // (GUARD_FLAG=2), freezes the remaining work like the convergence
    // break (STOP + SKIP_UPDATE + zeroed indirect args), and reports the
    // best residual.
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
                Some(block(vec![if_block_expr(
                    best.clone().le(Expr::lit_f32(0.0)) | r.clone().lt(best),
                    block(vec![
                        assign_expr(
                            Expr::ident("scalars").index(Expr::ident("SCALAR_BEST_RESID")),
                            r,
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
                )])),
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

    KernelWgsl::new(m)
}
