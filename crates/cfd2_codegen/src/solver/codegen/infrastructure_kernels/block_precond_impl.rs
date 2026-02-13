//! Block Jacobi preconditioner kernel generator implementation.
//!
//! This module contains the implementation of generate_block_precond()
//! which was ported from the handwritten
//! src/solver/gpu/shaders/block_precond.wgsl file.
//!
//! Implements the cell-block Jacobi preconditioner for generic-coupled FGMRES with
//! 2 entry points: build_block_inv (Gauss-Jordan inversion), apply_block_precond
//! (block matrix-vector multiply).

use crate::solver::codegen::kernel_wgsl::KernelWgsl;
use crate::solver::codegen::wgsl_ast::*;
use crate::solver::codegen::wgsl_dsl::*;

/// Generate the block_precond.wgsl kernel with 2 entry points:
/// - build_block_inv: Gauss-Jordan block inversion with partial pivoting
/// - apply_block_precond: Block matrix-vector multiply (M^{-1} * x -> y)
pub fn generate_block_precond() -> KernelWgsl {
    let mut m = Module::new();

    m.push(Item::Comment(
        "Cell-block Jacobi preconditioner for generic-coupled FGMRES.".into(),
    ));

    // ── Structs ────────────────────────────────────────────────────────────

    m.push(Item::Struct(StructDef::new(
        "GmresParams",
        vec![
            StructField::new("n", Type::U32),
            StructField::new("num_cells", Type::U32),
            StructField::new("num_iters", Type::U32),
            StructField::new("omega", Type::F32),
            StructField::new("dispatch_x", Type::U32),
            StructField::new("_pad1", Type::U32),
            StructField::new("_pad2", Type::U32),
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

    // ── Group 0: vectors ───────────────────────────────────────────────────

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

    // ── Group 1: matrix (CSR) ──────────────────────────────────────────────

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

    // ── Group 2: block inverse ─────────────────────────────────────────────

    m.push(Item::GlobalVar(GlobalVar::new(
        "block_inv",
        Type::array(Type::F32),
        StorageClass::Storage,
        Some(AccessMode::ReadWrite),
        vec![Attribute::Group(2), Attribute::Binding(0)],
    )));

    // ── Group 3: params ────────────────────────────────────────────────────

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

    // ── safe_inverse ───────────────────────────────────────────────────────

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
                    sign(Expr::ident("val")) * Expr::lit_f32(1e12),
                )]),
                None,
            ),
            return_expr(Expr::lit_f32(0.0)),
        ]),
    )));

    // ── const MAX_BLOCK ────────────────────────────────────────────────────

    m.push(Item::Const {
        name: "MAX_BLOCK".into(),
        ty: Type::U32,
        expr: Expr::lit_u32(16),
    });

    // ── swap_rows ──────────────────────────────────────────────────────────
    // The nested array type: array<array<f32, MAX_BLOCK>, MAX_BLOCK>
    // Since MAX_BLOCK is a const, we use Type::Custom to reference it by name.
    let nested_array_ty = Type::Custom("array<array<f32, MAX_BLOCK>, MAX_BLOCK>".into());
    let ptr_ty = Type::Ptr(Box::new(nested_array_ty.clone()), AddressSpace::Function);

    m.push(Item::Function(Function::new(
        "swap_rows",
        vec![
            Param::new("a", ptr_ty.clone(), vec![]),
            Param::new("b", ptr_ty, vec![]),
            Param::new("r0", Type::U32, vec![]),
            Param::new("r1", Type::U32, vec![]),
            Param::new("n", Type::U32, vec![]),
        ],
        None,
        vec![],
        block(vec![
            if_block_expr(
                Expr::ident("r0").eq(Expr::ident("r1")),
                block(vec![return_void()]),
                None,
            ),
            for_loop_expr(
                for_init_var_expr("c", Expr::lit_u32(0)),
                Expr::ident("c").lt(Expr::ident("n")),
                for_step_assign_expr(Expr::ident("c"), Expr::ident("c") + 1u32),
                block(vec![
                    // let tmp = (*a)[r0][c];
                    let_expr(
                        "tmp",
                        Expr::ident("a")
                            .deref()
                            .index(Expr::ident("r0"))
                            .index(Expr::ident("c")),
                    ),
                    // (*a)[r0][c] = (*a)[r1][c];
                    assign_expr(
                        Expr::ident("a")
                            .deref()
                            .index(Expr::ident("r0"))
                            .index(Expr::ident("c")),
                        Expr::ident("a")
                            .deref()
                            .index(Expr::ident("r1"))
                            .index(Expr::ident("c")),
                    ),
                    // (*a)[r1][c] = tmp;
                    assign_expr(
                        Expr::ident("a")
                            .deref()
                            .index(Expr::ident("r1"))
                            .index(Expr::ident("c")),
                        Expr::ident("tmp"),
                    ),
                    // let tmp_b = (*b)[r0][c];
                    let_expr(
                        "tmp_b",
                        Expr::ident("b")
                            .deref()
                            .index(Expr::ident("r0"))
                            .index(Expr::ident("c")),
                    ),
                    // (*b)[r0][c] = (*b)[r1][c];
                    assign_expr(
                        Expr::ident("b")
                            .deref()
                            .index(Expr::ident("r0"))
                            .index(Expr::ident("c")),
                        Expr::ident("b")
                            .deref()
                            .index(Expr::ident("r1"))
                            .index(Expr::ident("c")),
                    ),
                    // (*b)[r1][c] = tmp_b;
                    assign_expr(
                        Expr::ident("b")
                            .deref()
                            .index(Expr::ident("r1"))
                            .index(Expr::ident("c")),
                        Expr::ident("tmp_b"),
                    ),
                ]),
            ),
        ]),
    )));

    // ── Standard params for compute entry points ───────────────────────────
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

    // ── build_block_inv ────────────────────────────────────────────────────

    let params = Expr::ident("params");
    let a = Expr::ident("a");
    let inv = Expr::ident("inv");
    let b_var = Expr::ident("b");

    let build_body = block(vec![
        // Compute cell index
        let_expr("stride_x", Expr::ident("num_workgroups").field("x") * 64u32),
        let_expr(
            "cell",
            Expr::ident("global_id").field("y") * Expr::ident("stride_x")
                + Expr::ident("global_id").field("x"),
        ),
        if_block_expr(
            Expr::ident("cell").ge(params.field("num_cells")),
            block(vec![return_void()]),
            None,
        ),
        if_block_expr(
            params.field("num_cells").eq(0u32),
            block(vec![return_void()]),
            None,
        ),
        let_expr("b", params.field("n") / params.field("num_cells")),
        if_block_expr(
            b_var.eq(0u32) | b_var.gt(Expr::ident("MAX_BLOCK")),
            block(vec![return_void()]),
            None,
        ),
        let_expr("base", Expr::ident("cell") * b_var),
        // Declare local arrays
        var_typed_expr("a", nested_array_ty.clone(), None),
        var_typed_expr("inv", nested_array_ty.clone(), None),
        var_typed_expr(
            "diag_orig",
            Type::Custom("array<f32, MAX_BLOCK>".into()),
            None,
        ),
        // Initialize a and inv, extract diagonal block from CSR
        for_loop_expr(
            for_init_var_expr("r", Expr::lit_u32(0)),
            Expr::ident("r").lt(b_var),
            for_step_assign_expr(Expr::ident("r"), Expr::ident("r") + 1u32),
            block(vec![
                // Zero out a[r][c] and inv[r][c]
                for_loop_expr(
                    for_init_var_expr("c", Expr::lit_u32(0)),
                    Expr::ident("c").lt(b_var),
                    for_step_assign_expr(Expr::ident("c"), Expr::ident("c") + 1u32),
                    block(vec![
                        assign_expr(
                            a.index(Expr::ident("r")).index(Expr::ident("c")),
                            Expr::lit_f32(0.0),
                        ),
                        assign_expr(
                            inv.index(Expr::ident("r")).index(Expr::ident("c")),
                            Expr::lit_f32(0.0),
                        ),
                    ]),
                ),
                let_expr("row", Expr::ident("base") + Expr::ident("r")),
                let_expr(
                    "start",
                    Expr::ident("row_offsets").index(Expr::ident("row")),
                ),
                let_expr(
                    "end",
                    Expr::ident("row_offsets").index(Expr::ident("row") + 1u32),
                ),
                // Extract diagonal block from CSR
                for_loop_expr(
                    for_init_var_expr("k", Expr::ident("start")),
                    Expr::ident("k").lt(Expr::ident("end")),
                    for_step_assign_expr(Expr::ident("k"), Expr::ident("k") + 1u32),
                    block(vec![
                        let_expr("col", Expr::ident("col_indices").index(Expr::ident("k"))),
                        if_block_expr(
                            Expr::ident("col").ge(Expr::ident("base"))
                                & Expr::ident("col").lt(Expr::ident("base") + b_var),
                            block(vec![
                                let_expr("local", Expr::ident("col") - Expr::ident("base")),
                                assign_expr(
                                    a.index(Expr::ident("r")).index(Expr::ident("local")),
                                    Expr::ident("matrix_values").index(Expr::ident("k")),
                                ),
                            ]),
                            None,
                        ),
                    ]),
                ),
                // inv[r][r] = 1.0
                assign_expr(
                    inv.index(Expr::ident("r")).index(Expr::ident("r")),
                    Expr::lit_f32(1.0),
                ),
                // diag_orig[r] = a[r][r]
                assign_expr(
                    Expr::ident("diag_orig").index(Expr::ident("r")),
                    a.index(Expr::ident("r")).index(Expr::ident("r")),
                ),
            ]),
        ),
        // Gauss-Jordan elimination with partial pivoting
        var_expr("singular", Expr::lit_bool(false)),
        for_loop_expr(
            for_init_var_expr("i", Expr::lit_u32(0)),
            Expr::ident("i").lt(b_var),
            for_step_assign_expr(Expr::ident("i"), Expr::ident("i") + 1u32),
            block(vec![
                // Find pivot
                var_expr("pivot", Expr::ident("i")),
                var_expr(
                    "pivot_val",
                    abs(a.index(Expr::ident("i")).index(Expr::ident("i"))),
                ),
                for_loop_expr(
                    for_init_var_expr("r", Expr::ident("i") + 1u32),
                    Expr::ident("r").lt(b_var),
                    for_step_assign_expr(Expr::ident("r"), Expr::ident("r") + 1u32),
                    block(vec![
                        let_expr(
                            "val",
                            abs(a.index(Expr::ident("r")).index(Expr::ident("i"))),
                        ),
                        if_block_expr(
                            Expr::ident("val").gt(Expr::ident("pivot_val")),
                            block(vec![
                                assign_expr(Expr::ident("pivot_val"), Expr::ident("val")),
                                assign_expr(Expr::ident("pivot"), Expr::ident("r")),
                            ]),
                            None,
                        ),
                    ]),
                ),
                if_block_expr(
                    Expr::ident("pivot_val").lt(Expr::lit_f32(1e-12)),
                    block(vec![assign_expr(
                        Expr::ident("singular"),
                        Expr::lit_bool(true),
                    )]),
                    None,
                ),
                // swap_rows(&a, &inv, i, pivot, b)
                call_stmt_expr(Expr::call_named(
                    "swap_rows",
                    vec![
                        a.addr_of(),
                        inv.addr_of(),
                        Expr::ident("i"),
                        Expr::ident("pivot"),
                        b_var,
                    ],
                )),
                // Clamp pivot value
                var_expr("piv", a.index(Expr::ident("i")).index(Expr::ident("i"))),
                if_block_expr(
                    abs(Expr::ident("piv")).lt(Expr::lit_f32(1e-12)),
                    block(vec![assign_expr(
                        Expr::ident("piv"),
                        select(
                            Expr::lit_f32(1e-12),
                            Expr::lit_f32(-1e-12),
                            Expr::ident("piv").lt(Expr::lit_f32(0.0)),
                        ),
                    )]),
                    None,
                ),
                let_expr("inv_piv", Expr::lit_f32(1.0) / Expr::ident("piv")),
                // Scale pivot row
                for_loop_expr(
                    for_init_var_expr("c", Expr::lit_u32(0)),
                    Expr::ident("c").lt(b_var),
                    for_step_assign_expr(Expr::ident("c"), Expr::ident("c") + 1u32),
                    block(vec![
                        assign_expr(
                            a.index(Expr::ident("i")).index(Expr::ident("c")),
                            a.index(Expr::ident("i")).index(Expr::ident("c"))
                                * Expr::ident("inv_piv"),
                        ),
                        assign_expr(
                            inv.index(Expr::ident("i")).index(Expr::ident("c")),
                            inv.index(Expr::ident("i")).index(Expr::ident("c"))
                                * Expr::ident("inv_piv"),
                        ),
                    ]),
                ),
                // Eliminate column i from all other rows
                for_loop_expr(
                    for_init_var_expr("r", Expr::lit_u32(0)),
                    Expr::ident("r").lt(b_var),
                    for_step_assign_expr(Expr::ident("r"), Expr::ident("r") + 1u32),
                    block(vec![
                        if_block_expr(
                            Expr::ident("r").eq(Expr::ident("i")),
                            block(vec![continue_stmt()]),
                            None,
                        ),
                        let_expr("factor", a.index(Expr::ident("r")).index(Expr::ident("i"))),
                        for_loop_expr(
                            for_init_var_expr("c", Expr::lit_u32(0)),
                            Expr::ident("c").lt(b_var),
                            for_step_assign_expr(Expr::ident("c"), Expr::ident("c") + 1u32),
                            block(vec![
                                assign_expr(
                                    a.index(Expr::ident("r")).index(Expr::ident("c")),
                                    a.index(Expr::ident("r")).index(Expr::ident("c"))
                                        - Expr::ident("factor")
                                            * a.index(Expr::ident("i")).index(Expr::ident("c")),
                                ),
                                assign_expr(
                                    inv.index(Expr::ident("r")).index(Expr::ident("c")),
                                    inv.index(Expr::ident("r")).index(Expr::ident("c"))
                                        - Expr::ident("factor")
                                            * inv.index(Expr::ident("i")).index(Expr::ident("c")),
                                ),
                            ]),
                        ),
                    ]),
                ),
            ]),
        ),
        // Singular fallback: diagonal inverse
        if_block_expr(
            Expr::ident("singular"),
            block(vec![for_loop_expr(
                for_init_var_expr("r", Expr::lit_u32(0)),
                Expr::ident("r").lt(b_var),
                for_step_assign_expr(Expr::ident("r"), Expr::ident("r") + 1u32),
                block(vec![
                    for_loop_expr(
                        for_init_var_expr("c", Expr::lit_u32(0)),
                        Expr::ident("c").lt(b_var),
                        for_step_assign_expr(Expr::ident("c"), Expr::ident("c") + 1u32),
                        block(vec![assign_expr(
                            inv.index(Expr::ident("r")).index(Expr::ident("c")),
                            Expr::lit_f32(0.0),
                        )]),
                    ),
                    assign_expr(
                        inv.index(Expr::ident("r")).index(Expr::ident("r")),
                        Expr::call_named(
                            "safe_inverse",
                            vec![Expr::ident("diag_orig").index(Expr::ident("r"))],
                        ),
                    ),
                ]),
            )]),
            None,
        ),
        // Write result to block_inv buffer
        let_expr("offset", Expr::ident("cell") * (b_var * b_var)),
        for_loop_expr(
            for_init_var_expr("r", Expr::lit_u32(0)),
            Expr::ident("r").lt(b_var),
            for_step_assign_expr(Expr::ident("r"), Expr::ident("r") + 1u32),
            block(vec![for_loop_expr(
                for_init_var_expr("c", Expr::lit_u32(0)),
                Expr::ident("c").lt(b_var),
                for_step_assign_expr(Expr::ident("c"), Expr::ident("c") + 1u32),
                block(vec![assign_expr(
                    Expr::ident("block_inv")
                        .index(Expr::ident("offset") + Expr::ident("r") * b_var + Expr::ident("c")),
                    inv.index(Expr::ident("r")).index(Expr::ident("c")),
                )]),
            )]),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "build_block_inv",
        std_params.clone(),
        None,
        std_attrs.clone(),
        build_body,
    )));

    // ── apply_block_precond ────────────────────────────────────────────────

    let apply_body = block(vec![
        let_expr("stride_x", Expr::ident("num_workgroups").field("x") * 64u32),
        let_expr(
            "cell",
            Expr::ident("global_id").field("y") * Expr::ident("stride_x")
                + Expr::ident("global_id").field("x"),
        ),
        if_block_expr(
            Expr::ident("cell").ge(params.field("num_cells")),
            block(vec![return_void()]),
            None,
        ),
        if_block_expr(
            params.field("num_cells").eq(0u32),
            block(vec![return_void()]),
            None,
        ),
        let_expr("b", params.field("n") / params.field("num_cells")),
        if_block_expr(
            Expr::ident("b").eq(0u32) | Expr::ident("b").gt(Expr::ident("MAX_BLOCK")),
            block(vec![return_void()]),
            None,
        ),
        let_expr("base", Expr::ident("cell") * Expr::ident("b")),
        let_expr(
            "offset",
            Expr::ident("cell") * (Expr::ident("b") * Expr::ident("b")),
        ),
        for_loop_expr(
            for_init_var_expr("r", Expr::lit_u32(0)),
            Expr::ident("r").lt(Expr::ident("b")),
            for_step_assign_expr(Expr::ident("r"), Expr::ident("r") + 1u32),
            block(vec![
                var_expr("sum", Expr::lit_f32(0.0)),
                for_loop_expr(
                    for_init_var_expr("c", Expr::lit_u32(0)),
                    Expr::ident("c").lt(Expr::ident("b")),
                    for_step_assign_expr(Expr::ident("c"), Expr::ident("c") + 1u32),
                    block(vec![assign_op_expr(
                        AssignOp::Add,
                        Expr::ident("sum"),
                        Expr::ident("block_inv").index(
                            Expr::ident("offset")
                                + Expr::ident("r") * Expr::ident("b")
                                + Expr::ident("c"),
                        ) * Expr::ident("vec_x").index(Expr::ident("base") + Expr::ident("c")),
                    )]),
                ),
                assign_expr(
                    Expr::ident("vec_y").index(Expr::ident("base") + Expr::ident("r")),
                    Expr::ident("sum"),
                ),
            ]),
        ),
    ]);

    m.push(Item::Function(Function::new(
        "apply_block_precond",
        std_params,
        None,
        std_attrs,
        apply_body,
    )));

    KernelWgsl::new(m)
}
