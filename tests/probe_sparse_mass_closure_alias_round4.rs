//! Isolated audit probe: a symbolic mass determinant can be nonzero when
//! model-declared algebraic primitive closures make it identically zero at
//! every RK stage.

use cfd2::solver::model::backend::ast::{
    fvm, surface_scalar_dim, vol_scalar_dim, Coefficient, Equation, EquationSystem,
};
use cfd2::solver::model::backend::state_layout::StateLayout;
use cfd2::solver::model::{generic_diffusion_demo_model, PrimitiveDerivations};
use cfd2_codegen::solver::codegen::explicit_rk::{generate_rk4_stage_kernel_program, Rk4Stage};
use cfd2_codegen::solver::codegen::fusion::lower_kernel_program_to_wgsl;
use cfd2_codegen::solver::codegen::ir::lower_system;
use cfd2_codegen::solver::codegen::unified_assembly::generate_matrix_free_residual_kernel_program;
use cfd2_codegen::solver::codegen::wgsl_ast::Expr as WgslExpr;
use cfd2_codegen::solver::ir::ports::{
    PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec,
};
use cfd2_codegen::solver::ir::SchemeRegistry;
use cfd2_codegen::solver::scheme::Scheme;
use cfd2_ir::ast::Expr;
use cfd2_ir::dimensions::{Dimensionless, DivDim, Time, Volume};

#[test]
fn compact_rhs_keeps_full_coupled_flux_and_bc_ranks() {
    let q = vol_scalar_dim::<Dimensionless>("q");
    let a = vol_scalar_dim::<Dimensionless>("a");
    let r = vol_scalar_dim::<Dimensionless>("r");
    let phi = surface_scalar_dim::<DivDim<Volume, Time>>("phi");

    let mut q_eq = Equation::new(q);
    q_eq.add_term(fvm::ddt(q));
    q_eq.add_term(fvm::div(phi, q));
    let a_eq = Equation::new(a);
    let mut r_eq = Equation::new(r);
    r_eq.add_term(fvm::ddt(r));
    r_eq.add_term(fvm::div(phi, r));
    let mut system = EquationSystem::new();
    system.add_equation(q_eq);
    system.add_equation(a_eq);
    system.add_equation(r_eq);

    let discrete = lower_system(&system, &SchemeRegistry::new(Scheme::Upwind)).unwrap();
    let slots = ResolvedStateSlotsSpec {
        stride: 3,
        slots: [q, a, r]
            .into_iter()
            .enumerate()
            .map(|(offset, field)| ResolvedStateSlotSpec {
                name: field.name().to_string(),
                kind: PortFieldKind::Scalar,
                unit: field.unit(),
                base_offset: offset as u32,
            })
            .collect(),
    };
    let residual = generate_matrix_free_residual_kernel_program(
        "interleaved_flux_residual",
        &discrete,
        &slots,
        3,
        false,
        &[],
    )
    .unwrap();
    let wgsl = lower_kernel_program_to_wgsl(&residual).unwrap().to_wgsl();

    // Face fluxes and BC tables stay keyed by the full q,a,r coupled rank,
    // while only final RHS materialization is projected to compact q,r ranks.
    assert!(wgsl.contains("fluxes[face_idx * 3u + 0u]"));
    assert!(wgsl.contains("fluxes[face_idx * 3u + 2u]"));
    assert!(wgsl.contains("bc_kind[face_idx * 3u + 0u]"));
    assert!(wgsl.contains("bc_kind[face_idx * 3u + 2u]"));
    assert!(wgsl.contains("rhs[idx * 2u + 0u] = rhs_0"));
    assert!(wgsl.contains("rhs[idx * 2u + 1u] = rhs_2"));
}

#[test]
fn validator_rejects_mass_singular_under_constant_algebraic_closure() {
    let q = vol_scalar_dim::<Dimensionless>("q");
    let a = vol_scalar_dim::<Dimensionless>("a");
    let r = vol_scalar_dim::<Dimensionless>("r");

    // Differential compact block is M = [[a, 1], [1, 1]]. The validator sees
    // det(M) = a - 1, but the ordered primitive closure makes a := 1 exactly.
    let mut q_eq = Equation::new(q);
    q_eq.add_term(fvm::ddt_coeff(Coefficient::field(a).unwrap(), q));
    q_eq.add_term(fvm::ddt(r));
    let a_eq = Equation::new(a);
    let mut r_eq = Equation::new(r);
    r_eq.add_term(fvm::ddt(q));
    r_eq.add_term(fvm::ddt(r));

    let mut system = EquationSystem::new();
    system.add_equation(q_eq);
    system.add_equation(a_eq);
    system.add_equation(r_eq);

    let discrete = lower_system(&system, &SchemeRegistry::new(Scheme::Upwind)).unwrap();
    let slots = ResolvedStateSlotsSpec {
        stride: 3,
        slots: [q, a, r]
            .into_iter()
            .enumerate()
            .map(|(offset, field)| ResolvedStateSlotSpec {
                name: field.name().to_string(),
                kind: PortFieldKind::Scalar,
                unit: field.unit(),
                base_offset: offset as u32,
            })
            .collect(),
    };
    let residual = generate_matrix_free_residual_kernel_program(
        "closure_singular_residual",
        &discrete,
        &slots,
        0,
        false,
        &[],
    )
    .unwrap();
    let residual_wgsl = lower_kernel_program_to_wgsl(&residual).unwrap().to_wgsl();
    let stage = generate_rk4_stage_kernel_program(
        "closure_singular_stage",
        &discrete,
        &slots,
        &[(1, WgslExpr::lit_f32(1.0))],
        Rk4Stage::First,
        &[],
    )
    .unwrap();
    let stage_wgsl = lower_kernel_program_to_wgsl(&stage).unwrap().to_wgsl();

    assert!(residual_wgsl.contains("rhs[idx * 2u + 0u] = rhs_0"));
    assert!(residual_wgsl.contains("rhs[idx * 2u + 1u] = rhs_2"));
    assert!(stage_wgsl.contains("mass_0_0 += state[idx * 3u + 1u]"));
    assert!(stage_wgsl.contains("mass_0_1 += 1.0"));
    assert!(stage_wgsl.contains("mass_1_0 += 1.0"));
    assert!(stage_wgsl.contains("mass_1_1 += 1.0"));
    assert!(stage_wgsl.contains("mass_1_1 -= factor_0_1 * mass_0_1"));
    assert!(stage_wgsl.contains("rate_1 = rate_1 / select(mass_1_1"));

    for line in residual_wgsl
        .lines()
        .chain(stage_wgsl.lines())
        .filter(|line| {
            line.contains("rhs[idx * 2u")
                || line.contains("mass_0_0 +=")
                || line.contains("mass_0_1 +=")
                || line.contains("mass_1_0 +=")
                || line.contains("mass_1_1 +=")
                || line.contains("mass_1_1 -=")
                || line.contains("rate_1 = rate_1 / select")
        })
    {
        println!("{line}");
    }

    let mut model = generic_diffusion_demo_model().expect("base model");
    model.system = system;
    model.state_layout = StateLayout::new(vec![q, a, r]);
    model.primitives = PrimitiveDerivations::identity();
    model.explicit_primitives = Some(PrimitiveDerivations {
        derivations: [("a".to_string(), Expr::lit_f32(1.0))]
            .into_iter()
            .collect(),
    });

    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(
        error.contains("identically singular"),
        "unexpected closure-aware rejection: {error}"
    );
}

#[test]
fn validator_rejects_mass_singular_under_declared_closure_aliases() {
    let q = vol_scalar_dim::<Dimensionless>("q");
    let a = vol_scalar_dim::<Dimensionless>("a");
    let r = vol_scalar_dim::<Dimensionless>("r");
    let b = vol_scalar_dim::<Dimensionless>("b");

    // Differential compact block, after removing the interleaved algebraic
    // rows, is M = [[a, 1], [b, 1]], with symbolic det(M) = a - b.
    let mut q_eq = Equation::new(q);
    q_eq.add_term(fvm::ddt_coeff(Coefficient::field(a).unwrap(), q));
    q_eq.add_term(fvm::ddt(r));
    let a_eq = Equation::new(a);
    let mut r_eq = Equation::new(r);
    r_eq.add_term(fvm::ddt_coeff(Coefficient::field(b).unwrap(), q));
    r_eq.add_term(fvm::ddt(r));
    let b_eq = Equation::new(b);

    let mut system = EquationSystem::new();
    system.add_equation(q_eq);
    system.add_equation(a_eq);
    system.add_equation(r_eq);
    system.add_equation(b_eq);

    let mut model = generic_diffusion_demo_model().expect("base model");
    model.system = system;
    model.state_layout = StateLayout::new(vec![q, a, r, b]);
    model.primitives = PrimitiveDerivations::identity();
    model.explicit_primitives = Some(PrimitiveDerivations {
        derivations: [
            ("a".to_string(), Expr::ident("q")),
            ("b".to_string(), Expr::ident("q")),
        ]
        .into_iter()
        .collect(),
    });

    // Both closure kernels write the same f32 state value, so a == b bitwise
    // before every residual and after every stage. The runtime block is thus
    // [[q, 1], [q, 1]] and its second pivot is exactly zero.
    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(
        error.contains("identically singular"),
        "unexpected closure-alias rejection: {error}"
    );
}
