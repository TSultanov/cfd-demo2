//! Isolated adversarial re-audit of the closure-aware explicit mass proof.
//! Production solver files are deliberately untouched.

use cfd2::solver::model::backend::ast::{
    fvm, vol_scalar_dim, Coefficient, Equation, EquationSystem,
};
use cfd2::solver::model::backend::state_layout::StateLayout;
use cfd2::solver::model::{
    generic_diffusion_demo_model, ModelSpec, PrimitiveDerivations,
};
use cfd2_codegen::solver::codegen::explicit_rk::{
    generate_rk4_stage_kernel_program, Rk4Stage,
};
use cfd2_codegen::solver::codegen::fusion::lower_kernel_program_to_wgsl;
use cfd2_codegen::solver::codegen::ir::lower_system;
use cfd2_codegen::solver::ir::ports::{
    PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec,
};
use cfd2_codegen::solver::ir::SchemeRegistry;
use cfd2_codegen::solver::scheme::Scheme;
use cfd2_ir::ast::Expr;
use cfd2_ir::dimensions::Dimensionless;

/// Build the differential mass block [[a, 1], [b, 1]], with the recoverable
/// algebraic fields interleaved among differential rows as q,a,r,b.
fn closure_alias_model(a_expr: Expr, b_expr: Expr) -> ModelSpec {
    let q = vol_scalar_dim::<Dimensionless>("q");
    let a = vol_scalar_dim::<Dimensionless>("a");
    let r = vol_scalar_dim::<Dimensionless>("r");
    let b = vol_scalar_dim::<Dimensionless>("b");

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
            ("a".to_string(), a_expr),
            ("b".to_string(), b_expr),
        ]
        .into_iter()
        .collect(),
    });
    model
}

fn assert_singular_rejected(a: Expr, b: Expr, label: &str) {
    let error = match closure_alias_model(a, b).validate_explicit_rk4() {
        Ok(()) => panic!("{label}: accepted an always-singular closure alias"),
        Err(error) => error,
    };
    assert!(
        error.contains("identically singular"),
        "{label}: unexpected rejection: {error}"
    );
}

#[test]
fn fixed_proof_rejects_requested_polynomial_and_identical_opaque_variants() {
    assert_singular_rejected(
        Expr::ident("q"),
        Expr::ident("q"),
        "direct aliases",
    );
    assert_singular_rejected(
        Expr::ident("q"),
        Expr::ident("a"),
        "transitive aliases",
    );
    assert_singular_rejected(
        Expr::ident("q") + Expr::ident("r"),
        Expr::ident("r") + Expr::ident("q"),
        "commuted sums",
    );
    assert_singular_rejected(
        Expr::ident("q") * Expr::ident("r"),
        Expr::ident("r") * Expr::ident("q"),
        "commuted products",
    );
    assert_singular_rejected(
        Expr::ident("q") / Expr::lit_f32(2.0),
        Expr::lit_f32(0.5) * Expr::ident("q"),
        "constant division",
    );
    assert_singular_rejected(
        Expr::call_named("abs", vec![Expr::ident("q")]),
        Expr::call_named("abs", vec![Expr::ident("q")]),
        "identical opaque calls",
    );
    assert_singular_rejected(
        Expr::call_named("abs", vec![Expr::ident("q")]),
        Expr::ident("a"),
        "direct alias of an opaque closure",
    );
}

#[test]
fn rejects_idempotent_transitive_opaque_closure_without_model_domain_proof() {
    // The ordered closure writes:
    //   a = abs(q)
    //   b = abs(a) = abs(abs(q)) = abs(q)
    // Therefore [[a,1],[b,1]] has bitwise-identical rows for every f32 q. The
    // A syntactic polynomial proof cannot establish this identity in general,
    // so the safe default must reject the remaining opaque determinant.
    let model = closure_alias_model(
        Expr::call_named("abs", vec![Expr::ident("q")]),
        Expr::call_named("abs", vec![Expr::ident("a")]),
    );
    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(error.contains("nonlinear/opaque primitive closure"), "{error}");

    let ordered = model
        .explicit_primitives
        .as_ref()
        .unwrap()
        .ordered()
        .unwrap();
    assert_eq!(ordered[0].0, "a");
    assert_eq!(ordered[1].0, "b");
}

#[test]
fn rejects_bitwise_select_identity_without_model_domain_proof() {
    // Both select arms are the same value, so this returns q's exact f32 bit
    // pattern for either branch. The comparison supplies a valid WGSL bool
    // without relying on the primitive resolver's bool-literal handling.
    let q = Expr::ident("q");
    let selected = Expr::call_named(
        "select",
        vec![
            q.clone(),
            q.clone(),
            q.clone().gt(Expr::lit_f32(0.0)),
        ],
    );
    let model = closure_alias_model(selected, q);
    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(error.contains("nonlinear/opaque primitive closure"), "{error}");

    let discrete = lower_system(&model.system, &SchemeRegistry::new(Scheme::Upwind)).unwrap();
    let slots = ResolvedStateSlotsSpec {
        stride: 4,
        slots: model
            .state_layout
            .fields()
            .iter()
            .enumerate()
            .map(|(offset, field)| ResolvedStateSlotSpec {
                name: field.name().to_string(),
                kind: PortFieldKind::Scalar,
                unit: field.unit(),
                base_offset: offset as u32,
            })
            .collect(),
    };
    let primitives = model
        .explicit_primitives
        .as_ref()
        .unwrap()
        .ordered()
        .unwrap()
        .into_iter()
        .map(|(name, expr)| {
            let offset = match name.as_str() {
                "a" => 1,
                "b" => 3,
                _ => unreachable!(),
            };
            (offset, expr)
        })
        .collect::<Vec<_>>();
    let stage = generate_rk4_stage_kernel_program(
        "select_identity_survivor",
        &discrete,
        &slots,
        &primitives,
        Rk4Stage::First,
        &[],
    )
    .unwrap();
    let wgsl = lower_kernel_program_to_wgsl(&stage).unwrap().to_wgsl();
    let assignment = wgsl
        .lines()
        .find(|line| line.contains("state[idx * 4u + 1u] = select("))
        .expect("generated closure must materialize the select identity");
    assert!(assignment.matches("state[idx * 4u + 0u]").count() >= 3);
    assert!(assignment.contains("> 0.0"));
    println!("rejected opaque closure WGSL: {assignment}");
}
