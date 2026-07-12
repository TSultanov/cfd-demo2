//! Isolated adversarial audit of closure-aware explicit mass validation.
//!
//! These tests document both the protected polynomial/syntactic cases and the
//! conservative rejection of semantically related opaque closures.

use cfd2::solver::model::backend::ast::{
    fvm, vol_scalar_dim, Coefficient, Equation, EquationSystem,
};
use cfd2::solver::model::backend::state_layout::StateLayout;
use cfd2::solver::model::{
    generic_diffusion_demo_model, ExplicitMassClosureProof, ModelSpec, PrimitiveDerivations,
};
use cfd2_ir::ast::{Expr, ExprNode, Literal};
use cfd2_ir::dimensions::Dimensionless;

fn float_spelling(text: &str) -> Expr {
    Expr::alloc_node(ExprNode::Literal(Literal::Float(text.to_string())))
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::call_named(name, args)
}

/// Build interleaved rows q, a, r, b with compact differential block
///
///     M = [[a, 1],
///          [b, 1]].
///
/// Hence every pair of closures that always produces a == b must be rejected
/// as identically singular.
fn alias_block(a_expr: Expr, b_expr: Expr) -> ModelSpec {
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
        derivations: [("a".to_string(), a_expr), ("b".to_string(), b_expr)]
            .into_iter()
            .collect(),
    });
    model
}

fn singular(error: Result<(), String>) -> bool {
    error.unwrap_err().contains("identically singular")
}

fn opaque_rejected(error: Result<(), String>) -> bool {
    error
        .unwrap_err()
        .contains("nonlinear/opaque primitive closure")
}

#[test]
fn controls_reject_polynomial_commutation_and_identical_opaque_calls() {
    let q = Expr::ident("q");
    let r = Expr::ident("r");

    assert!(singular(
        alias_block(q.clone() + r.clone(), r.clone() + q.clone()).validate_explicit_rk4()
    ));
    assert!(singular(
        alias_block(q.clone() * r.clone(), r.clone() * q.clone()).validate_explicit_rk4()
    ));
    assert!(singular(
        alias_block(call("abs", vec![q.clone()]), call("abs", vec![q.clone()]),)
            .validate_explicit_rk4()
    ));
    let abs_q = call("abs", vec![q.clone()]);
    assert!(singular(
        alias_block(abs_q.clone() + r.clone(), r.clone() + abs_q.clone(),).validate_explicit_rk4()
    ));
    assert!(singular(
        alias_block(abs_q.clone() * r.clone(), r * abs_q).validate_explicit_rk4()
    ));
    assert!(singular(
        alias_block(q.clone() / Expr::lit_f32(1.0), q).validate_explicit_rk4()
    ));
}

#[test]
fn rejects_idempotent_builtin_vs_identity_as_opaque() {
    let q = Expr::ident("q");
    let result = alias_block(call("min", vec![q.clone(), q.clone()]), q).validate_explicit_rk4();
    assert!(opaque_rejected(result));
}

#[test]
fn rejects_bitwise_identity_select_vs_identity_as_opaque() {
    let q = Expr::ident("q");
    // select(q, q, true) returns the exact input bit pattern; unlike identities
    // based on arithmetic or min/max, this does not depend on rounding or zero
    // sign behavior. For every finite state, the two mass rows are identical.
    let result = alias_block(
        call("select", vec![q.clone(), q.clone(), Expr::lit_bool(true)]),
        q,
    )
    .validate_explicit_rk4();
    assert!(opaque_rejected(result));
}

#[test]
fn rejects_equal_opaque_calls_despite_different_literal_spellings() {
    let q = Expr::ident("q");
    let a = call("min", vec![q.clone(), float_spelling("1.0")]);
    let b = call("min", vec![q, float_spelling("1.00")]);
    let result = alias_block(a, b).validate_explicit_rk4();
    assert!(opaque_rejected(result));
}

#[test]
fn rejects_equivalent_opaque_forms_without_domain_proof() {
    let q = Expr::ident("q");

    // Both idempotent calls return q for every finite q.
    let result = alias_block(
        call("min", vec![q.clone(), q.clone()]),
        call("max", vec![q.clone(), q.clone()]),
    )
    .validate_explicit_rk4();
    assert!(opaque_rejected(result));

    // abs(1.0) is exactly 1.0, so both divisions return the same f32 q.
    let result = alias_block(q.clone() / call("abs", vec![Expr::lit_f32(1.0)]), q.clone())
        .validate_explicit_rk4();
    assert!(opaque_rejected(result));

    // Adding +0 after an opaque call does not change its finite f32 value.
    let abs_q = call("abs", vec![q]);
    let result = alias_block(abs_q.clone() + Expr::lit_f32(0.0), abs_q).validate_explicit_rk4();
    assert!(opaque_rejected(result));
}

#[test]
fn runtime_pivot_policy_requires_an_explicit_nonempty_model_invariant() {
    let q = Expr::ident("q");
    let opaque = call("max", vec![q, Expr::lit_f32(1.0)]);

    let mut missing = alias_block(opaque.clone(), Expr::lit_f32(0.0));
    missing.explicit_mass_closure_proof = ExplicitMassClosureProof::RuntimePivoted {
        justification: "   ",
    };
    let error = missing.validate_explicit_rk4().unwrap_err();
    assert!(error.contains("requires a nonempty justification"), "{error}");

    let mut declared = alias_block(opaque, Expr::lit_f32(0.0));
    declared.explicit_mass_closure_proof = ExplicitMassClosureProof::RuntimePivoted {
        justification: "test domain excludes equal rows; generated full pivoting guards every stage",
    };
    declared
        .validate_explicit_rk4()
        .expect("a model-owned runtime domain may opt into guarded opaque mass coefficients");
}

#[test]
fn constant_division_lowering_is_safe_but_not_runtime_bit_exact() {
    // The symbolic lowerer rewrites q/3 to q*(f32(1/3)). WGSL still executes
    // division, and these operations can round differently.
    let sample = f32::from_bits(0x3f00_0002);
    let divided = sample / 3.0_f32;
    let multiplied = sample * (1.0_f32 / 3.0_f32);
    assert_ne!(divided.to_bits(), multiplied.to_bits());

    let q = Expr::ident("q");
    let error = alias_block(
        q.clone() / Expr::lit_f32(3.0),
        q * Expr::lit_f32(1.0_f32 / 3.0_f32),
    )
    .validate_explicit_rk4();
    assert!(
        singular(error),
        "the current reciprocal rewrite should conservatively conflate the forms"
    );
}

#[test]
fn rejects_exact_and_semantic_transitive_opaque_aliases() {
    let q = vol_scalar_dim::<Dimensionless>("q");
    let a = vol_scalar_dim::<Dimensionless>("a");
    let c = vol_scalar_dim::<Dimensionless>("c");
    let r = vol_scalar_dim::<Dimensionless>("r");
    let b = vol_scalar_dim::<Dimensionless>("b");

    let mut q_eq = Equation::new(q);
    q_eq.add_term(fvm::ddt_coeff(Coefficient::field(a).unwrap(), q));
    q_eq.add_term(fvm::ddt(r));
    let a_eq = Equation::new(a);
    let c_eq = Equation::new(c);
    let mut r_eq = Equation::new(r);
    r_eq.add_term(fvm::ddt_coeff(Coefficient::field(b).unwrap(), q));
    r_eq.add_term(fvm::ddt(r));
    let b_eq = Equation::new(b);
    let mut system = EquationSystem::new();
    system.add_equation(q_eq);
    system.add_equation(a_eq);
    system.add_equation(c_eq);
    system.add_equation(r_eq);
    system.add_equation(b_eq);

    let build = |a_expr: Expr, b_expr: Expr| {
        let mut model = generic_diffusion_demo_model().expect("base model");
        model.system = system.clone();
        model.state_layout = StateLayout::new(vec![q, a, c, r, b]);
        model.primitives = PrimitiveDerivations::identity();
        model.explicit_primitives = Some(PrimitiveDerivations {
            derivations: [
                ("a".to_string(), a_expr),
                ("c".to_string(), Expr::ident("q")),
                ("b".to_string(), b_expr),
            ]
            .into_iter()
            .collect(),
        });
        model
    };

    let q_expr = Expr::ident("q");
    let c_expr = Expr::ident("c");
    let exact = build(
        call("abs", vec![q_expr.clone()]),
        call("abs", vec![c_expr.clone()]),
    )
    .validate_explicit_rk4();
    assert!(singular(exact));

    let semantic = build(call("min", vec![q_expr.clone(), q_expr]), c_expr).validate_explicit_rk4();
    assert!(opaque_rejected(semantic));
}
