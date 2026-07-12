//! Final independent adversarial probes for the compact explicit-RK mass solve.
//!
//! This file deliberately does not modify the production implementation.  The
//! tests preserve concrete finite-f32 counterexamples that the capability gate
//! and row-scaled runtime solve must reject.

use cfd2::solver::model::backend::ast::{
    fvm, vol_scalar_dim, Coefficient, Equation, EquationSystem,
};
use cfd2::solver::model::backend::state_layout::StateLayout;
use cfd2::solver::model::{
    generic_diffusion_demo_model, ExplicitMassClosureProof, ModelSpec, PrimitiveDerivations,
};
use cfd2_codegen::solver::codegen::explicit_rk::{generate_rk4_stage_kernel_program, Rk4Stage};
use cfd2_codegen::solver::codegen::fusion::lower_kernel_program_to_wgsl;
use cfd2_codegen::solver::codegen::ir::lower_system;
use cfd2_codegen::solver::ir::ports::{
    PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec,
};
use cfd2_codegen::solver::ir::SchemeRegistry;
use cfd2_codegen::solver::scheme::Scheme;
use cfd2_ir::ast::Expr;
use cfd2_ir::dimensions::Dimensionless;

/// Construct a two-row scalar model whose compact differential mass block is
/// exactly the supplied constant matrix.
fn constant_mass_model(a: [[f32; 2]; 2]) -> ModelSpec {
    let q = vol_scalar_dim::<Dimensionless>("q");
    let r = vol_scalar_dim::<Dimensionless>("r");
    let mut q_eq = Equation::new(q);
    q_eq.add_term(fvm::ddt_coeff(Coefficient::constant(a[0][0] as f64), q));
    q_eq.add_term(fvm::ddt_coeff(Coefficient::constant(a[0][1] as f64), r));
    let mut r_eq = Equation::new(r);
    r_eq.add_term(fvm::ddt_coeff(Coefficient::constant(a[1][0] as f64), q));
    r_eq.add_term(fvm::ddt_coeff(Coefficient::constant(a[1][1] as f64), r));
    let mut system = EquationSystem::new();
    system.add_equation(q_eq);
    system.add_equation(r_eq);

    let mut model = generic_diffusion_demo_model().expect("base model");
    model.system = system;
    model.state_layout = StateLayout::new(vec![q, r]);
    model.primitives = PrimitiveDerivations::identity();
    model.explicit_primitives = None;
    model.explicit_mass_closure_proof = ExplicitMassClosureProof::ExactSymbolic;
    model
}

fn generated_scalar_stage_wgsl(model: &ModelSpec, id: &str) -> String {
    let discrete = lower_system(&model.system, &SchemeRegistry::new(Scheme::Upwind))
        .expect("lower scalar-only adversarial model");
    let slots = ResolvedStateSlotsSpec {
        stride: model.state_layout.fields().len() as u32,
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
    let stage = generate_rk4_stage_kernel_program(
        id,
        &discrete,
        &slots,
        &[],
        Rk4Stage::First,
        &[],
    )
    .expect("generate adversarial explicit stage");
    lower_kernel_program_to_wgsl(&stage)
        .expect("lower adversarial stage WGSL")
        .to_wgsl()
}

/// The operations emitted by `explicit_rk.rs` for a 2x2 block, including its
/// absolute (not scale-relative) pivot cutoff.
fn generated_solve_2x2(mut a: [[f32; 2]; 2], mut b: [f32; 2]) -> ([f32; 2], [[f32; 2]; 2]) {
    if a[1][0].abs() > a[0][0].abs() {
        a.swap(0, 1);
        b.swap(0, 1);
    }
    let pivot = if a[0][0].abs() < 1.0e-20 {
        f32::NAN
    } else {
        a[0][0]
    };
    let factor = a[1][0] / pivot;
    a[1][0] -= factor * a[0][0];
    a[1][1] -= factor * a[0][1];
    b[1] -= factor * b[0];
    let pivot_1 = if a[1][1].abs() < 1.0e-20 {
        f32::NAN
    } else {
        a[1][1]
    };
    let x1 = b[1] / pivot_1;
    let x0 = (b[0] - a[0][1] * x1) / a[0][0];
    ([x0, x1], a)
}

#[test]
fn exact_symbolic_rejects_f32_absorption_in_polynomial_closure() {
    let q = vol_scalar_dim::<Dimensionless>("q");
    let a = vol_scalar_dim::<Dimensionless>("a");
    let r = vol_scalar_dim::<Dimensionless>("r");
    let b = vol_scalar_dim::<Dimensionless>("b");

    // M = [[a, 1], [b, 1]].  The exact-polynomial validator sees
    // det(M) = 2^-26*q.  In emitted f32 arithmetic, however,
    // fl(q + 2^-26*q) == q for every finite f32 q: for normal q the increment
    // is at most one quarter ulp, and for subnormal q the product underflows to
    // zero.  Thus a and b are bitwise equal and the runtime block is singular
    // at every finite state.
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

    let delta = Expr::lit_f32(2.0f32.powi(-26));
    let q_expr = Expr::ident("q");
    let mut model = generic_diffusion_demo_model().expect("base model");
    model.system = system;
    model.state_layout = StateLayout::new(vec![q, a, r, b]);
    model.primitives = PrimitiveDerivations::identity();
    model.explicit_primitives = Some(PrimitiveDerivations {
        derivations: [
            ("a".to_string(), q_expr.clone() + delta * q_expr.clone()),
            ("b".to_string(), q_expr),
        ]
        .into_iter()
        .collect(),
    });
    model.explicit_mass_closure_proof = ExplicitMassClosureProof::ExactSymbolic;

    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(error.contains("dynamic f32 mass arithmetic"), "{error}");

    // Exercise signs, binade boundaries, subnormals, and a deterministic broad
    // bit-pattern sample.  The mathematical ulp bound above covers all finite
    // f32 values; this executable sample guards the concrete premise.
    let mut bits = 0x1234_5678u32;
    for &q_value in &[
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1.0,
        -1.0,
        f32::MAX,
        -f32::MAX,
    ] {
        assert_eq!(
            (q_value + 2.0f32.powi(-26) * q_value).to_bits(),
            q_value.to_bits()
        );
    }
    for _ in 0..1_000_000 {
        bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let q_value = f32::from_bits(bits);
        if q_value.is_finite() {
            assert_eq!(
                (q_value + 2.0f32.powi(-26) * q_value).to_bits(),
                q_value.to_bits()
            );
        }
    }
}

#[test]
fn exact_symbolic_rejects_f32_absorption_without_a_primitive_closure() {
    let q = vol_scalar_dim::<Dimensionless>("q");
    let r = vol_scalar_dim::<Dimensionless>("r");
    let delta = 2.0f32.powi(-26);

    // The same absorbed increment can be expressed directly in the DDT tree:
    //
    //   M00 = q + 2^-26*q, M01 = 1
    //   M10 = q,            M11 = 1.
    //
    // This has no primitive closure at all, so merely tainting nonlinear
    // primitive recovery cannot repair the ExactSymbolic contract.
    let q_coeff = Coefficient::field(q).unwrap();
    let delta_q = Coefficient::product(Coefficient::constant(delta as f64), q_coeff.clone())
        .expect("scalar product");
    let mut q_eq = Equation::new(q);
    q_eq.add_term(fvm::ddt_coeff(q_coeff.clone(), q));
    q_eq.add_term(fvm::ddt_coeff(delta_q, q));
    q_eq.add_term(fvm::ddt(r));
    let mut r_eq = Equation::new(r);
    r_eq.add_term(fvm::ddt_coeff(q_coeff, q));
    r_eq.add_term(fvm::ddt(r));
    let mut system = EquationSystem::new();
    system.add_equation(q_eq);
    system.add_equation(r_eq);

    let mut model = generic_diffusion_demo_model().expect("base model");
    model.system = system;
    model.state_layout = StateLayout::new(vec![q, r]);
    model.primitives = PrimitiveDerivations::identity();
    model.explicit_primitives = None;
    model.explicit_mass_closure_proof = ExplicitMassClosureProof::ExactSymbolic;
    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(error.contains("dynamic f32 mass arithmetic"), "{error}");

    let discrete = lower_system(&model.system, &SchemeRegistry::new(Scheme::Upwind)).unwrap();
    let slots = ResolvedStateSlotsSpec {
        stride: 2,
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
    let stage = generate_rk4_stage_kernel_program(
        "direct_f32_absorption",
        &discrete,
        &slots,
        &[],
        Rk4Stage::First,
        &[],
    )
    .unwrap();
    let wgsl = lower_kernel_program_to_wgsl(&stage).unwrap().to_wgsl();
    assert!(wgsl.contains("mass_0_0 += state[idx * 2u + 0u]"));
    assert!(wgsl.contains("mass_0_0 += 0.000000014901161 * state[idx * 2u + 0u]"));
    assert!(wgsl.contains("mass_1_0 += state[idx * 2u + 0u]"));
}

#[test]
fn exact_symbolic_rejects_opaque_mass_before_real_algebra_cancels_its_atom() {
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

    // `a` is opaque and always at least 2^25. The lowerer represents `b` as
    // the polynomial A+1, then det([[A,1],[A+1,1]]) simplifies to -1 and no
    // PrimitiveClosure atom survives the final `contains_opaque_closure` test.
    // Emitted f32 evaluates b=fl(a+1)==a because ulp(a)>=4, so the actual rows
    // are bitwise identical for every finite q.
    let opaque_a = Expr::call_named(
        "max",
        vec![
            Expr::call_named("abs", vec![Expr::ident("q")]),
            Expr::lit_f32(2.0f32.powi(25)),
        ],
    );
    let mut model = generic_diffusion_demo_model().expect("base model");
    model.system = system;
    model.state_layout = StateLayout::new(vec![q, a, r, b]);
    model.primitives = PrimitiveDerivations::identity();
    model.explicit_primitives = Some(PrimitiveDerivations {
        derivations: [
            ("a".to_string(), opaque_a),
            ("b".to_string(), Expr::ident("a") + Expr::lit_f32(1.0)),
        ]
        .into_iter()
        .collect(),
    });
    model.explicit_mass_closure_proof = ExplicitMassClosureProof::ExactSymbolic;
    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(
        error.contains("nonlinear/opaque primitive closure"),
        "{error}"
    );

    let mut bits = 0x9e37_79b9u32;
    for _ in 0..1_000_000 {
        bits = bits.wrapping_mul(22_695_477).wrapping_add(1);
        let q_value = f32::from_bits(bits);
        if q_value.is_finite() {
            let a_value = q_value.abs().max(2.0f32.powi(25));
            assert_eq!((a_value + 1.0).to_bits(), a_value.to_bits());
        }
    }
}

#[test]
fn row_relative_pivot_cutoff_rejects_a_sixtyfold_forward_error_block() {
    // Every entry and both post-pivot diagonals are finite and far above 1e-20,
    // so the current constant-mass validator accepts this block.  Its column
    // scaling makes it nearly rank one, though, and ordinary f32 partial
    // pivoting loses the physically important first rate.
    let a = [[1.0e-10_f32, 1.0], [1.0e-7, 1.0000001e3]];
    let model = constant_mass_model(a);
    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(
        error.contains("2^-19 safety floor") || error.contains("1e-6 equilibrated cutoff"),
        "{error}"
    );

    let b = [-3.9999900_f32, -3.9999905e3_f32];
    let (got, eliminated) = generated_solve_2x2(a, b);
    assert!(got.iter().all(|value| value.is_finite()));
    assert!(eliminated[0][0].abs() > 1.0e-20);
    assert!(eliminated[1][1].abs() > 1.0e-20);

    // Closed-form f64 solve of the exact matrix/RHS represented by those f32
    // inputs.  This is about the local solve, not discretization error.
    let a00 = a[0][0] as f64;
    let a01 = a[0][1] as f64;
    let a10 = a[1][0] as f64;
    let a11 = a[1][1] as f64;
    let det = a00 * a11 - a01 * a10;
    let reference = [
        ((b[0] as f64) * a11 - a01 * (b[1] as f64)) / det,
        (a00 * (b[1] as f64) - (b[0] as f64) * a10) / det,
    ];
    let error =
        ((got[0] as f64 - reference[0]).powi(2) + (got[1] as f64 - reference[1]).powi(2)).sqrt();
    let reference_norm = (reference[0].powi(2) + reference[1].powi(2)).sqrt();
    assert!(
        error / reference_norm > 60.0,
        "expected gross silent forward error: got={got:?}, reference={reference:?}"
    );

    let discrete = lower_system(&model.system, &SchemeRegistry::new(Scheme::Upwind)).unwrap();
    let slots = ResolvedStateSlotsSpec {
        stride: 2,
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
    let stage = generate_rk4_stage_kernel_program(
        "ill_conditioned_constant_mass",
        &discrete,
        &slots,
        &[],
        Rk4Stage::First,
        &[],
    )
    .unwrap();
    let wgsl = lower_kernel_program_to_wgsl(&stage).unwrap().to_wgsl();
    assert!(wgsl.contains("mass_1_0 = select(mass_1_0 / mass_scale_1"));
    assert!(wgsl.contains("rate_1 / mass_scale_1"));
    assert!(wgsl.contains("mass_conditioning_scale_0"));
    assert!(wgsl.contains("mass_column_scale_0"));
    assert!(wgsl.contains("mass_column_scale_1"));
    assert!(wgsl.contains("< 0.000001"));
}

#[test]
fn equilibrated_guard_rejects_an_unrepresentable_column_unit_rescaling() {
    // This is diag(1e15, 1e15) * [[0.2, 1], [1, 1]]
    // * diag(1e-30, 1).  Every raw entry and pivot is finite and above the
    // absolute cutoff, while the first column is 1e-30 of the second in the
    // row-normalized matrix.  Treating that dimensionless column scale as the
    // physical 1e-20 pivot floor used to reject this perfectly regular block.
    let a = [[2.0e-16_f32, 1.0e15], [1.0e-15, 1.0e15]];
    let model = constant_mass_model(a);
    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(error.contains("2^-19 safety floor"), "{error}");

    let row_scales = [
        a[0].iter()
            .fold(0.0_f32, |scale, value| scale.max(value.abs())),
        a[1].iter()
            .fold(0.0_f32, |scale, value| scale.max(value.abs())),
    ];
    let column_scales = [
        (a[0][0].abs() / row_scales[0]).max(a[1][0].abs() / row_scales[1]),
        (a[0][1].abs() / row_scales[0]).max(a[1][1].abs() / row_scales[1]),
    ];
    assert!(column_scales[0] < 1.0e-20);
    let normalized =
        |row: usize, column: usize| a[row][column] / (row_scales[row] * column_scales[column]);
    let diagonal = normalized(0, 0) * normalized(1, 1);
    let cross = normalized(0, 1) * normalized(1, 0);
    let quality = (diagonal - cross).abs() / diagonal.abs().max(cross.abs());
    assert!((quality - 0.8).abs() < 2.0e-6, "quality={quality:e}");
}

#[test]
fn generated_guard_tracks_volume_division_precision_for_coupled_and_scalar_blocks() {
    // With these normal f32 inputs, rhs/vol rounds to zero even though the
    // subsequent mass inverse would recover a normal physical rate:
    //
    //   M = 1e-30 [[1,.25],[.5,1]],
    //   rhs = [MIN_POSITIVE,2 MIN_POSITIVE], vol = 1e8.
    //
    // The exact rates are [6.7171e-17,2.0151e-16], while forgetting the raw
    // RHS makes the emitted solve silently return zero.  The generated kernel
    // must therefore retain the pre-division nonzero predicate and combine it
    // with the row/column/conditioning scale before deciding materiality.
    let coupled = constant_mass_model([
        [1.0e-30_f32, 2.5e-31],
        [5.0e-31, 1.0e-30],
    ]);
    let coupled_wgsl = generated_scalar_stage_wgsl(&coupled, "volume_precision_coupled");
    assert!(coupled_wgsl.contains("let raw_rhs_0 = rhs["), "{coupled_wgsl}");
    assert!(
        coupled_wgsl.contains("raw_rhs_0 != 0.0")
            && coupled_wgsl.contains("abs(initial_rate_0) < bitcast<f32>(8388608u)"),
        "missing pre-volume-division precision predicate:\n{coupled_wgsl}"
    );
    assert!(
        coupled_wgsl.contains("let mass_volume_conditioning_scale_0")
            && coupled_wgsl.contains("volume_precision_risk_0 || volume_precision_risk_1")
            && coupled_wgsl.contains("mass_volume_conditioning_scale_0 <"),
        "missing coupled physical-conditioning gate:\n{coupled_wgsl}"
    );

    let q = vol_scalar_dim::<Dimensionless>("q");
    let mut equation = Equation::new(q);
    equation.add_term(fvm::ddt_coeff(Coefficient::constant(1.0e-30), q));
    let mut system = EquationSystem::new();
    system.add_equation(equation);
    let mut scalar = generic_diffusion_demo_model().expect("base model");
    scalar.system = system;
    scalar.state_layout = StateLayout::new(vec![q]);
    scalar.primitives = PrimitiveDerivations::identity();
    scalar.explicit_primitives = None;
    scalar.explicit_mass_closure_proof = ExplicitMassClosureProof::ExactSymbolic;
    let scalar_wgsl = generated_scalar_stage_wgsl(&scalar, "volume_precision_scalar");
    assert!(
        scalar_wgsl.contains("raw_rhs_0 != 0.0")
            && scalar_wgsl.contains("volume_precision_risk_0 && abs(mass_0_0) <"),
        "scalar fast path lost its material volume-underflow gate:\n{scalar_wgsl}"
    );
}

#[test]
fn row_normalization_prevents_finite_factor_underflow_wrong_solve() {
    // The row-equilibrated matrix is [[1, 1], [1, 0.05]], whose matching
    // quality is 0.95.  Raw elimination nevertheless forms 1e-17/1e30,
    // which underflows to zero; the lost multiplier has an O(1) effect after
    // it is multiplied by the first row.  This is a silent finite failure, not
    // an overflow that a later non-finite state check can catch.
    let a = [[1.0e30_f32, 1.0e30], [1.0e-17, 5.0e-19]];
    let exact_solution = [1.0_f32, 1.0];
    let rhs = [
        a[0][0] * exact_solution[0] + a[0][1] * exact_solution[1],
        a[1][0] * exact_solution[0] + a[1][1] * exact_solution[1],
    ];
    let (raw, _) = generated_solve_2x2(a, rhs);
    let raw_error =
        ((raw[0] as f64 - 1.0).powi(2) + (raw[1] as f64 - 1.0).powi(2)).sqrt() / 2.0_f64.sqrt();
    assert_eq!((a[1][0] / a[0][0]).to_bits(), 0.0_f32.to_bits());
    assert!(raw.iter().all(|value| value.is_finite()));
    assert!(
        raw_error > 19.0,
        "raw={raw:?}, relative error={raw_error:e}"
    );

    let row_scales = [1.0e30_f32, 1.0e-17];
    let normalized_a = [
        [a[0][0] / row_scales[0], a[0][1] / row_scales[0]],
        [a[1][0] / row_scales[1], a[1][1] / row_scales[1]],
    ];
    let normalized_rhs = [rhs[0] / row_scales[0], rhs[1] / row_scales[1]];
    let (normalized, _) = generated_solve_2x2(normalized_a, normalized_rhs);
    let normalized_error =
        ((normalized[0] as f64 - 1.0).powi(2) + (normalized[1] as f64 - 1.0).powi(2)).sqrt()
            / 2.0_f64.sqrt();
    assert!(
        normalized_error < 2.0e-6,
        "normalized={normalized:?}, relative error={normalized_error:e}"
    );

    constant_mass_model(a)
        .validate_explicit_rk4()
        .expect("the generated row-normalized solve must accept this regular block");
}

#[test]
fn lost_nonzero_row_normalized_rhs_is_routed_to_nonfinite_gate() {
    // The fully equilibrated mass is [[1, 0], [1, 1]] with matching quality
    // one. Row-normalizing the second RHS produces one subnormal quantum, while
    // the 1e-8 column scale would amplify its quantization into a normal
    // physical rate (~1e-37). Generated code must therefore route this
    // materially unrepresentable scaled RHS to the NaN/state gate.
    let a = [[1.0e35_f32, 0.0], [1.0e25, 1.0e17]];
    let rhs = [0.0_f32, 1.5e-20];
    let row_scale = a[1][0].abs().max(a[1][1].abs());
    assert_ne!(rhs[1].to_bits(), 0.0_f32.to_bits());
    assert_ne!((rhs[1] / row_scale).to_bits(), 0.0_f32.to_bits());
    assert!((rhs[1] / row_scale).abs() < f32::MIN_POSITIVE);

    let model = constant_mass_model(a);
    let validation_error = model.validate_explicit_rk4().unwrap_err();
    assert!(validation_error.contains("2^-19 safety floor"), "{validation_error}");
    let discrete = lower_system(&model.system, &SchemeRegistry::new(Scheme::Upwind)).unwrap();
    let slots = ResolvedStateSlotsSpec {
        stride: 2,
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
    let stage = generate_rk4_stage_kernel_program(
        "lost_nonzero_scaled_rhs",
        &discrete,
        &slots,
        &[],
        Rk4Stage::First,
        &[],
    )
    .unwrap();
    let wgsl = lower_kernel_program_to_wgsl(&stage).unwrap().to_wgsl();
    assert!(wgsl.contains("mass_conditioning_scale_0 <"), "{wgsl}");
    assert!(
        wgsl.contains("bitcast<f32>(8388608u)"),
        "missing minimum-normal rejection guard:\n{wgsl}"
    );
}

#[test]
fn constant_rcond_is_audited_per_disconnected_mass_component() {
    // H is well conditioned (rcond_inf=0.25, norm=4, inverse norm=1).
    // B is independently safe after equilibration (rcond_inf=1.499e-5,
    // norm=2, inverse norm=3.336e4).  A global audit spuriously combines H's
    // norm with B's inverse norm and reports 7.495e-6, even though no solve or
    // pivot can cross the structural zero block boundary.
    let h = [
        [1.0_f32, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ];
    let b = [[1.0_f32, 1.0], [1.0, 1.00006]];
    let mut a = [[0.0_f32; 6]; 6];
    for row in 0..4 {
        for column in 0..4 {
            a[row][column] = h[row][column];
        }
    }
    for row in 0..2 {
        for column in 0..2 {
            a[row + 4][column + 4] = b[row][column];
        }
    }

    let names = ["h0", "h1", "h2", "h3", "b0", "b1"];
    let fields = names.map(vol_scalar_dim::<Dimensionless>);
    let mut system = EquationSystem::new();
    for row in 0..6 {
        let mut equation = Equation::new(fields[row]);
        for column in 0..6 {
            if a[row][column] != 0.0 {
                equation.add_term(fvm::ddt_coeff(
                    Coefficient::constant(a[row][column] as f64),
                    fields[column],
                ));
            }
        }
        system.add_equation(equation);
    }
    let mut model = generic_diffusion_demo_model().expect("base model");
    model.system = system;
    model.state_layout = StateLayout::new(fields.to_vec());
    model.primitives = PrimitiveDerivations::identity();
    model.explicit_primitives = None;
    model.explicit_mass_closure_proof = ExplicitMassClosureProof::ExactSymbolic;
    model
        .validate_explicit_rk4()
        .expect("independent safe mass components must not cross-poison rcond");
}

fn scaled_generated_solve<const N: usize>(
    mut a: [[f32; N]; N],
    mut b: [f32; N],
) -> ([f32; N], f32) {
    let mut scales = [0.0_f32; N];
    for row in 0..N {
        for column in 0..N {
            scales[row] = scales[row].max(a[row][column].abs());
        }
    }
    let mut min_ratio = f32::INFINITY;
    for pivot in 0..N {
        for candidate in (pivot + 1)..N {
            let pivot_score = a[pivot][pivot].abs() / scales[pivot].max(1.0e-20);
            let candidate_score = a[candidate][pivot].abs() / scales[candidate].max(1.0e-20);
            if candidate_score > pivot_score {
                a.swap(pivot, candidate);
                b.swap(pivot, candidate);
                scales.swap(pivot, candidate);
            }
        }
        let ratio = a[pivot][pivot].abs() / scales[pivot].max(1.0e-20);
        min_ratio = min_ratio.min(ratio);
        assert!(a[pivot][pivot].abs() >= (1.0e-6 * scales[pivot]).max(1.0e-20));
        for row in (pivot + 1)..N {
            let factor = a[row][pivot] / a[pivot][pivot];
            for column in pivot..N {
                a[row][column] -= factor * a[pivot][column];
            }
            b[row] -= factor * b[pivot];
        }
    }
    for row in (0..N).rev() {
        let mut solved = b[row];
        for column in (row + 1)..N {
            solved -= a[row][column] * b[column];
        }
        b[row] = solved / a[row][row];
    }
    (b, min_ratio)
}

fn dense_reference_solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> [f64; N] {
    for pivot in 0..N {
        let mut best = pivot;
        for candidate in (pivot + 1)..N {
            if a[candidate][pivot].abs() > a[best][pivot].abs() {
                best = candidate;
            }
        }
        a.swap(pivot, best);
        b.swap(pivot, best);
        for row in (pivot + 1)..N {
            let factor = a[row][pivot] / a[pivot][pivot];
            for column in pivot..N {
                a[row][column] -= factor * a[pivot][column];
            }
            b[row] -= factor * b[pivot];
        }
    }
    for row in (0..N).rev() {
        let mut solved = b[row];
        for column in (row + 1)..N {
            solved -= a[row][column] * b[column];
        }
        b[row] = solved / a[row][row];
    }
    b
}

#[test]
fn reciprocal_condition_audit_rejects_a_four_hundred_percent_forward_error_block() {
    // Deterministic adversarial 3x3 block with condition number about 1.17e8.
    // Scaled partial pivoting's smallest original-row-relative pivot is
    // 1.39374e-6, above the generated 1e-6 cutoff. Every generated rate
    // remains finite, but the f32 solve has >400% normwise forward error.
    // This rank is below the shipped all-Mach differential rank, so the
    // survivor is not an artifact of the validator's maximum rank eight.
    let a = [
        [0.20359954, -0.21969615, -0.02607520],
        [0.63720125, -0.68757610, -0.08189549],
        [0.10513478, -0.11344578, -0.01358128],
    ];
    // Row/column-equilibrated representative of the same block. Its column
    // scales are all one, so the 2^-19 representability gate passes and the
    // independent rcond audit remains the decisive rejection.
    let audit_a = [
        [0.9999914, -1.0, -0.9914090],
        [0.9999949, -1.0, -0.9949170],
        [1.0, -1.0, -1.0],
    ];
    let rhs = [0.92463905, 0.48254293, 0.31806943];

    let names = ["x0", "x1", "x2"];
    let fields = names.map(vol_scalar_dim::<Dimensionless>);
    let mut system = EquationSystem::new();
    for row in 0..3 {
        let mut equation = Equation::new(fields[row]);
        for column in 0..3 {
            equation.add_term(fvm::ddt_coeff(
                Coefficient::constant(audit_a[row][column] as f64),
                fields[column],
            ));
        }
        system.add_equation(equation);
    }
    let mut model = generic_diffusion_demo_model().expect("base model");
    model.system = system;
    model.state_layout = StateLayout::new(fields.to_vec());
    model.primitives = PrimitiveDerivations::identity();
    model.explicit_primitives = None;
    model.explicit_mass_closure_proof = ExplicitMassClosureProof::ExactSymbolic;
    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(
        error.contains("reciprocal infinity-norm condition")
            && error.contains("below the f32 safety floor 1e-5"),
        "{error}"
    );

    let (got, min_ratio) = scaled_generated_solve(a, rhs);
    assert!(got.iter().all(|value| value.is_finite()));
    assert!(min_ratio > 1.3e-6 && min_ratio < 1.5e-6, "{min_ratio:e}");
    let reference = dense_reference_solve(
        a.map(|row| row.map(|value| value as f64)),
        rhs.map(|value| value as f64),
    );
    let error_norm = got
        .iter()
        .zip(reference)
        .map(|(&actual, expected)| (actual as f64 - expected).powi(2))
        .sum::<f64>()
        .sqrt();
    let reference_norm = reference
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(
        error_norm / reference_norm > 4.0,
        "expected a gross finite survivor: got={got:?}, reference={reference:?}, rel={}",
        error_norm / reference_norm,
    );
}

#[test]
fn unrelated_dynamic_scalar_cannot_disable_constant_component_rcond_audit() {
    // The isolated dynamic scalar is a separate structural component.  Its
    // RuntimePivoted contract must not suppress the exact build-time audit of
    // this disconnected constant 3x3 block.  The latter has equilibrated
    // rcond_inf = 9.935e-9, yet all of its emitted equilibrated pivots clear
    // 1e-6; without the per-component audit its finite f32 solve has a 1.57%
    // normwise forward error for the RHS in the test above.
    let a = [
        [0.9999914, -1.0, -0.9914090],
        [0.9999949, -1.0, -0.9949170],
        [1.0, -1.0, -1.0],
    ];
    let dynamic = vol_scalar_dim::<Dimensionless>("dynamic");
    let x0 = vol_scalar_dim::<Dimensionless>("x0");
    let x1 = vol_scalar_dim::<Dimensionless>("x1");
    let x2 = vol_scalar_dim::<Dimensionless>("x2");
    let fields = [dynamic, x0, x1, x2];

    let mut dynamic_equation = Equation::new(dynamic);
    dynamic_equation.add_term(fvm::ddt_coeff(
        Coefficient::field(dynamic).expect("scalar dynamic coefficient"),
        dynamic,
    ));
    let mut system = EquationSystem::new();
    system.add_equation(dynamic_equation);
    for row in 0..3 {
        let mut equation = Equation::new(fields[row + 1]);
        for column in 0..3 {
            equation.add_term(fvm::ddt_coeff(
                Coefficient::constant(a[row][column] as f64),
                fields[column + 1],
            ));
        }
        system.add_equation(equation);
    }

    let mut model = generic_diffusion_demo_model().expect("base model");
    model.system = system;
    model.state_layout = StateLayout::new(fields.to_vec());
    model.primitives = PrimitiveDerivations::identity();
    model.explicit_primitives = None;
    model.explicit_mass_closure_proof = ExplicitMassClosureProof::RuntimePivoted {
        justification: "the isolated scalar is runtime-pivot guarded",
    };

    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(
        error.contains("constant mass component")
            && error.contains("reciprocal infinity-norm condition")
            && error.contains("below the f32 safety floor 1e-5"),
        "{error}"
    );
}

#[test]
fn matching_quality_is_tracked_independently_for_two_safe_connected_blocks() {
    // Each B block has normalized pivot product about 5e-3 and rcond_inf about
    // 1.25e-3. Both clear their respective runtime/build-time safety floors.
    // Multiplying qualities across independent connected components, however,
    // yields about 2.5e-5 and poisons every backsolve rate.
    let b = [[1.0_f32, 1.0], [1.0, 1.005025]];
    let a = [
        [b[0][0], b[0][1], 0.0, 0.0],
        [b[1][0], b[1][1], 0.0, 0.0],
        [0.0, 0.0, b[0][0], b[0][1]],
        [0.0, 0.0, b[1][0], b[1][1]],
    ];
    let names = ["x0", "x1", "x2", "x3"];
    let fields = names.map(vol_scalar_dim::<Dimensionless>);
    let mut system = EquationSystem::new();
    for row in 0..4 {
        let mut equation = Equation::new(fields[row]);
        for column in 0..4 {
            if a[row][column] != 0.0 {
                equation.add_term(fvm::ddt_coeff(
                    Coefficient::constant(a[row][column] as f64),
                    fields[column],
                ));
            }
        }
        system.add_equation(equation);
    }
    let mut model = generic_diffusion_demo_model().expect("base model");
    model.system = system;
    model.state_layout = StateLayout::new(fields.to_vec());
    model.primitives = PrimitiveDerivations::identity();
    model.explicit_primitives = None;
    model.explicit_mass_closure_proof = ExplicitMassClosureProof::ExactSymbolic;
    model
        .validate_explicit_rk4()
        .expect("two independent condition-800 blocks pass the exact constant rcond gate");

    let pivot_1 = b[1][1] - b[1][0] * b[0][1];
    let one_block_quality = pivot_1.abs() / b[1][1].abs();
    assert!(one_block_quality > 1.0e-4);
    assert!(one_block_quality * one_block_quality < 1.0e-4);

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
    let stage = generate_rk4_stage_kernel_program(
        "independent_safe_mass_blocks",
        &discrete,
        &slots,
        &[],
        Rk4Stage::First,
        &[],
    )
    .unwrap();
    let wgsl = lower_kernel_program_to_wgsl(&stage).unwrap().to_wgsl();
    assert_eq!(
        wgsl.matches("let mass_match_quality_0").count(),
        1,
        "{wgsl}"
    );
    assert_eq!(
        wgsl.matches("let mass_match_quality_1").count(),
        1,
        "{wgsl}"
    );
    assert!(!wgsl.contains("mass_quality_0 *="), "{wgsl}");
    assert!(!wgsl.contains("mass_quality_1 *="), "{wgsl}");
    assert!(wgsl.contains("mass_quality_bad_0"), "{wgsl}");
    assert!(wgsl.contains("mass_quality_bad_1"), "{wgsl}");
}

#[test]
fn opposite_sign_matching_products_cannot_overstate_conditioning_scale() {
    // For E=[[1,1],[-1,1]], |ad-bc|/max(|ad|,|bc|)=2.  That is a useful
    // nonsingularity score, but inverse-error amplification cannot improve by
    // a factor greater than one.  Without the cap, a column at 2^-19 would
    // appear to clear the 2^-19 equilibration floor even though ordinary RHS
    // roundoff can still be amplified by 2^20.
    let column_scale = 2.0_f32.powi(-20);
    let model = constant_mass_model([
        [column_scale, 1.0],
        [-column_scale, 1.0],
    ]);
    let error = model.validate_explicit_rk4().unwrap_err();
    assert!(error.contains("2^-19 safety floor"), "{error}");

    let wgsl = generated_scalar_stage_wgsl(&model, "opposite_sign_matching_quality");
    assert!(
        wgsl.contains(
            "let mass_match_quality_0 = min(mass_match_raw_quality_0, 1.0)"
        ),
        "matching quality was not capped before conditioning:\n{wgsl}"
    );
}
