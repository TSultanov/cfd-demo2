//! Adversarial audit harness for compact explicit RHS and sparse local-mass solve.
//!
//! This is intentionally isolated from production solver files.

use cfd2::solver::model::generic_diffusion_demo_model;
use cfd2::solver::model::primitives::PrimitiveDerivations;
use cfd2_codegen::solver::codegen::explicit_rk::{generate_rk4_stage_kernel_program, Rk4Stage};
use cfd2_codegen::solver::codegen::fusion::lower_kernel_program_to_wgsl;
use cfd2_codegen::solver::codegen::ir::lower_system_unchecked;
use cfd2_codegen::solver::codegen::unified_assembly::generate_matrix_free_residual_kernel_program;
use cfd2_ir::dimensions::Dimensionless;
use cfd2_ir::kernel::{
    fvm, surface_scalar_dim, vol_scalar_dim, Coefficient, Equation, EquationSystem, SchemeRegistry,
    StateLayout,
};
use cfd2_ir::ports::{PortFieldKind, ResolvedStateSlotSpec, ResolvedStateSlotsSpec};
use cfd2_ir::scheme::Scheme;

fn slots(fields: &[cfd2_ir::kernel::FieldRef]) -> ResolvedStateSlotsSpec {
    let mut base_offset = 0u32;
    let slots = fields
        .iter()
        .map(|field| {
            let slot = ResolvedStateSlotSpec {
                name: field.name().to_string(),
                kind: PortFieldKind::Scalar,
                unit: field.unit(),
                base_offset,
            };
            base_offset += 1;
            slot
        })
        .collect();
    ResolvedStateSlotsSpec {
        stride: base_offset,
        slots,
    }
}

fn generated_interleaved_index_audit() {
    let q = vol_scalar_dim::<Dimensionless>("q");
    let algebraic = vol_scalar_dim::<Dimensionless>("a");
    let r = vol_scalar_dim::<Dimensionless>("r");
    let phi = surface_scalar_dim::<Dimensionless>("phi");

    let mut q_eq = Equation::new(q);
    q_eq.add_term(fvm::ddt(q));
    let algebraic_eq = Equation::new(algebraic);
    let mut r_eq = Equation::new(r);
    r_eq.add_term(fvm::ddt(r));
    r_eq.add_term(fvm::div_flux(phi, r));
    r_eq.add_term(fvm::laplacian(Coefficient::constant(1.0), r));

    let mut system = EquationSystem::new();
    system.add_equation(q_eq);
    system.add_equation(algebraic_eq);
    system.add_equation(r_eq);
    let discrete = lower_system_unchecked(&system, &SchemeRegistry::new(Scheme::Upwind));
    let slots = slots(&[q, algebraic, r]);

    let residual = generate_matrix_free_residual_kernel_program(
        "audit_interleaved_residual",
        &discrete,
        &slots,
        3,
        false,
        &[],
    )
    .expect("interleaved residual");
    let residual = lower_kernel_program_to_wgsl(&residual)
        .expect("lower interleaved residual")
        .to_wgsl();
    assert!(residual.contains("rhs[idx * 2u + 0u] = rhs_0"));
    assert!(residual.contains("rhs[idx * 2u + 1u] = rhs_2"));
    assert!(!residual.contains("rhs[idx * 3u"));
    assert!(
        residual.contains("fluxes[face_idx * 3u + 2u]"),
        "face flux lost full coupled rank:\n{residual}"
    );
    assert!(
        residual.contains("bc_kind[face_idx * 3u + 2u]"),
        "BC lookup lost full coupled rank:\n{residual}"
    );

    let stage = generate_rk4_stage_kernel_program(
        "audit_interleaved_stage",
        &discrete,
        &slots,
        &[],
        Rk4Stage::First,
        &[],
    )
    .expect("interleaved stage");
    let stage = lower_kernel_program_to_wgsl(&stage)
        .expect("lower interleaved stage")
        .to_wgsl();
    assert!(stage.contains("rhs[idx * 2u + 0u]"));
    assert!(stage.contains("rhs[idx * 2u + 1u]"));
    assert!(stage.contains("state[idx * 3u + 2u]"));

    // The codegen seam now rejects a nonzero compact stride before it can emit
    // a full-rank offset into an undersized face tuple.
    let undersized = std::panic::catch_unwind(|| {
        generate_matrix_free_residual_kernel_program(
            "audit_undersized_flux_stride",
            &discrete,
            &slots,
            2,
            false,
            &[],
        )
    });
    assert!(undersized.is_err());

    println!("interleaved-index audit: compact RHS and full-rank BC/flux indices passed");
    println!("codegen seam: undersized flux_stride=2 was rejected for full rank 3");
}

fn generated_partial_pivot_solve(mut a: [[f32; 2]; 2], mut b: [f32; 2]) -> [f32; 2] {
    const EPS: f32 = 1.0e-20;
    for pivot in 0..2 {
        for candidate in (pivot + 1)..2 {
            if a[candidate][pivot].abs() > a[pivot][pivot].abs() {
                a.swap(pivot, candidate);
                b.swap(pivot, candidate);
            }
        }
        let raw = a[pivot][pivot];
        let safe = if raw.abs() < EPS {
            if raw >= 0.0 {
                EPS
            } else {
                -EPS
            }
        } else {
            raw
        };
        for row in (pivot + 1)..2 {
            let factor = a[row][pivot] / safe;
            for column in pivot..2 {
                a[row][column] -= factor * a[pivot][column];
            }
            b[row] -= factor * b[pivot];
        }
    }
    for row in (0..2).rev() {
        for column in (row + 1)..2 {
            b[row] -= a[row][column] * b[column];
        }
        let raw = a[row][row];
        let safe = if raw.abs() < 1.0e-20 {
            if raw >= 0.0 {
                1.0e-20
            } else {
                -1.0e-20
            }
        } else {
            raw
        };
        b[row] /= safe;
    }
    b
}

fn full_partial_pivot_solve(mut a: [[f32; 2]; 2], mut b: [f32; 2]) -> [f32; 2] {
    for pivot in 0..2 {
        let mut best = pivot;
        for candidate in (pivot + 1)..2 {
            if a[candidate][pivot].abs() > a[best][pivot].abs() {
                best = candidate;
            }
        }
        if best != pivot {
            a.swap(pivot, best);
            b.swap(pivot, best);
        }
        for row in (pivot + 1)..2 {
            let factor = a[row][pivot] / a[pivot][pivot];
            for column in pivot..2 {
                a[row][column] -= factor * a[pivot][column];
            }
            b[row] -= factor * b[pivot];
        }
    }
    for row in (0..2).rev() {
        for column in (row + 1)..2 {
            b[row] -= a[row][column] * b[column];
        }
        b[row] /= a[row][row];
    }
    b
}

fn stable_constant_model_uses_full_partial_pivoting() {
    let q = vol_scalar_dim::<Dimensionless>("q");
    let r = vol_scalar_dim::<Dimensionless>("r");
    let mut q_eq = Equation::new(q);
    q_eq.add_term(fvm::ddt_coeff(Coefficient::constant(1.0e-19), q));
    q_eq.add_term(fvm::ddt(r));
    let mut r_eq = Equation::new(r);
    r_eq.add_term(fvm::ddt(q));
    r_eq.add_term(fvm::ddt(r));
    let mut system = EquationSystem::new();
    system.add_equation(q_eq);
    system.add_equation(r_eq);

    let mut model = generic_diffusion_demo_model().expect("base model");
    model.system = system;
    model.state_layout = StateLayout::new(vec![q, r]);
    model.primitives = PrimitiveDerivations::identity();
    model.explicit_primitives = None;
    model
        .validate_explicit_rk4()
        .expect("validator currently accepts the well-conditioned mass block");

    let matrix = [[1.0e-19_f32, 1.0], [1.0, 1.0]];
    let rhs = [1.0_f32, 2.0];
    let generated = generated_partial_pivot_solve(matrix, rhs);
    let pivoted = full_partial_pivot_solve(matrix, rhs);
    assert_eq!(generated, [1.0, 1.0]);
    assert_eq!(pivoted, [1.0, 1.0]);
    println!(
        "corrected pivot: M=[[1e-19,1],[1,1]], rhs=[1,2], generated={generated:?}, full-partial-pivot={pivoted:?}"
    );
}

fn main() {
    generated_interleaved_index_audit();
    stable_constant_model_uses_full_partial_pivoting();
}
