//! Compare compressible and incompressible lid-driven cavity solutions
//!
//! At low Mach numbers with isothermal conditions, compressible and incompressible
//! solutions should be very close. This test helps identify if the compressible solver
//! has excess viscous dissipation.

use cfd2::solver::gpu::enums::GpuBoundaryType;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverFieldAliasesExt, SolverRuntimeParamsExt,
};
use cfd2::solver::model::incompressible_momentum_model;
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

#[test]
fn lid_driven_cavity_compressible_vs_incompressible() {
    std::env::set_var("CFD2_QUIET", "1");

    let nx = 20usize;
    let ny = 20usize;
    let length = 1.0;
    let height = 1.0;
    let u_lid = 1.0f32;
    let viscosity = 1.0f32;
    let dt = 1e-5f32;
    let n_steps = 300;

    let mesh_inc = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundarySides {
            left: BoundaryType::Wall,
            right: BoundaryType::Wall,
            bottom: BoundaryType::Wall,
            top: BoundaryType::MovingWall,
        },
    );

    let mut solver_inc = pollster::block_on(UnifiedSolver::new(
        &mesh_inc,
        incompressible_momentum_model().expect("model"),
        SolverConfig {
            advection_scheme: Scheme::SecondOrderUpwind,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("incompressible solver init");

    solver_inc.set_dt(dt);
    solver_inc.set_dtau(0.0).unwrap();
    solver_inc.set_viscosity(viscosity).unwrap();
    solver_inc.set_uniform_state(1.0, [0.0, 0.0], 0.0);
    solver_inc
        .set_boundary_vec2(GpuBoundaryType::MovingWall, "U", [u_lid, 0.0])
        .unwrap();
    solver_inc.initialize_history();

    for _ in 0..n_steps {
        solver_inc.step();
    }

    let u_inc = pollster::block_on(solver_inc.get_u());

    let mesh_comp = generate_structured_rect_mesh(
        nx,
        ny,
        length,
        height,
        BoundarySides {
            left: BoundaryType::Wall,
            right: BoundaryType::Wall,
            bottom: BoundaryType::Wall,
            top: BoundaryType::MovingWall,
        },
    );

    let eos = EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature: 300.0,
    };

    let mut solver_comp = pollster::block_on(UnifiedSolver::new(
        &mesh_comp,
        compressible_model_with_eos(eos).expect("model"),
        SolverConfig {
            advection_scheme: Scheme::SecondOrderUpwindVanLeer,
            time_scheme: TimeScheme::BDF2,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("compressible solver init");

    let p0 = 101325.0f32;
    let rho0 = (p0 as f64 / (287.0 * 300.0)) as f32;

    solver_comp.set_eos(&eos).unwrap();
    solver_comp.set_dt(dt);
    solver_comp.set_dtau(0.0).unwrap();
    solver_comp.set_viscosity(viscosity).unwrap();
    solver_comp.set_density(rho0).unwrap();
    solver_comp
        .set_precond_model(cfd2::solver::GpuLowMachPrecondModel::Off)
        .unwrap();
    solver_comp.set_uniform_state(rho0, [0.0, 0.0], p0);
    solver_comp
        .set_boundary_vec2(GpuBoundaryType::MovingWall, "u", [u_lid, 0.0])
        .unwrap();
    solver_comp
        .set_boundary_vec2(GpuBoundaryType::MovingWall, "rho_u", [rho0 * u_lid, 0.0])
        .unwrap();
    solver_comp.initialize_history();

    for _ in 0..n_steps {
        solver_comp.step();
    }

    let u_comp = pollster::block_on(solver_comp.get_u());

    let u_inc_mag: Vec<f64> = u_inc
        .iter()
        .map(|(ux, uy)| (ux * ux + uy * uy).sqrt())
        .collect();
    let u_comp_mag: Vec<f64> = u_comp
        .iter()
        .map(|(ux, uy)| (ux * ux + uy * uy).sqrt())
        .collect();

    let max_inc = u_inc_mag.iter().cloned().fold(0.0, f64::max);
    let max_comp = u_comp_mag.iter().cloned().fold(0.0, f64::max);

    println!("Incompressible max velocity: {:.6}", max_inc);
    println!("Compressible max velocity: {:.6}", max_comp);
    println!("Ratio (comp/inc): {:.3}", max_comp / max_inc);

    let mut sum_sq_diff = 0.0;
    let mut sum_sq_inc = 0.0;
    for (i, ((ux_inc, uy_inc), (ux_comp, uy_comp))) in u_inc.iter().zip(u_comp.iter()).enumerate() {
        let dx = ux_comp - ux_inc;
        let dy = uy_comp - uy_inc;
        sum_sq_diff += dx * dx + dy * dy;
        sum_sq_inc += ux_inc * ux_inc + uy_inc * uy_inc;

        if i < 5 {
            let mag_inc = (ux_inc * ux_inc + uy_inc * uy_inc).sqrt();
            let mag_comp = (ux_comp * ux_comp + uy_comp * uy_comp).sqrt();
            println!(
                "Cell {}: inc=({:.4},{:.4}) mag={:.4}, comp=({:.4},{:.4}) mag={:.4}",
                i, ux_inc, uy_inc, mag_inc, ux_comp, uy_comp, mag_comp
            );
        }
    }

    let rms_diff = (sum_sq_diff / u_inc.len() as f64).sqrt();
    let rms_inc = (sum_sq_inc / u_inc.len() as f64).sqrt();
    let rel_error = rms_diff / rms_inc;

    println!("\nRMS incompressible velocity: {:.6}", rms_inc);
    println!("RMS difference: {:.6}", rms_diff);
    println!("Relative error (RMS): {:.2}%", rel_error * 100.0);

    // At low Mach these solutions should be close; a large gap indicates the
    // compressible path is under-dissipating.
    assert!(rel_error < 0.50,
        "Compressible and incompressible solutions differ by {:.1}%. This suggests viscous flux issues.",
        rel_error * 100.0);
}
