use cfd2::solver::mesh::{generate_cut_cell_mesh, BackwardsStep};
use cfd2::solver::model::eos::EosSpec;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverCompressibleInletExt, SolverFieldAliasesExt,
    SolverRuntimeParamsExt,
};
use cfd2::solver::model::compressible_model_with_eos;
use cfd2::solver::options::{PreconditionerType, SteppingMode, TimeScheme};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{SolverConfig, UnifiedSolver};
use nalgebra::Vector2;

#[test]
fn ui_compressible_air_backstep_smoke() {
    // Roughly match the UI defaults (BackwardsStep, CutCell, Air, inlet u=1).
    let length = 3.5;
    let domain_size = Vector2::new(length, 1.0);
    let geo = BackwardsStep {
        length,
        height_inlet: 0.5,
        height_outlet: 1.0,
        step_x: 0.5,
    };

    let min_cell_size = 0.025;
    let max_cell_size = 0.025;
    let mut mesh = generate_cut_cell_mesh(&geo, min_cell_size, max_cell_size, 1.2, domain_size);
    mesh.smooth(&geo, 0.3, 50);

    let density = 1.225f32;
    let viscosity = 1.81e-5f32;
    let inlet_u = 1.0f32;
    let eos = EosSpec::IdealGas {
        gamma: 1.4,
        gas_constant: 287.0,
        temperature: 300.0,
    };

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model_with_eos(eos),
        SolverConfig {
            advection_scheme: Scheme::Upwind,
            time_scheme: TimeScheme::Euler,
            preconditioner: PreconditionerType::Jacobi,
            stepping: SteppingMode::Implicit { outer_iters: 1 },
        },
        None,
        None,
    ))
    .expect("solver init");

    // Clear full state to avoid uninitialized auxiliary fields.
    let stride = solver.model().state_layout.stride() as usize;
    solver
        .write_state_f32(&vec![0.0f32; mesh.num_cells() * stride])
        .expect("clear state");

    // Initial dt matches the UI slider default, but we apply a CFL-based update
    // before stepping (mirrors the UI adaptive-dt behavior).
    solver.set_dt(0.001);

    solver.set_density(density).unwrap();
    solver.set_viscosity(viscosity).unwrap();
    solver.set_eos(&eos).unwrap();
    solver
        .set_compressible_inlet_isothermal_x(density, inlet_u, &eos)
        .unwrap();

    let p_ref = eos.pressure_for_density(density as f64) as f32;
    solver.set_uniform_state(density, [0.0, 0.0], p_ref);
    solver.initialize_history();

    let actual_min_cell_size = mesh
        .cell_vol
        .iter()
        .map(|&v| v.sqrt())
        .fold(f64::INFINITY, f64::min);
    let sound_speed = eos.sound_speed(density as f64);
    let target_cfl = 0.95f64;

    let mut prev_max_vel = 0.0f64;
    for step in 0..10 {
        let wave_speed = prev_max_vel + sound_speed;
        if wave_speed.is_finite() && wave_speed > 1e-12 {
            let current_dt = solver.dt() as f64;
            let mut next_dt = target_cfl * actual_min_cell_size / wave_speed;
            if next_dt > current_dt * 1.2 {
                next_dt = current_dt * 1.2;
            }
            next_dt = next_dt.clamp(1e-9, 100.0);
            solver.set_dt(next_dt as f32);
        }

        let stats = solver.step_with_stats().expect("step with stats");
        if let Some(last) = stats.last() {
            assert!(
                last.residual.is_finite() && last.residual < 1e12,
                "step {step}: linear residual blew up: {:?}",
                last
            );
            assert!(!last.diverged, "step {step}: linear solver diverged: {last:?}");
        }

        let u = pollster::block_on(solver.get_u());
        let p = pollster::block_on(solver.get_p());

        let mut max_vel = 0.0f64;
        for (vx, vy) in &u {
            assert!(vx.is_finite() && vy.is_finite(), "step {step}: non-finite u");
            let v = (vx * vx + vy * vy).sqrt();
            max_vel = max_vel.max(v);
        }
        for pv in &p {
            assert!(pv.is_finite(), "step {step}: non-finite p");
        }
        prev_max_vel = max_vel;
    }
}
