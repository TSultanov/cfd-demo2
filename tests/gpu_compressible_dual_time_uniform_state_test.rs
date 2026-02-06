use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};
use cfd2::solver::model::compressible_model;
use cfd2::solver::model::helpers::{
    SolverCompressibleIdealGasExt, SolverFieldAliasesExt, SolverRuntimeParamsExt,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};

#[test]
fn compressible_dual_time_preserves_uniform_state() {
    std::env::set_var("CFD2_QUIET", "1");

    let mesh = generate_structured_rect_mesh(
        4,
        2,
        1.0,
        0.5,
        BoundarySides {
            left: BoundaryType::SlipWall,
            right: BoundaryType::SlipWall,
            bottom: BoundaryType::SlipWall,
            top: BoundaryType::SlipWall,
        },
    );

    let mut solver = pollster::block_on(UnifiedSolver::new(
        &mesh,
        compressible_model(),
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

    solver.set_dt(0.01);
    solver.set_dtau(1e-3).expect("dtau");
    solver.set_viscosity(0.0).expect("viscosity");
    solver.set_uniform_state(1.0, [0.0, 0.0], 1.0);
    solver.initialize_history();

    let rho0 = pollster::block_on(solver.get_rho());
    let u0 = pollster::block_on(solver.get_u());
    let p0 = pollster::block_on(solver.get_p());

    for _ in 0..3 {
        solver.step();
    }

    let rho = pollster::block_on(solver.get_rho());
    let u = pollster::block_on(solver.get_u());
    let p = pollster::block_on(solver.get_p());

    let tol_rho = 1e-5;
    let tol_u = 1e-5;
    let tol_p = 1e-5;

    for (a, b) in rho.iter().zip(rho0.iter()) {
        assert!(
            (a - b).abs() < tol_rho,
            "rho drifted (a={a:.6e}, b={b:.6e})"
        );
    }
    for ((ax, ay), (bx, by)) in u.iter().zip(u0.iter()) {
        assert!(
            (ax - bx).abs() < tol_u && (ay - by).abs() < tol_u,
            "u drifted (a=({ax:.6e},{ay:.6e}), b=({bx:.6e},{by:.6e}))"
        );
    }
    for (a, b) in p.iter().zip(p0.iter()) {
        assert!((a - b).abs() < tol_p, "p drifted (a={a:.6e}, b={b:.6e})");
    }
}
