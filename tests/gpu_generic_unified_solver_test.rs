#![cfg(all(feature = "meshgen", feature = "dev-tests"))]

use cfd2::solver::mesh::{generate_cut_cell_mesh, Geometry, Mesh};
use cfd2::solver::model::{generic_diffusion_demo_model, generic_diffusion_demo_neumann_model};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::{PreconditionerType, SolverConfig, SteppingMode, TimeScheme, UnifiedSolver};
use nalgebra::{Point2, Vector2};

struct RectGeometry {
    length: f64,
    height: f64,
}

impl Geometry for RectGeometry {
    fn is_inside(&self, p: &Point2<f64>) -> bool {
        self.sdf(p) < 0.0
    }

    fn sdf(&self, p: &Point2<f64>) -> f64 {
        let dx = (p.x - self.length / 2.0).abs() - self.length / 2.0;
        let dy = (p.y - self.height / 2.0).abs() - self.height / 2.0;
        dx.max(dy).min(0.0) + Vector2::new(dx.max(0.0), dy.max(0.0)).norm()
    }

    fn get_boundary_points(&self, spacing: f64) -> Vec<Point2<f64>> {
        let mut points = Vec::new();
        let nx = (self.length / spacing).ceil() as usize;
        let ny = (self.height / spacing).ceil() as usize;

        for i in 0..=nx {
            let x = (i as f64 * spacing).min(self.length);
            points.push(Point2::new(x, 0.0));
            points.push(Point2::new(x, self.height));
        }
        for i in 0..=ny {
            let y = (i as f64 * spacing).min(self.height);
            points.push(Point2::new(0.0, y));
            points.push(Point2::new(self.length, y));
        }

        points
    }
}

fn build_rect_mesh(cell_size: f64) -> Mesh {
    let geo = RectGeometry {
        length: 1.0,
        height: 0.2,
    };
    generate_cut_cell_mesh(
        &geo,
        cell_size,
        cell_size,
        1.0,
        Vector2::new(geo.length, geo.height),
    )
}

#[test]
fn gpu_unified_solver_runs_generic_heat_step() {
    let mesh = build_rect_mesh(0.05);
    let dt = 1e-2;
    let model = generic_diffusion_demo_model();
    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Coupled,
    };

    let mut solver = pollster::block_on(UnifiedSolver::new(&mesh, model, config, None, None))
        .expect("solver init");
    solver.set_dt(dt as f32);

    let phi_old: Vec<f64> = mesh
        .cell_cx
        .iter()
        .map(|&x| (std::f64::consts::PI * x).sin())
        .collect();
    solver.set_field_scalar("phi", &phi_old).expect("set field");

    solver.step();

    let phi_new = pollster::block_on(solver.get_field_scalar("phi")).expect("get field");

    // demo model uses kappa=1.0
    let factor = 1.0 / (1.0 + std::f64::consts::PI * std::f64::consts::PI * dt);
    let expected: Vec<f64> = phi_old.iter().map(|v| v * factor).collect();

    let mut sum = 0.0;
    let mut dot = 0.0;
    let mut denom = 0.0;
    for (value, target) in phi_new.iter().zip(expected.iter()) {
        let diff = value - target;
        sum += diff * diff;
    }
    for (n, o) in phi_new.iter().zip(phi_old.iter()) {
        dot += n * o;
        denom += o * o;
    }
    let best_fit = if denom > 0.0 { dot / denom } else { 0.0 };
    let rms = (sum / phi_new.len() as f64).sqrt();
    assert!(
        rms < 3e-2,
        "rms error {:.3e} (expected factor {:.6}, best-fit {:.6})",
        rms,
        factor,
        best_fit
    );
}

#[test]
fn gpu_unified_solver_runs_generic_heat_step_neumann() {
    let mesh = build_rect_mesh(0.05);
    let dt = 1e-2;
    let model = generic_diffusion_demo_neumann_model();
    let config = SolverConfig {
        advection_scheme: Scheme::Upwind,
        time_scheme: TimeScheme::BDF2,
        preconditioner: PreconditionerType::Jacobi,
        stepping: SteppingMode::Coupled,
    };

    let mut solver = pollster::block_on(UnifiedSolver::new(&mesh, model, config, None, None))
        .expect("solver init");
    solver.set_dt(dt as f32);

    let phi_old: Vec<f64> = mesh
        .cell_cx
        .iter()
        .map(|&x| (std::f64::consts::PI * x).cos())
        .collect();
    solver.set_field_scalar("phi", &phi_old).expect("set field");

    solver.step();

    let phi_new = pollster::block_on(solver.get_field_scalar("phi")).expect("get field");

    // demo model uses kappa=1.0, and has homogeneous Neumann at inlet/outlet.
    let factor = 1.0 / (1.0 + std::f64::consts::PI * std::f64::consts::PI * dt);
    let expected: Vec<f64> = phi_old.iter().map(|v| v * factor).collect();

    let mut sum = 0.0;
    for (value, target) in phi_new.iter().zip(expected.iter()) {
        let diff = value - target;
        sum += diff * diff;
    }
    let rms = (sum / phi_new.len() as f64).sqrt();
    assert!(
        rms < 3e-2,
        "rms error {:.3e} (expected factor {:.6})",
        rms,
        factor
    );
}
