//! Method-of-Manufactured-Solutions convergence-order test for the COMPRESSIBLE
//! thermal all-Mach model (`allmach_thermal_compressible_mms`): the coupled
//! momentum + pressure + temperature system with the FULL production compressible
//! physics kept ON (unlike the barotropic `allmach_thermal_mms`, which strips it):
//!   * variable density recovered on-device from the REAL ideal-gas EOS
//!       rho = rho_t_ref/T + gamma*psi_ref*t_ref*p/T   (= p/(R*T))
//!   * viscous dissipation Phi = tau:grad(U) in the energy equation
//!   * the U.grad(p) compression-heating half (T2)
//! all driven at PSI > 0 so the compressible spatial operator is genuinely
//! exercised. The transient terms the model also carries (dp/dt heating, thermal
//! expansion, acoustic ddt) are identically zero at the steady manufactured
//! solution, so they leave the observed order untouched.
//!
//! Manufactured solution (steady, unit square, exact Dirichlet outlet anchor):
//!   Psi(x,y) = (A/pi) sin(pi x) sin(pi y)       (stream fn; div-free velocity)
//!   U*       = (dPsi/dy, -dPsi/dx)              => div U* = 0 (dev2/grad(divU)=0)
//!   T*       = T_ref (1.5 + 0.5 cos(pi x) cos(pi y))     in [T_ref, 2 T_ref]
//!   p*       = P_AMP (cos(2 pi x) + cos(2 pi y))
//!   rho*     = rho_t_ref/T* + gamma*PSI*t_ref*p*/T*      (the device recovery)
//!   m*       = rho* U*                                   (Rhie-Chow mass flux)
//!
//! Because the MASS flux m* is NOT divergence-free (div(rho*U*)=U*.grad(rho)!=0),
//! the continuity row is forced by S_p = +div(m*). The bounded momentum and
//! conservative extended-energy sources are 4th-order central finite differences
//! of the exact closures. The gate assembles at the exact state and measures
//! `(rhs - A*x_exact)/V`, avoiding nonlinear marching error and pressure-gauge drift.
#![cfg(all(feature = "dev-tests", feature = "cpu"))]

use std::f64::consts::PI;

use cfd2::solver::cpu::{CpuBackendConfig, CpuSolver};
use cfd2::solver::gpu::enums::{GpuBcKind, GpuBoundaryType};
use cfd2::solver::gpu::recipe::SteppingMode;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::{
    allmach_thermal_compressible_mms_model, ALLMACH_GAMMA, ALLMACH_K_OVER_CP,
    ALLMACH_MMS_SOURCE_P_FIELD, ALLMACH_MMS_SOURCE_T_FIELD, ALLMACH_MMS_SOURCE_U_FIELD,
    ALLMACH_T_REF,
};
use cfd2::solver::scheme::Scheme;
use cfd2::solver::TimeScheme;

const MU: f64 = 1.0;
const RHO_REF: f64 = 1.0;
const T_REF: f64 = ALLMACH_T_REF;
const K_OVER_CP: f64 = ALLMACH_K_OVER_CP;
const GAMMA: f64 = ALLMACH_GAMMA;
/// Compressibility ON: a healthy pressure->density coupling so the real-EOS
/// operator, Phi and the U.grad(p) heating are all exercised well above the f32
/// noise floor, while staying moderate enough to converge cleanly.
const PSI: f64 = 0.05;
const U_AMP: f64 = 0.1; // Taylor-Green velocity amplitude -> moderate Peclet
const P_AMP: f64 = 0.25;
const RHO_T_REF: f64 = RHO_REF * T_REF;

const DT: f64 = 0.3;

// ---- exact closures ---------------------------------------------------------

fn exact_t(x: f64, y: f64) -> f64 {
    T_REF * (1.5 + 0.5 * (PI * x).cos() * (PI * y).cos())
}

fn exact_p(x: f64, y: f64) -> f64 {
    P_AMP * ((2.0 * PI * x).cos() + (2.0 * PI * y).cos())
}

/// Real-EOS density, matching the on-device compressible recovery
/// rho = rho_t_ref/T + gamma*psi_ref*t_ref*p/T  (psi_ref = PSI, t_ref = T_REF).
fn exact_rho(x: f64, y: f64) -> f64 {
    let t = exact_t(x, y);
    RHO_T_REF / t + GAMMA * PSI * T_REF * exact_p(x, y) / t
}

/// Divergence-free Taylor-Green velocity (so dev2/grad(div U) and the -(1/3)(divU)^2
/// term of Phi vanish). NOT mass-flux-free for variable density.
fn exact_u(x: f64, y: f64) -> (f64, f64) {
    (
        U_AMP * (PI * x).sin() * (PI * y).cos(),
        -U_AMP * (PI * x).cos() * (PI * y).sin(),
    )
}

/// Rhie-Chow mass flux m* = rho* U* (variable density).
fn mass_flux(x: f64, y: f64) -> (f64, f64) {
    let (ux, uy) = exact_u(x, y);
    let r = exact_rho(x, y);
    (r * ux, r * uy)
}

// ---- 4th-order central finite differences over scalar closures --------------

const H: f64 = 1.0e-4;

fn d_dx(g: &impl Fn(f64, f64) -> f64, x: f64, y: f64) -> f64 {
    (g(x - 2.0 * H, y) - 8.0 * g(x - H, y) + 8.0 * g(x + H, y) - g(x + 2.0 * H, y)) / (12.0 * H)
}
fn d_dy(g: &impl Fn(f64, f64) -> f64, x: f64, y: f64) -> f64 {
    (g(x, y - 2.0 * H) - 8.0 * g(x, y - H) + 8.0 * g(x, y + H) - g(x, y + 2.0 * H)) / (12.0 * H)
}
fn d2_dx2(g: &impl Fn(f64, f64) -> f64, x: f64, y: f64) -> f64 {
    (-g(x - 2.0 * H, y) + 16.0 * g(x - H, y) - 30.0 * g(x, y) + 16.0 * g(x + H, y)
        - g(x + 2.0 * H, y))
        / (12.0 * H * H)
}
fn d2_dy2(g: &impl Fn(f64, f64) -> f64, x: f64, y: f64) -> f64 {
    (-g(x, y - 2.0 * H) + 16.0 * g(x, y - H) - 30.0 * g(x, y) + 16.0 * g(x, y + H)
        - g(x, y + 2.0 * H))
        / (12.0 * H * H)
}

fn u_x(x: f64, y: f64) -> f64 {
    exact_u(x, y).0
}
fn u_y(x: f64, y: f64) -> f64 {
    exact_u(x, y).1
}

// ---- manufactured sources ---------------------------------------------------

/// Momentum source for the bounded assembled equation
/// `div(m U) - U div(m) - mu lap(U) + grad(p) = S_U`.
fn source_u(x: f64, y: f64) -> (f64, f64) {
    let mx_ux = |x: f64, y: f64| mass_flux(x, y).0 * exact_u(x, y).0;
    let my_ux = |x: f64, y: f64| mass_flux(x, y).1 * exact_u(x, y).0;
    let mx_uy = |x: f64, y: f64| mass_flux(x, y).0 * exact_u(x, y).1;
    let my_uy = |x: f64, y: f64| mass_flux(x, y).1 * exact_u(x, y).1;
    let conv_x = d_dx(&mx_ux, x, y) + d_dy(&my_ux, x, y);
    let conv_y = d_dx(&mx_uy, x, y) + d_dy(&my_uy, x, y);
    let div_m = source_p(x, y);
    let (ux, uy) = exact_u(x, y);
    let bounded_conv_x = conv_x - ux * div_m;
    let bounded_conv_y = conv_y - uy * div_m;
    let visc_x = MU * (d2_dx2(&u_x, x, y) + d2_dy2(&u_x, x, y));
    let visc_y = MU * (d2_dx2(&u_y, x, y) + d2_dy2(&u_y, x, y));
    let gp_x = d_dx(&exact_p, x, y);
    let gp_y = d_dy(&exact_p, x, y);
    (
        bounded_conv_x - visc_x + gp_x,
        bounded_conv_y - visc_y + gp_y,
    )
}

/// Continuity source S_p = +div(m*) (= U*.grad(rho), non-zero for variable density).
fn source_p(x: f64, y: f64) -> f64 {
    let mx = |x: f64, y: f64| mass_flux(x, y).0;
    let my = |x: f64, y: f64| mass_flux(x, y).1;
    d_dx(&mx, x, y) + d_dy(&my, x, y)
}

/// Viscous dissipation `Phi = mu * 2[(du/dx)^2 + (dv/dy)^2 + 0.5(du/dy+dv/dx)^2 -
/// (1/3)(div U)^2]` scaled by the constant `1/cp = (gamma-1)*T_ref*psi_ref` — the
/// exact value the codegen adds to the energy RHS (div U = 0 here). Always >= 0.
fn phi_source(x: f64, y: f64) -> f64 {
    let dudx = d_dx(&u_x, x, y);
    let dudy = d_dy(&u_x, x, y);
    let dvdx = d_dx(&u_y, x, y);
    let dvdy = d_dy(&u_y, x, y);
    let shear = dudy + dvdx;
    let div_u = dudx + dvdy;
    let phi_grad =
        2.0 * (dudx * dudx + dvdy * dvdy + 0.5 * shear * shear - (1.0 / 3.0) * div_u * div_u);
    (GAMMA - 1.0) * T_REF * PSI * MU * phi_grad
}

/// U.grad(p) compression heating (T2): the codegen adds `(gamma-1)*T_ref*psi_ref *
/// U.grad(p)` to the energy RHS.
fn compression_source(x: f64, y: f64) -> f64 {
    let (ux, uy) = exact_u(x, y);
    let gpx = d_dx(&exact_p, x, y);
    let gpy = d_dy(&exact_p, x, y);
    (GAMMA - 1.0) * T_REF * PSI * (ux * gpx + uy * gpy)
}

/// Temperature source. The energy operator now carries the extra heating terms
/// Phi and T2 (both added to the energy RHS by the assembly), so the manufactured
/// source is the remaining RHS of
/// `div(m T) - (k/cp)lap(T) = Phi + T2 + S_T`.
fn source_t(x: f64, y: f64) -> f64 {
    let mx_t = |x: f64, y: f64| mass_flux(x, y).0 * exact_t(x, y);
    let my_t = |x: f64, y: f64| mass_flux(x, y).1 * exact_t(x, y);
    let conv = d_dx(&mx_t, x, y) + d_dy(&my_t, x, y);
    let conduction = K_OVER_CP * (d2_dx2(&exact_t, x, y) + d2_dy2(&exact_t, x, y));
    conv - conduction - phi_source(x, y) - compression_source(x, y)
}

fn assert_exact_outlet_bc(solver: &CpuSolver, mesh: &Mesh) {
    const STRIDE: usize = 4;
    let kinds = solver.debug_bc_kind();
    let values = solver.debug_bc_value();

    for face in 0..mesh.num_faces() {
        if mesh.face_boundary[face] != Some(BoundaryType::Outlet) {
            continue;
        }

        let x = mesh.face_cx[face];
        let y = mesh.face_cy[face];
        let (ux, uy) = exact_u(x, y);
        let exact = [ux, uy, exact_p(x, y), exact_t(x, y)];

        for row in 0..STRIDE {
            let index = face * STRIDE + row;
            assert_eq!(
                kinds[index],
                GpuBcKind::Dirichlet as u32,
                "Outlet row {row} is not Dirichlet"
            );
            assert!(
                (values[index] as f64 - exact[row]).abs() <= 2.0e-6,
                "Outlet row {row} value mismatch at face {face}: got {}, expected {}",
                values[index],
                exact[row]
            );
        }
    }
}

/// Assemble at the exact state and return the volume-weighted L2 norm of
/// `(rhs - A*x_exact)/V` for `[Ux, Uy, p, T]`.
///
/// Two boundary layers are excluded from the order norm because this is an
/// interior spatial-operator gate. Boundary kind and value correctness is
/// asserted independently above.
fn exact_residual_l2(n: usize) -> [f64; 4] {
    const STRIDE: usize = 4;

    let mesh = generate_structured_rect_mesh(
        n,
        n,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::MovingWall,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::MovingWall,
            top: BoundaryType::MovingWall,
        },
    );

    let model = allmach_thermal_compressible_mms_model().expect("MMS model");
    let mut solver = CpuSolver::with_stepping(
        &mesh,
        model,
        Scheme::SecondOrderUpwind,
        TimeScheme::BDF2,
        SteppingMode::Coupled,
        CpuBackendConfig::default(),
    )
    .expect("CPU solver");

    solver.set_dt(DT as f32);
    solver.set_dtau(0.0);
    solver.set_density(RHO_REF as f32);
    solver.set_viscosity(MU as f32);
    solver.set_alpha_u(0.7);
    solver.set_alpha_p(0.3);
    solver.set_outer_iters(1);

    let cells = mesh.num_cells();
    for (name, value) in [
        ("psi_ref", PSI),
        ("psi", PSI),
        ("psi_precond", 0.0),
        ("t_ref", T_REF),
        ("rho_t_ref", RHO_T_REF),
        ("rho_floor", PSI * 1.0e-5),
        ("u_ref", U_AMP),
        ("precond_mask", 0.0),
    ] {
        solver
            .set_field_scalar(name, &vec![value; cells])
            .unwrap_or_else(|error| panic!("seed {name}: {error}"));
    }

    let fx = mesh.face_cx.clone();
    let fy = mesh.face_cy.clone();
    for boundary in [GpuBoundaryType::MovingWall, GpuBoundaryType::Outlet] {
        for component in 0..2u32 {
            let (fx, fy) = (fx.clone(), fy.clone());
            solver
                .set_boundary_values_per_face(boundary, "U", component, &move |face: u32| {
                    let i = face as usize;
                    let (ux, uy) = exact_u(fx[i], fy[i]);
                    (if component == 0 { ux } else { uy }) as f32
                })
                .expect("exact U boundary");
        }

        let (fx, fy) = (fx.clone(), fy.clone());
        solver
            .set_boundary_values_per_face(boundary, "T", 0, &move |face: u32| {
                let i = face as usize;
                exact_t(fx[i], fy[i]) as f32
            })
            .expect("exact T boundary");
    }

    let (fxp, fyp) = (fx.clone(), fy.clone());
    solver
        .set_boundary_values_per_face(GpuBoundaryType::Outlet, "p", 0, &move |face: u32| {
            let i = face as usize;
            exact_p(fxp[i], fyp[i]) as f32
        })
        .expect("exact p boundary");

    assert_exact_outlet_bc(&solver, &mesh);

    let u: Vec<_> = (0..cells)
        .map(|i| exact_u(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let p: Vec<_> = (0..cells)
        .map(|i| exact_p(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let t: Vec<_> = (0..cells)
        .map(|i| exact_t(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let rho: Vec<_> = (0..cells)
        .map(|i| exact_rho(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let d_p: Vec<_> = rho.iter().map(|&r| 0.7 * DT / r).collect();

    let src_u: Vec<_> = (0..cells)
        .map(|i| source_u(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let src_p: Vec<_> = (0..cells)
        .map(|i| source_p(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let src_t: Vec<_> = (0..cells)
        .map(|i| source_t(mesh.cell_cx[i], mesh.cell_cy[i]))
        .collect();
    let u_dot_grad_p: Vec<_> = (0..cells)
        .map(|i| {
            let x = mesh.cell_cx[i];
            let y = mesh.cell_cy[i];
            let (ux, uy) = exact_u(x, y);
            ux * d_dx(&exact_p, x, y) + uy * d_dy(&exact_p, x, y)
        })
        .collect();

    solver.set_field_vec2("U", &u).expect("exact U");
    solver.set_field_scalar("p", &p).expect("exact p");
    solver.set_field_scalar("T", &t).expect("exact T");
    solver.set_field_scalar("rho", &rho).expect("exact rho");
    solver.set_field_scalar("d_p", &d_p).expect("exact d_p");
    solver
        .set_field_scalar("u_dot_grad_p", &u_dot_grad_p)
        .expect("exact U dot grad(p)");
    solver
        .set_field_vec2(ALLMACH_MMS_SOURCE_U_FIELD, &src_u)
        .expect("source U");
    solver
        .set_field_scalar(ALLMACH_MMS_SOURCE_P_FIELD, &src_p)
        .expect("source p");
    solver
        .set_field_scalar(ALLMACH_MMS_SOURCE_T_FIELD, &src_t)
        .expect("source T");
    solver.initialize_history();

    let (matrix, rhs) = solver.debug_assemble();
    let (row_offsets, columns, _, stride) = solver.debug_topology();
    assert_eq!(stride, STRIDE);

    let mut x_exact = vec![0.0_f64; cells * STRIDE];
    for i in 0..cells {
        x_exact[i * STRIDE] = u[i].0;
        x_exact[i * STRIDE + 1] = u[i].1;
        x_exact[i * STRIDE + 2] = p[i];
        x_exact[i * STRIDE + 3] = t[i];
    }

    let mut squared = [0.0_f64; STRIDE];
    let mut interior_volume = 0.0;

    for i in 0..cells {
        let ix = i % n;
        let iy = i / n;
        if ix < 2 || ix + 2 >= n || iy < 2 || iy + 2 >= n {
            continue;
        }

        let scalar_start = row_offsets[i] as usize;
        let neighbors = row_offsets[i + 1] as usize - scalar_start;
        let volume = mesh.cell_vol[i];
        interior_volume += volume;

        for row in 0..STRIDE {
            let block_start = scalar_start * STRIDE * STRIDE + neighbors * STRIDE * row;
            let mut ax = 0.0_f64;

            for rank in 0..neighbors {
                let neighbor = columns[scalar_start + rank] as usize;
                for column in 0..STRIDE {
                    ax += matrix[block_start + rank * STRIDE + column] as f64
                        * x_exact[neighbor * STRIDE + column];
                }
            }

            let residual = (rhs[i * STRIDE + row] as f64 - ax) / volume;
            squared[row] += residual * residual * volume;
        }
    }

    squared.map(|sum| (sum / interior_volume).sqrt())
}

#[test]
fn allmach_thermal_compressible_exact_residual_order() {
    let levels = [16usize, 32, 64];
    let mut errors: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());

    for n in levels {
        let residual = exact_residual_l2(n);
        println!(
            "[compressible-mms/residual] n={n:>3} \
             Ux={:.6e} Uy={:.6e} p={:.6e} T={:.6e}",
            residual[0], residual[1], residual[2], residual[3]
        );
        for row in 0..4 {
            errors[row].push(residual[row]);
        }
    }

    let labels = ["U_x", "U_y", "p", "T"];
    let finest_caps = [3.0e-3, 3.0e-3, 8.0e-3, 1.2e-2];

    for row in 0..4 {
        let pair_orders: Vec<_> = errors[row]
            .windows(2)
            .map(|w| (w[0] / w[1]).ln() / 2.0_f64.ln())
            .collect();
        let worst_order = pair_orders.iter().copied().fold(f64::INFINITY, f64::min);

        println!(
            "[compressible-mms/residual] {} errors={:?} pair_orders={pair_orders:?}",
            labels[row], errors[row]
        );

        assert!(
            errors[row].windows(2).all(|w| w[1] < w[0]),
            "{} exact-state residual did not decrease: {:?}",
            labels[row],
            errors[row]
        );
        assert!(
            worst_order >= 1.75,
            "{} exact-state residual order regressed: errors={:?}, orders={pair_orders:?}",
            labels[row],
            errors[row]
        );
        assert!(
            errors[row][2] <= finest_caps[row],
            "{} finest residual {} exceeds cap {}",
            labels[row],
            errors[row][2],
            finest_caps[row]
        );
    }
}
