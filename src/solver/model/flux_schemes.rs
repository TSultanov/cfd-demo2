use crate::solver::ir::{
    FaceScalarExpr as S, FaceSide, FaceVec2Expr as V, FluxLayout, FluxModuleKernelSpec, LimiterSpec,
};
use crate::solver::model::backend::ast::EquationSystem;
use crate::solver::model::flux_module::FluxSchemeSpec;
use crate::solver::scheme::Scheme;

use crate::solver::ir::reconstruction::{
    limited_linear_face_value, quick_face_value, FaceExprBuilder,
};

pub fn lower_flux_scheme(
    flux_scheme: &FluxSchemeSpec,
    system: &EquationSystem,
    reconstruction: Scheme,
) -> Result<FluxModuleKernelSpec, String> {
    match *flux_scheme {
        FluxSchemeSpec::EulerCentralUpwind => euler_central_upwind(system, reconstruction),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_state_named(expr: &S, name: &str) -> bool {
        matches!(
            expr,
            S::State {
                name: n,
                ..
            } if n == name
        )
    }

    fn is_lit(expr: &S, value: f32) -> bool {
        matches!(expr, S::Literal(v) if *v == value)
    }

    fn is_mul_named_states(expr: &S, a: &str, b: &str) -> bool {
        match expr {
            S::Mul(x, y) => {
                (is_state_named(x, a) && is_state_named(y, b))
                    || (is_state_named(x, b) && is_state_named(y, a))
            }
            _ => false,
        }
    }

    fn contains_sign_guard_for_states(expr: &S, a: &str, b: &str) -> bool {
        fn contains_state(expr: &S, name: &str) -> bool {
            match expr {
                S::State { name: n, .. } => n == name,
                S::Add(x, y)
                | S::Sub(x, y)
                | S::Mul(x, y)
                | S::Div(x, y)
                | S::Max(x, y)
                | S::Min(x, y)
                | S::Lerp(x, y) => contains_state(x, name) || contains_state(y, name),
                S::Neg(x) | S::Abs(x) | S::Sqrt(x) => contains_state(x, name),
                S::Dot(_, _)
                | S::Literal(_)
                | S::Builtin(_)
                | S::Constant { .. }
                | S::LowMachParam(_)
                | S::Primitive { .. } => false,
            }
        }

        fn depends_on_states(expr: &S, a: &str, b: &str) -> bool {
            contains_state(expr, a) && contains_state(expr, b)
        }

        fn walk(expr: &S, f: &mut impl FnMut(&S)) {
            f(expr);
            match expr {
                S::Add(x, y)
                | S::Sub(x, y)
                | S::Mul(x, y)
                | S::Div(x, y)
                | S::Max(x, y)
                | S::Min(x, y)
                | S::Lerp(x, y) => {
                    walk(x, f);
                    walk(y, f);
                }
                S::Neg(x) | S::Abs(x) | S::Sqrt(x) => walk(x, f),
                S::Dot(_, _) => {}
                S::Literal(_)
                | S::Builtin(_)
                | S::Constant { .. }
                | S::LowMachParam(_)
                | S::State { .. }
                | S::Primitive { .. } => {}
            }
        }

        let mut found = false;
        walk(expr, &mut |node| {
            let S::Div(num, denom) = node else {
                return;
            };

            fn is_max_of(
                expr: &S,
                mut a: impl FnMut(&S) -> bool,
                mut b: impl FnMut(&S) -> bool,
            ) -> bool {
                match expr {
                    S::Max(x, y) => (a(x) && b(y)) || (a(y) && b(x)),
                    _ => false,
                }
            }

            let p = |e: &S| depends_on_states(e, a, b);
            let abs_p = |e: &S| matches!(e, S::Abs(inner) if p(inner));
            let num_ok = is_max_of(num, p, |e| is_lit(e, 0.0));
            let denom_ok = is_max_of(denom, abs_p, |e| is_lit(e, crate::solver::ir::VANLEER_EPS));

            if num_ok && denom_ok {
                found = true;
            }
        });

        found
    }

    #[test]
    fn contract_flux_module_vanleer_has_opposite_slope_sign_guard() {
        // This is a structural/IR contract (not a brittle WGSL string match):
        // VanLeer MUSCL must include a guard that zeros the correction when diff*delta <= 0.
        let diff = S::state(FaceSide::Owner, "diff");
        let delta = S::state(FaceSide::Owner, "delta");
        let delta_limited = crate::solver::ir::reconstruction::vanleer_delta_limited::<
            FaceExprBuilder,
        >(diff, delta);

        assert!(
            contains_sign_guard_for_states(&delta_limited, "diff", "delta"),
            "expected VanLeer sign-guard structure (max(p,0)/max(abs(p),eps))"
        );
    }
}

fn euler_central_upwind(
    system: &EquationSystem,
    reconstruction: Scheme,
) -> Result<FluxModuleKernelSpec, String> {
    let flux_layout = FluxLayout::from_system(system);
    let components: Vec<String> = flux_layout
        .components
        .iter()
        .map(|c| c.name.clone())
        .collect();

    let ex = V::vec2(S::lit(1.0), S::lit(0.0));
    let ey = V::vec2(S::lit(0.0), S::lit(1.0));

    // Conventional compressible field names (model-side convention).
    let rho_name = "rho";
    let rho_u_name = "rho_u";
    let rho_e_name = "rho_e";

    let other_side = |side: FaceSide| {
        if side == FaceSide::Owner {
            FaceSide::Neighbor
        } else {
            FaceSide::Owner
        }
    };

    let limiter_for_scheme = |scheme: Scheme| -> LimiterSpec {
        match scheme {
            Scheme::SecondOrderUpwindMinMod | Scheme::QUICKMinMod => LimiterSpec::MinMod,
            Scheme::SecondOrderUpwindVanLeer | Scheme::QUICKVanLeer => LimiterSpec::VanLeer,
            _ => LimiterSpec::None,
        }
    };

    let reconstruct_scalar = |side: FaceSide, phi_cell: S, phi_other: S, grad: V| -> S {
        match reconstruction {
            Scheme::Upwind => phi_cell,

            Scheme::SecondOrderUpwind
            | Scheme::SecondOrderUpwindMinMod
            | Scheme::SecondOrderUpwindVanLeer => limited_linear_face_value::<FaceExprBuilder>(
                phi_cell,
                phi_other,
                grad,
                V::cell_to_face(side),
                limiter_for_scheme(reconstruction),
            ),

            Scheme::QUICK | Scheme::QUICKMinMod | Scheme::QUICKVanLeer => {
                let d = V::Sub(
                    Box::new(V::cell_to_face(side)),
                    Box::new(V::cell_to_face(other_side(side))),
                );
                quick_face_value::<FaceExprBuilder>(
                    phi_cell,
                    phi_other,
                    grad,
                    d,
                    limiter_for_scheme(reconstruction),
                )
            }
        }
    };

    let rho_raw = |side: FaceSide| S::state(side, rho_name);
    let t_raw = |side: FaceSide| S::state(side, "T");

    // OpenFOAM's `vanLeer` / `vanLeerV` reconstruction (as used by rhoCentralFoam) is an NVD/TVD
    // limited interpolation that blends between central differencing and upwind, driven by a
    // limiter function:
    //   psi(r) = (r + |r|) / (1 + |r|)
    // with r computed from the upwind gradient along the cell-center vector `d`.
    //
    // OpenFOAM uses:
    //   r = 2*(gradcf/gradf) - 1,
    // guarded by a `1000*` threshold that effectively clamps r to [-2001, 1999].
    //
    // We mirror this with a branchless, divide-by-zero-safe ratio:
    //   gradcf/gradf ≈ (gradcf * gradf) / (gradf^2 + eps2)
    // which matches the exact ratio for nonzero gradf and yields r=-1 when gradf=0 (=> psi=0).
    let eps2 = S::lit(1e-30);
    let vanleer_limiter = |gradf: S, gradcf: S| {
        let gradf2 = S::Mul(Box::new(gradf.clone()), Box::new(gradf.clone()));
        let ratio = S::Div(
            Box::new(S::Mul(Box::new(gradcf), Box::new(gradf.clone()))),
            Box::new(S::Add(Box::new(gradf2), Box::new(eps2.clone()))),
        );
        let r_raw = S::Sub(
            Box::new(S::Mul(Box::new(S::lit(2.0)), Box::new(ratio))),
            Box::new(S::lit(1.0)),
        );

        // OpenFOAM's NVDTVD/NVDVTVDV guards extreme ratios via a `1000*` threshold,
        // which corresponds to clamping r to [-2001, 1999].
        let r = S::Max(
            Box::new(S::lit(-2001.0)),
            Box::new(S::Min(Box::new(r_raw), Box::new(S::lit(1999.0)))),
        );

        let abs_r = S::Abs(Box::new(r.clone()));
        S::Div(
            Box::new(S::Add(Box::new(r), Box::new(abs_r.clone()))),
            Box::new(S::Add(Box::new(S::lit(1.0)), Box::new(abs_r))),
        )
    };

    let d = V::Sub(
        Box::new(V::cell_to_face(FaceSide::Owner)),
        Box::new(V::cell_to_face(FaceSide::Neighbor)),
    );

    let reconstruct_vanleer_scalar =
        |side: FaceSide, phi_p: S, phi_n: S, grad_p: V, grad_n: V| -> S {
            let gradf = S::Sub(Box::new(phi_n.clone()), Box::new(phi_p.clone()));
            let grad = if side == FaceSide::Owner {
                grad_p
            } else {
                grad_n
            };
            let gradcf = S::Dot(Box::new(d.clone()), Box::new(grad));
            let limiter = vanleer_limiter(gradf.clone(), gradcf);
            let delta = match side {
                FaceSide::Owner => S::Mul(Box::new(limiter), Box::new(S::lambda_other())),
                FaceSide::Neighbor => S::Mul(Box::new(limiter), Box::new(S::lambda())),
            };
            match side {
                FaceSide::Owner => S::Add(
                    Box::new(phi_p),
                    Box::new(S::Mul(Box::new(delta), Box::new(gradf))),
                ),
                FaceSide::Neighbor => S::Sub(
                    Box::new(phi_n),
                    Box::new(S::Mul(Box::new(delta), Box::new(gradf))),
                ),
            }
        };

    let reconstruct_vanleer_vec2 =
        |side: FaceSide, phi_p: V, phi_n: V, grad_px: V, grad_py: V, grad_nx: V, grad_ny: V| -> V {
            let gradf_v = V::Sub(Box::new(phi_n.clone()), Box::new(phi_p.clone()));
            let gradf = S::Dot(Box::new(gradf_v.clone()), Box::new(gradf_v.clone()));

            let (gx, gy) = if side == FaceSide::Owner {
                (grad_px, grad_py)
            } else {
                (grad_nx, grad_ny)
            };

            // Match OpenFOAM's `vanLeerV` (NVDVTVDV) form:
            //   gradcf = gradfV & (d & gradcP)
            //
            // With our stored per-component gradients:
            //   gx = grad(phi_x) = [dphi_x/dx, dphi_x/dy]
            //   gy = grad(phi_y) = [dphi_y/dx, dphi_y/dy]
            // and d = [dx, dy], we interpret this as directional derivatives per component:
            //   gradcf_x = d · gx
            //   gradcf_y = d · gy
            //   gradcf   = gradfV · [gradcf_x, gradcf_y]
            let gradcf_x = S::Dot(Box::new(d.clone()), Box::new(gx.clone()));
            let gradcf_y = S::Dot(Box::new(d.clone()), Box::new(gy.clone()));
            let gradcf = S::Dot(
                Box::new(gradf_v.clone()),
                Box::new(V::vec2(gradcf_x, gradcf_y)),
            );
            let limiter = vanleer_limiter(gradf, gradcf);
            let delta = match side {
                FaceSide::Owner => S::Mul(Box::new(limiter), Box::new(S::lambda_other())),
                FaceSide::Neighbor => S::Mul(Box::new(limiter), Box::new(S::lambda())),
            };
            let corr = V::MulScalar(Box::new(gradf_v), Box::new(delta));
            match side {
                FaceSide::Owner => V::Add(Box::new(phi_p), Box::new(corr)),
                FaceSide::Neighbor => V::Sub(Box::new(phi_n), Box::new(corr)),
            }
        };

    let rho = |side: FaceSide| match reconstruction {
        Scheme::SecondOrderUpwindVanLeer => reconstruct_vanleer_scalar(
            side,
            rho_raw(FaceSide::Owner),
            rho_raw(FaceSide::Neighbor),
            V::state_vec2(FaceSide::Owner, "grad_rho"),
            V::state_vec2(FaceSide::Neighbor, "grad_rho"),
        ),
        _ => reconstruct_scalar(
            side,
            rho_raw(side),
            rho_raw(other_side(side)),
            V::state_vec2(side, "grad_rho"),
        ),
    };

    let t = |side: FaceSide| match reconstruction {
        Scheme::SecondOrderUpwindVanLeer => reconstruct_vanleer_scalar(
            side,
            t_raw(FaceSide::Owner),
            t_raw(FaceSide::Neighbor),
            V::state_vec2(FaceSide::Owner, "grad_T"),
            V::state_vec2(FaceSide::Neighbor, "grad_T"),
        ),
        _ => reconstruct_scalar(
            side,
            t_raw(side),
            t_raw(other_side(side)),
            V::state_vec2(side, "grad_T"),
        ),
    };

    // Match OpenFOAM's rhoCentralFoam: reconstruct conserved momentum `rhoU` using `vanLeerV`
    // and derive face velocity as `U = rhoU/rho`.
    let rho_u = |side: FaceSide| match reconstruction {
        Scheme::SecondOrderUpwindVanLeer => reconstruct_vanleer_vec2(
            side,
            V::state_vec2(FaceSide::Owner, rho_u_name),
            V::state_vec2(FaceSide::Neighbor, rho_u_name),
            V::state_vec2(FaceSide::Owner, "grad_rho_u_x"),
            V::state_vec2(FaceSide::Owner, "grad_rho_u_y"),
            V::state_vec2(FaceSide::Neighbor, "grad_rho_u_x"),
            V::state_vec2(FaceSide::Neighbor, "grad_rho_u_y"),
        ),
        _ => {
            let rho_u_owner = V::state_vec2(side, rho_u_name);
            let rho_u_other = V::state_vec2(other_side(side), rho_u_name);
            let x = reconstruct_scalar(
                side,
                S::Dot(Box::new(rho_u_owner.clone()), Box::new(ex.clone())),
                S::Dot(Box::new(rho_u_other.clone()), Box::new(ex.clone())),
                V::state_vec2(side, "grad_rho_u_x"),
            );
            let y = reconstruct_scalar(
                side,
                S::Dot(Box::new(rho_u_owner), Box::new(ey.clone())),
                S::Dot(Box::new(rho_u_other), Box::new(ey.clone())),
                V::state_vec2(side, "grad_rho_u_y"),
            );
            V::vec2(x, y)
        }
    };

    let u_vec = |side: FaceSide| {
        let inv_rho = S::Div(Box::new(S::lit(1.0)), Box::new(rho(side)));
        V::MulScalar(Box::new(rho_u(side)), Box::new(inv_rho))
    };

    // Ideal-gas pressure from reconstructed rho and T: p = rho * R * T.
    // (For barotropic closures, the model still enforces p = rho * R * T via the temperature
    // recovery constraint, so this remains consistent.)
    let p = |side: FaceSide| {
        S::Mul(
            Box::new(S::Mul(Box::new(rho(side)), Box::new(S::constant("eos_r")))),
            Box::new(t(side)),
        )
    };

    // Total energy density reconstructed from primitive variables:
    //   rho_e = p/(gamma-1) + 0.5 * rho * |u|^2
    //
    // OpenFOAM's rhoCentralFoam reconstructs rho/U/T and derives energy/enthalpy consistently
    // at faces. Doing the same here improves agreement vs OpenFOAM reference cases.
    let rho_e = |side: FaceSide| {
        let gm1 = S::Max(Box::new(S::constant("eos_gm1")), Box::new(S::lit(1e-12)));
        let u = u_vec(side);
        let u2 = S::Dot(Box::new(u.clone()), Box::new(u));
        let ke = S::Mul(
            Box::new(S::Mul(Box::new(S::lit(0.5)), Box::new(rho(side)))),
            Box::new(u2),
        );
        S::Add(
            Box::new(S::Div(Box::new(p(side)), Box::new(gm1))),
            Box::new(ke),
        )
    };

    let rho_u_x = |side: FaceSide| S::Dot(Box::new(rho_u(side)), Box::new(ex.clone()));
    let rho_u_y = |side: FaceSide| S::Dot(Box::new(rho_u(side)), Box::new(ey.clone()));
    let u_n = |side: FaceSide| S::Dot(Box::new(u_vec(side)), Box::new(V::normal()));

    let c2 = |side: FaceSide| {
        // Generalized wave speed:
        //   c^2 = gamma * p / rho + dp_drho
        // where dp_drho is nonzero for barotropic closures (e.g. linear compressibility).
        S::Add(
            Box::new(S::Div(
                Box::new(S::Mul(
                    Box::new(S::constant("eos_gamma")),
                    Box::new(p(side)),
                )),
                Box::new(rho(side)),
            )),
            Box::new(S::constant("eos_dp_drho")),
        )
    };

    let u_n2 = |side: FaceSide| {
        let un = u_n(side);
        S::Mul(Box::new(un.clone()), Box::new(un))
    };

    let low_mach_model = S::low_mach_model();
    let low_mach_theta_floor = S::low_mach_theta_floor();

    let weight_for = |value: f32| {
        S::Max(
            Box::new(S::lit(0.0)),
            Box::new(S::Sub(
                Box::new(S::lit(1.0)),
                Box::new(S::Abs(Box::new(S::Sub(
                    Box::new(low_mach_model.clone()),
                    Box::new(S::lit(value)),
                )))),
            )),
        )
    };

    let w_legacy = weight_for(0.0);
    let w_weiss_smith = weight_for(1.0);
    let w_off = weight_for(2.0);

    let c_eff2_legacy = |side: FaceSide| {
        let c2_side = c2(side);
        S::Min(Box::new(u_n2(side)), Box::new(c2_side))
    };

    let c_eff2_weiss_smith = |side: FaceSide| {
        let c2_side = c2(side);
        let floor = S::Mul(
            Box::new(low_mach_theta_floor.clone()),
            Box::new(c2_side.clone()),
        );
        S::Min(
            Box::new(S::Max(Box::new(u_n2(side)), Box::new(floor))),
            Box::new(c2_side),
        )
    };

    let c_eff2 = |side: FaceSide| {
        let c2_side = c2(side);
        S::Add(
            Box::new(S::Add(
                Box::new(S::Mul(Box::new(w_off.clone()), Box::new(c2_side))),
                Box::new(S::Mul(
                    Box::new(w_legacy.clone()),
                    Box::new(c_eff2_legacy(side)),
                )),
            )),
            Box::new(S::Mul(
                Box::new(w_weiss_smith.clone()),
                Box::new(c_eff2_weiss_smith(side)),
            )),
        )
    };

    let c_eff = |side: FaceSide| S::Sqrt(Box::new(c_eff2(side)));

    // --- Low-Mach pressure coupling ---
    //
    // At low Mach numbers, a purely density-based continuity flux can decouple pressure on
    // collocated grids and produce checkerboarding. Add a preconditioning-gated pressure
    // perturbation contribution to the density state used by the central-upwind dissipation
    // term (leaves the physical mass flux unchanged).
    let pressure_coupling_alpha = S::low_mach_pressure_coupling_alpha();
    let low_mach_enabled = S::Sub(Box::new(S::lit(1.0)), Box::new(w_off.clone()));

    // Convert a pressure perturbation to an equivalent density perturbation using the *physical*
    // compressibility (ρ' ≈ p'/c^2).
    //
    // Using the *preconditioned* wave speed here can severely over-amplify the coupling at very
    // low Mach numbers (since c_eff^2 ~ O(|u|^2)), causing unphysical density states and solver
    // instability.
    let c_couple2_safe = |side: FaceSide| S::Max(Box::new(c2(side)), Box::new(S::lit(1e-12)));
    let rho_diss = |side: FaceSide| {
        // Only apply the pressure-coupling fix when low-Mach preconditioning is explicitly enabled.
        // This keeps the default (low_mach_model=Off) behavior aligned with OpenFOAM's rhoCentralFoam.
        let coupling = S::Mul(
            Box::new(low_mach_enabled.clone()),
            Box::new(pressure_coupling_alpha.clone()),
        );
        S::Add(
            Box::new(rho(side)),
            Box::new(S::Mul(
                Box::new(coupling),
                Box::new(S::Div(Box::new(p(side)), Box::new(c_couple2_safe(side)))),
            )),
        )
    };

    let n_x = S::Dot(Box::new(V::normal()), Box::new(ex.clone()));
    let n_y = S::Dot(Box::new(V::normal()), Box::new(ey.clone()));

    let flux_mass = |side: FaceSide| S::Mul(Box::new(rho(side)), Box::new(u_n(side)));

    // OpenFOAM rhoCentralFoam viscous split:
    //
    //   - fvm::laplacian(muEff, U)
    //   - fvc::div(tauMC)
    //
    // with:
    //   tauMC = muEff * dev2(T(grad(U)))
    //
    // Important: OpenFOAM's Gauss div-scheme uses `dotInterpolate(Sf, tauMC)`, i.e. `Sf & tauMC`,
    // which corresponds to a traction based on `tauMC^T`. Since `tauMC` is already constructed
    // with the transpose gradient, the effective face traction uses `dev2(grad(U))` (not
    // `dev2(T(grad(U)))`).
    //
    // Add this explicit correction to the momentum flux to better match OpenFOAM viscous cases.
    let visc_mu = S::constant("viscosity");
    // Note: neighbor-side `grad_*` vectors are forced to zero on boundary faces to preserve
    // boundary semantics for reconstruction. Flip the lerp order so boundary faces default
    // to the owner-cell gradient (while interior faces remain unchanged on our symmetric meshes).
    let grad_u_x_face_raw = V::Lerp(
        Box::new(V::state_vec2(FaceSide::Neighbor, "grad_u_x")),
        Box::new(V::state_vec2(FaceSide::Owner, "grad_u_x")),
    );
    let grad_u_y_face_raw = V::Lerp(
        Box::new(V::state_vec2(FaceSide::Neighbor, "grad_u_y")),
        Box::new(V::state_vec2(FaceSide::Owner, "grad_u_y")),
    );

    // Match OpenFOAM's Gauss gradient boundary correction.
    //
    // OpenFOAM computes cell-centered gradients via Gauss' theorem and then corrects the
    // *boundary* gradient field so that the normal component matches `snGrad(U)`:
    //   grad += n ⊗ (snGrad(U) - (n & grad))
    //
    // Our `grad_*` fields are cell-centered. Apply the equivalent correction on boundary faces
    // when forming the face gradient used by the `div(tauMC)` correction.
    let n = V::normal();
    let is_boundary = S::is_boundary();
    let dist_safe = S::Max(Box::new(S::dist()), Box::new(S::lit(1e-6)));

    let u_face = V::state_vec2(FaceSide::Owner, "u");
    let u_cell = V::cell_state_vec2(FaceSide::Owner, "u");
    let u_face_x = S::Dot(Box::new(u_face.clone()), Box::new(ex.clone()));
    let u_face_y = S::Dot(Box::new(u_face), Box::new(ey.clone()));
    let u_cell_x = S::Dot(Box::new(u_cell.clone()), Box::new(ex.clone()));
    let u_cell_y = S::Dot(Box::new(u_cell), Box::new(ey.clone()));

    let sn_grad_u_x = S::Div(
        Box::new(S::Sub(Box::new(u_face_x), Box::new(u_cell_x))),
        Box::new(dist_safe.clone()),
    );
    let sn_grad_u_y = S::Div(
        Box::new(S::Sub(Box::new(u_face_y), Box::new(u_cell_y))),
        Box::new(dist_safe),
    );

    let grad_u_x_face = V::Add(
        Box::new(grad_u_x_face_raw.clone()),
        Box::new(V::MulScalar(
            Box::new(n.clone()),
            Box::new(S::Mul(
                Box::new(is_boundary.clone()),
                Box::new(S::Sub(
                    Box::new(sn_grad_u_x),
                    Box::new(S::Dot(Box::new(n.clone()), Box::new(grad_u_x_face_raw))),
                )),
            )),
        )),
    );
    let grad_u_y_face = V::Add(
        Box::new(grad_u_y_face_raw.clone()),
        Box::new(V::MulScalar(
            Box::new(n.clone()),
            Box::new(S::Mul(
                Box::new(is_boundary),
                Box::new(S::Sub(
                    Box::new(sn_grad_u_y),
                    Box::new(S::Dot(Box::new(n), Box::new(grad_u_y_face_raw))),
                )),
            )),
        )),
    );

    let dux_dx = S::Dot(Box::new(grad_u_x_face.clone()), Box::new(ex.clone()));
    let dux_dy = S::Dot(Box::new(grad_u_x_face.clone()), Box::new(ey.clone()));
    let duy_dx = S::Dot(Box::new(grad_u_y_face.clone()), Box::new(ex.clone()));
    let duy_dy = S::Dot(Box::new(grad_u_y_face.clone()), Box::new(ey.clone()));
    let div_u = S::Add(Box::new(dux_dx.clone()), Box::new(duy_dy.clone()));
    let two_thirds = S::lit(2.0 / 3.0);

    // OpenFOAM details:
    // - `fvc::grad(U)` returns a tensor `G` with indices `G_ij = ∂u_j/∂x_i` (direction-first).
    // - `tauMC = muEff*dev2(T(G))` therefore corresponds to `tauMC_ij = ∂u_i/∂x_j - 2/3 δ_ij div(u)`
    //   (the transpose gradient, with the 2/3 deviatoric correction).
    // - `fvc::div(tauMC)` uses a Gauss div-scheme based on `Sf & tauMC`, which contracts `Sf_i*tauMC_ij`.
    //   This produces the `∇(div u)` contribution required for the full Newtonian viscous stress.
    //
    // Here we form the corresponding face flux contribution per unit area: (tauMC · n).
    // `fvc::div(tauMC)` uses a Gauss div-scheme based on `Sf & tauMC`.
    // This corresponds to the flux contribution (per unit area): (n · tauMC), i.e.
    //   (n_i * tauMC_{i j}) for momentum component j.
    //
    // With tauMC = muEff * dev2(T(grad(U))) and OpenFOAM's tensor conventions, this yields:
    //   flux_x = muEff * [ (d u_x/dx - 2/3 divU) n_x + (d u_x/dy) n_y ]
    //   flux_y = muEff * [ (d u_y/dx) n_x + (d u_y/dy - 2/3 divU) n_y ]
    let tau_mc_dot_n_x = S::Mul(
        Box::new(visc_mu.clone()),
        Box::new(S::Add(
            Box::new(S::Mul(
                Box::new(S::Sub(
                    Box::new(dux_dx),
                    Box::new(S::Mul(
                        Box::new(two_thirds.clone()),
                        Box::new(div_u.clone()),
                    )),
                )),
                Box::new(n_x.clone()),
            )),
            Box::new(S::Mul(Box::new(dux_dy.clone()), Box::new(n_y.clone()))),
        )),
    );
    let tau_mc_dot_n_y = S::Mul(
        Box::new(visc_mu),
        Box::new(S::Add(
            Box::new(S::Mul(Box::new(duy_dx), Box::new(n_x.clone()))),
            Box::new(S::Mul(
                Box::new(S::Sub(
                    Box::new(duy_dy),
                    Box::new(S::Mul(Box::new(two_thirds), Box::new(div_u))),
                )),
                Box::new(n_y.clone()),
            )),
        )),
    );

    let flux_mom_x = |side: FaceSide| {
        S::Sub(
            Box::new(S::Add(
                Box::new(S::Mul(Box::new(rho_u_x(side)), Box::new(u_n(side)))),
                Box::new(S::Mul(Box::new(p(side)), Box::new(n_x.clone()))),
            )),
            Box::new(tau_mc_dot_n_x.clone()),
        )
    };
    let flux_mom_y = |side: FaceSide| {
        S::Sub(
            Box::new(S::Add(
                Box::new(S::Mul(Box::new(rho_u_y(side)), Box::new(u_n(side)))),
                Box::new(S::Mul(Box::new(p(side)), Box::new(n_y.clone()))),
            )),
            Box::new(tau_mc_dot_n_y.clone()),
        )
    };

    // Wave speed bounds (OpenFOAM rhoCentralFoam Kurganov).
    // These are used by the central-upwind numerical flux and also for the viscous-work
    // energy correction (sigmaDotU).
    let a_plus = S::Max(
        Box::new(S::lit(0.0)),
        Box::new(S::Max(
            Box::new(S::Add(
                Box::new(u_n(FaceSide::Owner)),
                Box::new(c_eff(FaceSide::Owner)),
            )),
            Box::new(S::Add(
                Box::new(u_n(FaceSide::Neighbor)),
                Box::new(c_eff(FaceSide::Neighbor)),
            )),
        )),
    );
    let a_minus = S::Min(
        Box::new(S::lit(0.0)),
        Box::new(S::Min(
            Box::new(S::Sub(
                Box::new(u_n(FaceSide::Owner)),
                Box::new(c_eff(FaceSide::Owner)),
            )),
            Box::new(S::Sub(
                Box::new(u_n(FaceSide::Neighbor)),
                Box::new(c_eff(FaceSide::Neighbor)),
            )),
        )),
    );

    // Viscous work term for total energy (OpenFOAM rhoCentralFoam):
    //   solve(ddt(rhoE) + div(phiEp) - div(sigmaDotU))
    // where:
    //   sigmaDotU = ( muEff*magSf*snGrad(U) + dotInterpolate(Sf, tauMC) ) & (a_pos*U_pos + a_neg*U_neg)
    //
    // We add this as a *face-flux correction* (same value on both sides) so it is not
    // upwinded/dissipated by the central-upwind blending.
    let denom_speed = S::Max(
        Box::new(S::Sub(Box::new(a_plus.clone()), Box::new(a_minus.clone()))),
        Box::new(S::lit(1e-6)),
    );
    let a_pos = S::Div(Box::new(a_plus.clone()), Box::new(denom_speed));
    let a_neg = S::Sub(Box::new(S::lit(1.0)), Box::new(a_pos.clone()));
    let u_sigma = V::Add(
        Box::new(V::MulScalar(
            Box::new(u_vec(FaceSide::Owner)),
            Box::new(a_pos.clone()),
        )),
        Box::new(V::MulScalar(
            Box::new(u_vec(FaceSide::Neighbor)),
            Box::new(a_neg.clone()),
        )),
    );

    // Approximate `snGrad(U)` via (grad(U) · n) at the face (matches Gauss on orthogonal meshes).
    let sn_grad_u_x_face = S::Dot(Box::new(grad_u_x_face.clone()), Box::new(V::normal()));
    let sn_grad_u_y_face = S::Dot(Box::new(grad_u_y_face.clone()), Box::new(V::normal()));
    let sn_grad_u_face = V::vec2(sn_grad_u_x_face, sn_grad_u_y_face);

    let traction_tau = V::vec2(tau_mc_dot_n_x.clone(), tau_mc_dot_n_y.clone());
    let traction_lapl = V::MulScalar(Box::new(sn_grad_u_face), Box::new(S::constant("viscosity")));
    let traction = V::Add(Box::new(traction_lapl), Box::new(traction_tau));
    let sigma_dot_u_per_area = S::Dot(Box::new(traction), Box::new(u_sigma));

    let flux_energy = |side: FaceSide| {
        S::Sub(
            Box::new(S::Mul(
                Box::new(S::Add(Box::new(rho_e(side)), Box::new(p(side)))),
                Box::new(u_n(side)),
            )),
            Box::new(sigma_dot_u_per_area.clone()),
        )
    };

    let mut u_left = Vec::new();
    let mut u_right = Vec::new();
    let mut flux_left = Vec::new();
    let mut flux_right = Vec::new();

    for name in &components {
        if name == rho_name {
            u_left.push(rho_diss(FaceSide::Owner));
            u_right.push(rho_diss(FaceSide::Neighbor));
            flux_left.push(flux_mass(FaceSide::Owner));
            flux_right.push(flux_mass(FaceSide::Neighbor));
        } else if name == &format!("{rho_u_name}_x") {
            u_left.push(rho_u_x(FaceSide::Owner));
            u_right.push(rho_u_x(FaceSide::Neighbor));
            flux_left.push(flux_mom_x(FaceSide::Owner));
            flux_right.push(flux_mom_x(FaceSide::Neighbor));
        } else if name == &format!("{rho_u_name}_y") {
            u_left.push(rho_u_y(FaceSide::Owner));
            u_right.push(rho_u_y(FaceSide::Neighbor));
            flux_left.push(flux_mom_y(FaceSide::Owner));
            flux_right.push(flux_mom_y(FaceSide::Neighbor));
        } else if name == rho_e_name {
            u_left.push(rho_e(FaceSide::Owner));
            u_right.push(rho_e(FaceSide::Neighbor));
            flux_left.push(flux_energy(FaceSide::Owner));
            flux_right.push(flux_energy(FaceSide::Neighbor));
        } else {
            // Auxiliary coupled unknowns (e.g. primitive fields coupled via diffusion/constraints)
            // get zero face flux by default.
            u_left.push(S::lit(0.0));
            u_right.push(S::lit(0.0));
            flux_left.push(S::lit(0.0));
            flux_right.push(S::lit(0.0));
        }
    }

    Ok(FluxModuleKernelSpec::CentralUpwind {
        reconstruction,
        components,
        u_left,
        u_right,
        flux_left,
        flux_right,
        a_plus,
        a_minus,
    })
}
