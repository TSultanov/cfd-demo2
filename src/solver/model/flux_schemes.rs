use crate::solver::ir::{
    FaceScalarExpr as S, FaceSide, FaceVec2Expr as V, FluxLayout, FluxModuleKernelSpec, LimiterSpec,
};
use crate::solver::model::backend::algebraic::AlgExpr;
use crate::solver::model::backend::ast::EquationSystem;
use crate::solver::model::flux_module::FluxSchemeSpec;
use crate::solver::scheme::Scheme;

use crate::solver::ir::reconstruction::{
    limited_linear_face_value, quick_face_value, FaceExprBuilder,
};

/// Declaration of a compressible conservation system for the central-upwind
/// (Kurganov-style) flux derivation: which state fields play which role,
/// plus the EOS relations as algebraic expressions. The derivation supplies
/// the numerics (reconstruction schemes, Kurganov wave splitting, low-Mach
/// preconditioning, OpenFOAM-matching viscous corrections); the declaration
/// supplies the physics.
#[derive(Debug, Clone, PartialEq)]
pub struct CentralUpwindDecl {
    /// Conserved density field.
    pub density: &'static str,
    /// Conserved momentum-density field (Vector2).
    pub momentum: &'static str,
    /// Conserved total-energy-density field.
    pub energy: &'static str,
    /// Primitive temperature field (drives acoustic reconstruction).
    pub temperature: &'static str,
    /// Primitive velocity field (Vector2).
    pub velocity: &'static str,
    /// Primitive pressure field (named by the generalized wave speed).
    pub pressure_field: &'static str,
    /// Face pressure from reconstructed conserved variables. Scalar atoms may
    /// reference `density` and `energy`; `mag_sqr(momentum)` is lowered from
    /// the same scheme/limiter's reconstructed momentum vector.
    pub pressure: AlgExpr,
    /// Squared acoustic speed for the Kurganov wave bounds; atom:
    /// `temperature` field (e.g. `gamma * R * T`).
    pub wave_speed_sq: AlgExpr,
    /// Generalized squared wave speed for low-Mach dissipation scaling;
    /// atoms: `pressure_field` (mapped to the reconstructed face pressure)
    /// and `density` (e.g. `gamma * p / rho + dp_drho`).
    pub generalized_wave_speed_sq: AlgExpr,
}

/// Lower a (cell-)algebraic expression to a face expression by mapping its
/// field atoms through `map_field` (typically to reconstructed face states
/// of a chosen side). Params become uniform constants; vector `mag_sqr`
/// atoms are supplied separately so the caller must choose an explicitly
/// reconstructed vector rather than accidentally reading a raw cell value.
fn lower_alg_to_face(
    expr: &AlgExpr,
    map_field: &dyn Fn(&str) -> Result<S, String>,
    map_mag_sqr: &dyn Fn(&str) -> Result<S, String>,
) -> Result<S, String> {
    let rec = |e: &AlgExpr| lower_alg_to_face(e, map_field, map_mag_sqr);
    Ok(match expr {
        AlgExpr::Constant { value, .. } => S::lit(*value as f32),
        AlgExpr::Param(p) => S::constant(p.name()),
        AlgExpr::Field(f) => map_field(f.name())?,
        AlgExpr::MagSqr(f) => map_mag_sqr(f.name())?,
        AlgExpr::Mul(a, b) => S::Mul(Box::new(rec(a)?), Box::new(rec(b)?)),
        AlgExpr::Div(a, b) => S::Div(Box::new(rec(a)?), Box::new(rec(b)?)),
        AlgExpr::Add(a, b) => S::Add(Box::new(rec(a)?), Box::new(rec(b)?)),
        AlgExpr::Sub(a, b) => S::Sub(Box::new(rec(a)?), Box::new(rec(b)?)),
        AlgExpr::Neg(a) => S::Neg(Box::new(rec(a)?)),
    })
}

pub fn lower_flux_scheme(
    flux_scheme: &FluxSchemeSpec,
    system: &EquationSystem,
    reconstruction: Scheme,
) -> Result<FluxModuleKernelSpec, String> {
    match flux_scheme {
        FluxSchemeSpec::CentralUpwind(decl) => derive_central_upwind(system, decl, reconstruction),
    }
}

fn derive_central_upwind(
    system: &EquationSystem,
    decl: &CentralUpwindDecl,
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

    // Field roles from the declaration.
    let rho_name = decl.density;
    let rho_u_name = decl.momentum;
    let rho_e_name = decl.energy;

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
    let rho_e_raw = |side: FaceSide| S::state(side, rho_e_name);
    let t_raw = |side: FaceSide| S::state(side, decl.temperature);

    // Derived gradient-field names (state-layout convention: `grad_<field>`).
    let grad_rho_name = format!("grad_{}", rho_name);
    let grad_rho_e_name = format!("grad_{}", rho_e_name);
    let grad_t_name = format!("grad_{}", decl.temperature);
    let grad_rho_u_x_name = format!("grad_{}_x", rho_u_name);
    let grad_rho_u_y_name = format!("grad_{}_y", rho_u_name);
    let grad_u_x_name = format!("grad_{}_x", decl.velocity);
    let grad_u_y_name = format!("grad_{}_y", decl.velocity);

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

    // Minmod limiter in the same OpenFOAM NVDTVD ratio convention (r = 2*gradcf/gradf - 1):
    //   psi(r) = max(0, min(1, r))
    // Caps psi at 1 (no anti-diffusion for r>1, where vanLeer rises toward 2 and adds a
    // compressive/destabilizing correction) and is more dissipative for r<1. min/max already
    // bound the output to [0,1], so the vanLeer 1000x guard clamp is unnecessary here.
    let minmod_limiter = |gradf: S, gradcf: S| {
        let gradf2 = S::Mul(Box::new(gradf.clone()), Box::new(gradf.clone()));
        let ratio = S::Div(
            Box::new(S::Mul(Box::new(gradcf), Box::new(gradf.clone()))),
            Box::new(S::Add(Box::new(gradf2), Box::new(eps2.clone()))),
        );
        let r = S::Sub(
            Box::new(S::Mul(Box::new(S::lit(2.0)), Box::new(ratio))),
            Box::new(S::lit(1.0)),
        );
        S::Max(
            Box::new(S::lit(0.0)),
            Box::new(S::Min(Box::new(r), Box::new(S::lit(1.0)))),
        )
    };

    let d = V::Sub(
        Box::new(V::cell_to_face(FaceSide::Owner)),
        Box::new(V::cell_to_face(FaceSide::Neighbor)),
    );

    let reconstruct_limited_scalar = |limiter_fn: &dyn Fn(S, S) -> S,
                                      side: FaceSide,
                                      phi_p: S,
                                      phi_n: S,
                                      grad_p: V,
                                      grad_n: V|
     -> S {
        let gradf = S::Sub(Box::new(phi_n.clone()), Box::new(phi_p.clone()));
        let grad = if side == FaceSide::Owner {
            grad_p
        } else {
            grad_n
        };
        let gradcf = S::Dot(Box::new(d.clone()), Box::new(grad));
        let limiter = limiter_fn(gradf.clone(), gradcf);
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

    let reconstruct_limited_vec2 = |limiter_fn: &dyn Fn(S, S) -> S,
                                    side: FaceSide,
                                    phi_p: V,
                                    phi_n: V,
                                    grad_px: V,
                                    grad_py: V,
                                    grad_nx: V,
                                    grad_ny: V|
     -> V {
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
        let limiter = limiter_fn(gradf, gradcf);
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
        Scheme::SecondOrderUpwindVanLeer => reconstruct_limited_scalar(
            &vanleer_limiter,
            side,
            rho_raw(FaceSide::Owner),
            rho_raw(FaceSide::Neighbor),
            V::state_vec2(FaceSide::Owner, grad_rho_name.clone()),
            V::state_vec2(FaceSide::Neighbor, grad_rho_name.clone()),
        ),
        Scheme::SecondOrderUpwindMinMod => reconstruct_limited_scalar(
            &minmod_limiter,
            side,
            rho_raw(FaceSide::Owner),
            rho_raw(FaceSide::Neighbor),
            V::state_vec2(FaceSide::Owner, grad_rho_name.clone()),
            V::state_vec2(FaceSide::Neighbor, grad_rho_name.clone()),
        ),
        _ => reconstruct_scalar(
            side,
            rho_raw(side),
            rho_raw(other_side(side)),
            V::state_vec2(side, grad_rho_name.clone()),
        ),
    };

    // Total energy is a conserved unknown for every EOS family. Reconstruct
    // that authoritative value directly; deriving it back from p/(gamma-1)
    // is equivalent only for a perfectly consistent ideal-gas state and is
    // singular for barotropic closures where gamma-1 == 0.
    let rho_e = |side: FaceSide| match reconstruction {
        Scheme::SecondOrderUpwindVanLeer => reconstruct_limited_scalar(
            &vanleer_limiter,
            side,
            rho_e_raw(FaceSide::Owner),
            rho_e_raw(FaceSide::Neighbor),
            V::state_vec2(FaceSide::Owner, grad_rho_e_name.clone()),
            V::state_vec2(FaceSide::Neighbor, grad_rho_e_name.clone()),
        ),
        Scheme::SecondOrderUpwindMinMod => reconstruct_limited_scalar(
            &minmod_limiter,
            side,
            rho_e_raw(FaceSide::Owner),
            rho_e_raw(FaceSide::Neighbor),
            V::state_vec2(FaceSide::Owner, grad_rho_e_name.clone()),
            V::state_vec2(FaceSide::Neighbor, grad_rho_e_name.clone()),
        ),
        _ => reconstruct_scalar(
            side,
            rho_e_raw(side),
            rho_e_raw(other_side(side)),
            V::state_vec2(side, grad_rho_e_name.clone()),
        ),
    };

    // Match OpenFOAM's rhoCentralFoam: reconstruct conserved momentum `rhoU` using `vanLeerV`
    // and derive face velocity as `U = rhoU/rho`.
    let rho_u = |side: FaceSide| match reconstruction {
        Scheme::SecondOrderUpwindVanLeer => reconstruct_limited_vec2(
            &vanleer_limiter,
            side,
            V::state_vec2(FaceSide::Owner, rho_u_name),
            V::state_vec2(FaceSide::Neighbor, rho_u_name),
            V::state_vec2(FaceSide::Owner, grad_rho_u_x_name.clone()),
            V::state_vec2(FaceSide::Owner, grad_rho_u_y_name.clone()),
            V::state_vec2(FaceSide::Neighbor, grad_rho_u_x_name.clone()),
            V::state_vec2(FaceSide::Neighbor, grad_rho_u_y_name.clone()),
        ),
        Scheme::SecondOrderUpwindMinMod => reconstruct_limited_vec2(
            &minmod_limiter,
            side,
            V::state_vec2(FaceSide::Owner, rho_u_name),
            V::state_vec2(FaceSide::Neighbor, rho_u_name),
            V::state_vec2(FaceSide::Owner, grad_rho_u_x_name.clone()),
            V::state_vec2(FaceSide::Owner, grad_rho_u_y_name.clone()),
            V::state_vec2(FaceSide::Neighbor, grad_rho_u_x_name.clone()),
            V::state_vec2(FaceSide::Neighbor, grad_rho_u_y_name.clone()),
        ),
        _ => {
            let rho_u_owner = V::state_vec2(side, rho_u_name);
            let rho_u_other = V::state_vec2(other_side(side), rho_u_name);
            let x = reconstruct_scalar(
                side,
                S::Dot(Box::new(rho_u_owner.clone()), Box::new(ex.clone())),
                S::Dot(Box::new(rho_u_other.clone()), Box::new(ex.clone())),
                V::state_vec2(side, grad_rho_u_x_name.clone()),
            );
            let y = reconstruct_scalar(
                side,
                S::Dot(Box::new(rho_u_owner), Box::new(ey.clone())),
                S::Dot(Box::new(rho_u_other), Box::new(ey.clone())),
                V::state_vec2(side, grad_rho_u_y_name.clone()),
            );
            V::vec2(x, y)
        }
    };

    let u_vec = |side: FaceSide| {
        // Gauge storage: rho(side) is the STORED (gauge) reconstruction; the
        // velocity divides by the ABSOLUTE density (reference zero when off).
        let rho_abs = S::Add(
            Box::new(rho(side)),
            Box::new(S::constant("eos_gauge_rho_ref")),
        );
        let inv_rho = S::Div(Box::new(S::lit(1.0)), Box::new(rho_abs));
        V::MulScalar(Box::new(rho_u(side)), Box::new(inv_rho))
    };

    // Evaluate the declared EOS over one coherent reconstructed conserved
    // state. Every atom uses the selected scheme/limiter on rho, rho_u, and
    // rho_e before the nonlinear pressure closure is applied. This avoids the
    // non-commuting rho_f*T_f surrogate that violates an affine barotropic EOS
    // and can disagree with conserved energy for ideal-gas high-order faces.
    let p_lowered = |side: FaceSide| -> Result<S, String> {
        lower_alg_to_face(
            &decl.pressure,
            &|name| {
                if name == rho_name {
                    Ok(rho(side))
                } else if name == rho_e_name {
                    Ok(rho_e(side))
                } else {
                    Err(format!(
                        "pressure declaration references scalar '{name}', expected only '{}' or \
                         '{}'",
                        rho_name, rho_e_name
                    ))
                }
            },
            &|name| {
                if name == rho_u_name {
                    let momentum = rho_u(side);
                    Ok(S::Dot(Box::new(momentum.clone()), Box::new(momentum)))
                } else {
                    Err(format!(
                        "pressure declaration references mag_sqr({name}), expected only \
                         mag_sqr({rho_u_name})"
                    ))
                }
            },
        )
    };
    let p_own = p_lowered(FaceSide::Owner)?;
    let p_neigh = p_lowered(FaceSide::Neighbor)?;
    let p = |side: FaceSide| {
        if side == FaceSide::Owner {
            p_own.clone()
        } else {
            p_neigh.clone()
        }
    };

    let rho_u_x = |side: FaceSide| S::Dot(Box::new(rho_u(side)), Box::new(ex.clone()));
    let rho_u_y = |side: FaceSide| S::Dot(Box::new(rho_u(side)), Box::new(ey.clone()));

    // Generalized squared wave speed over reconstructed states, from the
    // declaration (atoms: pressure_field -> p(side), density -> rho(side)).
    let c2_lowered = |side: FaceSide| -> Result<S, String> {
        lower_alg_to_face(
            &decl.generalized_wave_speed_sq,
            &|name| {
                if name == decl.pressure_field {
                    Ok(p(side))
                } else if name == rho_name {
                    Ok(rho(side))
                } else {
                    Err(format!(
                        "generalized wave-speed declaration references '{name}', expected only \
                         '{}' or '{}'",
                        decl.pressure_field, rho_name
                    ))
                }
            },
            &|name| {
                Err(format!(
                    "generalized wave-speed declaration unexpectedly references mag_sqr({name})"
                ))
            },
        )
    };
    let c2_own = c2_lowered(FaceSide::Owner)?;
    let c2_neigh = c2_lowered(FaceSide::Neighbor)?;
    let c2 = |side: FaceSide| {
        if side == FaceSide::Owner {
            c2_own.clone()
        } else {
            c2_neigh.clone()
        }
    };

    // Low-Mach preconditioning should be driven by a representative local Mach number.
    //
    // Using only the *normal* velocity can drive `c_eff` unrealistically low for shear-driven
    // flows (e.g., lid-driven cavity, wall-bounded shear) where `u_n ≈ 0` but `|u|` is not.
    // This can severely reduce dissipation and destabilize the central-upwind flux.
    let u_mag2 = |side: FaceSide| {
        let u = u_vec(side);
        S::Dot(Box::new(u.clone()), Box::new(u))
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
        S::Min(Box::new(u_mag2(side)), Box::new(c2_side))
    };

    let c_eff2_weiss_smith = |side: FaceSide| {
        let c2_side = c2(side);
        let floor = S::Mul(
            Box::new(low_mach_theta_floor.clone()),
            Box::new(c2_side.clone()),
        );
        S::Min(
            Box::new(S::Max(Box::new(u_mag2(side)), Box::new(floor))),
            Box::new(c2_side),
        )
    };

    // Low-Mach preconditioning in `cfd2` is intended for pseudo-transient/dual-time stepping
    // (`dtau > 0`). In time-accurate mode (`dtau == 0`), it can destabilize low-Mach transient
    // cases and distort transient acoustics.
    //
    // Gate preconditioning on `dtau > 0` so the same solver config can be used safely across
    // both modes.
    let dtau = S::constant("dtau");
    let dtau_eps = S::lit(1e-12);
    let dtau_enable = S::Div(
        Box::new(dtau.clone()),
        Box::new(S::Add(Box::new(dtau), Box::new(dtau_eps))),
    );
    let dtau_enable = S::Min(
        Box::new(S::lit(1.0)),
        Box::new(S::Max(Box::new(S::lit(0.0)), Box::new(dtau_enable))),
    );
    let dtau_disable = S::Sub(Box::new(S::lit(1.0)), Box::new(dtau_enable.clone()));

    // Effective sound speed squared for preconditioning (blended across models).
    let c_eff2_low_mach = |side: FaceSide| {
        let c2_side = c2(side);
        S::Add(
            Box::new(S::Add(
                Box::new(S::Mul(Box::new(w_off.clone()), Box::new(c2_side.clone()))),
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

    // Disable preconditioning in time-accurate mode by falling back to the physical wave speed.
    let c_eff2 = |side: FaceSide| {
        let c2_side = c2(side);
        S::Add(
            Box::new(S::Mul(
                Box::new(dtau_enable.clone()),
                Box::new(c_eff2_low_mach(side)),
            )),
            Box::new(S::Mul(Box::new(dtau_disable.clone()), Box::new(c2_side))),
        )
    };

    // --- Low-Mach pressure coupling ---
    //
    // At low Mach numbers, a purely density-based continuity flux can decouple pressure on
    // collocated grids and produce checkerboarding. Add a preconditioning-gated pressure
    // perturbation contribution to the density state used by the central-upwind dissipation
    // term (leaves the physical mass flux unchanged).
    let pressure_coupling_alpha = S::low_mach_pressure_coupling_alpha();
    let low_mach_enabled = S::Mul(
        Box::new(dtau_enable.clone()),
        Box::new(S::Sub(Box::new(S::lit(1.0)), Box::new(w_off.clone()))),
    );

    // Convert a pressure perturbation to an equivalent density perturbation using the *physical*
    // compressibility (ρ' ≈ p'/c^2).
    //
    // Using the *preconditioned* wave speed here can severely over-amplify the coupling at very
    // low Mach numbers (since c_eff^2 ~ O(|u|^2)), causing unphysical density states and solver
    // instability.
    let c_couple2_safe = |side: FaceSide| S::Max(Box::new(c2(side)), Box::new(S::lit(1e-12)));
    let n_x = S::Dot(Box::new(V::normal()), Box::new(ex.clone()));
    let n_y = S::Dot(Box::new(V::normal()), Box::new(ey.clone()));

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
        Box::new(V::state_vec2(FaceSide::Neighbor, grad_u_x_name.clone())),
        Box::new(V::state_vec2(FaceSide::Owner, grad_u_x_name.clone())),
    );
    let grad_u_y_face_raw = V::Lerp(
        Box::new(V::state_vec2(FaceSide::Neighbor, grad_u_y_name.clone())),
        Box::new(V::state_vec2(FaceSide::Owner, grad_u_y_name.clone())),
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

    let u_face = V::state_vec2(FaceSide::Owner, decl.velocity);
    let u_cell = V::cell_state_vec2(FaceSide::Owner, decl.velocity);
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

    let two_thirds = S::lit(2.0 / 3.0);

    // OpenFOAM rhoCentralFoam viscous split uses:
    //   tauMC = muEff*dev2(T(fvc::grad(U)))
    //   - fvm::laplacian(muEff, U)
    //   - fvc::div(tauMC)
    //
    // dev2(A) = A - 2/3 tr(A) I, so tauMC contains ONE velocity-gradient
    // tensor (the transpose part); the full deviatoric Newtonian stress
    //   tau = mu * (grad(U) + grad(U)^T - 2/3 I div(U))
    // only arises as the SUM laplacian + tauMC. Since this model also
    // assembles the implicit `laplacian(mu, u)` (traction mu * (grad u_i).n
    // per component), the traction built here must be exactly the remainder
    //   tauMC . n = mu * ((J^T - 2/3 I div u) . n),   J_ij = du_i/dx_j
    // i.e. traction_x = mu*((dux_dx - 2/3 div)*n_x + duy_dx*n_y).
    //
    // The Gauss div scheme forms the face traction via `Sf & tauMC` (units: force).
    // Here we build the equivalent traction per unit area (tauMC · n) from face gradients,
    // then multiply by area when assembling integrated fluxes.
    // OpenFOAM's `dotInterpolate(Sf, tauMC)` interpolates the *cell-centered* tauMC tensor to
    // faces and then contracts with Sf. Mirror that by building tauMC from each side's stored
    // cell gradient and linearly interpolating the resulting traction.
    let tau_mc_dot_n_components = |side: FaceSide| {
        let grad_u_x_side = V::state_vec2(side, grad_u_x_name.clone());
        let grad_u_y_side = V::state_vec2(side, grad_u_y_name.clone());

        let dux_dx_s = S::Dot(Box::new(grad_u_x_side.clone()), Box::new(ex.clone()));
        let dux_dy_s = S::Dot(Box::new(grad_u_x_side), Box::new(ey.clone()));
        let duy_dx_s = S::Dot(Box::new(grad_u_y_side.clone()), Box::new(ex.clone()));
        let duy_dy_s = S::Dot(Box::new(grad_u_y_side), Box::new(ey.clone()));

        let div_u_s = S::Add(Box::new(dux_dx_s.clone()), Box::new(duy_dy_s.clone()));
        // tauMC row x: (dux_dx - 2/3 div, duy_dx); row y: (dux_dy, duy_dy - 2/3 div).
        let tau_xx_s = S::Sub(
            Box::new(dux_dx_s),
            Box::new(S::Mul(
                Box::new(two_thirds.clone()),
                Box::new(div_u_s.clone()),
            )),
        );
        let tau_yy_s = S::Sub(
            Box::new(duy_dy_s),
            Box::new(S::Mul(Box::new(two_thirds.clone()), Box::new(div_u_s))),
        );

        let traction_x = S::Mul(
            Box::new(visc_mu.clone()),
            Box::new(S::Add(
                Box::new(S::Mul(Box::new(tau_xx_s), Box::new(n_x.clone()))),
                Box::new(S::Mul(Box::new(duy_dx_s), Box::new(n_y.clone()))),
            )),
        );
        let traction_y = S::Mul(
            Box::new(visc_mu.clone()),
            Box::new(S::Add(
                Box::new(S::Mul(Box::new(dux_dy_s), Box::new(n_x.clone()))),
                Box::new(S::Mul(Box::new(tau_yy_s), Box::new(n_y.clone()))),
            )),
        );

        (traction_x, traction_y)
    };

    let (tau_mc_dot_n_x_own, tau_mc_dot_n_y_own) = tau_mc_dot_n_components(FaceSide::Owner);
    let (tau_mc_dot_n_x_neigh, tau_mc_dot_n_y_neigh) = tau_mc_dot_n_components(FaceSide::Neighbor);

    let tau_mc_dot_n_x = S::Lerp(Box::new(tau_mc_dot_n_x_neigh), Box::new(tau_mc_dot_n_x_own));
    let tau_mc_dot_n_y = S::Lerp(Box::new(tau_mc_dot_n_y_neigh), Box::new(tau_mc_dot_n_y_own));

    // Viscous work term for total energy (OpenFOAM rhoCentralFoam):
    //   solve(ddt(rhoE) + div(phiEp) - div(sigmaDotU))
    // where:
    //   sigmaDotU = ( interpolate(muEff)*magSf*snGrad(U) + dotInterpolate(Sf, tauMC) )
    //              & (a_pos*U_pos + a_neg*U_neg)
    //
    // We compute `sigmaDotU` later in the integrated-flux section using the same `a_pos/a_neg`
    // weights as OpenFOAM's Kurganov flux split.

    // --- OpenFOAM rhoCentralFoam-style integrated Kurganov fluxes ---
    //
    // OpenFOAM forms Kurganov fluxes using Sf-weighted wave speeds and blends the pos/neg
    // directed-interpolated states:
    //   phi   = aphiv_pos*rho_pos + aphiv_neg*rho_neg
    //   phiUp = aphiv_pos*rhoU_pos + aphiv_neg*rhoU_neg + (a_pos*p_pos + a_neg*p_neg)*Sf
    //   phiEp = aphiv_pos*(rhoE_pos + p_pos) + aphiv_neg*(rhoE_neg + p_neg) + aSf*(p_pos - p_neg)
    // and solves energy with an explicit viscous-work correction -div(sigmaDotU).
    //
    // We compute the same integrated face fluxes directly and return them as
    // `ScalarPerComponent` entries so WGSL writes them verbatim into the packed flux table.
    let sf = V::MulScalar(Box::new(V::normal()), Box::new(S::area()));
    let sf_x = S::Mul(Box::new(S::area()), Box::new(n_x.clone()));
    let sf_y = S::Mul(Box::new(S::area()), Box::new(n_y.clone()));

    let phiv_pos = S::Dot(Box::new(u_vec(FaceSide::Owner)), Box::new(sf.clone()));
    let phiv_neg = S::Dot(Box::new(u_vec(FaceSide::Neighbor)), Box::new(sf));

    // Match OpenFOAM: reconstruct the acoustic speed `c` as a scalar field using the same
    // `reconstruct(T)` scheme (vanLeer) and then multiply by `magSf`.
    //
    // `c = sqrt(declared wave_speed_sq)` over raw cell temperatures and
    // runtime EOS constants. For the production declaration this is the full
    // physical `sqrt(gamma*R*T + dp_drho)`: ideal gas is algebraically and
    // numerically identical at dp_drho=0, while a linear EOS obtains its
    // nonzero acoustic base.
    let c_cell_lowered = |side: FaceSide| -> Result<S, String> {
        Ok(S::Sqrt(Box::new(lower_alg_to_face(
            &decl.wave_speed_sq,
            &|name| {
                if name == decl.temperature {
                    Ok(t_raw(side))
                } else {
                    Err(format!(
                        "wave-speed declaration references '{name}', expected only '{}'",
                        decl.temperature
                    ))
                }
            },
            &|name| {
                Err(format!(
                    "wave-speed declaration unexpectedly references mag_sqr({name})"
                ))
            },
        )?)))
    };
    let c_cell_own = c_cell_lowered(FaceSide::Owner)?;
    let c_cell_neigh = c_cell_lowered(FaceSide::Neighbor)?;
    let c_cell = |side: FaceSide| {
        if side == FaceSide::Owner {
            c_cell_own.clone()
        } else {
            c_cell_neigh.clone()
        }
    };
    // Analytic gradient of the runtime-EOS acoustic speed:
    //   grad(c) = 0.5 * (gamma * R / c) * grad(T)
    // for c²=gamma*R*T+dp_drho. The affine dp_drho term is spatially constant,
    // so it contributes no gradient. A more general non-ideal EOS would need
    // its own gradient form here (no symbolic differentiation by design).
    let grad_c = |side: FaceSide| {
        let denom = S::Max(Box::new(c_cell(side)), Box::new(S::lit(1e-12)));
        let factor = S::Div(
            Box::new(S::Mul(
                Box::new(S::lit(0.5)),
                Box::new(S::Mul(
                    Box::new(S::constant("eos_gamma")),
                    Box::new(S::constant("eos_r")),
                )),
            )),
            Box::new(denom),
        );
        V::MulScalar(
            Box::new(V::state_vec2(side, grad_t_name.clone())),
            Box::new(factor),
        )
    };
    let c_face_raw = |side: FaceSide| match reconstruction {
        Scheme::SecondOrderUpwindVanLeer => reconstruct_limited_scalar(
            &vanleer_limiter,
            side,
            c_cell(FaceSide::Owner),
            c_cell(FaceSide::Neighbor),
            grad_c(FaceSide::Owner),
            grad_c(FaceSide::Neighbor),
        ),
        Scheme::SecondOrderUpwindMinMod => reconstruct_limited_scalar(
            &minmod_limiter,
            side,
            c_cell(FaceSide::Owner),
            c_cell(FaceSide::Neighbor),
            grad_c(FaceSide::Owner),
            grad_c(FaceSide::Neighbor),
        ),
        _ => reconstruct_scalar(side, c_cell(side), c_cell(other_side(side)), grad_c(side)),
    };

    // Apply low-Mach preconditioning to the acoustic speed for dissipation scaling.
    // scale = sqrt(c_eff^2 / c^2) gives the ratio of effective to physical wave speed.
    let scale_factor = |side: FaceSide| {
        let c2_side = c2(side);
        S::Sqrt(Box::new(S::Div(
            Box::new(c_eff2(side)),
            Box::new(S::Max(Box::new(c2_side), Box::new(S::lit(1e-12)))),
        )))
    };

    let c_face_pre =
        |side: FaceSide| S::Mul(Box::new(c_face_raw(side)), Box::new(scale_factor(side)));

    let c_sf_pos = S::Mul(Box::new(c_face_pre(FaceSide::Owner)), Box::new(S::area()));
    let c_sf_neg = S::Mul(
        Box::new(c_face_pre(FaceSide::Neighbor)),
        Box::new(S::area()),
    );

    let ap = S::Max(
        Box::new(S::Max(
            Box::new(S::Add(
                Box::new(phiv_pos.clone()),
                Box::new(c_sf_pos.clone()),
            )),
            Box::new(S::Add(
                Box::new(phiv_neg.clone()),
                Box::new(c_sf_neg.clone()),
            )),
        )),
        Box::new(S::lit(0.0)),
    );
    let am = S::Min(
        Box::new(S::Min(
            Box::new(S::Sub(
                Box::new(phiv_pos.clone()),
                Box::new(c_sf_pos.clone()),
            )),
            Box::new(S::Sub(
                Box::new(phiv_neg.clone()),
                Box::new(c_sf_neg.clone()),
            )),
        )),
        Box::new(S::lit(0.0)),
    );

    let denom_sf = S::Max(
        Box::new(S::Sub(Box::new(ap.clone()), Box::new(am.clone()))),
        Box::new(S::lit(1e-6)),
    );
    let a_pos_sf = S::Div(Box::new(ap.clone()), Box::new(denom_sf.clone()));
    let a_neg_sf = S::Sub(Box::new(S::lit(1.0)), Box::new(a_pos_sf.clone()));
    let a_sf = S::Mul(Box::new(am.clone()), Box::new(a_pos_sf.clone()));

    // Approximate `snGrad(U)` via (grad(U) · n) at the face (matches Gauss on orthogonal meshes).
    let sn_grad_u_x_face = S::Dot(Box::new(grad_u_x_face.clone()), Box::new(V::normal()));
    let sn_grad_u_y_face = S::Dot(Box::new(grad_u_y_face.clone()), Box::new(V::normal()));
    let sn_grad_u_face = V::vec2(sn_grad_u_x_face, sn_grad_u_y_face);

    // OpenFOAM uses the same `a_pos/a_neg` (Sf-weighted split) for `sigmaDotU`.
    let u_sigma = V::Add(
        Box::new(V::MulScalar(
            Box::new(u_vec(FaceSide::Owner)),
            Box::new(a_pos_sf.clone()),
        )),
        Box::new(V::MulScalar(
            Box::new(u_vec(FaceSide::Neighbor)),
            Box::new(a_neg_sf.clone()),
        )),
    );

    let traction_tau = V::vec2(tau_mc_dot_n_x.clone(), tau_mc_dot_n_y.clone());
    let traction_lapl = V::MulScalar(Box::new(sn_grad_u_face), Box::new(S::constant("viscosity")));
    let traction = V::Add(Box::new(traction_lapl), Box::new(traction_tau));
    let sigma_dot_u_per_area = S::Dot(Box::new(traction), Box::new(u_sigma));

    let aphiv_pos = S::Sub(
        Box::new(S::Mul(
            Box::new(phiv_pos.clone()),
            Box::new(a_pos_sf.clone()),
        )),
        Box::new(a_sf.clone()),
    );
    let aphiv_neg = S::Add(
        Box::new(S::Mul(
            Box::new(phiv_neg.clone()),
            Box::new(a_neg_sf.clone()),
        )),
        Box::new(a_sf.clone()),
    );

    // Analytic `aphiv_pos + aphiv_neg`: the +/- a_sf dissipation parts cancel
    // SYMBOLICALLY. This multiplies the constant gauge references in the
    // regrouped integrated fluxes below, so the reference part of a
    // gauge-stored conserved state contributes its exact advective flux
    // without round-tripping through c-scale dissipation arithmetic in f32
    // (and the whole term vanishes exactly at zero references).
    let aphiv_sum = S::Add(
        Box::new(S::Mul(
            Box::new(phiv_pos.clone()),
            Box::new(a_pos_sf.clone()),
        )),
        Box::new(S::Mul(
            Box::new(phiv_neg.clone()),
            Box::new(a_neg_sf.clone()),
        )),
    );

    // --- Low-Mach pressure coupling for mass flux ---
    // Add pressure perturbation contribution to prevent checkerboarding at low Mach numbers.
    // The coupling adds a term: alpha * (p_pos - p_neg) / c^2 * area
    // which acts as an artificial compressibility to couple pressure and velocity.
    let p_pos = p(FaceSide::Owner);
    let p_neg = p(FaceSide::Neighbor);
    let pressure_diff = S::Sub(Box::new(p_pos.clone()), Box::new(p_neg.clone()));

    // Pressure coupling term: rho' = alpha * (p' / c^2) with a face-averaged compressibility.
    //
    // Note: keep this symmetric across owner/neighbor. Using a single face term avoids
    // double-counting the coupling.
    let inv_c2_pos = S::Div(
        Box::new(S::lit(1.0)),
        Box::new(c_couple2_safe(FaceSide::Owner)),
    );
    let inv_c2_neg = S::Div(
        Box::new(S::lit(1.0)),
        Box::new(c_couple2_safe(FaceSide::Neighbor)),
    );
    let inv_c2_face = S::Mul(
        Box::new(S::lit(0.5)),
        Box::new(S::Add(Box::new(inv_c2_pos), Box::new(inv_c2_neg))),
    );
    let rho_couple = S::Mul(
        Box::new(S::Mul(
            Box::new(pressure_coupling_alpha.clone()),
            Box::new(pressure_diff.clone()),
        )),
        Box::new(inv_c2_face),
    );
    let phi_couple = S::Mul(
        Box::new(S::Mul(
            Box::new(rho_couple),
            Box::new(low_mach_enabled.clone()),
        )),
        Box::new(S::area()),
    );

    let phi = {
        let rho_pos = rho(FaceSide::Owner);
        let rho_neg = rho(FaceSide::Neighbor);
        // Gauge storage: absolute mass flux = flux of the stored deviation
        // plus the reference density advected by the analytic weight sum.
        let deviation = S::Add(
            Box::new(S::Mul(Box::new(aphiv_pos.clone()), Box::new(rho_pos))),
            Box::new(S::Mul(Box::new(aphiv_neg.clone()), Box::new(rho_neg))),
        );
        let reference = S::Mul(
            Box::new(S::constant("eos_gauge_rho_ref")),
            Box::new(aphiv_sum.clone()),
        );
        S::Add(
            Box::new(S::Add(Box::new(deviation), Box::new(reference))),
            Box::new(phi_couple),
        )
    };

    let phi_u_x = {
        let rho_u_pos = rho_u_x(FaceSide::Owner);
        let rho_u_neg = rho_u_x(FaceSide::Neighbor);
        S::Add(
            Box::new(S::Mul(Box::new(aphiv_pos.clone()), Box::new(rho_u_pos))),
            Box::new(S::Mul(Box::new(aphiv_neg.clone()), Box::new(rho_u_neg))),
        )
    };
    let phi_u_y = {
        let rho_u_pos = rho_u_y(FaceSide::Owner);
        let rho_u_neg = rho_u_y(FaceSide::Neighbor);
        S::Add(
            Box::new(S::Mul(Box::new(aphiv_pos.clone()), Box::new(rho_u_pos))),
            Box::new(S::Mul(Box::new(aphiv_neg.clone()), Box::new(rho_u_neg))),
        )
    };

    let p_blend = S::Add(
        Box::new(S::Mul(Box::new(a_pos_sf.clone()), Box::new(p_pos.clone()))),
        Box::new(S::Mul(Box::new(a_neg_sf.clone()), Box::new(p_neg.clone()))),
    );

    let phi_up_x = {
        let conv = S::Add(
            Box::new(phi_u_x),
            Box::new(S::Mul(Box::new(p_blend.clone()), Box::new(sf_x))),
        );
        // Subtract the explicit tauMC traction contribution (integrated over face area).
        S::Sub(
            Box::new(conv),
            Box::new(S::Mul(
                Box::new(tau_mc_dot_n_x.clone()),
                Box::new(S::area()),
            )),
        )
    };
    let phi_up_y = {
        let conv = S::Add(
            Box::new(phi_u_y),
            Box::new(S::Mul(Box::new(p_blend), Box::new(sf_y))),
        );
        S::Sub(
            Box::new(conv),
            Box::new(S::Mul(
                Box::new(tau_mc_dot_n_y.clone()),
                Box::new(S::area()),
            )),
        )
    };

    let phi_ep = {
        let rho_e_pos = rho_e(FaceSide::Owner);
        let rho_e_neg = rho_e(FaceSide::Neighbor);

        let term_pos = S::Add(Box::new(rho_e_pos), Box::new(p_pos.clone()));
        let term_neg = S::Add(Box::new(rho_e_neg), Box::new(p_neg.clone()));

        let deviation = S::Add(
            Box::new(S::Mul(Box::new(aphiv_pos), Box::new(term_pos))),
            Box::new(S::Mul(Box::new(aphiv_neg), Box::new(term_neg))),
        );
        // Gauge storage: the reference total enthalpy density
        // (gauge_e_ref + gauge_p_ref) advects by the analytic weight sum;
        // the a_sf pressure-jump dissipation below sees only differences and
        // is gauge-invariant as-is.
        let reference_enthalpy = S::Add(
            Box::new(S::constant("eos_gauge_e_ref")),
            Box::new(S::constant("eos_gauge_p_ref")),
        );
        let conv = S::Add(
            Box::new(deviation),
            Box::new(S::Mul(
                Box::new(reference_enthalpy),
                Box::new(aphiv_sum.clone()),
            )),
        );

        let pressure_jump = S::Sub(Box::new(p_pos), Box::new(p_neg));
        let phi_ep_conv = S::Add(
            Box::new(conv),
            Box::new(S::Mul(Box::new(a_sf), Box::new(pressure_jump))),
        );

        // Subtract viscous work (sigmaDotU) as an integrated face-power flux.
        S::Sub(
            Box::new(phi_ep_conv),
            Box::new(S::Mul(Box::new(sigma_dot_u_per_area), Box::new(S::area()))),
        )
    };

    // Biharmonic dissipation is handled implicitly in the matrix (auxiliary
    // `lap_<conserved>` unknowns + `laplacian(bih_eps4, lap_X)` per conserved
    // equation), not on the explicit flux here.

    let mut flux = Vec::new();
    for name in &components {
        if name == rho_name {
            flux.push(phi.clone());
        } else if name == &format!("{rho_u_name}_x") {
            flux.push(phi_up_x.clone());
        } else if name == &format!("{rho_u_name}_y") {
            flux.push(phi_up_y.clone());
        } else if name == rho_e_name {
            flux.push(phi_ep.clone());
        } else {
            // Auxiliary coupled unknowns (e.g. primitive fields coupled via diffusion/constraints)
            // get zero face flux by default.
            flux.push(S::lit(0.0));
        }
    }

    Ok(FluxModuleKernelSpec::ScalarPerComponent { components, flux })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_lit(expr: &S, value: f32) -> bool {
        matches!(expr, S::Literal(v) if *v == value)
    }

    fn contains_state(expr: &S, wanted: &str) -> bool {
        match expr {
            S::State { name, .. } | S::CellState { name, .. } => name == wanted,
            S::Add(a, b)
            | S::Sub(a, b)
            | S::Mul(a, b)
            | S::Div(a, b)
            | S::Max(a, b)
            | S::Min(a, b)
            | S::Lerp(a, b) => contains_state(a, wanted) || contains_state(b, wanted),
            S::Neg(a) | S::Abs(a) | S::Sqrt(a) => contains_state(a, wanted),
            S::Dot(_, _)
            | S::Literal(_)
            | S::Builtin(_)
            | S::Constant { .. }
            | S::LowMachParam(_)
            | S::BoundaryDirichlet { .. }
            | S::MeshFlux
            | S::Primitive { .. } => false,
        }
    }

    fn contains_constant(expr: &S, wanted: &str) -> bool {
        match expr {
            S::Constant { name } => name == wanted,
            S::Add(a, b)
            | S::Sub(a, b)
            | S::Mul(a, b)
            | S::Div(a, b)
            | S::Max(a, b)
            | S::Min(a, b)
            | S::Lerp(a, b) => contains_constant(a, wanted) || contains_constant(b, wanted),
            S::Neg(a) | S::Abs(a) | S::Sqrt(a) => contains_constant(a, wanted),
            S::Dot(_, _)
            | S::Literal(_)
            | S::Builtin(_)
            | S::LowMachParam(_)
            | S::State { .. }
            | S::CellState { .. }
            | S::BoundaryDirichlet { .. }
            | S::MeshFlux
            | S::Primitive { .. } => false,
        }
    }

    fn contains_low_mach_parameter(expr: &S) -> bool {
        match expr {
            S::LowMachParam(_) => true,
            S::Add(a, b)
            | S::Sub(a, b)
            | S::Mul(a, b)
            | S::Div(a, b)
            | S::Max(a, b)
            | S::Min(a, b)
            | S::Lerp(a, b) => {
                contains_low_mach_parameter(a) || contains_low_mach_parameter(b)
            }
            S::Neg(a) | S::Abs(a) | S::Sqrt(a) => contains_low_mach_parameter(a),
            S::Dot(_, _)
            | S::Literal(_)
            | S::Builtin(_)
            | S::Constant { .. }
            | S::State { .. }
            | S::CellState { .. }
            | S::BoundaryDirichlet { .. }
            | S::MeshFlux
            | S::Primitive { .. } => false,
        }
    }

    fn contains_physical_barotropic_acoustic_sqrt(expr: &S) -> bool {
        match expr {
            S::Sqrt(inner)
                if contains_constant(inner, "eos_dp_drho")
                    && !contains_low_mach_parameter(inner) =>
            {
                true
            }
            S::Add(a, b)
            | S::Sub(a, b)
            | S::Mul(a, b)
            | S::Div(a, b)
            | S::Max(a, b)
            | S::Min(a, b)
            | S::Lerp(a, b) => {
                contains_physical_barotropic_acoustic_sqrt(a)
                    || contains_physical_barotropic_acoustic_sqrt(b)
            }
            S::Neg(a) | S::Abs(a) | S::Sqrt(a) => {
                contains_physical_barotropic_acoustic_sqrt(a)
            }
            S::Dot(_, _)
            | S::Literal(_)
            | S::Builtin(_)
            | S::Constant { .. }
            | S::LowMachParam(_)
            | S::State { .. }
            | S::CellState { .. }
            | S::BoundaryDirichlet { .. }
            | S::MeshFlux
            | S::Primitive { .. } => false,
        }
    }

    fn contains_divisor_constant(expr: &S, wanted: &str) -> bool {
        fn is_constant_floor(expr: &S, wanted: &str) -> bool {
            match expr {
                S::Constant { name } => name == wanted,
                S::Max(a, b) | S::Min(a, b) => {
                    (is_constant_floor(a, wanted) && matches!(b.as_ref(), S::Literal(_)))
                        || (is_constant_floor(b, wanted)
                            && matches!(a.as_ref(), S::Literal(_)))
                }
                S::Neg(a) | S::Abs(a) => is_constant_floor(a, wanted),
                _ => false,
            }
        }

        match expr {
            S::Div(a, b) => {
                is_constant_floor(b, wanted)
                    || contains_divisor_constant(a, wanted)
                    || contains_divisor_constant(b, wanted)
            }
            S::Add(a, b)
            | S::Sub(a, b)
            | S::Mul(a, b)
            | S::Max(a, b)
            | S::Min(a, b)
            | S::Lerp(a, b) => {
                contains_divisor_constant(a, wanted)
                    || contains_divisor_constant(b, wanted)
            }
            S::Neg(a) | S::Abs(a) | S::Sqrt(a) => contains_divisor_constant(a, wanted),
            S::Dot(_, _)
            | S::Literal(_)
            | S::Builtin(_)
            | S::Constant { .. }
            | S::LowMachParam(_)
            | S::State { .. }
            | S::CellState { .. }
            | S::BoundaryDirichlet { .. }
            | S::MeshFlux
            | S::Primitive { .. } => false,
        }
    }

    #[test]
    fn compressible_flux_uses_conserved_energy_and_barotropic_acoustic_base() {
        let system = crate::solver::model::compressible_system();
        let spec = derive_central_upwind(
            &system,
            &crate::solver::model::compressible_central_upwind_decl(),
            Scheme::Upwind,
        )
        .expect("derive production compressible flux");
        let FluxModuleKernelSpec::ScalarPerComponent { components, flux } = spec else {
            panic!("compressible flux did not lower per component");
        };
        let energy = &flux[components
            .iter()
            .position(|name| name == "rho_e")
            .expect("rho_e flux component")];
        assert!(
            contains_state(energy, "rho_e"),
            "energy flux no longer reads the conserved energy"
        );
        assert!(
            !contains_divisor_constant(energy, "eos_gm1"),
            "energy flux reintroduced a singular division by gamma-1"
        );
        assert!(
            flux.iter()
                .any(contains_physical_barotropic_acoustic_sqrt),
            "physical Kurganov wave bound does not contain the barotropic sound-speed term"
        );
    }

    #[test]
    fn ideal_and_water_enthalpy_fluxes_are_finite_from_conserved_energy() {
        let rho = 1.2_f32;
        let velocity = 30.0_f32;
        let pressure = rho * 287.0 * 300.0;
        let kinetic = 0.5 * rho * velocity * velocity;
        let conserved_energy = pressure / 0.4 + kinetic;
        let old_ideal_flux = (pressure / 0.4 + kinetic + pressure) * velocity;
        let conserved_ideal_flux = (conserved_energy + pressure) * velocity;
        assert_eq!(conserved_ideal_flux.to_bits(), old_ideal_flux.to_bits());

        let water = crate::solver::model::eos::EosSpec::LinearCompressibility {
            bulk_modulus: 2.2e9,
            rho_ref: 1000.0,
            p_ref: 1.0e5,
        }
        .runtime_params();
        for rho in [1000.0_f32, 1000.125] {
            let pressure = water.dp_drho * (rho - water.rho_ref) + water.p_ref;
            let arbitrary_conserved_energy = -12_345.0_f32 + 7.0 * (rho - 1000.0);
            let flux = (arbitrary_conserved_energy + pressure) * 2.0;
            assert!(pressure.is_finite() && flux.is_finite());
            assert!(flux.abs() < 1.0e7, "barotropic energy flux exploded: {flux}");
        }
    }

    fn eval_declared_pressure(
        expr: &AlgExpr,
        rho: f64,
        rho_u: [f64; 2],
        rho_e: f64,
        gm1: f64,
        dp_drho: f64,
        rho_ref: f64,
        p_ref: f64,
    ) -> f64 {
        let rec = |e: &AlgExpr| {
            eval_declared_pressure(e, rho, rho_u, rho_e, gm1, dp_drho, rho_ref, p_ref)
        };
        match expr {
            AlgExpr::Constant { value, .. } => *value,
            AlgExpr::Param(param) => match param.name() {
                "eos_gm1" => gm1,
                "eos_dp_drho" => dp_drho,
                "eos_rho_ref" => rho_ref,
                "eos_p_ref" => p_ref,
                // Zero-gauge evaluation (absolute storage): the state-form
                // closure's affine tail equals the historical `+ p_ref`.
                "eos_gauge_rho_ref" | "eos_gauge_p_ref" | "eos_gauge_e_ref" => 0.0,
                "eos_gauge_p_bias" => p_ref,
                other => panic!("unexpected pressure parameter {other}"),
            },
            AlgExpr::Field(field) => match field.name() {
                "rho" => rho,
                "rho_e" => rho_e,
                other => panic!("unexpected scalar pressure field {other}"),
            },
            AlgExpr::MagSqr(field) => match field.name() {
                "rho_u" => rho_u[0] * rho_u[0] + rho_u[1] * rho_u[1],
                other => panic!("unexpected pressure magnitude field {other}"),
            },
            AlgExpr::Mul(a, b) => rec(a) * rec(b),
            AlgExpr::Div(a, b) => rec(a) / rec(b),
            AlgExpr::Add(a, b) => rec(a) + rec(b),
            AlgExpr::Sub(a, b) => rec(a) - rec(b),
            AlgExpr::Neg(a) => -rec(a),
        }
    }

    fn alg_contains_field(expr: &AlgExpr, wanted: &str) -> bool {
        match expr {
            AlgExpr::Field(field) | AlgExpr::MagSqr(field) => field.name() == wanted,
            AlgExpr::Mul(a, b)
            | AlgExpr::Div(a, b)
            | AlgExpr::Add(a, b)
            | AlgExpr::Sub(a, b) => {
                alg_contains_field(a, wanted) || alg_contains_field(b, wanted)
            }
            AlgExpr::Neg(a) => alg_contains_field(a, wanted),
            AlgExpr::Constant { .. } | AlgExpr::Param(_) => false,
        }
    }

    fn alg_contains_param(expr: &AlgExpr, wanted: &str) -> bool {
        match expr {
            AlgExpr::Param(param) => param.name() == wanted,
            AlgExpr::Mul(a, b)
            | AlgExpr::Div(a, b)
            | AlgExpr::Add(a, b)
            | AlgExpr::Sub(a, b) => {
                alg_contains_param(a, wanted) || alg_contains_param(b, wanted)
            }
            AlgExpr::Neg(a) => alg_contains_param(a, wanted),
            AlgExpr::Constant { .. } | AlgExpr::Field(_) | AlgExpr::MagSqr(_) => false,
        }
    }

    fn alg_contains_mag_sqr(expr: &AlgExpr, wanted: &str) -> bool {
        match expr {
            AlgExpr::MagSqr(field) => field.name() == wanted,
            AlgExpr::Mul(a, b)
            | AlgExpr::Div(a, b)
            | AlgExpr::Add(a, b)
            | AlgExpr::Sub(a, b) => {
                alg_contains_mag_sqr(a, wanted) || alg_contains_mag_sqr(b, wanted)
            }
            AlgExpr::Neg(a) => alg_contains_mag_sqr(a, wanted),
            AlgExpr::Constant { .. } | AlgExpr::Param(_) | AlgExpr::Field(_) => false,
        }
    }

    fn has_grouped_centered_barotropic_product(expr: &AlgExpr) -> bool {
        match expr {
            AlgExpr::Mul(a, b) => {
                let is_dp = |e: &AlgExpr| {
                    matches!(e, AlgExpr::Param(param) if param.name() == "eos_dp_drho")
                };
                // Gauge-storage grouped form: rho + (gauge_rho_ref - rho_ref).
                // The two reference constants subtract FIRST (exact constant
                // arithmetic), so neither storage convention round-trips a
                // liquid density through its large absolute value in f32.
                let is_centered_density = |e: &AlgExpr| {
                    matches!(
                        e,
                        AlgExpr::Add(rho, offset)
                            if matches!(rho.as_ref(), AlgExpr::Field(field) if field.name() == "rho")
                                && matches!(
                                    offset.as_ref(),
                                    AlgExpr::Sub(gauge, rho_ref)
                                        if matches!(gauge.as_ref(), AlgExpr::Param(param) if param.name() == "eos_gauge_rho_ref")
                                            && matches!(rho_ref.as_ref(), AlgExpr::Param(param) if param.name() == "eos_rho_ref")
                                )
                    )
                };
                (is_dp(a) && is_centered_density(b))
                    || (is_dp(b) && is_centered_density(a))
                    || has_grouped_centered_barotropic_product(a)
                    || has_grouped_centered_barotropic_product(b)
            }
            AlgExpr::Div(a, b)
            | AlgExpr::Add(a, b)
            | AlgExpr::Sub(a, b) => {
                has_grouped_centered_barotropic_product(a)
                    || has_grouped_centered_barotropic_product(b)
            }
            AlgExpr::Neg(a) => has_grouped_centered_barotropic_product(a),
            AlgExpr::Constant { .. }
            | AlgExpr::Param(_)
            | AlgExpr::Field(_)
            | AlgExpr::MagSqr(_) => false,
        }
    }

    fn has_momentum_squared_over_density(expr: &AlgExpr) -> bool {
        match expr {
            AlgExpr::Div(numerator, denominator) => {
                // Gauge storage: the kinetic-energy division runs on the
                // ABSOLUTE density `rho + eos_gauge_rho_ref` (reference zero
                // for absolute storage).
                let is_absolute_density = |e: &AlgExpr| {
                    matches!(
                        e,
                        AlgExpr::Add(rho, gauge)
                            if matches!(rho.as_ref(), AlgExpr::Field(field) if field.name() == "rho")
                                && matches!(gauge.as_ref(), AlgExpr::Param(param) if param.name() == "eos_gauge_rho_ref")
                    )
                };
                let numerator_has_momentum = alg_contains_mag_sqr(numerator, "rho_u")
                    && is_absolute_density(denominator);
                numerator_has_momentum
                    || has_momentum_squared_over_density(numerator)
                    || has_momentum_squared_over_density(denominator)
            }
            AlgExpr::Mul(a, b)
            | AlgExpr::Add(a, b)
            | AlgExpr::Sub(a, b) => {
                has_momentum_squared_over_density(a) || has_momentum_squared_over_density(b)
            }
            AlgExpr::Neg(a) => has_momentum_squared_over_density(a),
            AlgExpr::Constant { .. }
            | AlgExpr::Param(_)
            | AlgExpr::Field(_)
            | AlgExpr::MagSqr(_) => false,
        }
    }

    #[test]
    fn declared_face_pressure_is_the_grouped_conserved_eos_closure() {
        let pressure = crate::solver::model::compressible_central_upwind_decl().pressure;
        for field in ["rho", "rho_u", "rho_e"] {
            assert!(alg_contains_field(&pressure, field), "missing {field}");
        }
        for parameter in [
            "eos_gm1",
            "eos_dp_drho",
            "eos_rho_ref",
            "eos_gauge_rho_ref",
            "eos_gauge_p_bias",
        ] {
            assert!(
                alg_contains_param(&pressure, parameter),
                "missing {parameter}"
            );
        }
        assert!(!alg_contains_field(&pressure, "T"));
        assert!(!alg_contains_param(&pressure, "eos_r"));
        assert!(has_momentum_squared_over_density(&pressure));
        assert!(has_grouped_centered_barotropic_product(&pressure));
    }

    #[derive(Clone, Copy)]
    struct LineState {
        owner: f64,
        neighbor: f64,
        grad_owner: f64,
        grad_neighbor: f64,
    }

    fn reconstruct_1d(line: LineState, scheme: Scheme, side: FaceSide) -> f64 {
        let (cell, other, gradient, cell_to_face, other_minus_cell) = match side {
            FaceSide::Owner => (
                line.owner,
                line.neighbor,
                line.grad_owner,
                0.5,
                1.0,
            ),
            FaceSide::Neighbor => (
                line.neighbor,
                line.owner,
                line.grad_neighbor,
                -0.5,
                -1.0,
            ),
        };
        match scheme {
            Scheme::Upwind => cell,
            Scheme::SecondOrderUpwind => cell + gradient * cell_to_face,
            Scheme::QUICK => {
                0.625 * cell + 0.375 * other + 0.125 * gradient * other_minus_cell
            }
            other => panic!("numeric pressure fixture does not cover {other:?}"),
        }
    }

    #[test]
    fn exact_contact_pressure_survives_three_schemes_and_both_flow_directions() {
        // Exact ideal-gas contact: p=10 and |u|=3 on both sides, but density
        // jumps 2 -> 4. Conserved variables are linear across this two-cell
        // fixture, so SOU and QUICK both reconstruct the exact midpoint
        // (rho,m,E)=(3,+/-9,38.5). Reconstructing primitive T independently
        // instead produces rho_f*T_f=3*3.75=11.25.
        let rho = LineState {
            owner: 2.0,
            neighbor: 4.0,
            grad_owner: 2.0,
            grad_neighbor: 2.0,
        };
        let energy = LineState {
            owner: 34.0,
            neighbor: 43.0,
            grad_owner: 9.0,
            grad_neighbor: 9.0,
        };
        let exact_t = LineState {
            owner: 5.0,
            neighbor: 2.5,
            grad_owner: -2.5,
            grad_neighbor: -2.5,
        };
        let pressure_expr =
            crate::solver::model::compressible_central_upwind_decl().pressure;

        for velocity_sign in [1.0, -1.0] {
            let momentum = LineState {
                owner: velocity_sign * 6.0,
                neighbor: velocity_sign * 12.0,
                grad_owner: velocity_sign * 6.0,
                grad_neighbor: velocity_sign * 6.0,
            };
            for scheme in [Scheme::Upwind, Scheme::SecondOrderUpwind, Scheme::QUICK] {
                for side in [FaceSide::Owner, FaceSide::Neighbor] {
                    let rho_f = reconstruct_1d(rho, scheme, side);
                    let pressure = eval_declared_pressure(
                        &pressure_expr,
                        rho_f,
                        [reconstruct_1d(momentum, scheme, side), 0.0],
                        reconstruct_1d(energy, scheme, side),
                        0.4,
                        0.0,
                        0.0,
                        0.0,
                    );
                    assert!(
                        (pressure - 10.0).abs() <= 2.0e-15,
                        "sign={velocity_sign} {scheme:?} {side:?}: pressure={pressure}"
                    );

                    let old_rho_t = rho_f * reconstruct_1d(exact_t, scheme, side);
                    if scheme == Scheme::Upwind {
                        assert!((old_rho_t - 10.0).abs() <= 2.0e-15);
                    } else {
                        assert!((old_rho_t - 11.25).abs() <= 2.0e-15);
                    }
                }
            }

            // Upwind is cellwise exact when T is synchronized, but the new
            // closure is also robust to a stale primitive snapshot: its
            // authoritative owner pressure remains 10 while rho*T would be 12.
            let upwind_pressure = eval_declared_pressure(
                &pressure_expr,
                rho.owner,
                [velocity_sign * 6.0, 0.0],
                energy.owner,
                0.4,
                0.0,
                0.0,
                0.0,
            );
            assert_eq!(upwind_pressure, 10.0);
            assert_eq!(rho.owner * 6.0, 12.0);
        }
    }

    #[test]
    fn exact_barotropic_cells_expose_high_order_rho_t_noncommutation() {
        let rho = LineState {
            owner: 1000.0,
            neighbor: 1001.0,
            grad_owner: 1.0,
            grad_neighbor: 1.0,
        };
        let p_owner = 100_000.0;
        let p_neighbor = 2_300_000.0;
        let temperature = LineState {
            owner: p_owner / rho.owner,
            neighbor: p_neighbor / rho.neighbor,
            grad_owner: p_neighbor / rho.neighbor - p_owner / rho.owner,
            grad_neighbor: p_neighbor / rho.neighbor - p_owner / rho.owner,
        };
        let pressure_expr =
            crate::solver::model::compressible_central_upwind_decl().pressure;

        for scheme in [Scheme::SecondOrderUpwind, Scheme::QUICK] {
            for side in [FaceSide::Owner, FaceSide::Neighbor] {
                let rho_f = reconstruct_1d(rho, scheme, side);
                let pressure = eval_declared_pressure(
                    &pressure_expr,
                    rho_f,
                    [17.0, -9.0],
                    -1234.0,
                    0.0,
                    2.2e6,
                    1000.0,
                    100_000.0,
                );
                assert!((pressure - 1_200_000.0).abs() <= 1.0e-9);

                let old_rho_t = rho_f * reconstruct_1d(temperature, scheme, side);
                assert!(
                    ((old_rho_t - pressure).abs() - 549.425_574_425_6).abs() <= 1.0e-8,
                    "{scheme:?} {side:?}: rho*T error was {} Pa",
                    old_rho_t - pressure
                );
            }
        }
    }

    #[test]
    fn asymmetric_pressure_table_pins_owner_neighbor_and_momentum_parity() {
        // Asymmetric rational fixture: unlike the contact test, the two
        // high-order side states differ, so this catches owner/neighbor swaps
        // as well as a pressure expression that is not even in momentum.
        let rho = LineState {
            owner: 2.0,
            neighbor: 4.0,
            grad_owner: 1.0,
            grad_neighbor: 2.0,
        };
        let mx_positive = LineState {
            owner: 4.0,
            neighbor: 8.0,
            grad_owner: 2.0,
            grad_neighbor: 4.0,
        };
        let my = LineState {
            owner: 2.0,
            neighbor: -4.0,
            grad_owner: -1.0,
            grad_neighbor: -2.0,
        };
        let energy = LineState {
            owner: 30.0,
            neighbor: 40.0,
            grad_owner: 4.0,
            grad_neighbor: 8.0,
        };
        let cases = [
            (Scheme::Upwind, FaceSide::Owner, 10.0),
            (Scheme::Upwind, FaceSide::Neighbor, 12.0),
            (Scheme::SecondOrderUpwind, FaceSide::Owner, 531.0 / 50.0),
            (Scheme::SecondOrderUpwind, FaceSide::Neighbor, 57.0 / 5.0),
            (Scheme::QUICK, FaceSide::Owner, 10_479.0 / 920.0),
            (Scheme::QUICK, FaceSide::Neighbor, 231.0 / 20.0),
        ];
        let pressure_expr =
            crate::solver::model::compressible_central_upwind_decl().pressure;

        for momentum_sign in [1.0, -1.0] {
            let mx = LineState {
                owner: momentum_sign * mx_positive.owner,
                neighbor: momentum_sign * mx_positive.neighbor,
                grad_owner: momentum_sign * mx_positive.grad_owner,
                grad_neighbor: momentum_sign * mx_positive.grad_neighbor,
            };
            for (scheme, side, expected) in cases {
                let pressure = eval_declared_pressure(
                    &pressure_expr,
                    reconstruct_1d(rho, scheme, side),
                    [
                        reconstruct_1d(mx, scheme, side),
                        reconstruct_1d(my, scheme, side),
                    ],
                    reconstruct_1d(energy, scheme, side),
                    0.4,
                    0.0,
                    0.0,
                    0.0,
                );
                assert!(
                    (pressure - expected).abs() <= 3.0e-15,
                    "sign={momentum_sign} {scheme:?} {side:?}: {pressure} != {expected}"
                );
            }
        }
    }

    #[test]
    fn ideal_uniform_exact_state_pressure_is_invariant_for_every_reconstruction() {
        let rho_value = 1.2;
        let velocity = [30.0, -4.0];
        let pressure_value = 101_325.0;
        let momentum = [rho_value * velocity[0], rho_value * velocity[1]];
        let energy = pressure_value / 0.4
            + 0.5 * rho_value * (velocity[0] * velocity[0] + velocity[1] * velocity[1]);
        let uniform = |value| LineState {
            owner: value,
            neighbor: value,
            grad_owner: 0.0,
            grad_neighbor: 0.0,
        };
        let pressure_expr =
            crate::solver::model::compressible_central_upwind_decl().pressure;

        for scheme in [Scheme::Upwind, Scheme::SecondOrderUpwind, Scheme::QUICK] {
            for side in [FaceSide::Owner, FaceSide::Neighbor] {
                let pressure = eval_declared_pressure(
                    &pressure_expr,
                    reconstruct_1d(uniform(rho_value), scheme, side),
                    [
                        reconstruct_1d(uniform(momentum[0]), scheme, side),
                        reconstruct_1d(uniform(momentum[1]), scheme, side),
                    ],
                    reconstruct_1d(uniform(energy), scheme, side),
                    0.4,
                    0.0,
                    0.0,
                    0.0,
                );
                assert!(
                    (pressure - pressure_value).abs() <= pressure_value * 2.0e-15,
                    "{scheme:?} {side:?}: uniform pressure drifted to {pressure}"
                );
            }
        }
    }

    #[test]
    fn derived_momentum_flux_pressure_reads_reconstructed_conserved_state() {
        let system = crate::solver::model::compressible_system();
        for scheme in [
            Scheme::Upwind,
            Scheme::SecondOrderUpwind,
            Scheme::SecondOrderUpwindMinMod,
            Scheme::SecondOrderUpwindVanLeer,
            Scheme::QUICK,
            Scheme::QUICKMinMod,
            Scheme::QUICKVanLeer,
        ] {
            let spec = derive_central_upwind(
                &system,
                &crate::solver::model::compressible_central_upwind_decl(),
                scheme,
            )
            .expect("derive production compressible flux");
            let FluxModuleKernelSpec::ScalarPerComponent { components, flux } = spec else {
                panic!("compressible flux did not lower per component");
            };
            let momentum_x = &flux[components
                .iter()
                .position(|name| name == "rho_u_x")
                .expect("rho_u_x flux component")];
            assert!(
                contains_state(momentum_x, "rho_e"),
                "{scheme:?}: pressure path does not read reconstructed rho_e"
            );
            for constant in [
                "eos_gm1",
                "eos_dp_drho",
                "eos_rho_ref",
                "eos_gauge_rho_ref",
                "eos_gauge_p_bias",
            ] {
                assert!(
                    contains_constant(momentum_x, constant),
                    "{scheme:?}: pressure path lost {constant}"
                );
            }
        }
    }

    #[test]
    fn generated_compressible_fluxes_pin_two_conserved_pressure_closures_per_scheme() {
        let generated = [
            (
                "compressible",
                include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/gpu/shaders/generated/flux_module_compressible.wgsl"
                )),
            ),
            (
                "compressible_mms",
                include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/gpu/shaders/generated/flux_module_compressible_mms.wgsl"
                )),
            ),
            (
                "compressible_mms_biharmonic",
                include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/gpu/shaders/generated/flux_module_compressible_mms_biharmonic.wgsl"
                )),
            ),
            (
                "compressible_structured",
                include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/src/solver/gpu/shaders/generated/flux_module_compressible_structured.wgsl"
                )),
            ),
        ];

        for (name, wgsl) in generated {
            assert_eq!(
                wgsl.matches("constants.eos_gm1 * (").count(),
                14,
                "{name}: expected owner+neighbor closure for all seven schemes"
            );
            assert_eq!(
                wgsl.matches(
                    "+ (constants.eos_gauge_rho_ref - constants.eos_rho_ref)) + constants.eos_gauge_p_bias"
                )
                .count(),
                14,
                "{name}: gauge-grouped centered affine closure was expanded or omitted"
            );
            assert!(wgsl.contains("s_own_rho_e"));
            assert!(wgsl.contains("s_neigh_rho_e"));
            assert!(
                !wgsl.contains("/ max(constants.eos_gm1"),
                "{name}: singular energy reconstruction returned"
            );
        }
    }

    fn contains_sign_guard_for_states(expr: &S, a: &str, b: &str) -> bool {
        fn contains_state(expr: &S, name: &str) -> bool {
            match expr {
                S::State { name: n, .. } | S::CellState { name: n, .. } => n == name,
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
                | S::BoundaryDirichlet { .. }
                | S::MeshFlux
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
                | S::CellState { .. }
                | S::BoundaryDirichlet { .. }
                | S::MeshFlux
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
