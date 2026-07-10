// Derivation of flux-module kernels from declarative flux definitions.
//
// Models declare *what* a flux is in math terms; this module lowers the
// declaration to a `FluxModuleKernelSpec` for the generic flux-module codegen.
// This file is also include!()'d into build.rs, so only regular comments here.

use crate::solver::ir::{
    FaceScalarExpr as S, FaceSide, FaceVec2Expr as V, FluxLayout, FluxModuleKernelSpec,
};
use crate::solver::model::backend::ast::EquationSystem;
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::model::ports::dimensions::{Density, Temperature, Velocity};
use crate::solver::model::ports::PortRegistry;

/// A derived Rhie–Chow flux: the flux-module kernel plus the auxiliary
/// kernel bundle (dp_init / dp_update / grad_p maintenance) it requires.
/// Both are inferred from the declared momentum/pressure equation system;
/// the model attaches `aux_module` to its module list and feeds
/// `flux_kernel` to `FluxModuleSpec::Kernel`.
#[derive(Debug)]
pub struct DerivedRhieChow {
    pub flux_kernel: FluxModuleKernelSpec,
    pub aux_module: crate::solver::model::module::KernelBundleModule,
}

/// Derive the Rhie–Chow mass flux from the declared equation system.
///
/// Inference (all from the math, no model-specific configuration):
/// - the (momentum, pressure) pair: the unique vector equation with a
///   transport term whose `grad(p)` scalar has its own equation containing
///   a Laplacian;
/// - `d_p`: the unique state-layout scalar with `D_P` units referenced by
///   the pressure-Laplacian coefficient;
/// - density: the state-layout `rho` if present, else the uniform density.
///
/// The flux is the Rhie–Chow-style mass flux
///   `phi = rho_f * (HbyA_f . n) * A  -  rho_f * d_p_f * ((p_N - p_O)/dist) * A`
/// with `HbyA ~= U + d_p * grad(p)`; the pressure row receives the predicted
/// flux (first term only) and all other rows the corrected flux. The
/// pressure-gradient correction uses the same face-normal distance
/// projection as the Laplacian discretization so the two stay consistent.
pub fn derive_rhie_chow(
    system: &EquationSystem,
    layout: &StateLayout,
) -> Result<DerivedRhieChow, String> {
    derive_rhie_chow_with_dp(
        system,
        layout,
        crate::solver::model::modules::rhie_chow::DpFormulation::ClosedForm,
    )
}

/// [`derive_rhie_chow`] with an explicit coupling-coefficient formulation
/// (see [`crate::solver::model::modules::rhie_chow::DpFormulation`]).
pub fn derive_rhie_chow_with_dp(
    system: &EquationSystem,
    layout: &StateLayout,
    dp_formulation: crate::solver::model::modules::rhie_chow::DpFormulation,
) -> Result<DerivedRhieChow, String> {
    let (flux_kernel, d_p) = derive_rhie_chow_flux(system, layout)?;
    // The aux-module port manifest requires 'static names (same pattern as
    // the module's own grad_p name interning).
    let d_p_static: &'static str = Box::leak(d_p.into_boxed_str());
    let aux_module = crate::solver::model::modules::rhie_chow::rhie_chow_aux_module(
        system,
        d_p_static,
        true,
        true,
        dp_formulation,
    )
    .map_err(|e| format!("derive_rhie_chow: aux module: {e}"))?;
    Ok(DerivedRhieChow {
        flux_kernel,
        aux_module,
    })
}

fn derive_rhie_chow_flux(
    system: &EquationSystem,
    layout: &StateLayout,
) -> Result<(FluxModuleKernelSpec, String), String> {
    use crate::solver::model::backend::{Coefficient as BackendCoeff, FieldKind, FieldRef, TermOp};
    use std::collections::{HashMap, HashSet};

    fn collect_coeff_fields(coeff: &BackendCoeff, out: &mut Vec<FieldRef>) {
        match coeff {
            BackendCoeff::Constant { .. } => {}
            BackendCoeff::Field(field) => out.push(*field),
            BackendCoeff::MagSqr(field) => out.push(*field),
            BackendCoeff::Product(lhs, rhs) => {
                collect_coeff_fields(lhs, out);
                collect_coeff_fields(rhs, out);
            }
        }
    }

    let registry = PortRegistry::new(layout.clone());

    fn density_face_expr(registry: &PortRegistry, ale: bool) -> Result<S, String> {
        // Prefer a state-layout density when present; else fall back to the
        // global constant density uniform.
        //
        // ALE INVARIANT (do not break): whatever face density `rho_f` this flux
        // kernel bakes into the mass flux `phi = rho_f*(U_f.n)*A`, the unified
        // assembly's mesh-relative subtraction `phi_rel = phi - rho_f*mesh_flux`
        // must reconstruct the EXACT same expression on the same states
        // (crates/cfd2_codegen .../unified_assembly.rs, `ale_relative_flux_expr`
        // + `ale_thermal_upwind_face_density_stmts`). If the two sites disagree,
        // every face with a density gradient on a moving mesh gains a spurious
        // mass source ∝ (rho mismatch) × (mesh velocity). If you change the
        // blend, the lambda convention, or the sgn() argument here, change the
        // assembly mirror in the same commit.
        match registry.validate_scalar_field::<Density>("derive_rhie_chow", "rho") {
            Ok(()) => {
                let rho_o = S::state(FaceSide::Owner, "rho");
                let rho_n = S::state(FaceSide::Neighbor, "rho");
                // Upwind face density (by face-normal velocity sign) for the
                // real-EOS compressible model, gated on the marker field `t_ref`:
                // central averaging is dispersive and over-expands the diverging
                // section (M_throat > M_exit). Without `t_ref` keep the central
                // `Lerp`, byte-identical, so MMS order is unaffected.
                let upwind = registry
                    .validate_scalar_field::<Temperature>("derive_rhie_chow", "t_ref")
                    .is_ok();
                if !upwind {
                    return Ok(S::Lerp(Box::new(rho_o), Box::new(rho_n)));
                }
                // sgn(u_n) = u_n / max(|u_n|, eps),  u_n = U_face . n  (central face U).
                //
                // Under ALE the CONVECTING velocity is the mesh-relative
                // `(U - U_mesh).n = U.n - mesh_flux/A` (the mesh flux is the
                // owner-signed volumetric swept rate, zero-filled on a static
                // mesh so `u_n - 0.0/A` is bitwise `u_n` there): where the mesh
                // outruns the flow through a face, the absolute `U.n` would pick
                // the DOWNWIND side of the relative transport. The assembly's
                // subtraction uses the same relative sgn (see invariant above).
                let u_face = V::Lerp(
                    Box::new(V::state_vec2(FaceSide::Owner, "U")),
                    Box::new(V::state_vec2(FaceSide::Neighbor, "U")),
                );
                let mut u_n = S::Dot(Box::new(u_face), Box::new(V::normal()));
                if ale {
                    u_n = S::Sub(
                        Box::new(u_n),
                        Box::new(S::Div(Box::new(S::mesh_flux()), Box::new(S::area()))),
                    );
                }
                let sgn = S::Div(
                    Box::new(u_n.clone()),
                    Box::new(S::Max(
                        Box::new(S::Abs(Box::new(u_n))),
                        Box::new(S::lit(1.0e-12)),
                    )),
                );
                // rho_f = 0.5*(rho_o+rho_n) + 0.5*sgn*(rho_o-rho_n)
                //       = rho_o if u_n>=0 (owner is upwind), rho_n if u_n<0.
                let avg = S::Mul(
                    Box::new(S::lit(0.5)),
                    Box::new(S::Add(Box::new(rho_o.clone()), Box::new(rho_n.clone()))),
                );
                let half_diff = S::Mul(
                    Box::new(S::lit(0.5)),
                    Box::new(S::Sub(Box::new(rho_o), Box::new(rho_n))),
                );
                Ok(S::Add(
                    Box::new(avg),
                    Box::new(S::Mul(Box::new(sgn), Box::new(half_diff))),
                ))
            }
            Err(crate::solver::model::ports::PortValidationError::MissingField { .. }) => {
                // rho is not in state layout; fall back to uniform density
                Ok(S::constant("density"))
            }
            Err(e) => Err(format!("rho field exists but has wrong kind or unit: {}", e)),
        }
    }

    // Infer (momentum, pressure) coupling from the declared equation system.
    let equations = system.equations();
    let mut eq_by_target: HashMap<FieldRef, usize> = HashMap::new();
    for (idx, eq) in equations.iter().enumerate() {
        eq_by_target.insert(*eq.target(), idx);
    }

    let mut candidates = Vec::new();
    for eq in equations {
        if !matches!(eq.target().kind(), FieldKind::Vector2 | FieldKind::Vector3) {
            continue;
        }

        let has_transport = eq
            .terms()
            .iter()
            .any(|t| matches!(t.op, TermOp::Div | TermOp::Laplacian));
        if !has_transport {
            continue;
        }

        let mut grad_scalars: HashSet<FieldRef> = HashSet::new();
        for term in eq.terms() {
            if term.op == TermOp::Grad && term.field.kind() == FieldKind::Scalar {
                grad_scalars.insert(term.field);
            }
        }

        for pressure in grad_scalars {
            let Some(&p_eq_idx) = eq_by_target.get(&pressure) else {
                continue;
            };
            let p_eq = &equations[p_eq_idx];
            let p_has_laplacian = p_eq.terms().iter().any(|t| t.op == TermOp::Laplacian);
            if p_has_laplacian {
                candidates.push((eq.target().name().to_string(), pressure.name().to_string()));
            }
        }
    }

    let (momentum, pressure) = match candidates.as_slice() {
        [(m, p)] => (m.clone(), p.clone()),
        [] => return Err("no unique momentum-pressure coupling found for Rhie–Chow".to_string()),
        many => {
            return Err(format!(
                "Rhie–Chow requires a unique momentum-pressure coupling, found {} candidates: [{}]",
                many.len(),
                many.iter()
                    .map(|(m, p)| format!("{m}↔{p}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    };

    // Infer d_p from the pressure equation laplacian coefficient: pick the unique scalar field
    // referenced by that coefficient that is present in the model's state layout.
    let pressure_eq = equations
        .iter()
        .find(|eq| eq.target().name() == pressure)
        .ok_or_else(|| {
            format!("missing pressure equation for inferred pressure field '{pressure}'")
        })?;
    let pressure_laplacian = pressure_eq
        .terms()
        .iter()
        .find(|t| t.op == TermOp::Laplacian)
        .ok_or_else(|| {
            format!("pressure equation for '{pressure}' must include a laplacian term")
        })?;
    let Some(coeff) = &pressure_laplacian.coeff else {
        return Err(format!(
            "pressure laplacian coefficient for '{pressure}' is missing"
        ));
    };
    let mut coeff_fields = Vec::new();
    collect_coeff_fields(coeff, &mut coeff_fields);
    let mut d_p_candidates: Vec<String> = Vec::new();
    for f in &coeff_fields {
        match registry
            .validate_scalar_field::<cfd2_ir::dimensions::D_P>("derive_rhie_chow", f.name())
        {
            Ok(()) => d_p_candidates.push(f.name().to_string()),
            Err(crate::solver::model::ports::PortValidationError::MissingField { .. }) => {
                // Field is not in state layout; not a candidate
            }
            Err(_) => {
                // Field exists but has wrong kind or unit; not a candidate
            }
        }
    }
    let d_p = match d_p_candidates.as_slice() {
        [only] => only.clone(),
        [] => {
            return Err(format!(
                "pressure laplacian coefficient for '{pressure}' does not reference any state-layout scalar fields with D_P units"
            ));
        }
        many => {
            return Err(format!(
                "pressure laplacian coefficient for '{pressure}' references multiple state-layout fields with D_P units; cannot derive unique d_p: [{}]",
                many.join(", ")
            ));
        }
    };

    // Rhie–Chow-style mass flux:
    //   phi = rho * (u_f · n) * area  -  rho * d_p_f * ((p_N - p_O) / dist) * area
    // The pressure-gradient term uses the same face-normal distance projection
    // (`dist`) as the Laplacian discretization, so the two stay consistent.
    let rho_face = density_face_expr(&registry, system.is_ale())?;

    let d_p_face = S::Lerp(
        Box::new(S::state(FaceSide::Owner, d_p.clone())),
        Box::new(S::state(FaceSide::Neighbor, d_p.clone())),
    );

    // `HbyA` is the momentum predictor for the "predicted" mass flux (pressure
    // equation RHS); the explicit pressure correction is subtracted after.
    //   HbyA ≈ U + d_p * grad(p)
    let grad_p_field = format!("grad_{}", pressure);
    let u_face = V::Lerp(
        Box::new(V::state_vec2(FaceSide::Owner, momentum.clone())),
        Box::new(V::state_vec2(FaceSide::Neighbor, momentum.clone())),
    );
    let grad_p_face = V::Lerp(
        Box::new(V::state_vec2(FaceSide::Owner, grad_p_field.clone())),
        Box::new(V::state_vec2(FaceSide::Neighbor, grad_p_field)),
    );
    let hby_a_face = V::Add(
        Box::new(u_face),
        Box::new(V::MulScalar(
            Box::new(grad_p_face),
            Box::new(d_p_face.clone()),
        )),
    );
    let u_n = S::Dot(Box::new(hby_a_face), Box::new(V::normal()));
    let phi_pred = S::Mul(
        Box::new(S::Mul(Box::new(rho_face.clone()), Box::new(u_n))),
        Box::new(S::area()),
    );
    let dp = S::Sub(
        Box::new(S::state(FaceSide::Neighbor, pressure.clone())),
        Box::new(S::state(FaceSide::Owner, pressure.clone())),
    );
    let dp_over_dist = S::Div(Box::new(dp), Box::new(S::dist()));
    let phi_p = S::Mul(
        Box::new(S::Mul(
            Box::new(S::Mul(Box::new(rho_face), Box::new(d_p_face))),
            Box::new(dp_over_dist),
        )),
        Box::new(S::area()),
    );
    let phi_corr = S::Sub(Box::new(phi_pred.clone()), Box::new(phi_p));

    // Pressure equation uses the *predicted* mass flux (phi_pred) on its RHS;
    // momentum convection uses the corrected flux (phi_corr) to reduce
    // pressure–velocity decoupling on collocated grids (Rhie–Chow). The flux
    // buffer is indexed by coupled unknown component, so `p` and `U` slots can
    // differ within a single scalar flux-module kernel.
    let flux_layout = FluxLayout::from_system(system);
    let components: Vec<String> = flux_layout
        .components
        .iter()
        .map(|c| c.name.clone())
        .collect();
    let flux: Vec<S> = components
        .iter()
        .map(|name| {
            if name == &pressure {
                phi_pred.clone()
            } else {
                phi_corr.clone()
            }
        })
        .collect();

    Ok((
        FluxModuleKernelSpec::ScalarPerComponent { components, flux },
        d_p,
    ))
}

/// Declarative flux specification: what the flux *is*, in math terms.
#[derive(Debug, Clone, PartialEq)]
pub enum FluxExprSpec {
    /// Advective flux of a velocity field: `phi_f = [rho_f] * (U_f . n) * A`,
    /// with `U_f` the arithmetic face average of the cell-centered velocity.
    ///
    /// With `density: None` this is a volumetric flux (unit Volume/Time);
    /// with a density field it is a mass flux. The same flux value is used for
    /// every coupled unknown component.
    AdvectingVelocity {
        /// Name of a `Vector2` velocity field in the state layout.
        velocity: &'static str,
        /// Optional name of a scalar density field in the state layout.
        density: Option<&'static str>,
    },
}

/// Lower a declarative flux spec to a flux-module kernel for `system`.
pub fn derive_flux_module_kernel(
    system: &EquationSystem,
    layout: &StateLayout,
    spec: &FluxExprSpec,
) -> Result<FluxModuleKernelSpec, String> {
    match spec {
        FluxExprSpec::AdvectingVelocity { velocity, density } => {
            derive_advecting_velocity_flux(system, layout, velocity, *density)
        }
    }
}

fn derive_advecting_velocity_flux(
    system: &EquationSystem,
    layout: &StateLayout,
    velocity: &str,
    density: Option<&'static str>,
) -> Result<FluxModuleKernelSpec, String> {
    let registry = PortRegistry::new(layout.clone());
    registry
        .validate_vector2_field::<Velocity>("derive_flux_module_kernel", velocity)
        .map_err(|e| format!("advecting velocity field invalid: {e}"))?;
    if let Some(rho) = density {
        registry
            .validate_scalar_field::<Density>("derive_flux_module_kernel", rho)
            .map_err(|e| format!("advecting density field invalid: {e}"))?;
    }

    // u_f . n, with u_f the arithmetic face average of the cell-centered velocity.
    let u_face = V::Lerp(
        Box::new(V::state_vec2(FaceSide::Owner, velocity)),
        Box::new(V::state_vec2(FaceSide::Neighbor, velocity)),
    );
    let u_n = S::Dot(Box::new(u_face), Box::new(V::normal()));

    let mut phi = S::Mul(Box::new(u_n), Box::new(S::area()));
    if let Some(rho) = density {
        let rho_face = S::Lerp(
            Box::new(S::state(FaceSide::Owner, rho)),
            Box::new(S::state(FaceSide::Neighbor, rho)),
        );
        phi = S::Mul(Box::new(rho_face), Box::new(phi));
    }

    let flux_layout = FluxLayout::from_system(system);
    let components: Vec<String> = flux_layout
        .components
        .iter()
        .map(|c| c.name.clone())
        .collect();
    if components.is_empty() {
        return Err("flux derivation requires at least one coupled unknown".to_string());
    }
    let flux = vec![phi; components.len()];

    Ok(FluxModuleKernelSpec::ScalarPerComponent { components, flux })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_rhie_chow_matches_model_construction() {
        // The incompressible model consumes the deriver; pin the inferred
        // structure here: 3 coupled components (U_x, U_y, p), pressure slot
        // receives the predicted flux, momentum slots the corrected flux.
        let fields = crate::solver::model::definitions::IncompressibleMomentumFields::new();
        let system = crate::solver::model::definitions::incompressible_momentum_system();
        let layout = PortRegistry::from_fields(vec![
            fields.u,
            fields.p,
            fields.d_p,
            fields.grad_p,
            fields.grad_p_old,
        ])
        .into_state_layout();

        let derived = derive_rhie_chow(&system, &layout).expect("derive");
        let FluxModuleKernelSpec::ScalarPerComponent { components, flux } = derived.flux_kernel
        else {
            panic!("expected ScalarPerComponent");
        };
        assert_eq!(components, ["U_x", "U_y", "p"]);
        assert_eq!(flux[0], flux[1], "momentum slots share the corrected flux");
        assert_ne!(
            flux[0], flux[2],
            "pressure slot gets the predicted (uncorrected) flux"
        );
        assert_eq!(derived.aux_module.name, "rhie_chow_aux");
    }

    #[test]
    fn derive_rhie_chow_rejects_systems_without_coupling() {
        let system = crate::solver::model::definitions::scalar_transport_model()
            .expect("model")
            .system;
        let layout = crate::solver::model::definitions::scalar_transport_model()
            .expect("model")
            .state_layout;
        let err = derive_rhie_chow(&system, &layout).unwrap_err();
        assert!(err.contains("no unique momentum-pressure coupling"), "{err}");
    }
}
