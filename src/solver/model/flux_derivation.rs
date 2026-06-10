// Derivation of flux-module kernels from declarative flux definitions.
//
// Models declare *what* a flux is in math terms (e.g. "the volumetric flux of
// advecting velocity field `U_adv`"); this module lowers the declaration to a
// `FluxModuleKernelSpec` consumed by the generic flux-module codegen. No
// model-specific WGSL or face-expression assembly lives in the model definition.
// (Regular comments: this file is also include!()'d into build.rs.)

use crate::solver::ir::{
    FaceScalarExpr as S, FaceSide, FaceVec2Expr as V, FluxLayout, FluxModuleKernelSpec,
};
use crate::solver::model::backend::ast::EquationSystem;
use crate::solver::model::backend::state_layout::StateLayout;
use crate::solver::model::ports::dimensions::{Density, Velocity};
use crate::solver::model::ports::PortRegistry;

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
