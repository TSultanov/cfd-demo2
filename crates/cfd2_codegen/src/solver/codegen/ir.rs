use crate::solver::scheme::Scheme;

use crate::solver::ir::{
    Coefficient, Discretization, EquationSystem, FieldRef, FluxRef, SchemeRegistry, Term, TermOp,
    UnitValidationError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiscreteOpKind {
    TimeDerivative,
    Convection,
    Gradient,
    Diffusion,
    Source,
}

impl DiscreteOpKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DiscreteOpKind::TimeDerivative => "ddt",
            DiscreteOpKind::Convection => "div",
            DiscreteOpKind::Gradient => "grad",
            DiscreteOpKind::Diffusion => "laplacian",
            DiscreteOpKind::Source => "source",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteOp {
    pub target: FieldRef,
    pub kind: DiscreteOpKind,
    pub term_op: TermOp,
    pub discretization: Discretization,
    pub scheme: Scheme,
    /// True when the scheme was declared on the term itself (model math
    /// declaration) rather than resolved from the registry default. Declared
    /// schemes are baked as literals in the generated WGSL and are not
    /// affected by the runtime `constants.scheme` knob.
    pub scheme_declared: bool,
    /// Bounded convection form: subtract the continuity defect
    /// `(div phi) * phi_P` from the diagonal (only meaningful on implicit
    /// `Div` convection ops).
    pub bounded: bool,
    /// Per-component direction multipliers for explicit sources on vector
    /// targets (`coeff * direction[c] * V` per component).
    pub direction: Option<Vec<f64>>,
    /// Explicit transpose/deviatoric viscous form: assemble as
    /// `-div(coeff * dev2((grad field)^T))` from grad_state cell gradients
    /// (only meaningful on explicit Diffusion ops with Vector2 fields).
    pub transpose_dev2: bool,
    /// Static implicit diagonal: an implicit `Source` op contributes `coeff`
    /// to the matrix diagonal WITHOUT the `* V` cell-volume factor (see
    /// `Term::static_diag`). Used for pointwise identity/constraint rows.
    pub static_diag: bool,
    /// Deferred-correction Newton linearization of a `DivFlux` mass-flux term
    /// against the pressure: `Some(coeff)` carries the flux pressure-sensitivity
    /// `d(rho_face)/dp` (the compressibility). Assembly emits an implicit upwind
    /// convection of the pressure by `coeff_face * (U.n) * A` plus a deferred
    /// RHS correction (frozen-state pressure) that cancels it at convergence
    /// (see `Term::linearize_pressure_flux`).
    pub linearize_pressure_flux: Option<Coefficient>,
    /// ALE (mesh-relative) convection: assembly consumes this op's face flux
    /// as `phi_rel = phi - rho_f * mesh_fluxes[face]` (owner-signed exactly
    /// like `phi`; `rho_f` = the constant `rho` coefficient for incompressible
    /// models). Gates the `mesh_fluxes` storage binding in the assembly
    /// kernels. See `Term::relative_to_mesh`.
    pub relative_to_mesh: bool,
    pub field: FieldRef,
    pub flux: Option<FluxRef>,
    pub coeff: Option<Coefficient>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteEquation {
    pub target: FieldRef,
    pub ops: Vec<DiscreteOp>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteSystem {
    pub equations: Vec<DiscreteEquation>,
}

impl DiscreteSystem {
    /// ALE marker, derived: a system is ALE iff any op consumes its face flux
    /// relative to the mesh (`Term::relative_to_mesh`). Gates the ALE codegen:
    /// the `mesh_fluxes` / `cell_vols_old{,_old}` storage bindings, the
    /// moving-volume ddt lowering, the ALE bounded-correction augmentation and
    /// the continuity volume source. Static (non-ALE) systems take the exact
    /// pre-ALE emission paths, so their generated code stays byte-identical.
    pub fn is_ale(&self) -> bool {
        self.equations
            .iter()
            .any(|eq| eq.ops.iter().any(|op| op.relative_to_mesh))
    }
}

/// Lower an equation system to a discrete system without validation.
///
/// This is a "trusted" variant for systems that have already been validated
/// (e.g., built with the typed builder). Prefer `lower_system` for untyped
/// construction paths where validation is needed as a backstop.
pub fn lower_system_unchecked(system: &EquationSystem, schemes: &SchemeRegistry) -> DiscreteSystem {
    let mut equations = Vec::new();
    for equation in system.equations() {
        let mut ops = Vec::new();
        for term in equation.terms() {
            ops.push(lower_term(equation.target(), term, schemes));
        }
        equations.push(DiscreteEquation {
            target: *equation.target(),
            ops,
        });
    }
    DiscreteSystem { equations }
}

/// Lower an equation system to a discrete system with validation.
///
/// This is the standard entry point that validates units before lowering.
/// For systems already known to be valid (e.g., typed-built), use
/// `lower_system_unchecked` to avoid redundant validation.
pub fn lower_system(
    system: &EquationSystem,
    schemes: &SchemeRegistry,
) -> Result<DiscreteSystem, UnitValidationError> {
    system.validate_units()?;
    Ok(lower_system_unchecked(system, schemes))
}

fn lower_term(target: &FieldRef, term: &Term, schemes: &SchemeRegistry) -> DiscreteOp {
    let kind = match term.op {
        TermOp::Ddt => DiscreteOpKind::TimeDerivative,
        TermOp::Div | TermOp::DivFlux => DiscreteOpKind::Convection,
        TermOp::Grad => DiscreteOpKind::Gradient,
        TermOp::Laplacian => DiscreteOpKind::Diffusion,
        TermOp::Source => DiscreteOpKind::Source,
    };
    DiscreteOp {
        target: *target,
        kind,
        term_op: term.op,
        discretization: term.discretization,
        scheme: term.scheme.unwrap_or_else(|| schemes.scheme_for(term)),
        scheme_declared: term.scheme.is_some(),
        bounded: term.bounded,
        direction: term.direction.clone(),
        transpose_dev2: term.transpose_dev2,
        static_diag: term.static_diag,
        linearize_pressure_flux: term.linearize_pressure_flux.clone(),
        relative_to_mesh: term.relative_to_mesh,
        field: term.field,
        flux: term.flux,
        coeff: term.coeff.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ir::{fvc, fvm, surface_scalar_dim, vol_scalar_dim, vol_vector_dim};
    use cfd2_ir::dimensions::{
        Density, DynamicViscosity, MassFlux, Pressure, PressureGradient, UnitDimension, Velocity,
    };

    #[test]
    fn lower_system_maps_all_term_kinds() {
        let u = vol_vector_dim::<Velocity>("U");
        let p = vol_scalar_dim::<Pressure>("p");
        let rho = vol_scalar_dim::<Density>("rho");
        let phi = surface_scalar_dim::<MassFlux>("phi");
        let mu = vol_scalar_dim::<DynamicViscosity>("mu");

        let mut eqn = crate::solver::ir::Equation::new(u);
        eqn.add_term(fvm::ddt_coeff(Coefficient::field(rho).unwrap(), u));
        eqn.add_term(fvm::div(phi, u));
        eqn.add_term(fvc::grad(p));
        eqn.add_term(fvm::laplacian(Coefficient::field(mu).unwrap(), u));
        eqn.add_term(fvc::source_coeff(
            Coefficient::constant_unit(1.0, PressureGradient::UNIT),
            u,
        ));

        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let registry = SchemeRegistry::new(Scheme::Upwind);
        let discrete = lower_system(&system, &registry).unwrap();

        assert_eq!(discrete.equations.len(), 1);
        let ops = &discrete.equations[0].ops;
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].kind, DiscreteOpKind::TimeDerivative);
        assert_eq!(ops[1].kind, DiscreteOpKind::Convection);
        assert_eq!(ops[2].kind, DiscreteOpKind::Gradient);
        assert_eq!(ops[3].kind, DiscreteOpKind::Diffusion);
        assert_eq!(ops[4].kind, DiscreteOpKind::Source);
        assert_eq!(ops[0].scheme, Scheme::Upwind);
    }

    #[test]
    fn lower_system_applies_scheme_registry() {
        let u = vol_vector_dim::<Velocity>("U");
        let phi = surface_scalar_dim::<MassFlux>("phi");

        let mut eqn = crate::solver::ir::Equation::new(u);
        eqn.add_term(fvm::div(phi, u));

        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let mut registry = SchemeRegistry::new(Scheme::Upwind);
        registry.set_for_term(TermOp::Div, Some(&phi), &u, Scheme::QUICK);

        let discrete = lower_system(&system, &registry).unwrap();
        assert_eq!(discrete.equations[0].ops[0].scheme, Scheme::QUICK);
    }
}
