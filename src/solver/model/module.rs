use crate::solver::model::eos::EosSpec;
use crate::solver::model::kernel::{ModelKernelGeneratorSpec, ModelKernelSpec};

/// Re-export IR-safe port manifest types for convenience.
pub use cfd2_ir::solver::ir::ports::{
    BufferAccess, BufferSpec, FieldSpec, ParamSpec, PortFieldKind, PortManifest,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NamedParamKey {
    Key(&'static str),
}

impl NamedParamKey {
    pub fn as_str(self) -> &'static str {
        match self {
            NamedParamKey::Key(s) => s,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKindReq {
    Scalar,
    Vector2,
    Vector3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleInvariant {
    /// Require a named field in the state layout, optionally with a specific kind.
    RequireStateField {
        name: &'static str,
        kind: Option<FieldKindReq>,
    },

    /// Require a unique momentum-pressure coupling whose pressure Laplacian coefficient
    /// references the given `d_p` field name, and (optionally) require a Vector2 momentum
    /// and a `grad_<pressure>` Vector2 field in the state layout.
    RequireUniqueMomentumPressureCouplingReferencingDp {
        dp_field: &'static str,
        require_vector2_momentum: bool,
        require_pressure_gradient: bool,
    },
}

#[derive(Debug, Clone, Default)]
pub struct ModuleManifest {
    /// Optional solver/method selection contributed by this module.
    pub method: Option<crate::solver::model::method::MethodSpec>,

    /// Optional flux module configuration contributed by this module.
    pub flux_module: Option<crate::solver::model::flux_module::FluxModuleSpec>,

    /// Named parameters supported by this module.
    ///
    /// These keys control which `plan.set_named_param()` entries are accepted.
    pub named_params: Vec<NamedParamKey>,

    /// Typed invariant requirements declared by this module.
    pub invariants: Vec<ModuleInvariant>,

    /// Optional port-based manifest for this module.
    ///
    /// This provides a structured declaration of params, fields, and buffers
    /// that can be used for validation and code generation.
    pub port_manifest: Option<PortManifest>,
}

/// Object-safe interface for model-defined numerical modules.
///
/// Modules are small bundles of kernel passes + optional WGSL generators.
/// They are composed by the model (and/or method selection) to produce a recipe.
pub trait ModelModule {
    fn name(&self) -> &'static str;
    fn kernel_specs(&self) -> &[ModelKernelSpec];
    fn kernel_generators(&self) -> &[ModelKernelGeneratorSpec];

    fn manifest(&self) -> &ModuleManifest;
}

/// A simple data-driven module: a named bundle of kernel specs + WGSL generators.
#[derive(Debug, Clone, Default)]
pub struct KernelBundleModule {
    pub name: &'static str,
    pub kernels: Vec<ModelKernelSpec>,
    pub generators: Vec<ModelKernelGeneratorSpec>,

    /// Optional EOS configuration carried by this module.
    ///
    /// When present, the owning model is considered to have an EOS.
    pub eos: Option<EosSpec>,

    pub manifest: ModuleManifest,
}

impl ModelModule for KernelBundleModule {
    fn name(&self) -> &'static str {
        self.name
    }

    fn kernel_specs(&self) -> &[ModelKernelSpec] {
        &self.kernels
    }

    fn kernel_generators(&self) -> &[ModelKernelGeneratorSpec] {
        &self.generators
    }

    fn manifest(&self) -> &ModuleManifest {
        &self.manifest
    }
}
