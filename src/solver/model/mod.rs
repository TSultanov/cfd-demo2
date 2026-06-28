#[doc(hidden)]
pub mod backend;
mod definitions;
pub mod eos;
pub mod flux_derivation;
pub mod flux_layout;
pub mod flux_module;
pub mod flux_schemes;
pub mod gpu_spec;
pub mod helpers;
pub mod invariants;
pub mod kernel;
pub mod linear_solver;
pub mod method;
pub mod module;
pub mod modules;
pub mod ports;
pub mod primitives;

pub use crate::solver::ir::LimiterSpec;
pub use definitions::{
    all_models, buoyant_incompressible_mms_model, buoyant_incompressible_model,
    compressible_central_upwind_decl, compressible_generalized_wave_speed_sq,
    compressible_mms_biharmonic_model, compressible_mms_model, compressible_model,
    compressible_model_with_eos, compressible_system,
    compressible_wave_speed_sq, generic_diffusion_demo_mms_dirichlet_model,
    generic_diffusion_demo_mms_model,
    generic_diffusion_demo_mms_neumann_model, generic_diffusion_demo_model,
    generic_diffusion_demo_neumann_model, incompressible_momentum_mms_model,
    incompressible_momentum_model, incompressible_momentum_system, scalar_transport_model,
    scalar_transport_sou_model,
    allmach_pressure_mms_model, allmach_pressure_model, allmach_pressure_system,
    allmach_thermal_mms_model, allmach_thermal_model,
    BcValue, BoundaryCondition, BoundarySpec, CompressibleFields, FieldBoundarySpec,
    GenericCoupledFields,
    AllMachPressureFields, IncompressibleMomentumFields, ModelSpec, ADVECTING_VELOCITY_FIELD,
    ALLMACH_K_OVER_CP, ALLMACH_MMS_SOURCE_P_FIELD, ALLMACH_MMS_SOURCE_T_FIELD,
    ALLMACH_MMS_SOURCE_U_FIELD, ALLMACH_RHO_T_REF_FIELD, ALLMACH_TEMPERATURE_FIELD, ALLMACH_T_REF,
    BUOYANT_BETA_G,
    BUOYANT_K_OVER_CP, BUOYANT_MMS_SOURCE_T_FIELD, BUOYANT_MMS_SOURCE_U_FIELD, BUOYANT_T0,
    BUOYANT_TEMPERATURE_FIELD, COMPRESSIBLE_MMS_SOURCE_RHO_E_FIELD,
    COMPRESSIBLE_MMS_SOURCE_RHO_FIELD, COMPRESSIBLE_MMS_SOURCE_RHO_U_FIELD,
    INCOMPRESSIBLE_MMS_SOURCE_FIELD, MMS_SOURCE_FIELD,
    SCALAR_TRANSPORT_FIELD, SCALAR_TRANSPORT_KAPPA, SCALAR_TRANSPORT_MMS_SOURCE_FIELD,
};
pub use eos::EosSpec;
pub use flux_layout::{FluxComponent, FluxLayout};
pub use flux_module::FluxModuleSpec;
pub use gpu_spec::{expand_field_components, GradientStorage};
pub use kernel::KernelId;
pub use linear_solver::SchurBlockLayout;
pub use linear_solver::{ModelLinearSolverSpec, ModelPreconditionerSpec};
pub use method::MethodSpec;
pub use module::{KernelBundleModule, ModelModule};
pub use primitives::PrimitiveDerivations;
