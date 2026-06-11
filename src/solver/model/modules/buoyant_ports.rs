/// Buoyant Boussinesq port definitions - separate from the model definition to
/// avoid build script issues (this file is also compiled into build.rs).
use crate::solver::ir::ports::ParamSpec;
use crate::solver::model::module::PortManifest as ModulePortManifest;
use cfd2_ir::dimensions::{
    Acceleration, Density, DivDim, Length, MulDim, Temperature, Time, UnitDimension, Volume,
};

/// Port manifest for the buoyant model's runtime uniform params.
///
/// LAYOUT CONTRACT: these specs become Constants-struct tail fields appended
/// AFTER the canonical EOS block (see `extract_eos_params`), and the host
/// `GpuConstants` POD must carry the same fields in the same order at its
/// tail — the buffer is written wholesale, so WGSL offsets are only correct
/// if the two declarations mirror each other.
pub fn buoyant_uniform_port_manifest() -> ModulePortManifest {
    // beta*|g|: acceleration per kelvin.
    let beta_g_unit = DivDim::<Acceleration, Temperature>::UNIT;
    // k/cp = rho * thermal diffusivity: kg/(m*s), expressed as
    // Density*Volume/(Length*Time) to match the T-equation laplacian coeff.
    let k_over_cp_unit = DivDim::<MulDim<Density, Volume>, MulDim<Length, Time>>::UNIT;

    ModulePortManifest {
        params: vec![
            ParamSpec {
                key: "buoyant.beta_g",
                wgsl_field: "buoyant_beta_g",
                wgsl_type: "f32",
                unit: beta_g_unit,
            },
            ParamSpec {
                key: "buoyant.t0",
                wgsl_field: "buoyant_t0",
                wgsl_type: "f32",
                unit: Temperature::UNIT,
            },
            ParamSpec {
                key: "buoyant.k_over_cp",
                wgsl_field: "buoyant_k_over_cp",
                wgsl_type: "f32",
                unit: k_over_cp_unit,
            },
        ],
        fields: vec![],
        buffers: vec![],
        gradient_targets: vec![],
        resolved_state_slots: None,
    }
}

/// Module declaring the buoyant runtime params (no kernels of its own).
pub fn buoyant_params_module() -> crate::solver::model::module::KernelBundleModule {
    crate::solver::model::module::KernelBundleModule {
        name: "buoyant_params",
        port_manifest: Some(buoyant_uniform_port_manifest()),
        ..Default::default()
    }
}
