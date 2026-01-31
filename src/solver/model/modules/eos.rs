use crate::solver::model::eos::EosSpec;
use crate::solver::model::module::{KernelBundleModule, ModuleManifest, NamedParamKey};

/// EOS is treated as a model-defined module so its named parameters and
/// resource requirements are declared via manifests (rather than implied
/// by top-level `ModelSpec` fields).
///
/// This module does not contribute kernels.
pub fn eos_module(eos: EosSpec) -> KernelBundleModule {
    let mut manifest = ModuleManifest::default();

    // For now, EOS tuning + low-mach knobs are only meaningful for compressible EOS variants.
    // Keep the contract consistent with the previous EOS-implied behavior.
    let requires_low_mach_params = matches!(
        eos,
        EosSpec::IdealGas { .. } | EosSpec::LinearCompressibility { .. }
    );

    if requires_low_mach_params {
        // Set the port-based manifest for uniform params
        // This is done via a helper function in a separate module to avoid
        // proc-macro issues in build scripts
        manifest.port_manifest =
            Some(crate::solver::model::modules::eos_ports::eos_uniform_port_manifest());

        // Keep named_params for non-uniform parameters and backward compatibility
        // Note: low_mach.model is a u32 enum, not representable as ParamPort<F32, _>
        // EOS uniform params are now declared via port_manifest, so we only list
        // low_mach params here.
        manifest.named_params = vec![
            NamedParamKey::Key("low_mach.model"),
            NamedParamKey::Key("low_mach.theta_floor"),
            NamedParamKey::Key("low_mach.pressure_coupling_alpha"),
        ];
    }

    KernelBundleModule {
        name: "eos",
        eos: Some(eos),
        manifest,
        ..Default::default()
    }
}
