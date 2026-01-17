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
    let requires_low_mach_params = matches!(eos, EosSpec::IdealGas { .. } | EosSpec::LinearCompressibility { .. });

    if requires_low_mach_params {
        manifest.named_params = vec![
            NamedParamKey::Key("eos.gamma"),
            NamedParamKey::Key("eos.gm1"),
            NamedParamKey::Key("eos.r"),
            NamedParamKey::Key("eos.dp_drho"),
            NamedParamKey::Key("eos.p_offset"),
            NamedParamKey::Key("eos.theta_ref"),
            NamedParamKey::Key("low_mach.model"),
            NamedParamKey::Key("low_mach.theta_floor"),
        ];
    }

    KernelBundleModule {
        name: "eos",
        eos: Some(eos),
        manifest,
        ..Default::default()
    }
}
