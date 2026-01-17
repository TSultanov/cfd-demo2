use crate::solver::model::kernel::{ModelKernelGeneratorSpec, ModelKernelSpec};
use crate::solver::model::module::{KernelBundleModule, ModuleInvariant, ModuleManifest};
use crate::solver::model::KernelId;
use crate::solver::model::kernel::{DispatchKindId, KernelPhaseId};

pub fn rhie_chow_aux_module(
    dp_field: &'static str,
    require_vector2_momentum: bool,
    require_pressure_gradient: bool,
) -> KernelBundleModule {
    KernelBundleModule {
        name: "rhie_chow_aux",
        kernels: vec![
            ModelKernelSpec {
                id: KernelId::DP_INIT,
                phase: KernelPhaseId::Preparation,
                dispatch: DispatchKindId::Cells,
            },
            ModelKernelSpec {
                id: KernelId::RHIE_CHOW_CORRECT_VELOCITY,
                phase: KernelPhaseId::Gradients,
                dispatch: DispatchKindId::Cells,
            },
            ModelKernelSpec {
                id: KernelId::DP_UPDATE_FROM_DIAG,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
            },
        ],
        generators: vec![
            ModelKernelGeneratorSpec {
                id: KernelId::DP_INIT,
                generator: crate::solver::model::kernel::generate_dp_init_kernel_wgsl,
            },
            ModelKernelGeneratorSpec {
                id: KernelId::DP_UPDATE_FROM_DIAG,
                generator: crate::solver::model::kernel::generate_dp_update_from_diag_kernel_wgsl,
            },
            ModelKernelGeneratorSpec {
                id: KernelId::RHIE_CHOW_CORRECT_VELOCITY,
                generator: crate::solver::model::kernel::generate_rhie_chow_correct_velocity_kernel_wgsl,
            },
        ],
        manifest: ModuleManifest {
            invariants: vec![
                ModuleInvariant::RequireUniqueMomentumPressureCouplingReferencingDp {
                    dp_field,
                    require_vector2_momentum,
                    require_pressure_gradient,
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}
