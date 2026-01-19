use crate::solver::model::kernel::{KernelConditionId, ModelKernelGeneratorSpec, ModelKernelSpec};
use crate::solver::model::module::{KernelBundleModule, ModuleInvariant, ModuleManifest};
use crate::solver::model::KernelId;
use crate::solver::model::kernel::{DispatchKindId, KernelPhaseId};

pub fn rhie_chow_aux_module(
    dp_field: &'static str,
    require_vector2_momentum: bool,
    require_pressure_gradient: bool,
) -> KernelBundleModule {
    let kernel_dp_init = KernelId("dp_init");
    let kernel_dp_update_from_diag = KernelId("dp_update_from_diag");
    let kernel_rhie_chow_correct_velocity = KernelId("rhie_chow/correct_velocity");

    KernelBundleModule {
        name: "rhie_chow_aux",
        kernels: vec![
            ModelKernelSpec {
                id: kernel_dp_init,
                phase: KernelPhaseId::Preparation,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
            ModelKernelSpec {
                id: kernel_rhie_chow_correct_velocity,
                phase: KernelPhaseId::Gradients,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
            ModelKernelSpec {
                id: kernel_dp_update_from_diag,
                phase: KernelPhaseId::Update,
                dispatch: DispatchKindId::Cells,
                condition: KernelConditionId::Always,
            },
        ],
        generators: vec![
            ModelKernelGeneratorSpec::new(
                kernel_dp_init,
                crate::solver::model::kernel::generate_dp_init_kernel_wgsl,
            ),
            ModelKernelGeneratorSpec::new(
                kernel_dp_update_from_diag,
                crate::solver::model::kernel::generate_dp_update_from_diag_kernel_wgsl,
            ),
            ModelKernelGeneratorSpec::new(
                kernel_rhie_chow_correct_velocity,
                crate::solver::model::kernel::generate_rhie_chow_correct_velocity_kernel_wgsl,
            ),
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
