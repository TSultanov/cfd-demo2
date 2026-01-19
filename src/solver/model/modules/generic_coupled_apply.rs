use crate::solver::model::kernel::{DispatchKindId, KernelConditionId, KernelPhaseId, ModelKernelSpec};
use crate::solver::model::module::KernelBundleModule;
use crate::solver::model::KernelId;

pub fn generic_coupled_apply_module() -> KernelBundleModule {
    KernelBundleModule {
        name: "generic_coupled_apply",
        kernels: vec![ModelKernelSpec {
            id: KernelId::GENERIC_COUPLED_APPLY,
            phase: KernelPhaseId::Apply,
            dispatch: DispatchKindId::Cells,
            condition: KernelConditionId::RequiresImplicitStepping,
        }],
        ..Default::default()
    }
}
