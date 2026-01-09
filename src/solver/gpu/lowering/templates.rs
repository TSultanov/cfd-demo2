use crate::solver::gpu::execution_plan::GraphExecMode;
use crate::solver::gpu::plans::program::{
    ProgramCondId, ProgramCountId, ProgramGraphId, ProgramHostId, ProgramSpec, ProgramSpecBuilder,
    ProgramSpecNode,
};
use crate::solver::model::{KernelKind, ModelSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProgramTemplateKind {
    Compressible,
    IncompressibleCoupled,
    GenericCoupledScalar,
}

impl ProgramTemplateKind {
    pub(crate) fn for_model(model: &ModelSpec) -> Result<Self, String> {
        let kernel_plan = model.kernel_plan();
        let kernels = kernel_plan.kernels();

        if kernels.iter().any(|k| {
            matches!(
                k,
                KernelKind::CompressibleAssembly
                    | KernelKind::CompressibleApply
                    | KernelKind::CompressibleFluxKt
                    | KernelKind::CompressibleGradients
                    | KernelKind::CompressibleUpdate
            )
        }) {
            return Ok(Self::Compressible);
        }

        if kernels.iter().any(|k| {
            matches!(
                k,
                KernelKind::PrepareCoupled
                    | KernelKind::CoupledAssembly
                    | KernelKind::PressureAssembly
                    | KernelKind::UpdateFieldsFromCoupled
                    | KernelKind::FluxRhieChow
            )
        }) {
            return Ok(Self::IncompressibleCoupled);
        }

        if kernels.iter().any(|k| {
            matches!(
                k,
                KernelKind::GenericCoupledAssembly
                    | KernelKind::GenericCoupledApply
                    | KernelKind::GenericCoupledUpdate
            )
        }) {
            return Ok(Self::GenericCoupledScalar);
        }

        Err(format!(
            "no lowering template for model id '{}' (kernel_plan={kernels:?})",
            model.id
        ))
    }
}

pub(crate) fn build_program_spec(kind: ProgramTemplateKind) -> ProgramSpec {
    match kind {
        ProgramTemplateKind::Compressible => compressible::build_program_spec(),
        ProgramTemplateKind::IncompressibleCoupled => incompressible_coupled::build_program_spec(),
        ProgramTemplateKind::GenericCoupledScalar => generic_coupled_scalar::build_program_spec(),
    }
}

pub(crate) mod compressible {
    use super::*;

    pub(crate) const G_EXPLICIT_GRAPH: ProgramGraphId = ProgramGraphId(0);
    pub(crate) const G_IMPLICIT_GRAD_ASSEMBLY: ProgramGraphId = ProgramGraphId(1);
    pub(crate) const G_IMPLICIT_SNAPSHOT: ProgramGraphId = ProgramGraphId(2);
    pub(crate) const G_IMPLICIT_APPLY: ProgramGraphId = ProgramGraphId(3);
    pub(crate) const G_PRIMITIVE_UPDATE: ProgramGraphId = ProgramGraphId(4);

    pub(crate) const H_EXPLICIT_PREPARE: ProgramHostId = ProgramHostId(0);
    pub(crate) const H_EXPLICIT_FINALIZE: ProgramHostId = ProgramHostId(1);
    pub(crate) const H_IMPLICIT_PREPARE: ProgramHostId = ProgramHostId(2);
    pub(crate) const H_IMPLICIT_SET_ITER_PARAMS: ProgramHostId = ProgramHostId(3);
    pub(crate) const H_IMPLICIT_SOLVE_FGMRES: ProgramHostId = ProgramHostId(4);
    pub(crate) const H_IMPLICIT_RECORD_STATS: ProgramHostId = ProgramHostId(5);
    pub(crate) const H_IMPLICIT_SET_ALPHA: ProgramHostId = ProgramHostId(6);
    pub(crate) const H_IMPLICIT_RESTORE_ALPHA: ProgramHostId = ProgramHostId(7);
    pub(crate) const H_IMPLICIT_ADVANCE_OUTER_IDX: ProgramHostId = ProgramHostId(8);
    pub(crate) const H_IMPLICIT_FINALIZE: ProgramHostId = ProgramHostId(9);

    pub(crate) const C_SHOULD_USE_EXPLICIT: ProgramCondId = ProgramCondId(0);
    pub(crate) const N_IMPLICIT_OUTER_ITERS: ProgramCountId = ProgramCountId(0);

    pub(crate) fn build_program_spec() -> ProgramSpec {
        let mut program = ProgramSpecBuilder::new();
        let root = program.root();
        let explicit_block = program.new_block();
        let implicit_iter_block = program.new_block();
        let implicit_block = program.new_block();

        program.push(
            explicit_block,
            ProgramSpecNode::Host {
                label: "compressible:explicit_prepare",
                id: H_EXPLICIT_PREPARE,
            },
        );
        program.push(
            explicit_block,
            ProgramSpecNode::Graph {
                label: "compressible:explicit_graph",
                id: G_EXPLICIT_GRAPH,
                mode: GraphExecMode::SplitTimed,
            },
        );
        program.push(
            explicit_block,
            ProgramSpecNode::Host {
                label: "compressible:explicit_finalize",
                id: H_EXPLICIT_FINALIZE,
            },
        );

        for node in [
            ProgramSpecNode::Host {
                label: "compressible:implicit_set_iter_params",
                id: H_IMPLICIT_SET_ITER_PARAMS,
            },
            ProgramSpecNode::Graph {
                label: "compressible:implicit_grad_assembly",
                id: G_IMPLICIT_GRAD_ASSEMBLY,
                mode: GraphExecMode::SplitTimed,
            },
            ProgramSpecNode::Host {
                label: "compressible:implicit_fgmres",
                id: H_IMPLICIT_SOLVE_FGMRES,
            },
            ProgramSpecNode::Host {
                label: "compressible:implicit_record_stats",
                id: H_IMPLICIT_RECORD_STATS,
            },
            ProgramSpecNode::Graph {
                label: "compressible:implicit_snapshot",
                id: G_IMPLICIT_SNAPSHOT,
                mode: GraphExecMode::SingleSubmit,
            },
            ProgramSpecNode::Host {
                label: "compressible:implicit_set_alpha",
                id: H_IMPLICIT_SET_ALPHA,
            },
            ProgramSpecNode::Graph {
                label: "compressible:implicit_apply",
                id: G_IMPLICIT_APPLY,
                mode: GraphExecMode::SingleSubmit,
            },
            ProgramSpecNode::Host {
                label: "compressible:implicit_restore_alpha",
                id: H_IMPLICIT_RESTORE_ALPHA,
            },
            ProgramSpecNode::Host {
                label: "compressible:implicit_outer_idx_inc",
                id: H_IMPLICIT_ADVANCE_OUTER_IDX,
            },
        ] {
            program.push(implicit_iter_block, node);
        }

        program.push(
            implicit_block,
            ProgramSpecNode::Host {
                label: "compressible:implicit_prepare",
                id: H_IMPLICIT_PREPARE,
            },
        );
        program.push(
            implicit_block,
            ProgramSpecNode::Repeat {
                label: "compressible:implicit_outer_loop",
                times: N_IMPLICIT_OUTER_ITERS,
                body: implicit_iter_block,
            },
        );
        program.push(
            implicit_block,
            ProgramSpecNode::Graph {
                label: "compressible:primitive_update",
                id: G_PRIMITIVE_UPDATE,
                mode: GraphExecMode::SingleSubmit,
            },
        );
        program.push(
            implicit_block,
            ProgramSpecNode::Host {
                label: "compressible:implicit_finalize",
                id: H_IMPLICIT_FINALIZE,
            },
        );

        program.push(
            root,
            ProgramSpecNode::If {
                label: "compressible:select_step_path",
                cond: C_SHOULD_USE_EXPLICIT,
                then_block: explicit_block,
                else_block: Some(implicit_block),
            },
        );

        program.build()
    }
}

pub(crate) mod incompressible_coupled {
    use super::*;

    pub(crate) const G_COUPLED_PREPARE_ASSEMBLY: ProgramGraphId = ProgramGraphId(0);
    pub(crate) const G_COUPLED_ASSEMBLY: ProgramGraphId = ProgramGraphId(1);
    pub(crate) const G_COUPLED_UPDATE: ProgramGraphId = ProgramGraphId(2);
    pub(crate) const G_COUPLED_INIT_PREPARE: ProgramGraphId = ProgramGraphId(3);

    pub(crate) const H_COUPLED_BEGIN_STEP: ProgramHostId = ProgramHostId(0);
    pub(crate) const H_COUPLED_BEFORE_ITER: ProgramHostId = ProgramHostId(1);
    pub(crate) const H_COUPLED_SOLVE: ProgramHostId = ProgramHostId(2);
    pub(crate) const H_COUPLED_CLEAR_MAX_DIFF: ProgramHostId = ProgramHostId(3);
    pub(crate) const H_COUPLED_CONVERGENCE_ADVANCE: ProgramHostId = ProgramHostId(4);
    pub(crate) const H_COUPLED_FINALIZE_STEP: ProgramHostId = ProgramHostId(5);

    pub(crate) const C_HAS_COUPLED_RESOURCES: ProgramCondId = ProgramCondId(0);
    pub(crate) const C_COUPLED_NEEDS_PREPARE: ProgramCondId = ProgramCondId(1);
    pub(crate) const C_COUPLED_SHOULD_CONTINUE: ProgramCondId = ProgramCondId(2);

    pub(crate) const N_COUPLED_MAX_ITERS: ProgramCountId = ProgramCountId(0);

    pub(crate) fn build_program_spec() -> ProgramSpec {
        let mut program = ProgramSpecBuilder::new();
        let root = program.root();
        let coupled_iter_block = program.new_block();
        let coupled_prepare_block = program.new_block();
        let coupled_assembly_block = program.new_block();
        let coupled_step_block = program.new_block();

        program.push(
            coupled_prepare_block,
            ProgramSpecNode::Graph {
                label: "incompressible:coupled_prepare_assembly",
                id: G_COUPLED_PREPARE_ASSEMBLY,
                mode: GraphExecMode::SingleSubmit,
            },
        );
        program.push(
            coupled_assembly_block,
            ProgramSpecNode::Graph {
                label: "incompressible:coupled_assembly",
                id: G_COUPLED_ASSEMBLY,
                mode: GraphExecMode::SingleSubmit,
            },
        );

        for node in [
            ProgramSpecNode::Host {
                label: "incompressible:coupled_before_iter",
                id: H_COUPLED_BEFORE_ITER,
            },
            ProgramSpecNode::If {
                label: "incompressible:coupled_prepare_or_assembly",
                cond: C_COUPLED_NEEDS_PREPARE,
                then_block: coupled_prepare_block,
                else_block: Some(coupled_assembly_block),
            },
            ProgramSpecNode::Host {
                label: "incompressible:coupled_solve",
                id: H_COUPLED_SOLVE,
            },
            ProgramSpecNode::Host {
                label: "incompressible:coupled_clear_max_diff",
                id: H_COUPLED_CLEAR_MAX_DIFF,
            },
            ProgramSpecNode::Graph {
                label: "incompressible:coupled_update_fields_max_diff",
                id: G_COUPLED_UPDATE,
                mode: GraphExecMode::SingleSubmit,
            },
            ProgramSpecNode::Host {
                label: "incompressible:coupled_convergence_and_advance",
                id: H_COUPLED_CONVERGENCE_ADVANCE,
            },
        ] {
            program.push(coupled_iter_block, node);
        }

        for node in [
            ProgramSpecNode::Host {
                label: "incompressible:coupled_begin_step",
                id: H_COUPLED_BEGIN_STEP,
            },
            ProgramSpecNode::Graph {
                label: "incompressible:coupled_init_prepare",
                id: G_COUPLED_INIT_PREPARE,
                mode: GraphExecMode::SingleSubmit,
            },
            ProgramSpecNode::While {
                label: "incompressible:coupled_outer_loop",
                max_iters: N_COUPLED_MAX_ITERS,
                cond: C_COUPLED_SHOULD_CONTINUE,
                body: coupled_iter_block,
            },
            ProgramSpecNode::Host {
                label: "incompressible:coupled_finalize_step",
                id: H_COUPLED_FINALIZE_STEP,
            },
        ] {
            program.push(coupled_step_block, node);
        }

        program.push(
            root,
            ProgramSpecNode::If {
                label: "incompressible:step",
                cond: C_HAS_COUPLED_RESOURCES,
                then_block: coupled_step_block,
                else_block: None,
            },
        );

        program.build()
    }
}

pub(crate) mod generic_coupled_scalar {
    use super::*;

    pub(crate) const G_ASSEMBLY: ProgramGraphId = ProgramGraphId(0);
    pub(crate) const G_UPDATE: ProgramGraphId = ProgramGraphId(1);

    pub(crate) const H_PREPARE: ProgramHostId = ProgramHostId(0);
    pub(crate) const H_SOLVE: ProgramHostId = ProgramHostId(1);

    pub(crate) fn build_program_spec() -> ProgramSpec {
        let mut program = ProgramSpecBuilder::new();
        let root = program.root();

        for node in [
            ProgramSpecNode::Host {
                label: "generic_coupled:prepare",
                id: H_PREPARE,
            },
            ProgramSpecNode::Graph {
                label: "generic_coupled:assembly",
                id: G_ASSEMBLY,
                mode: GraphExecMode::SplitTimed,
            },
            ProgramSpecNode::Host {
                label: "generic_coupled:solve",
                id: H_SOLVE,
            },
            ProgramSpecNode::Graph {
                label: "generic_coupled:update",
                id: G_UPDATE,
                mode: GraphExecMode::SingleSubmit,
            },
        ] {
            program.push(root, node);
        }

        program.build()
    }
}

