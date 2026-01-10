use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::modules::graph::{
    ComputeSpec, DispatchKind, ModuleGraph, ModuleNode, RuntimeDims,
};
use crate::solver::gpu::modules::model_kernels::{
    KernelBindGroups, KernelPipeline, ModelKernelsModule,
};
use crate::solver::gpu::modules::unified_graph::build_optional_graph_for_phase;
use crate::solver::gpu::recipe::{KernelPhase, SolverRecipe};
use crate::solver::model::KernelId;

pub struct CompressibleGraphs {
    explicit_module_graph: ModuleGraph<ModelKernelsModule>,
    explicit_module_graph_first_order: ModuleGraph<ModelKernelsModule>,
    implicit_grad_assembly_module_graph: ModuleGraph<ModelKernelsModule>,
    implicit_assembly_module_graph_first_order: ModuleGraph<ModelKernelsModule>,
    implicit_apply_module_graph: ModuleGraph<ModelKernelsModule>,
    primitive_update_module_graph: ModuleGraph<ModelKernelsModule>,
}

impl CompressibleGraphs {
    /// Create graphs from a SolverRecipe, falling back to hardcoded graphs
    /// when the recipe doesn't define the needed phases.
    pub fn from_recipe(recipe: &SolverRecipe, kernels: &ModelKernelsModule) -> Result<Self, String> {
        // Try to build graphs from recipe phases; fall back to hardcoded
        // TODO: Use these once we have composite graph building from recipe
        let _gradients_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Gradients,
            kernels,
            "compressible",
        )?;
        let _assembly_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Assembly,
            kernels,
            "compressible",
        )?;

        // These are used today; treat non-empty phases as required if present.
        let apply_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Apply,
            kernels,
            "compressible",
        )?;
        let update_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::PrimitiveRecovery,
            kernels,
            "compressible",
        )?;

        // For now, use hardcoded graphs since compressible has complex multi-phase sequences
        // TODO: Once recipe fully describes all phases, switch to recipe-driven construction
        Ok(Self {
            explicit_module_graph: Self::build_explicit_module_graph(true),
            explicit_module_graph_first_order: Self::build_explicit_module_graph(false),
            implicit_grad_assembly_module_graph: Self::build_implicit_grad_assembly_module_graph(
                true,
            ),
            implicit_assembly_module_graph_first_order:
                Self::build_implicit_grad_assembly_module_graph(false),
            implicit_apply_module_graph: apply_graph
                .unwrap_or_else(|| Self::build_implicit_apply_module_graph()),
            primitive_update_module_graph: update_graph
                .unwrap_or_else(|| Self::build_primitive_update_module_graph()),
        })
    }

    pub fn new() -> Self {
        Self {
            explicit_module_graph: Self::build_explicit_module_graph(true),
            explicit_module_graph_first_order: Self::build_explicit_module_graph(false),
            implicit_grad_assembly_module_graph: Self::build_implicit_grad_assembly_module_graph(
                true,
            ),
            implicit_assembly_module_graph_first_order:
                Self::build_implicit_grad_assembly_module_graph(false),
            implicit_apply_module_graph: Self::build_implicit_apply_module_graph(),
            primitive_update_module_graph: Self::build_primitive_update_module_graph(),
        }
    }

    fn build_explicit_module_graph(include_gradients: bool) -> ModuleGraph<ModelKernelsModule> {
        let mut nodes = Vec::new();
        if include_gradients {
            nodes.push(ModuleNode::Compute(ComputeSpec {
                label: "compressible:gradients",
                pipeline: KernelPipeline::Kernel(KernelId::COMPRESSIBLE_GRADIENTS),
                bind: KernelBindGroups::MeshFields,
                dispatch: DispatchKind::Cells,
            }));
        }
        nodes.push(ModuleNode::Compute(ComputeSpec {
            label: "compressible:flux_kt",
            pipeline: KernelPipeline::Kernel(KernelId::COMPRESSIBLE_FLUX_KT),
            bind: KernelBindGroups::MeshFields,
            dispatch: DispatchKind::Faces,
        }));
        nodes.push(ModuleNode::Compute(ComputeSpec {
            label: "compressible:explicit_update",
            pipeline: KernelPipeline::Kernel(KernelId::COMPRESSIBLE_EXPLICIT_UPDATE),
            bind: KernelBindGroups::MeshFields,
            dispatch: DispatchKind::Cells,
        }));
        nodes.push(ModuleNode::Compute(ComputeSpec {
            label: "compressible:primitive_update",
            pipeline: KernelPipeline::Kernel(KernelId::COMPRESSIBLE_UPDATE),
            bind: KernelBindGroups::FieldsOnly,
            dispatch: DispatchKind::Cells,
        }));
        ModuleGraph::new(nodes)
    }

    fn build_implicit_grad_assembly_module_graph(
        include_gradients: bool,
    ) -> ModuleGraph<ModelKernelsModule> {
        let mut nodes = Vec::new();
        if include_gradients {
            nodes.push(ModuleNode::Compute(ComputeSpec {
                label: "compressible:gradients",
                pipeline: KernelPipeline::Kernel(KernelId::COMPRESSIBLE_GRADIENTS),
                bind: KernelBindGroups::MeshFields,
                dispatch: DispatchKind::Cells,
            }));
        }
        nodes.push(ModuleNode::Compute(ComputeSpec {
            label: "compressible:assembly",
            pipeline: KernelPipeline::Kernel(KernelId::COMPRESSIBLE_ASSEMBLY),
            bind: KernelBindGroups::MeshFieldsSolver,
            dispatch: DispatchKind::Cells,
        }));
        ModuleGraph::new(nodes)
    }

    fn build_implicit_apply_module_graph() -> ModuleGraph<ModelKernelsModule> {
        ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
            label: "compressible:apply",
            pipeline: KernelPipeline::Kernel(KernelId::COMPRESSIBLE_APPLY),
            bind: KernelBindGroups::ApplyFieldsSolver,
            dispatch: DispatchKind::Cells,
        })])
    }

    fn build_primitive_update_module_graph() -> ModuleGraph<ModelKernelsModule> {
        ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
            label: "compressible:primitive_update",
            pipeline: KernelPipeline::Kernel(KernelId::COMPRESSIBLE_UPDATE),
            bind: KernelBindGroups::FieldsOnly,
            dispatch: DispatchKind::Cells,
        })])
    }

    pub fn run_explicit(
        &self,
        context: &GpuContext,
        kernels: &ModelKernelsModule,
        dims: RuntimeDims,
        mode: GraphExecMode,
        needs_gradients: bool,
    ) -> (f64, Option<GraphDetail>) {
        let graph = if needs_gradients {
            &self.explicit_module_graph
        } else {
            &self.explicit_module_graph_first_order
        };
        run_module_graph(graph, context, kernels, dims, mode)
    }

    pub fn run_implicit_grad_assembly(
        &self,
        context: &GpuContext,
        kernels: &ModelKernelsModule,
        dims: RuntimeDims,
        mode: GraphExecMode,
        needs_gradients: bool,
    ) -> (f64, Option<GraphDetail>) {
        let graph = if needs_gradients {
            &self.implicit_grad_assembly_module_graph
        } else {
            &self.implicit_assembly_module_graph_first_order
        };
        run_module_graph(graph, context, kernels, dims, mode)
    }

    pub fn run_implicit_apply(
        &self,
        context: &GpuContext,
        kernels: &ModelKernelsModule,
        dims: RuntimeDims,
        mode: GraphExecMode,
    ) -> (f64, Option<GraphDetail>) {
        run_module_graph(
            &self.implicit_apply_module_graph,
            context,
            kernels,
            dims,
            mode,
        )
    }

    pub fn run_primitive_update(
        &self,
        context: &GpuContext,
        kernels: &ModelKernelsModule,
        dims: RuntimeDims,
        mode: GraphExecMode,
    ) -> (f64, Option<GraphDetail>) {
        run_module_graph(
            &self.primitive_update_module_graph,
            context,
            kernels,
            dims,
            mode,
        )
    }
}
