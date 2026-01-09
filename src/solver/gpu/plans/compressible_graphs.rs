use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::modules::graph::{
    ComputeSpec, DispatchKind, ModuleGraph, ModuleNode, RuntimeDims,
};
use crate::solver::gpu::modules::model_kernels::{
    KernelBindGroups, KernelPipeline, ModelKernelsModule,
};
use crate::solver::model::KernelKind;

pub struct CompressibleGraphs {
    explicit_module_graph: ModuleGraph<ModelKernelsModule>,
    explicit_module_graph_first_order: ModuleGraph<ModelKernelsModule>,
    implicit_grad_assembly_module_graph: ModuleGraph<ModelKernelsModule>,
    implicit_assembly_module_graph_first_order: ModuleGraph<ModelKernelsModule>,
    implicit_apply_module_graph: ModuleGraph<ModelKernelsModule>,
    primitive_update_module_graph: ModuleGraph<ModelKernelsModule>,
}

impl CompressibleGraphs {
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
                pipeline: KernelPipeline::Kernel(KernelKind::CompressibleGradients),
                bind: KernelBindGroups::MeshFields,
                dispatch: DispatchKind::Cells,
            }));
        }
        nodes.push(ModuleNode::Compute(ComputeSpec {
            label: "compressible:flux_kt",
            pipeline: KernelPipeline::Kernel(KernelKind::CompressibleFluxKt),
            bind: KernelBindGroups::MeshFields,
            dispatch: DispatchKind::Faces,
        }));
        nodes.push(ModuleNode::Compute(ComputeSpec {
            label: "compressible:explicit_update",
            pipeline: KernelPipeline::Kernel(KernelKind::CompressibleExplicitUpdate),
            bind: KernelBindGroups::MeshFields,
            dispatch: DispatchKind::Cells,
        }));
        nodes.push(ModuleNode::Compute(ComputeSpec {
            label: "compressible:primitive_update",
            pipeline: KernelPipeline::Kernel(KernelKind::CompressibleUpdate),
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
                pipeline: KernelPipeline::Kernel(KernelKind::CompressibleGradients),
                bind: KernelBindGroups::MeshFields,
                dispatch: DispatchKind::Cells,
            }));
        }
        nodes.push(ModuleNode::Compute(ComputeSpec {
            label: "compressible:assembly",
            pipeline: KernelPipeline::Kernel(KernelKind::CompressibleAssembly),
            bind: KernelBindGroups::MeshFieldsSolver,
            dispatch: DispatchKind::Cells,
        }));
        ModuleGraph::new(nodes)
    }

    fn build_implicit_apply_module_graph() -> ModuleGraph<ModelKernelsModule> {
        ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
            label: "compressible:apply",
            pipeline: KernelPipeline::Kernel(KernelKind::CompressibleApply),
            bind: KernelBindGroups::ApplyFieldsSolver,
            dispatch: DispatchKind::Cells,
        })])
    }

    fn build_primitive_update_module_graph() -> ModuleGraph<ModelKernelsModule> {
        ModuleGraph::new(vec![ModuleNode::Compute(ComputeSpec {
            label: "compressible:primitive_update",
            pipeline: KernelPipeline::Kernel(KernelKind::CompressibleUpdate),
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
