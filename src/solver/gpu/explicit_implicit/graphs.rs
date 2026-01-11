use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::execution_plan::{run_module_graph, GraphDetail, GraphExecMode};
use crate::solver::gpu::modules::graph::{ModuleGraph, RuntimeDims};
use crate::solver::gpu::modules::model_kernels::ModelKernelsModule;
use crate::solver::gpu::modules::unified_graph::{
    build_optional_graph_for_phase, build_optional_graph_for_phases,
};
use crate::solver::gpu::recipe::{KernelPhase, SolverRecipe};

pub struct CompressibleGraphs {
    explicit_module_graph: ModuleGraph<ModelKernelsModule>,
    explicit_module_graph_first_order: ModuleGraph<ModelKernelsModule>,
    implicit_grad_assembly_module_graph: ModuleGraph<ModelKernelsModule>,
    implicit_assembly_module_graph_first_order: ModuleGraph<ModelKernelsModule>,
    implicit_apply_module_graph: ModuleGraph<ModelKernelsModule>,
    primitive_update_module_graph: ModuleGraph<ModelKernelsModule>,
}

impl CompressibleGraphs {
    /// Create graphs from a SolverRecipe, requiring the necessary phases to be present.
    pub fn from_recipe(
        recipe: &SolverRecipe,
        kernels: &ModelKernelsModule,
    ) -> Result<Self, String> {
        // Explicit path: gradients (optional) -> flux -> explicit update -> primitive recovery
        let explicit_graph = build_optional_graph_for_phases(
            recipe,
            &[
                KernelPhase::Gradients,
                KernelPhase::FluxComputation,
                KernelPhase::ExplicitUpdate,
                KernelPhase::PrimitiveRecovery,
            ],
            kernels,
            "compressible",
        )?
        .ok_or_else(|| {
            "compressible: missing recipe phases for explicit kernel sequence".to_string()
        })?;

        // First-order variant: omit gradients even if available.
        let explicit_graph_first_order = build_optional_graph_for_phases(
            recipe,
            &[
                KernelPhase::FluxComputation,
                KernelPhase::ExplicitUpdate,
                KernelPhase::PrimitiveRecovery,
            ],
            kernels,
            "compressible",
        )?
        .ok_or_else(|| {
            "compressible: missing recipe phases for first-order explicit kernel sequence"
                .to_string()
        })?;

        // Implicit path: gradients (optional) -> assembly
        let implicit_grad_assembly_graph = build_optional_graph_for_phases(
            recipe,
            &[KernelPhase::Gradients, KernelPhase::Assembly],
            kernels,
            "compressible",
        )?
        .ok_or_else(|| "compressible: missing recipe phases for implicit assembly".to_string())?;
        let implicit_assembly_graph_first_order = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Assembly,
            kernels,
            "compressible",
        )?
        .ok_or_else(|| "compressible: missing recipe phase for assembly".to_string())?;

        // These are used today; treat non-empty phases as required if present.
        let apply_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::Apply,
            kernels,
            "compressible",
        )?
        .ok_or_else(|| "compressible: missing recipe phase for apply".to_string())?;
        let primitive_update_graph = build_optional_graph_for_phase(
            recipe,
            KernelPhase::PrimitiveRecovery,
            kernels,
            "compressible",
        )?
        .ok_or_else(|| "compressible: missing recipe phase for primitive recovery".to_string())?;

        Ok(Self {
            explicit_module_graph: explicit_graph,
            explicit_module_graph_first_order: explicit_graph_first_order,
            implicit_grad_assembly_module_graph: implicit_grad_assembly_graph,
            implicit_assembly_module_graph_first_order: implicit_assembly_graph_first_order,
            implicit_apply_module_graph: apply_graph,
            primitive_update_module_graph: primitive_update_graph,
        })
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
