//! Unified Compute Graph Builder
//!
//! This module provides utilities for building compute graphs from a SolverRecipe.
//! Instead of each solver family having its own hardcoded graph construction,
//! this module derives the graph structure from the recipe's kernel specifications.

use crate::solver::gpu::modules::graph::{
    ComputeSpec, DispatchKind, GpuComputeModule, ModuleGraph, ModuleNode,
};
use crate::solver::gpu::recipe::{KernelPhase, SolverRecipe};
use crate::solver::model::KernelId;

/// Configuration for a unified compute graph.
#[derive(Debug, Clone)]
pub struct UnifiedGraphConfig {
    pub label_prefix: &'static str,
    pub dispatch_kind: DispatchKind,
}

impl Default for UnifiedGraphConfig {
    fn default() -> Self {
        Self {
            label_prefix: "unified",
            dispatch_kind: DispatchKind::Cells,
        }
    }
}

/// Trait for modules that can be used with the unified graph builder.
///
/// This trait maps kernel kinds to pipeline/bind group keys.
/// It requires GpuComputeModule as a supertrait since the graph builder
/// needs to create ModuleGraph instances.
pub trait UnifiedGraphModule: GpuComputeModule {
    /// Get the pipeline key for a kernel id.
    fn pipeline_for_kernel(&self, id: KernelId) -> Option<Self::PipelineKey>;

    /// Get the bind key for a kernel id.
    fn bind_for_kernel(&self, id: KernelId) -> Option<Self::BindKey>;
}

fn push_nodes_for_phase<M: UnifiedGraphModule>(
    nodes: &mut Vec<ModuleNode<M::PipelineKey, M::BindKey>>,
    recipe: &SolverRecipe,
    phase: KernelPhase,
    module: &M,
    label_prefix: &'static str,
) -> Result<(), String> {
    for kernel_spec in recipe.kernels_for_phase(phase) {
        let pipeline = module
            .pipeline_for_kernel(kernel_spec.id)
            .ok_or_else(|| format!("no pipeline for kernel {}", kernel_spec.id.as_str()))?;
        let bind = module
            .bind_for_kernel(kernel_spec.id)
            .ok_or_else(|| format!("no bind group for kernel {}", kernel_spec.id.as_str()))?;
        let dispatch = kernel_spec.dispatch.clone();

        let label = kernel_label(label_prefix, kernel_spec.id);

        nodes.push(ModuleNode::Compute(ComputeSpec {
            label,
            pipeline,
            bind,
            dispatch,
        }));
    }

    Ok(())
}

/// Build a compute graph from a recipe for a specific phase.
///
/// This is a generic function that works with any module implementing UnifiedGraphModule.
pub fn build_graph_for_phase<M: UnifiedGraphModule>(
    recipe: &SolverRecipe,
    phase: KernelPhase,
    module: &M,
    label_prefix: &'static str,
) -> Result<ModuleGraph<M>, String> {
    let mut nodes = Vec::new();
    push_nodes_for_phase(&mut nodes, recipe, phase, module, label_prefix)?;

    if nodes.is_empty() {
        return Err(format!("no kernels found for phase {phase:?}"));
    }

    Ok(ModuleGraph::new(nodes))
}

/// Build a compute graph from a recipe for a sequence of phases.
///
/// Phases are appended in the order provided.
pub fn build_graph_for_phases<M: UnifiedGraphModule>(
    recipe: &SolverRecipe,
    phases: &[KernelPhase],
    module: &M,
    label_prefix: &'static str,
) -> Result<ModuleGraph<M>, String> {
    let mut nodes = Vec::new();
    for &phase in phases {
        push_nodes_for_phase(&mut nodes, recipe, phase, module, label_prefix)?;
    }

    if nodes.is_empty() {
        return Err(format!("no kernels found for phases {phases:?}"));
    }

    Ok(ModuleGraph::new(nodes))
}

/// Build a compute graph for a sequence of phases, returning None when all phases are empty.
///
/// Use this for optional composite graphs (e.g. a model family may omit a whole path).
pub fn build_optional_graph_for_phases<M: UnifiedGraphModule>(
    recipe: &SolverRecipe,
    phases: &[KernelPhase],
    module: &M,
    label_prefix: &'static str,
) -> Result<Option<ModuleGraph<M>>, String> {
    match build_graph_for_phases(recipe, phases, module, label_prefix) {
        Ok(g) => Ok(Some(g)),
        Err(e) if e.starts_with("no kernels found for phases") => Ok(None),
        Err(e) => Err(e),
    }
}

/// Build a compute graph for a phase, returning None when the phase has no kernels.
///
/// Use this for optional phases (e.g. gradients when `needs_gradients == false`).
pub fn build_optional_graph_for_phase<M: UnifiedGraphModule>(
    recipe: &SolverRecipe,
    phase: KernelPhase,
    module: &M,
    label_prefix: &'static str,
) -> Result<Option<ModuleGraph<M>>, String> {
    match build_graph_for_phase(recipe, phase, module, label_prefix) {
        Ok(g) => Ok(Some(g)),
        Err(e) if e.starts_with("no kernels found for phase") => Ok(None),
        Err(e) => Err(e),
    }
}

/// Generate a static label for a kernel.
fn kernel_label(prefix: &'static str, id: KernelId) -> &'static str {
    // Leaked: the kernel set is fixed, so a static-lifetime label per id is bounded.
    let label = format!("{}:{}", prefix, id.as_str());
    Box::leak(label.into_boxed_str())
}

/// A set of graphs for a complete solver step.
#[derive(Default)]
pub struct UnifiedGraphSet<M: UnifiedGraphModule> {
    pub preparation: Option<ModuleGraph<M>>,
    pub gradients: Option<ModuleGraph<M>>,
    pub assembly: Option<ModuleGraph<M>>,
    pub update: Option<ModuleGraph<M>>,
}

impl<M: UnifiedGraphModule> UnifiedGraphSet<M> {
    /// Build a complete graph set from a recipe.
    pub fn from_recipe(
        recipe: &SolverRecipe,
        module: &M,
        label_prefix: &'static str,
    ) -> Result<Self, String> {
        let preparation =
            build_optional_graph_for_phase(recipe, KernelPhase::Preparation, module, label_prefix)?;
        let gradients =
            build_optional_graph_for_phase(recipe, KernelPhase::Gradients, module, label_prefix)?;
        let assembly =
            build_optional_graph_for_phase(recipe, KernelPhase::Assembly, module, label_prefix)?;
        let update =
            build_optional_graph_for_phase(recipe, KernelPhase::Update, module, label_prefix)?;

        Ok(Self {
            preparation,
            gradients,
            assembly,
            update,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Concrete-module coverage lives in the integration tests.

    #[test]
    fn test_kernel_label_format() {
        let label = kernel_label("test", KernelId::GENERIC_COUPLED_ASSEMBLY);
        assert!(label.starts_with("test:"));
        assert!(label.contains("generic_coupled_assembly"));
    }
}
