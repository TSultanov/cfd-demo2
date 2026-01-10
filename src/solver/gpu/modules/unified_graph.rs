//! Unified Compute Graph Builder
//!
//! This module provides utilities for building compute graphs from a SolverRecipe.
//! Instead of each solver family having its own hardcoded graph construction,
//! this module derives the graph structure from the recipe's kernel specifications.

use crate::solver::gpu::modules::graph::{ComputeSpec, DispatchKind, GpuComputeModule, ModuleGraph, ModuleNode};
use crate::solver::gpu::recipe::{KernelPhase, SolverRecipe};
use crate::solver::model::KernelId;

/// Configuration for a unified compute graph.
#[derive(Debug, Clone)]
pub struct UnifiedGraphConfig {
    /// Label prefix for nodes
    pub label_prefix: &'static str,
    /// Whether to dispatch per cell or per face
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
    
    for kernel_spec in recipe.kernels_for_phase(phase) {
        let pipeline = module
            .pipeline_for_kernel(kernel_spec.id)
            .ok_or_else(|| format!("no pipeline for kernel {}", kernel_spec.id.as_str()))?;
        let bind = module
            .bind_for_kernel(kernel_spec.id)
            .ok_or_else(|| format!("no bind group for kernel {}", kernel_spec.id.as_str()))?;
        let dispatch = kernel_spec.dispatch;

        let label = kernel_label(label_prefix, kernel_spec.id);
        
        nodes.push(ModuleNode::Compute(ComputeSpec {
            label,
            pipeline,
            bind,
            dispatch,
        }));
    }

    if nodes.is_empty() {
        return Err(format!("no kernels found for phase {phase:?}"));
    }
    
    Ok(ModuleGraph::new(nodes))
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
    // Use leaked strings for static labels
    // In practice, these are a fixed set so leaking is acceptable
    let label = format!("{}:{}", prefix, id.as_str());
    Box::leak(label.into_boxed_str())
}

/// A set of graphs for a complete solver step.
#[derive(Default)]
pub struct UnifiedGraphSet<M: UnifiedGraphModule> {
    /// Preparation phase graph (if any)
    pub preparation: Option<ModuleGraph<M>>,
    /// Gradient computation graph (if any)
    pub gradients: Option<ModuleGraph<M>>,
    /// Assembly phase graph
    pub assembly: Option<ModuleGraph<M>>,
    /// State update graph
    pub update: Option<ModuleGraph<M>>,
}

impl<M: UnifiedGraphModule> UnifiedGraphSet<M> {
    /// Build a complete graph set from a recipe.
    pub fn from_recipe(
        recipe: &SolverRecipe,
        module: &M,
        label_prefix: &'static str,
    ) -> Result<Self, String> {
        let preparation = build_optional_graph_for_phase(recipe, KernelPhase::Preparation, module, label_prefix)?;
        let gradients = build_optional_graph_for_phase(recipe, KernelPhase::Gradients, module, label_prefix)?;
        let assembly = build_optional_graph_for_phase(recipe, KernelPhase::Assembly, module, label_prefix)?;
        let update = build_optional_graph_for_phase(recipe, KernelPhase::Update, module, label_prefix)?;
        
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
    
    // Note: These tests require a concrete module implementation to test.
    // The module is tested through integration tests with actual solver modules.
    
    #[test]
    fn test_kernel_label_format() {
        let label = kernel_label("test", KernelId::GENERIC_COUPLED_ASSEMBLY);
        assert!(label.starts_with("test:"));
        assert!(label.contains("generic_coupled_assembly"));
    }
}
