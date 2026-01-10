//! Unified Compute Graph Builder
//!
//! This module provides utilities for building compute graphs from a SolverRecipe.
//! Instead of each solver family having its own hardcoded graph construction,
//! this module derives the graph structure from the recipe's kernel specifications.

use crate::solver::gpu::modules::graph::{ComputeSpec, DispatchKind, GpuComputeModule, ModuleGraph, ModuleNode};
use crate::solver::gpu::recipe::{KernelPhase, KernelSpec, SolverRecipe};
use crate::solver::model::KernelKind;

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
    /// Get the pipeline key for a kernel kind.
    fn pipeline_for_kernel(&self, kind: KernelKind) -> Option<Self::PipelineKey>;
    
    /// Get the bind key for a kernel kind.
    fn bind_for_kernel(&self, kind: KernelKind) -> Option<Self::BindKey>;
    
    /// Get the dispatch kind for a kernel kind.
    fn dispatch_for_kernel(&self, kind: KernelKind) -> DispatchKind {
        match kind {
            KernelKind::FluxRhieChow => DispatchKind::Faces,
            _ => DispatchKind::Cells,
        }
    }
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
            .pipeline_for_kernel(kernel_spec.kind)
            .ok_or_else(|| format!("no pipeline for kernel {:?}", kernel_spec.kind))?;
        let bind = module
            .bind_for_kernel(kernel_spec.kind)
            .ok_or_else(|| format!("no bind group for kernel {:?}", kernel_spec.kind))?;
        let dispatch = module.dispatch_for_kernel(kernel_spec.kind);
        
        let label = kernel_label(label_prefix, kernel_spec.kind);
        
        nodes.push(ModuleNode::Compute(ComputeSpec {
            label,
            pipeline,
            bind,
            dispatch,
        }));
    }
    
    Ok(ModuleGraph::new(nodes))
}

/// Generate a static label for a kernel.
fn kernel_label(prefix: &'static str, kind: KernelKind) -> &'static str {
    // Use leaked strings for static labels
    // In practice, these are a fixed set so leaking is acceptable
    let label = format!("{}:{:?}", prefix, kind);
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
        let preparation = build_graph_for_phase(recipe, KernelPhase::Preparation, module, label_prefix).ok();
        let gradients = build_graph_for_phase(recipe, KernelPhase::Gradients, module, label_prefix).ok();
        let assembly = build_graph_for_phase(recipe, KernelPhase::Assembly, module, label_prefix).ok();
        let update = build_graph_for_phase(recipe, KernelPhase::Update, module, label_prefix).ok();
        
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
        let label = kernel_label("test", KernelKind::GenericCoupledAssembly);
        assert!(label.starts_with("test:"));
        assert!(label.contains("GenericCoupledAssembly"));
    }
}
