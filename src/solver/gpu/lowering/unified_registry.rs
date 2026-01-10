//! Unified Op Registry Module
//!
//! This module provides a unified way to register operation handlers based on
//! a SolverRecipe. Instead of each solver family having its own hardcoded
//! op registry, this module builds the registry dynamically from the recipe.

use crate::solver::gpu::execution_plan::{GraphDetail, GraphExecMode};
use crate::solver::gpu::plans::program::{
    GpuProgramPlan, GraphOpHandler, GraphOpKind, HostOpHandler, HostOpKind, ProgramOpRegistry,
};
use crate::solver::gpu::recipe::{SolverRecipe, SteppingMode};

/// Trait for modules that can provide graph execution.
pub trait GraphExecutor: Send + 'static {
    fn execute(
        &self,
        plan: &GpuProgramPlan,
        mode: GraphExecMode,
    ) -> (f64, Option<GraphDetail>);
}

/// Configuration for building a unified op registry.
pub struct UnifiedOpRegistryConfig {
    /// Handler for prepare step
    pub prepare: Option<HostOpHandler>,
    /// Handler for finalize step
    pub finalize: Option<HostOpHandler>,
    /// Handler for linear solve
    pub solve: Option<HostOpHandler>,
    /// Handler for assembly graph
    pub assembly_graph: Option<GraphOpHandler>,
    /// Handler for update graph
    pub update_graph: Option<GraphOpHandler>,
    /// Handler for gradients graph (if needed)
    pub gradients_graph: Option<GraphOpHandler>,
    /// Handler for apply graph (for implicit solvers)
    pub apply_graph: Option<GraphOpHandler>,
}

impl Default for UnifiedOpRegistryConfig {
    fn default() -> Self {
        Self {
            prepare: None,
            finalize: None,
            solve: None,
            assembly_graph: None,
            update_graph: None,
            gradients_graph: None,
            apply_graph: None,
        }
    }
}

/// Build a unified op registry from a recipe and handler configuration.
///
/// This maps the generic op kinds from the recipe's program spec to actual
/// handler implementations provided in the config.
pub fn build_unified_registry(
    recipe: &SolverRecipe,
    config: UnifiedOpRegistryConfig,
) -> Result<ProgramOpRegistry, String> {
    let mut registry = ProgramOpRegistry::new();

    match recipe.stepping {
        SteppingMode::Explicit => {
            // Explicit: prepare, (gradients), update, finalize
            if let Some(h) = config.prepare {
                registry.register_host(HostOpKind("explicit:prepare"), h)?;
            }
            if let Some(h) = config.finalize {
                registry.register_host(HostOpKind("explicit:finalize"), h)?;
            }
            if recipe.needs_gradients() {
                if let Some(g) = config.gradients_graph {
                    registry.register_graph(GraphOpKind("explicit:gradients"), g)?;
                }
            }
            if let Some(g) = config.update_graph {
                registry.register_graph(GraphOpKind("explicit:update"), g)?;
            }
        }

        SteppingMode::Implicit { .. } => {
            // Implicit: prepare, (gradients, assembly, solve, apply)*, finalize
            if let Some(h) = config.prepare {
                registry.register_host(HostOpKind("implicit:prepare"), h)?;
            }
            if let Some(h) = config.finalize {
                registry.register_host(HostOpKind("implicit:finalize"), h)?;
            }
            if let Some(h) = config.solve {
                registry.register_host(HostOpKind("implicit:solve"), h)?;
            }
            if recipe.needs_gradients() {
                if let Some(g) = config.gradients_graph {
                    registry.register_graph(GraphOpKind("implicit:gradients"), g)?;
                }
            }
            if let Some(g) = config.assembly_graph {
                registry.register_graph(GraphOpKind("implicit:assembly"), g)?;
            }
            if let Some(g) = config.apply_graph {
                registry.register_graph(GraphOpKind("implicit:apply"), g)?;
            }
        }

        SteppingMode::Coupled { .. } => {
            // Coupled: prepare, assembly, solve, update, finalize
            if let Some(h) = config.prepare {
                registry.register_host(HostOpKind("coupled:prepare"), h)?;
            }
            if let Some(h) = config.finalize {
                registry.register_host(HostOpKind("coupled:finalize"), h)?;
            }
            if let Some(h) = config.solve {
                registry.register_host(HostOpKind("coupled:solve"), h)?;
            }
            if let Some(g) = config.assembly_graph {
                registry.register_graph(GraphOpKind("coupled:assembly"), g)?;
            }
            if let Some(g) = config.update_graph {
                registry.register_graph(GraphOpKind("coupled:update"), g)?;
            }
        }
    }

    Ok(registry)
}

/// Builder pattern for unified op registry configuration.
pub struct UnifiedOpRegistryBuilder {
    config: UnifiedOpRegistryConfig,
}

impl UnifiedOpRegistryBuilder {
    pub fn new() -> Self {
        Self {
            config: UnifiedOpRegistryConfig::default(),
        }
    }

    pub fn prepare(mut self, handler: HostOpHandler) -> Self {
        self.config.prepare = Some(handler);
        self
    }

    pub fn finalize(mut self, handler: HostOpHandler) -> Self {
        self.config.finalize = Some(handler);
        self
    }

    pub fn solve(mut self, handler: HostOpHandler) -> Self {
        self.config.solve = Some(handler);
        self
    }

    pub fn assembly_graph(mut self, handler: GraphOpHandler) -> Self {
        self.config.assembly_graph = Some(handler);
        self
    }

    pub fn update_graph(mut self, handler: GraphOpHandler) -> Self {
        self.config.update_graph = Some(handler);
        self
    }

    pub fn gradients_graph(mut self, handler: GraphOpHandler) -> Self {
        self.config.gradients_graph = Some(handler);
        self
    }

    pub fn apply_graph(mut self, handler: GraphOpHandler) -> Self {
        self.config.apply_graph = Some(handler);
        self
    }

    pub fn build(self, recipe: &SolverRecipe) -> Result<ProgramOpRegistry, String> {
        build_unified_registry(recipe, self.config)
    }
}

impl Default for UnifiedOpRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::gpu::enums::TimeScheme;
    use crate::solver::gpu::structs::PreconditionerType;
    use crate::solver::model::generic_diffusion_demo_model;
    use crate::solver::scheme::Scheme;

    fn dummy_host(_plan: &mut GpuProgramPlan) {}
    fn dummy_graph(
        _plan: &GpuProgramPlan,
        _ctx: &crate::solver::gpu::context::GpuContext,
        _mode: GraphExecMode,
    ) -> (f64, Option<GraphDetail>) {
        (0.0, None)
    }

    #[test]
    fn test_build_unified_registry_for_coupled() {
        let model = generic_diffusion_demo_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::Upwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
        )
        .expect("should create recipe");

        let registry = UnifiedOpRegistryBuilder::new()
            .prepare(dummy_host)
            .finalize(dummy_host)
            .solve(dummy_host)
            .assembly_graph(dummy_graph)
            .update_graph(dummy_graph)
            .build(&recipe)
            .expect("should build registry");

        // Registry should have the expected ops registered
        assert!(registry.has_host(&HostOpKind("coupled:prepare")));
        assert!(registry.has_host(&HostOpKind("coupled:solve")));
        assert!(registry.has_host(&HostOpKind("coupled:finalize")));
        assert!(registry.has_graph(&GraphOpKind("coupled:assembly")));
        assert!(registry.has_graph(&GraphOpKind("coupled:update")));
    }
}
