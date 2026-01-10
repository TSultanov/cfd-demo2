//! Unified Op Registry Module
//!
//! This module provides a unified way to register operation handlers based on
//! a SolverRecipe. Instead of each solver family having its own hardcoded
//! op registry, this module builds the registry dynamically from the recipe.

use crate::solver::gpu::execution_plan::{GraphDetail, GraphExecMode};
use crate::solver::gpu::plans::program::{
    CondOpHandler, CondOpKind, CountOpHandler, CountOpKind, GpuProgramPlan, GraphOpHandler,
    GraphOpKind, HostOpHandler, HostOpKind, ProgramOpRegistry,
};
use crate::solver::gpu::recipe::{SolverRecipe, SteppingMode};

/// Configuration for building a unified op registry.
///
/// Any handler left as None will be registered as a safe no-op default.
pub struct UnifiedOpRegistryConfig {
    pub prepare: Option<HostOpHandler>,
    pub finalize: Option<HostOpHandler>,
    pub solve: Option<HostOpHandler>,

    pub assembly_graph: Option<GraphOpHandler>,
    pub update_graph: Option<GraphOpHandler>,
    pub gradients_graph: Option<GraphOpHandler>,
    pub apply_graph: Option<GraphOpHandler>,

    pub implicit_update_graph: Option<GraphOpHandler>,
    pub implicit_snapshot_graph: Option<GraphOpHandler>,

    pub implicit_before_iter: Option<HostOpHandler>,
    pub implicit_after_solve: Option<HostOpHandler>,
    pub implicit_before_apply: Option<HostOpHandler>,
    pub implicit_after_apply: Option<HostOpHandler>,
    pub implicit_advance_outer_idx: Option<HostOpHandler>,
    pub implicit_outer_iters: Option<CountOpHandler>,

    pub coupled_enabled: Option<CondOpHandler>,
    pub coupled_init_prepare_graph: Option<GraphOpHandler>,
    pub coupled_before_iter: Option<HostOpHandler>,
    pub coupled_clear_max_diff: Option<HostOpHandler>,
    pub coupled_convergence_advance: Option<HostOpHandler>,
    pub coupled_should_continue: Option<CondOpHandler>,
    pub coupled_max_iters: Option<CountOpHandler>,
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
            implicit_update_graph: None,
            implicit_snapshot_graph: None,
            implicit_before_iter: None,
            implicit_after_solve: None,
            implicit_before_apply: None,
            implicit_after_apply: None,
            implicit_advance_outer_idx: None,
            implicit_outer_iters: None,
            coupled_enabled: None,
            coupled_init_prepare_graph: None,
            coupled_before_iter: None,
            coupled_clear_max_diff: None,
            coupled_convergence_advance: None,
            coupled_should_continue: None,
            coupled_max_iters: None,
        }
    }
}

fn noop_host(_plan: &mut GpuProgramPlan) {}

fn noop_graph(
    _plan: &GpuProgramPlan,
    _ctx: &crate::solver::gpu::context::GpuContext,
    _mode: GraphExecMode,
) -> (f64, Option<GraphDetail>) {
    (0.0, None)
}

fn cond_true(_plan: &GpuProgramPlan) -> bool {
    true
}

fn count_one(_plan: &GpuProgramPlan) -> usize {
    1
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
            registry.register_host(
                HostOpKind("explicit:prepare"),
                config.prepare.unwrap_or(noop_host),
            )?;
            registry.register_graph(
                GraphOpKind("explicit:update"),
                config.update_graph.unwrap_or(noop_graph),
            )?;
            registry.register_host(
                HostOpKind("explicit:finalize"),
                config.finalize.unwrap_or(noop_host),
            )?;

            // Back-compat / future hook.
            if recipe.needs_gradients() {
                registry.register_graph(
                    GraphOpKind("explicit:gradients"),
                    config.gradients_graph.unwrap_or(noop_graph),
                )?;
            }
        }

        SteppingMode::Implicit { .. } => {
            registry.register_host(
                HostOpKind("implicit:prepare"),
                config.prepare.unwrap_or(noop_host),
            )?;

            registry.register_count(
                CountOpKind("implicit:outer_iters"),
                config.implicit_outer_iters.unwrap_or(count_one),
            )?;

            registry.register_host(
                HostOpKind("implicit:before_iter"),
                config.implicit_before_iter.unwrap_or(noop_host),
            )?;
            registry.register_graph(
                GraphOpKind("implicit:assembly"),
                config.assembly_graph.unwrap_or(noop_graph),
            )?;
            registry.register_host(
                HostOpKind("implicit:solve"),
                config.solve.unwrap_or(noop_host),
            )?;
            registry.register_host(
                HostOpKind("implicit:after_solve"),
                config.implicit_after_solve.unwrap_or(noop_host),
            )?;
            registry.register_graph(
                GraphOpKind("implicit:snapshot"),
                config.implicit_snapshot_graph.unwrap_or(noop_graph),
            )?;
            registry.register_host(
                HostOpKind("implicit:before_apply"),
                config.implicit_before_apply.unwrap_or(noop_host),
            )?;
            registry.register_graph(
                GraphOpKind("implicit:apply"),
                config.apply_graph.unwrap_or(noop_graph),
            )?;
            registry.register_host(
                HostOpKind("implicit:after_apply"),
                config.implicit_after_apply.unwrap_or(noop_host),
            )?;
            registry.register_host(
                HostOpKind("implicit:advance_outer_idx"),
                config.implicit_advance_outer_idx.unwrap_or(noop_host),
            )?;

            registry.register_graph(
                GraphOpKind("implicit:update"),
                config.implicit_update_graph.unwrap_or(noop_graph),
            )?;
            registry.register_host(
                HostOpKind("implicit:finalize"),
                config.finalize.unwrap_or(noop_host),
            )?;
        }

        SteppingMode::Coupled { .. } => {
            registry.register_cond(
                CondOpKind("coupled:enabled"),
                config.coupled_enabled.unwrap_or(cond_true),
            )?;
            registry.register_count(
                CountOpKind("coupled:max_iters"),
                config.coupled_max_iters.unwrap_or(count_one),
            )?;
            registry.register_cond(
                CondOpKind("coupled:should_continue"),
                config.coupled_should_continue.unwrap_or(cond_true),
            )?;

            registry.register_host(
                HostOpKind("coupled:begin_step"),
                config.prepare.unwrap_or(noop_host),
            )?;
            registry.register_graph(
                GraphOpKind("coupled:init_prepare"),
                config.coupled_init_prepare_graph.unwrap_or(noop_graph),
            )?;

            registry.register_host(
                HostOpKind("coupled:before_iter"),
                config.coupled_before_iter.unwrap_or(noop_host),
            )?;
            registry.register_graph(
                GraphOpKind("coupled:assembly"),
                config.assembly_graph.unwrap_or(noop_graph),
            )?;
            registry.register_host(
                HostOpKind("coupled:solve"),
                config.solve.unwrap_or(noop_host),
            )?;
            registry.register_host(
                HostOpKind("coupled:clear_max_diff"),
                config.coupled_clear_max_diff.unwrap_or(noop_host),
            )?;
            registry.register_graph(
                GraphOpKind("coupled:update"),
                config.update_graph.unwrap_or(noop_graph),
            )?;
            registry.register_host(
                HostOpKind("coupled:convergence_advance"),
                config.coupled_convergence_advance.unwrap_or(noop_host),
            )?;

            registry.register_host(
                HostOpKind("coupled:finalize_step"),
                config.finalize.unwrap_or(noop_host),
            )?;
        }
    }

    Ok(registry)
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

        let config = UnifiedOpRegistryConfig {
            prepare: Some(dummy_host),
            finalize: Some(dummy_host),
            solve: Some(dummy_host),
            assembly_graph: Some(dummy_graph),
            update_graph: Some(dummy_graph),
            ..Default::default()
        };

        let registry = build_unified_registry(&recipe, config).expect("should build registry");

        // Registry should have the expected ops registered
        assert!(registry.has_cond(&CondOpKind("coupled:enabled")));
        assert!(registry.has_cond(&CondOpKind("coupled:should_continue")));
        assert!(registry.has_count(&CountOpKind("coupled:max_iters")));

        assert!(registry.has_host(&HostOpKind("coupled:begin_step")));
        assert!(registry.has_graph(&GraphOpKind("coupled:init_prepare")));
        assert!(registry.has_host(&HostOpKind("coupled:before_iter")));
        assert!(registry.has_graph(&GraphOpKind("coupled:assembly")));
        assert!(registry.has_host(&HostOpKind("coupled:solve")));
        assert!(registry.has_host(&HostOpKind("coupled:clear_max_diff")));
        assert!(registry.has_graph(&GraphOpKind("coupled:update")));
        assert!(registry.has_host(&HostOpKind("coupled:convergence_advance")));
        assert!(registry.has_host(&HostOpKind("coupled:finalize_step")));
    }
}
