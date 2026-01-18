//! Solver Recipe: A unified specification for deriving solver resources and execution plans.
//!
//! A `SolverRecipe` captures everything needed to construct a GPU solver:
//! - Required kernel passes (derived from the model's equation system)
//! - Required auxiliary buffers (gradients, history buffers, etc.)
//! - Linear solver configuration
//! - Time integration requirements
//!
//! The recipe is derived from `ModelSpec + SolverConfig` and serves as the
//! single source of truth for resource allocation and program construction.

use crate::solver::gpu::enums::TimeScheme;
use crate::solver::gpu::modules::graph::DispatchKind;
use crate::solver::gpu::structs::PreconditionerType;
use crate::solver::model::backend::{expand_schemes, SchemeRegistry};
use crate::solver::model::{
    expand_field_components, FluxSpec, GradientStorage, KernelId, ModelSpec,
};
use crate::solver::scheme::Scheme;
use std::collections::HashSet;

/// Specification for a linear solver.
#[derive(Debug, Clone)]
pub struct LinearSolverSpec {
    pub solver_type: LinearSolverType,
    pub preconditioner: PreconditionerType,
    pub max_iters: u32,
    pub tolerance: f32,
    pub tolerance_abs: f32,
}

impl Default for LinearSolverSpec {
    fn default() -> Self {
        Self {
            solver_type: LinearSolverType::Fgmres { max_restart: 30 },
            preconditioner: PreconditionerType::Jacobi,
            max_iters: 100,
            tolerance: 1e-6,
            tolerance_abs: 1e-10,
        }
    }
}

/// Type of linear solver to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverType {
    /// FGMRES with flexible preconditioning
    Fgmres { max_restart: usize },
    /// Conjugate Gradient (for SPD systems)
    Cg,
    /// BiCGSTAB
    BiCgStab,
}

impl LinearSolverSpec {
    pub fn is_implicit(&self) -> bool {
        true // All linear solvers are for implicit systems
    }
}

/// Specification for time integration.
#[derive(Debug, Clone, Copy)]
pub struct TimeIntegrationSpec {
    pub scheme: TimeScheme,
    /// Number of history buffers needed (1 for Euler, 2 for BDF2, etc.)
    pub history_levels: usize,
}

impl Default for TimeIntegrationSpec {
    fn default() -> Self {
        Self {
            scheme: TimeScheme::Euler,
            history_levels: 1,
        }
    }
}

impl TimeIntegrationSpec {
    pub fn for_scheme(scheme: TimeScheme) -> Self {
        let history_levels = match scheme {
            TimeScheme::Euler => 1,
            TimeScheme::BDF2 => 2,
        };
        Self {
            scheme,
            history_levels,
        }
    }
}

/// Specification for an auxiliary buffer.
#[derive(Debug, Clone)]
pub struct BufferSpec {
    pub name: &'static str,
    pub size_per_cell: usize,
    pub purpose: BufferPurpose,
}

/// Purpose of an auxiliary buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferPurpose {
    /// Gradient storage for high-order reconstruction
    Gradient,
    /// History buffer for time integration
    History,
    /// Temporary workspace
    Workspace,
    /// Iteration snapshot for implicit solvers
    IterationSnapshot,
}

/// Specification for a kernel pass.
#[derive(Debug, Clone)]
pub struct KernelSpec {
    pub id: KernelId,
    pub phase: KernelPhase,
    pub dispatch: DispatchKind,
}

/// Phase in which a kernel executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelPhase {
    /// Executed before the main solver loop
    Preparation,
    /// Gradient computation (if needed)
    Gradients,
    /// Flux computation (for explicit schemes)
    FluxComputation,
    /// Explicit update step
    ExplicitUpdate,
    /// Matrix/RHS assembly
    Assembly,
    /// Apply solution correction (implicit methods)
    Apply,
    /// Linear solve (handled by linear solver module)
    LinearSolve,
    /// Solution update/correction
    Update,
    /// Primitive variable recovery (e.g., p, u from conserved)
    PrimitiveRecovery,
}

/// Solver stepping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SteppingMode {
    /// Fully explicit time stepping
    Explicit,
    /// Implicit with Newton-like outer iterations
    Implicit { outer_iters: usize },
    /// Coupled system (momentum + pressure together)
    Coupled,
}

/// High-level program structure emitted for a recipe.
/// A complete recipe for constructing a GPU solver.
#[derive(Debug, Clone)]
pub struct SolverRecipe {
    /// Model identifier
    pub model_id: &'static str,

    /// Required kernel passes
    pub kernels: Vec<KernelSpec>,

    /// Required auxiliary buffers
    pub aux_buffers: Vec<BufferSpec>,

    /// Linear solver configuration
    pub linear_solver: LinearSolverSpec,

    /// Time integration requirements
    pub time_integration: TimeIntegrationSpec,

    /// Stepping mode (explicit, implicit, coupled)
    pub stepping: SteppingMode,

    /// Fields that require gradient computation
    pub gradient_fields: Vec<String>,

    /// Optional face-based flux storage requirements.
    pub flux: Option<FluxSpec>,

    /// Whether the model requires a low-mach params uniform buffer.
    pub requires_low_mach_params: bool,

    /// Number of unknowns per cell in the coupled system
    pub unknowns_per_cell: usize,
}

impl SolverRecipe {
    /// Derive a solver recipe from a model specification and solver configuration.
    pub fn from_model(
        model: &ModelSpec,
        advection_scheme: Scheme,
        time_scheme: TimeScheme,
        preconditioner: PreconditionerType,
        stepping: SteppingMode,
    ) -> Result<Self, String> {
        model.validate_module_manifests()?;

        let scheme_registry = SchemeRegistry::new(advection_scheme);
        let scheme_expansion = expand_schemes(&model.system, &scheme_registry)
            .map_err(|e| format!("scheme expansion failed: {e}"))?;

        let mut gradient_fields: Vec<String> = match model.gpu.gradient_storage {
            GradientStorage::None => Vec::new(),
            GradientStorage::PerFieldName => scheme_expansion
                .gradient_fields()
                .iter()
                .map(|f| f.name().to_string())
                .collect(),
            GradientStorage::PackedState => vec!["state".to_string()],
            GradientStorage::PerFieldComponents => scheme_expansion
                .gradient_fields()
                .iter()
                .flat_map(|&f| expand_field_components(f))
                .collect(),
        };

        gradient_fields.extend(model.gpu.required_gradient_fields.iter().cloned());

        // Dedupe while preserving a stable order.
        let mut seen = HashSet::new();
        gradient_fields.retain(|name| seen.insert(name.clone()));

        let needs_gradients = !gradient_fields.is_empty();
        let flux = model.gpu.flux;
        let requires_low_mach_params = model.gpu.requires_low_mach_params;

        // Emit kernel specs in terms of stable KernelIds.
        //
        // Kernel selection, phase membership, and dispatch kind are model-owned and derived
        // from the method + module configuration.
        let mut model_kernel_specs = crate::solver::model::kernel::derive_kernel_specs_for_model(model)?;

        // The implicit stepping program uses a dedicated Apply graph.
        // Keep apply mechanics out of model identity by composing a small apply module
        // into the derived recipe (rather than hard-coding kernel insertion here).
        if matches!(stepping, SteppingMode::Implicit { .. }) {
            let apply_module = crate::solver::model::modules::generic_coupled_apply::generic_coupled_apply_module();
            model_kernel_specs.extend_from_slice(&apply_module.kernels);
        }

        let mut kernels: Vec<KernelSpec> = Vec::new();
        for spec in model_kernel_specs {
            let phase = match spec.phase {
                crate::solver::model::kernel::KernelPhaseId::Preparation => KernelPhase::Preparation,
                crate::solver::model::kernel::KernelPhaseId::Gradients => KernelPhase::Gradients,
                crate::solver::model::kernel::KernelPhaseId::FluxComputation => {
                    KernelPhase::FluxComputation
                }
                crate::solver::model::kernel::KernelPhaseId::Assembly => KernelPhase::Assembly,
                crate::solver::model::kernel::KernelPhaseId::Apply => KernelPhase::Apply,
                crate::solver::model::kernel::KernelPhaseId::Update => KernelPhase::Update,
            };

            let dispatch = match spec.dispatch {
                crate::solver::model::kernel::DispatchKindId::Cells => DispatchKind::Cells,
                crate::solver::model::kernel::DispatchKindId::Faces => DispatchKind::Faces,
            };

            kernels.push(KernelSpec {
                id: spec.id,
                phase,
                dispatch,
            });
        }

        // Derive auxiliary buffers
        let mut aux_buffers = Vec::new();

        if needs_gradients {
            // Add gradient buffers for each field that needs them
            for field_name in &gradient_fields {
                let size_per_cell = if model.gpu.gradient_storage == GradientStorage::PackedState
                    && field_name == "state"
                {
                    model.state_layout.stride() as usize * 2
                } else {
                    2
                };
                aux_buffers.push(BufferSpec {
                    name: Box::leak(format!("grad_{field_name}").into_boxed_str()),
                    size_per_cell,
                    purpose: BufferPurpose::Gradient,
                });
            }
        }

        // Time integration buffers
        let time_integration = TimeIntegrationSpec::for_scheme(time_scheme);
        if time_integration.history_levels > 1 {
            aux_buffers.push(BufferSpec {
                name: "state_old2",
                size_per_cell: model.state_layout.stride() as usize,
                purpose: BufferPurpose::History,
            });
        }

        // Linear solver spec
        let max_restart = 30;
        let linear_solver = LinearSolverSpec {
            solver_type: LinearSolverType::Fgmres { max_restart },
            preconditioner,
            max_iters: 100,
            tolerance: 1e-6,
            tolerance_abs: 1e-10,
        };

        Ok(SolverRecipe {
            model_id: model.id,
            kernels,
            aux_buffers,
            linear_solver,
            time_integration,
            stepping,
            gradient_fields,
            flux,
            requires_low_mach_params,
            unknowns_per_cell: model.system.unknowns_per_cell() as usize,
        })
    }

    /// Check if this recipe requires gradient computation.
    pub fn needs_gradients(&self) -> bool {
        !self.gradient_fields.is_empty()
    }

    /// Check if this recipe uses implicit time stepping.
    pub fn is_implicit(&self) -> bool {
        matches!(
            self.stepping,
            SteppingMode::Implicit { .. } | SteppingMode::Coupled
        )
    }

    /// Get kernels for a specific phase.
    pub fn kernels_for_phase(&self, phase: KernelPhase) -> impl Iterator<Item = &KernelSpec> {
        self.kernels.iter().filter(move |k| k.phase == phase)
    }

    /// Build a ProgramSpec from this recipe.
    ///
    /// This generates the execution sequence (prepare → assembly → solve → update → finalize)
    /// based on the stepping mode and kernel requirements.
    pub(crate) fn build_program_spec(&self) -> crate::solver::gpu::plans::program::ProgramSpec {
        use crate::solver::gpu::execution_plan::GraphExecMode;
        use crate::solver::gpu::plans::program::{
            CondOpKind, CountOpKind, GraphOpKind, HostOpKind, ProgramSpecBuilder, ProgramSpecNode,
        };

        let mut program = ProgramSpecBuilder::new();
        let root = program.root();

        match &self.stepping {
            SteppingMode::Explicit => {
                // Explicit: prepare → update_graph → finalize
                program.push(
                    root,
                    ProgramSpecNode::Host {
                        label: "explicit:prepare",
                        kind: HostOpKind("explicit:prepare"),
                    },
                );
                program.push(
                    root,
                    ProgramSpecNode::Graph {
                        label: "explicit:update",
                        kind: GraphOpKind("explicit:update"),
                        mode: GraphExecMode::SplitTimed,
                    },
                );
                program.push(
                    root,
                    ProgramSpecNode::Host {
                        label: "explicit:finalize",
                        kind: HostOpKind("explicit:finalize"),
                    },
                );
            }

            SteppingMode::Implicit { outer_iters: _ } => {
                // Implicit: prepare → outer loop (before_iter → assembly → solve → after_solve → snapshot → before_apply → apply → after_apply → advance_outer_idx) → update → finalize
                let iter_block = program.new_block();

                program.push(
                    root,
                    ProgramSpecNode::Host {
                        label: "implicit:prepare",
                        kind: HostOpKind("implicit:prepare"),
                    },
                );

                for node in [
                    ProgramSpecNode::Host {
                        label: "implicit:before_iter",
                        kind: HostOpKind("implicit:before_iter"),
                    },
                    ProgramSpecNode::Graph {
                        label: "implicit:assembly",
                        kind: GraphOpKind("implicit:assembly"),
                        mode: GraphExecMode::SplitTimed,
                    },
                    ProgramSpecNode::Host {
                        label: "implicit:solve",
                        kind: HostOpKind("implicit:solve"),
                    },
                    ProgramSpecNode::Host {
                        label: "implicit:after_solve",
                        kind: HostOpKind("implicit:after_solve"),
                    },
                    ProgramSpecNode::Graph {
                        label: "implicit:snapshot",
                        kind: GraphOpKind("implicit:snapshot"),
                        mode: GraphExecMode::SingleSubmit,
                    },
                    ProgramSpecNode::Host {
                        label: "implicit:before_apply",
                        kind: HostOpKind("implicit:before_apply"),
                    },
                    ProgramSpecNode::Graph {
                        label: "implicit:apply",
                        kind: GraphOpKind("implicit:apply"),
                        mode: GraphExecMode::SingleSubmit,
                    },
                    ProgramSpecNode::Host {
                        label: "implicit:after_apply",
                        kind: HostOpKind("implicit:after_apply"),
                    },
                    ProgramSpecNode::Host {
                        label: "implicit:advance_outer_idx",
                        kind: HostOpKind("implicit:advance_outer_idx"),
                    },
                ] {
                    program.push(iter_block, node);
                }

                program.push(
                    root,
                    ProgramSpecNode::Repeat {
                        label: "implicit:outer_loop",
                        times: CountOpKind("implicit:outer_iters"),
                        body: iter_block,
                    },
                );

                program.push(
                    root,
                    ProgramSpecNode::Graph {
                        label: "implicit:update",
                        kind: GraphOpKind("implicit:update"),
                        mode: GraphExecMode::SingleSubmit,
                    },
                );

                program.push(
                    root,
                    ProgramSpecNode::Host {
                        label: "implicit:finalize",
                        kind: HostOpKind("implicit:finalize"),
                    },
                );
            }

            SteppingMode::Coupled => {
                // Coupled: if enabled → begin_step → init_prepare → while (before_iter → assembly → solve → clear_max_diff → update → convergence_advance) → finalize_step
                let step_block = program.new_block();
                let iter_block = program.new_block();

                for node in [
                    ProgramSpecNode::Host {
                        label: "coupled:begin_step",
                        kind: HostOpKind("coupled:begin_step"),
                    },
                    ProgramSpecNode::Graph {
                        label: "coupled:init_prepare",
                        kind: GraphOpKind("coupled:init_prepare"),
                        mode: GraphExecMode::SingleSubmit,
                    },
                    ProgramSpecNode::While {
                        label: "coupled:outer_loop",
                        max_iters: CountOpKind("coupled:max_iters"),
                        cond: CondOpKind("coupled:should_continue"),
                        body: iter_block,
                    },
                    ProgramSpecNode::Host {
                        label: "coupled:finalize_step",
                        kind: HostOpKind("coupled:finalize_step"),
                    },
                ] {
                    program.push(step_block, node);
                }

                for node in [
                    ProgramSpecNode::Host {
                        label: "coupled:before_iter",
                        kind: HostOpKind("coupled:before_iter"),
                    },
                    ProgramSpecNode::Graph {
                        label: "coupled:assembly",
                        kind: GraphOpKind("coupled:assembly"),
                        mode: GraphExecMode::SingleSubmit,
                    },
                    ProgramSpecNode::Host {
                        label: "coupled:solve",
                        kind: HostOpKind("coupled:solve"),
                    },
                    ProgramSpecNode::Host {
                        label: "coupled:clear_max_diff",
                        kind: HostOpKind("coupled:clear_max_diff"),
                    },
                    ProgramSpecNode::Graph {
                        label: "coupled:update",
                        kind: GraphOpKind("coupled:update"),
                        mode: GraphExecMode::SingleSubmit,
                    },
                    ProgramSpecNode::Host {
                        label: "coupled:convergence_advance",
                        kind: HostOpKind("coupled:convergence_advance"),
                    },
                ] {
                    program.push(iter_block, node);
                }

                program.push(
                    root,
                    ProgramSpecNode::If {
                        label: "coupled:step_if_enabled",
                        cond: CondOpKind("coupled:enabled"),
                        then_block: step_block,
                        else_block: None,
                    },
                );
            }
        }

        program.build()
    }
}

// Stepping selection is owned by the solver config (not the model).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::gpu::plans::program::ProgramSpecNode;
    use crate::solver::model::generic_diffusion_demo_model;
    use crate::solver::model::{
        compressible_model, incompressible_momentum_generic_model, incompressible_momentum_model,
    };

    #[test]
    fn implicit_recipe_includes_apply_kernel_via_module_composition() {
        let model = generic_diffusion_demo_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::Upwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
            SteppingMode::Implicit { outer_iters: 1 },
        )
        .expect("recipe build");

        assert!(
            recipe
                .kernels
                .iter()
                .any(|k| k.id.as_str() == "generic_coupled_apply" && k.phase == KernelPhase::Apply),
            "implicit stepping recipes must include the Apply kernel"
        );
    }

    #[test]
    fn test_recipe_from_generic_diffusion_model() {
        let model = generic_diffusion_demo_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::Upwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
            SteppingMode::Coupled,
        )
        .expect("should create recipe");

        assert_eq!(recipe.model_id, "generic_diffusion_demo");
        assert!(recipe.needs_gradients());
        assert_eq!(recipe.gradient_fields, vec!["state".to_string()]);
        assert_eq!(recipe.unknowns_per_cell, 1);
    }

    #[test]
    fn recipe_allocates_packed_gradients_for_packed_storage() {
        let model = compressible_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::Upwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
            SteppingMode::Implicit { outer_iters: 1 },
        )
        .expect("recipe build");

        assert!(recipe.needs_gradients());
        assert_eq!(recipe.gradient_fields, vec!["state".to_string()]);
        assert!(recipe.kernels.iter().any(|k|
            k.id.as_str() == "packed_state_gradients" && k.phase == KernelPhase::Gradients
        ));

        let grad_state = recipe
            .aux_buffers
            .iter()
            .find(|b| b.purpose == BufferPurpose::Gradient && b.name == "grad_state")
            .expect("recipe must allocate grad_state buffer");
        assert_eq!(grad_state.size_per_cell, model.state_layout.stride() as usize * 2);
    }

    #[test]
    fn test_time_integration_spec_history_levels() {
        assert_eq!(
            TimeIntegrationSpec::for_scheme(TimeScheme::Euler).history_levels,
            1
        );
        assert_eq!(
            TimeIntegrationSpec::for_scheme(TimeScheme::BDF2).history_levels,
            2
        );
    }

    #[test]
    fn test_build_program_spec_for_coupled() {
        let model = generic_diffusion_demo_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::Upwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
            SteppingMode::Coupled,
        )
        .expect("should create recipe");

        let spec = recipe.build_program_spec();
        // Verify spec has expected structure for coupled solver
        let root_block = spec.block(spec.root);
        assert!(
            !root_block.nodes.is_empty(),
            "program spec should have nodes"
        );
    }

    #[test]
    fn test_build_program_spec_for_compressible_emits_implicit_outer_loop() {
        let model = compressible_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::Upwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
            SteppingMode::Implicit { outer_iters: 1 },
        )
        .expect("should create recipe");

        let spec = recipe.build_program_spec();
        // Find a Repeat node (outer loop) anywhere in the program blocks.
        let has_repeat = spec
            .blocks
            .iter()
            .flat_map(|b| b.nodes.iter())
            .any(|n| matches!(n, ProgramSpecNode::Repeat { .. }));
        assert!(
            has_repeat,
            "implicit program spec should have a Repeat outer loop"
        );
    }

    #[test]
    fn test_build_program_spec_for_incompressible_emits_outer_loop() {
        let model = incompressible_momentum_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::Upwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
            SteppingMode::Coupled,
        )
        .expect("should create recipe");

        let spec = recipe.build_program_spec();

        // Find a While node (outer corrector loop) anywhere in the program blocks.
        let has_while = spec
            .blocks
            .iter()
            .flat_map(|b| b.nodes.iter())
            .any(|n| matches!(n, ProgramSpecNode::While { .. }));
        assert!(
            has_while,
            "incompressible coupled spec should have a While loop"
        );
    }

    #[test]
    fn test_recipe_for_incompressible_generic_includes_flux_and_generic_assembly() {
        let model = incompressible_momentum_generic_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::Upwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
            SteppingMode::Coupled,
        )
        .expect("should create recipe");

        assert!(recipe.flux.is_some(), "generic incompressible should allocate flux buffer");
        assert_eq!(
            recipe.flux.unwrap().stride,
            model.system.unknowns_per_cell(),
            "flux stride should match packed coupled unknown layout"
        );
        assert!(recipe
            .kernels
            .iter()
            .any(|k| k.id == KernelId::FLUX_MODULE));
        assert!(recipe
            .kernels
            .iter()
            .any(|k| k.id == KernelId::GENERIC_COUPLED_ASSEMBLY));
    }
}
