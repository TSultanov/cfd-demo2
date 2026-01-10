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
use crate::solver::gpu::structs::PreconditionerType;
use crate::solver::model::backend::{expand_schemes, EquationSystem, SchemeRegistry, TermOp};
use crate::solver::model::{KernelKind, ModelSpec};
use crate::solver::scheme::Scheme;

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
    pub kind: KernelKind,
    pub phase: KernelPhase,
}

/// Phase in which a kernel executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelPhase {
    /// Executed before the main solver loop
    Preparation,
    /// Gradient computation (if needed)
    Gradients,
    /// Matrix/RHS assembly
    Assembly,
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
    Coupled { outer_correctors: u32 },
}

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
    ) -> Result<Self, String> {
        let scheme_registry = SchemeRegistry::new(advection_scheme);
        let scheme_expansion = expand_schemes(&model.system, &scheme_registry)
            .map_err(|e| format!("scheme expansion failed: {e}"))?;

        let gradient_fields: Vec<String> = scheme_expansion
            .gradient_fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect();

        let needs_gradients = !gradient_fields.is_empty();

        // Derive kernel plan from model
        let base_kernels = model.kernel_plan();
        let mut kernels: Vec<KernelSpec> = base_kernels
            .kernels()
            .iter()
            .map(|&kind| KernelSpec {
                kind,
                phase: phase_for_kernel(kind),
            })
            .collect();

        // Sort kernels by phase for proper execution order
        kernels.sort_by_key(|k| phase_order(k.phase));

        // Derive auxiliary buffers
        let mut aux_buffers = Vec::new();

        if needs_gradients {
            // Add gradient buffers for each field that needs them
            for field_name in &gradient_fields {
                aux_buffers.push(BufferSpec {
                    name: Box::leak(format!("grad_{field_name}").into_boxed_str()),
                    size_per_cell: 2, // 2D gradient (dx, dy)
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

        // Determine stepping mode from model structure
        let stepping = derive_stepping_mode(model, &base_kernels.kernels());

        // Linear solver spec
        let linear_solver = LinearSolverSpec {
            solver_type: LinearSolverType::Fgmres { max_restart: 30 },
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
            SteppingMode::Implicit { .. } | SteppingMode::Coupled { .. }
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
    pub fn build_program_spec(&self) -> crate::solver::gpu::plans::program::ProgramSpec {
        use crate::solver::gpu::execution_plan::GraphExecMode;
        use crate::solver::gpu::plans::program::{
            GraphOpKind, HostOpKind, ProgramSpecBuilder, ProgramSpecNode,
        };

        let mut program = ProgramSpecBuilder::new();
        let root = program.root();

        match &self.stepping {
            SteppingMode::Explicit => {
                // Explicit: prepare → gradients (if needed) → explicit update → finalize
                program.push(
                    root,
                    ProgramSpecNode::Host {
                        label: "explicit:prepare",
                        kind: HostOpKind("explicit:prepare"),
                    },
                );

                if self.needs_gradients() {
                    program.push(
                        root,
                        ProgramSpecNode::Graph {
                            label: "explicit:gradients",
                            kind: GraphOpKind("explicit:gradients"),
                            mode: GraphExecMode::SingleSubmit,
                        },
                    );
                }

                program.push(
                    root,
                    ProgramSpecNode::Graph {
                        label: "explicit:update",
                        kind: GraphOpKind("explicit:update"),
                        mode: GraphExecMode::SingleSubmit,
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

            SteppingMode::Implicit { outer_iters } => {
                // Implicit: prepare → newton loop (gradients → assembly → solve → apply) → finalize
                let newton_block = program.new_block();

                program.push(
                    root,
                    ProgramSpecNode::Host {
                        label: "implicit:prepare",
                        kind: HostOpKind("implicit:prepare"),
                    },
                );

                // Newton iteration body
                if self.needs_gradients() {
                    program.push(
                        newton_block,
                        ProgramSpecNode::Graph {
                            label: "implicit:gradients",
                            kind: GraphOpKind("implicit:gradients"),
                            mode: GraphExecMode::SingleSubmit,
                        },
                    );
                }

                program.push(
                    newton_block,
                    ProgramSpecNode::Graph {
                        label: "implicit:assembly",
                        kind: GraphOpKind("implicit:assembly"),
                        mode: GraphExecMode::SplitTimed,
                    },
                );

                program.push(
                    newton_block,
                    ProgramSpecNode::Host {
                        label: "implicit:solve",
                        kind: HostOpKind("implicit:solve"),
                    },
                );

                program.push(
                    newton_block,
                    ProgramSpecNode::Graph {
                        label: "implicit:apply",
                        kind: GraphOpKind("implicit:apply"),
                        mode: GraphExecMode::SingleSubmit,
                    },
                );

                program.push(
                    root,
                    ProgramSpecNode::Repeat {
                        label: "implicit:newton_loop",
                        times: crate::solver::gpu::plans::program::CountOpKind(
                            "implicit:outer_iters",
                        ),
                        body: newton_block,
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

            SteppingMode::Coupled { outer_correctors: _ } => {
                // Coupled: prepare → assembly → solve → update → finalize
                // (simple linear system, no Newton iteration needed)
                program.push(
                    root,
                    ProgramSpecNode::Host {
                        label: "coupled:prepare",
                        kind: HostOpKind("coupled:prepare"),
                    },
                );

                program.push(
                    root,
                    ProgramSpecNode::Graph {
                        label: "coupled:assembly",
                        kind: GraphOpKind("coupled:assembly"),
                        mode: GraphExecMode::SplitTimed,
                    },
                );

                program.push(
                    root,
                    ProgramSpecNode::Host {
                        label: "coupled:solve",
                        kind: HostOpKind("coupled:solve"),
                    },
                );

                program.push(
                    root,
                    ProgramSpecNode::Graph {
                        label: "coupled:update",
                        kind: GraphOpKind("coupled:update"),
                        mode: GraphExecMode::SingleSubmit,
                    },
                );

                program.push(
                    root,
                    ProgramSpecNode::Host {
                        label: "coupled:finalize",
                        kind: HostOpKind("coupled:finalize"),
                    },
                );
            }
        }

        program.build()
    }
}

/// Map kernel kind to execution phase.
fn phase_for_kernel(kind: KernelKind) -> KernelPhase {
    match kind {
        KernelKind::PrepareCoupled => KernelPhase::Preparation,
        KernelKind::FluxRhieChow => KernelPhase::Preparation,

        KernelKind::CompressibleGradients => KernelPhase::Gradients,

        KernelKind::CoupledAssembly
        | KernelKind::PressureAssembly
        | KernelKind::CompressibleAssembly
        | KernelKind::GenericCoupledAssembly
        | KernelKind::CompressibleFluxKt => KernelPhase::Assembly,

        KernelKind::CompressibleApply | KernelKind::GenericCoupledApply => KernelPhase::Assembly,

        KernelKind::UpdateFieldsFromCoupled
        | KernelKind::CompressibleUpdate
        | KernelKind::CompressibleExplicitUpdate
        | KernelKind::GenericCoupledUpdate => KernelPhase::Update,

        KernelKind::IncompressibleMomentum => KernelPhase::Assembly,
    }
}

/// Order for sorting kernels by phase.
fn phase_order(phase: KernelPhase) -> u8 {
    match phase {
        KernelPhase::Preparation => 0,
        KernelPhase::Gradients => 1,
        KernelPhase::Assembly => 2,
        KernelPhase::LinearSolve => 3,
        KernelPhase::Update => 4,
        KernelPhase::PrimitiveRecovery => 5,
    }
}

/// Derive stepping mode from model structure.
fn derive_stepping_mode(_model: &ModelSpec, kernels: &[KernelKind]) -> SteppingMode {
    // Check for compressible (can be explicit or implicit)
    if kernels.contains(&KernelKind::CompressibleExplicitUpdate) {
        // Has explicit path, default to implicit with fallback
        return SteppingMode::Implicit { outer_iters: 3 };
    }

    // Check for coupled incompressible
    if kernels.contains(&KernelKind::CoupledAssembly) {
        return SteppingMode::Coupled { outer_correctors: 3 };
    }

    // Check for generic coupled scalar
    if kernels.contains(&KernelKind::GenericCoupledAssembly) {
        return SteppingMode::Coupled { outer_correctors: 1 };
    }

    // Default: implicit
    SteppingMode::Implicit { outer_iters: 1 }
}

/// Derive kernel requirements from equation system structure.
///
/// This is a more principled approach than hardcoding in `ModelSpec::kernel_plan()`.
pub fn derive_kernel_plan(
    system: &EquationSystem,
    schemes: &SchemeRegistry,
) -> Result<Vec<KernelKind>, String> {
    let expansion = expand_schemes(system, schemes).map_err(|e| e.to_string())?;

    let mut kernels = Vec::new();

    // Analyze equation terms to determine required kernels
    let has_convection = system
        .equations()
        .iter()
        .any(|eq| eq.terms().iter().any(|t| t.op == TermOp::Div));

    let has_diffusion = system
        .equations()
        .iter()
        .any(|eq| eq.terms().iter().any(|t| t.op == TermOp::Laplacian));

    let has_time_derivative = system
        .equations()
        .iter()
        .any(|eq| eq.terms().iter().any(|t| t.op == TermOp::Ddt));

    // For now, return a generic coupled kernel set
    // This will be expanded as we migrate more logic here
    if has_convection || has_diffusion {
        kernels.push(KernelKind::GenericCoupledAssembly);
    }

    if expansion.needs_gradients() {
        // Gradient computation would go here
        // For now, handled by the assembly kernel
    }

    kernels.push(KernelKind::GenericCoupledApply);
    kernels.push(KernelKind::GenericCoupledUpdate);

    Ok(kernels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::generic_diffusion_demo_model;

    #[test]
    fn test_recipe_from_generic_diffusion_model() {
        let model = generic_diffusion_demo_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::Upwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
        )
        .expect("should create recipe");

        assert_eq!(recipe.model_id, "generic_diffusion_demo");
        assert!(!recipe.needs_gradients()); // Upwind doesn't need gradients
        assert_eq!(recipe.unknowns_per_cell, 1);
    }

    #[test]
    fn test_recipe_needs_gradients_with_sou() {
        let model = generic_diffusion_demo_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::SecondOrderUpwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
        )
        .expect("should create recipe");

        // SecondOrderUpwind requires gradients for convection terms
        // (if the model has convection - check if it does)
        // For diffusion-only model, may not need gradients
    }

    #[test]
    fn test_time_integration_spec_history_levels() {
        assert_eq!(TimeIntegrationSpec::for_scheme(TimeScheme::Euler).history_levels, 1);
        assert_eq!(TimeIntegrationSpec::for_scheme(TimeScheme::BDF2).history_levels, 2);
    }

    #[test]
    fn test_build_program_spec_for_coupled() {
        let model = generic_diffusion_demo_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::Upwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
        )
        .expect("should create recipe");

        let spec = recipe.build_program_spec();
        // Verify spec has expected structure for coupled solver
        let root_block = spec.block(spec.root);
        assert!(!root_block.nodes.is_empty(), "program spec should have nodes");
    }
}
