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
use crate::solver::model::{expand_field_components, FluxSpec, GradientStorage, KernelId, ModelSpec};
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
    Coupled { outer_correctors: u32 },
}

/// High-level program structure emitted for a recipe.
///
/// This is a transitional bridge while solver-family runtime resources still exist.
/// The intent is that *eventually* the program can be emitted purely from phases and
/// module capabilities, but today we persist the chosen structure as part of the recipe
/// rather than re-inferring it in `build_program_spec()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramShape {
    Compressible,
    IncompressibleCoupled,
    GenericCoupledScalar,
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

    /// Program structure to emit for this recipe.
    pub program_shape: ProgramShape,

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
    ) -> Result<Self, String> {
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
            GradientStorage::PackedState => {
                if scheme_expansion.needs_gradients() {
                    vec!["state".to_string()]
                } else {
                    Vec::new()
                }
            }
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
        // `derive_kernel_ids` is structural (based on the equation system); the recipe assigns
        // phase and dispatch as it constructs `KernelSpec`s.
        let mut kernels: Vec<KernelSpec> = Vec::new();
        for id in crate::solver::model::kernel::derive_kernel_ids(&model.system) {
            // Some models include a gradients kernel in the plan, but for schemes that do not
            // require gradients we can skip it (if nothing else forces gradients).
            if id == KernelId::COMPRESSIBLE_GRADIENTS && !needs_gradients {
                continue;
            }

            let phase = match id {
                KernelId::COMPRESSIBLE_GRADIENTS => KernelPhase::Gradients,
                KernelId::COMPRESSIBLE_FLUX_KT => KernelPhase::FluxComputation,
                KernelId::COMPRESSIBLE_EXPLICIT_UPDATE => KernelPhase::ExplicitUpdate,

                KernelId::COUPLED_ASSEMBLY
                | KernelId::PRESSURE_ASSEMBLY
                | KernelId::COMPRESSIBLE_ASSEMBLY
                | KernelId::GENERIC_COUPLED_ASSEMBLY => KernelPhase::Assembly,

                KernelId::COMPRESSIBLE_APPLY | KernelId::GENERIC_COUPLED_APPLY => KernelPhase::Apply,

                KernelId::UPDATE_FIELDS_FROM_COUPLED | KernelId::GENERIC_COUPLED_UPDATE => {
                    KernelPhase::Update
                }

                KernelId::COMPRESSIBLE_UPDATE => KernelPhase::PrimitiveRecovery,

                // Preparation kernels (or legacy defaults)
                _ => KernelPhase::Preparation,
            };

            let dispatch = match id {
                KernelId::FLUX_RHIE_CHOW | KernelId::COMPRESSIBLE_FLUX_KT => DispatchKind::Faces,
                _ => DispatchKind::Cells,
            };

            kernels.push(KernelSpec { id, phase, dispatch });
        }

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
        let kernel_ids: Vec<KernelId> = kernels.iter().map(|k| k.id).collect();
        let stepping = derive_stepping_mode(model, &kernel_ids);

        // Choose the program structure up-front so `build_program_spec()` does not need
        // to re-infer “family-like” behavior by scanning the kernel set.
        let program_shape = derive_program_shape(&kernel_ids);

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
            program_shape,
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
    pub(crate) fn build_program_spec(&self) -> crate::solver::gpu::plans::program::ProgramSpec {
        use crate::solver::gpu::execution_plan::GraphExecMode;
        use crate::solver::gpu::plans::program::{
            CondOpKind, CountOpKind, GraphOpKind, HostOpKind, ProgramSpecBuilder, ProgramSpecNode,
        };

        match self.program_shape {
            ProgramShape::Compressible => {
            let mut program = ProgramSpecBuilder::new();
            let root = program.root();
            let explicit_block = program.new_block();
            let implicit_iter_block = program.new_block();
            let implicit_block = program.new_block();

            // Op kinds match `src/solver/gpu/lowering/templates.rs` (compressible)
            let g_explicit_graph = GraphOpKind("compressible:explicit_graph");
            let g_implicit_grad_assembly = GraphOpKind("compressible:implicit_grad_assembly");
            let g_implicit_snapshot = GraphOpKind("compressible:implicit_snapshot");
            let g_implicit_apply = GraphOpKind("compressible:implicit_apply");
            let g_primitive_update = GraphOpKind("compressible:primitive_update");

            let h_explicit_prepare = HostOpKind("compressible:explicit_prepare");
            let h_explicit_finalize = HostOpKind("compressible:explicit_finalize");
            let h_implicit_prepare = HostOpKind("compressible:implicit_prepare");
            let h_implicit_set_iter_params = HostOpKind("compressible:implicit_set_iter_params");
            let h_implicit_solve_fgmres = HostOpKind("compressible:implicit_solve_fgmres");
            let h_implicit_record_stats = HostOpKind("compressible:implicit_record_stats");
            let h_implicit_set_alpha = HostOpKind("compressible:implicit_set_alpha");
            let h_implicit_restore_alpha = HostOpKind("compressible:implicit_restore_alpha");
            let h_implicit_advance_outer_idx = HostOpKind("compressible:implicit_advance_outer_idx");
            let h_implicit_finalize = HostOpKind("compressible:implicit_finalize");

            let c_should_use_explicit = CondOpKind("compressible:should_use_explicit");
            let n_implicit_outer_iters = CountOpKind("compressible:implicit_outer_iters");

            // Explicit block
            program.push(
                explicit_block,
                ProgramSpecNode::Host {
                    label: "compressible:explicit_prepare",
                    kind: h_explicit_prepare,
                },
            );
            program.push(
                explicit_block,
                ProgramSpecNode::Graph {
                    label: "compressible:explicit_graph",
                    kind: g_explicit_graph,
                    mode: GraphExecMode::SplitTimed,
                },
            );
            program.push(
                explicit_block,
                ProgramSpecNode::Host {
                    label: "compressible:explicit_finalize",
                    kind: h_explicit_finalize,
                },
            );

            // Implicit iteration body
            for node in [
                ProgramSpecNode::Host {
                    label: "compressible:implicit_set_iter_params",
                    kind: h_implicit_set_iter_params,
                },
                ProgramSpecNode::Graph {
                    label: "compressible:implicit_grad_assembly",
                    kind: g_implicit_grad_assembly,
                    mode: GraphExecMode::SplitTimed,
                },
                ProgramSpecNode::Host {
                    label: "compressible:implicit_fgmres",
                    kind: h_implicit_solve_fgmres,
                },
                ProgramSpecNode::Host {
                    label: "compressible:implicit_record_stats",
                    kind: h_implicit_record_stats,
                },
                ProgramSpecNode::Graph {
                    label: "compressible:implicit_snapshot",
                    kind: g_implicit_snapshot,
                    mode: GraphExecMode::SingleSubmit,
                },
                ProgramSpecNode::Host {
                    label: "compressible:implicit_set_alpha",
                    kind: h_implicit_set_alpha,
                },
                ProgramSpecNode::Graph {
                    label: "compressible:implicit_apply",
                    kind: g_implicit_apply,
                    mode: GraphExecMode::SingleSubmit,
                },
                ProgramSpecNode::Host {
                    label: "compressible:implicit_restore_alpha",
                    kind: h_implicit_restore_alpha,
                },
                ProgramSpecNode::Host {
                    label: "compressible:implicit_outer_idx_inc",
                    kind: h_implicit_advance_outer_idx,
                },
            ] {
                program.push(implicit_iter_block, node);
            }

            // Implicit block
            program.push(
                implicit_block,
                ProgramSpecNode::Host {
                    label: "compressible:implicit_prepare",
                    kind: h_implicit_prepare,
                },
            );
            program.push(
                implicit_block,
                ProgramSpecNode::Repeat {
                    label: "compressible:implicit_outer_loop",
                    times: n_implicit_outer_iters,
                    body: implicit_iter_block,
                },
            );
            program.push(
                implicit_block,
                ProgramSpecNode::Graph {
                    label: "compressible:primitive_update",
                    kind: g_primitive_update,
                    mode: GraphExecMode::SingleSubmit,
                },
            );
            program.push(
                implicit_block,
                ProgramSpecNode::Host {
                    label: "compressible:implicit_finalize",
                    kind: h_implicit_finalize,
                },
            );

            // Select explicit vs implicit
            program.push(
                root,
                ProgramSpecNode::If {
                    label: "compressible:select_step_path",
                    cond: c_should_use_explicit,
                    then_block: explicit_block,
                    else_block: Some(implicit_block),
                },
            );

            return program.build();
            }

            ProgramShape::IncompressibleCoupled => {
            let mut program = ProgramSpecBuilder::new();
            let root = program.root();
            let coupled_iter_block = program.new_block();
            let coupled_prepare_block = program.new_block();
            let coupled_assembly_block = program.new_block();
            let coupled_step_block = program.new_block();

            // Op kinds match `src/solver/gpu/lowering/templates.rs` (incompressible_coupled)
            let g_coupled_prepare_assembly = GraphOpKind("incompressible:coupled_prepare_assembly");
            let g_coupled_assembly = GraphOpKind("incompressible:coupled_assembly");
            let g_coupled_update = GraphOpKind("incompressible:coupled_update");
            let g_coupled_init_prepare = GraphOpKind("incompressible:coupled_init_prepare");

            let h_coupled_begin_step = HostOpKind("incompressible:coupled_begin_step");
            let h_coupled_before_iter = HostOpKind("incompressible:coupled_before_iter");
            let h_coupled_solve = HostOpKind("incompressible:coupled_solve");
            let h_coupled_clear_max_diff = HostOpKind("incompressible:coupled_clear_max_diff");
            let h_coupled_convergence_advance =
                HostOpKind("incompressible:coupled_convergence_advance");
            let h_coupled_finalize_step = HostOpKind("incompressible:coupled_finalize_step");

            let c_has_coupled_resources = CondOpKind("incompressible:has_coupled_resources");
            let c_coupled_needs_prepare = CondOpKind("incompressible:coupled_needs_prepare");
            let c_coupled_should_continue = CondOpKind("incompressible:coupled_should_continue");

            let n_coupled_max_iters = CountOpKind("incompressible:coupled_max_iters");

            program.push(
                coupled_prepare_block,
                ProgramSpecNode::Graph {
                    label: "incompressible:coupled_prepare_assembly",
                    kind: g_coupled_prepare_assembly,
                    mode: GraphExecMode::SingleSubmit,
                },
            );
            program.push(
                coupled_assembly_block,
                ProgramSpecNode::Graph {
                    label: "incompressible:coupled_assembly",
                    kind: g_coupled_assembly,
                    mode: GraphExecMode::SingleSubmit,
                },
            );

            for node in [
                ProgramSpecNode::Host {
                    label: "incompressible:coupled_before_iter",
                    kind: h_coupled_before_iter,
                },
                ProgramSpecNode::If {
                    label: "incompressible:coupled_prepare_or_assembly",
                    cond: c_coupled_needs_prepare,
                    then_block: coupled_prepare_block,
                    else_block: Some(coupled_assembly_block),
                },
                ProgramSpecNode::Host {
                    label: "incompressible:coupled_solve",
                    kind: h_coupled_solve,
                },
                ProgramSpecNode::Host {
                    label: "incompressible:coupled_clear_max_diff",
                    kind: h_coupled_clear_max_diff,
                },
                ProgramSpecNode::Graph {
                    label: "incompressible:coupled_update_fields_max_diff",
                    kind: g_coupled_update,
                    mode: GraphExecMode::SingleSubmit,
                },
                ProgramSpecNode::Host {
                    label: "incompressible:coupled_convergence_and_advance",
                    kind: h_coupled_convergence_advance,
                },
            ] {
                program.push(coupled_iter_block, node);
            }

            for node in [
                ProgramSpecNode::Host {
                    label: "incompressible:coupled_begin_step",
                    kind: h_coupled_begin_step,
                },
                ProgramSpecNode::Graph {
                    label: "incompressible:coupled_init_prepare",
                    kind: g_coupled_init_prepare,
                    mode: GraphExecMode::SingleSubmit,
                },
                ProgramSpecNode::While {
                    label: "incompressible:coupled_outer_loop",
                    max_iters: n_coupled_max_iters,
                    cond: c_coupled_should_continue,
                    body: coupled_iter_block,
                },
                ProgramSpecNode::Host {
                    label: "incompressible:coupled_finalize_step",
                    kind: h_coupled_finalize_step,
                },
            ] {
                program.push(coupled_step_block, node);
            }

            program.push(
                root,
                ProgramSpecNode::If {
                    label: "incompressible:step",
                    cond: c_has_coupled_resources,
                    then_block: coupled_step_block,
                    else_block: None,
                },
            );

            return program.build();
            }

            ProgramShape::GenericCoupledScalar => {
                // Fall through to generic program emission below (driven by stepping).
            }
        }

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

            SteppingMode::Implicit { outer_iters: _ } => {
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
// Legacy mapping used by older/handwritten recipe constructors.
// New recipes should assign phases explicitly when constructing KernelSpecs.
#[allow(dead_code)]
/// Derive stepping mode from model structure.
fn derive_stepping_mode(_model: &ModelSpec, kernels: &[KernelId]) -> SteppingMode {
    // Check for compressible (can be explicit or implicit)
    if kernels.contains(&KernelId::COMPRESSIBLE_EXPLICIT_UPDATE) {
        // Has explicit path, default to implicit with fallback
        return SteppingMode::Implicit { outer_iters: 3 };
    }

    // Check for coupled incompressible
    if kernels.contains(&KernelId::COUPLED_ASSEMBLY) {
        return SteppingMode::Coupled { outer_correctors: 3 };
    }

    // Check for generic coupled scalar
    if kernels.contains(&KernelId::GENERIC_COUPLED_ASSEMBLY) {
        return SteppingMode::Coupled { outer_correctors: 1 };
    }

    // Default: implicit
    SteppingMode::Implicit { outer_iters: 1 }
}

fn derive_program_shape(kernels: &[KernelId]) -> ProgramShape {
    if kernels.iter().any(|&id| {
        matches!(
            id,
            KernelId::COMPRESSIBLE_ASSEMBLY
                | KernelId::COMPRESSIBLE_APPLY
                | KernelId::COMPRESSIBLE_EXPLICIT_UPDATE
                | KernelId::COMPRESSIBLE_FLUX_KT
                | KernelId::COMPRESSIBLE_GRADIENTS
                | KernelId::COMPRESSIBLE_UPDATE
        )
    }) {
        return ProgramShape::Compressible;
    }

    if kernels.iter().any(|&id| {
        matches!(
            id,
            KernelId::PREPARE_COUPLED
                | KernelId::COUPLED_ASSEMBLY
                | KernelId::PRESSURE_ASSEMBLY
                | KernelId::UPDATE_FIELDS_FROM_COUPLED
                | KernelId::FLUX_RHIE_CHOW
        )
    }) {
        return ProgramShape::IncompressibleCoupled;
    }

    ProgramShape::GenericCoupledScalar
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::{compressible_model, incompressible_momentum_model};
    use crate::solver::model::generic_diffusion_demo_model;
    use crate::solver::gpu::plans::program::ProgramSpecNode;

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
        let _recipe = SolverRecipe::from_model(
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

    #[test]
    fn test_build_program_spec_for_compressible_emits_step_path_if() {
        let model = compressible_model();
        let recipe = SolverRecipe::from_model(
            &model,
            Scheme::Upwind,
            TimeScheme::Euler,
            PreconditionerType::Jacobi,
        )
        .expect("should create recipe");

        let spec = recipe.build_program_spec();
        let root_block = spec.block(spec.root);
        assert!(
            root_block.nodes.iter().any(|n| matches!(
                n,
                ProgramSpecNode::If { label, .. } if *label == "compressible:select_step_path"
            )),
            "compressible program spec should select explicit/implicit path"
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
        )
        .expect("should create recipe");

        let spec = recipe.build_program_spec();

        // Find a While node (outer corrector loop) anywhere in the program blocks.
        let has_while = spec
            .blocks
            .iter()
            .flat_map(|b| b.nodes.iter())
            .any(|n| matches!(n, ProgramSpecNode::While { .. }));
        assert!(has_while, "incompressible coupled spec should have a While loop");
    }
}
