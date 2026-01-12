//! Generic Linear Solver Module
//!
//! A unified interface for linear solvers that can be used across all solver families.
//! This module provides a common abstraction over FGMRES, CG, and other Krylov methods,
//! with pluggable preconditioners.

use crate::solver::gpu::linear_solver::fgmres::{FgmresPrecondBindings, FgmresWorkspace};
use crate::solver::gpu::modules::krylov_precond::{DispatchGrids, FgmresPreconditionerModule};
use crate::solver::gpu::modules::krylov_solve::KrylovSolveModule;
use crate::solver::gpu::modules::linear_system::LinearSystemView;
use crate::solver::gpu::recipe::{LinearSolverSpec, LinearSolverType};
use crate::solver::gpu::structs::LinearSolverStats;

/// Configuration for the generic linear solver.
#[derive(Debug, Clone)]
pub struct GenericLinearSolverConfig {
    pub n: u32,
    pub num_cells: u32,
    pub max_restart: usize,
    pub tol: f32,
    pub tol_abs: f32,
}

impl Default for GenericLinearSolverConfig {
    fn default() -> Self {
        Self {
            n: 0,
            num_cells: 0,
            max_restart: 30,
            tol: 1e-6,
            tol_abs: 1e-10,
        }
    }
}

impl From<&LinearSolverSpec> for GenericLinearSolverConfig {
    fn from(spec: &LinearSolverSpec) -> Self {
        let max_restart = match spec.solver_type {
            LinearSolverType::Fgmres { max_restart } => max_restart,
            _ => 30,
        };
        Self {
            n: 0,
            num_cells: 0,
            max_restart,
            tol: spec.tolerance,
            tol_abs: spec.tolerance_abs,
        }
    }
}

/// A generic linear solver that wraps the Krylov solver infrastructure.
///
/// This is designed to be usable by any solver family (compressible, incompressible,
/// generic coupled) without duplication of solver logic.
pub struct GenericLinearSolverModule<P: FgmresPreconditionerModule> {
    resources: Option<KrylovSolveModule<P>>,
    config: GenericLinearSolverConfig,
    last_stats: LinearSolverStats,
}

impl<P: FgmresPreconditionerModule> GenericLinearSolverModule<P> {
    /// Create a new linear solver module with the given configuration.
    pub fn new(config: GenericLinearSolverConfig) -> Self {
        Self {
            resources: None,
            config,
            last_stats: LinearSolverStats::default(),
        }
    }

    /// Create from a LinearSolverSpec.
    pub fn from_spec(spec: &LinearSolverSpec) -> Self {
        Self::new(GenericLinearSolverConfig::from(spec))
    }

    /// Get the last solve statistics.
    pub fn last_stats(&self) -> LinearSolverStats {
        self.last_stats
    }

    /// Check if resources are initialized.
    pub fn has_resources(&self) -> bool {
        self.resources.is_some()
    }

    /// Access the underlying Krylov resources (if initialized).
    pub fn resources(&self) -> Option<&KrylovSolveModule<P>> {
        self.resources.as_ref()
    }

    /// Access the underlying Krylov resources mutably.
    pub fn resources_mut(&mut self) -> Option<&mut KrylovSolveModule<P>> {
        self.resources.as_mut()
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: GenericLinearSolverConfig) {
        // Invalidate resources if size changed
        if self.config.n != config.n || self.config.max_restart != config.max_restart {
            self.resources = None;
        }
        self.config = config;
    }

    /// Set tolerance.
    pub fn set_tolerance(&mut self, tol: f32, tol_abs: f32) {
        self.config.tol = tol;
        self.config.tol_abs = tol_abs;
    }
}

/// Trait for creating preconditioners for the generic linear solver.
///
/// This allows different solver families to plug in their own preconditioner
/// creation logic while sharing the solver infrastructure.
pub trait PreconditionerFactory<P: FgmresPreconditionerModule> {
    /// Create the preconditioner-specific buffers and return bindings for FGMRES.
    fn create_precond_bindings(
        &self,
        device: &wgpu::Device,
        n: u32,
        num_cells: u32,
    ) -> (P::Buffers, FgmresPrecondBindings<'_>)
    where
        P: PreconditionerWithBuffers;

    /// Create the preconditioner module from the workspace and buffers.
    fn create_precond(
        &self,
        device: &wgpu::Device,
        fgmres: &FgmresWorkspace,
        num_cells: u32,
        buffers: P::Buffers,
    ) -> P
    where
        P: PreconditionerWithBuffers;
}

/// Extended trait for preconditioners that own buffers.
pub trait PreconditionerWithBuffers: FgmresPreconditionerModule {
    type Buffers;
}

/// Solve result with detailed timing information.
#[derive(Debug, Clone, Default)]
pub struct SolveResult {
    pub stats: LinearSolverStats,
    pub precond_setup_time: f64,
    pub solve_time: f64,
}

/// Helper to create FGMRES resources for any preconditioner type.
pub fn create_fgmres_resources<P: FgmresPreconditionerModule>(
    device: &wgpu::Device,
    n: u32,
    num_cells: u32,
    max_restart: usize,
    system: LinearSystemView<'_>,
    precond_bindings: FgmresPrecondBindings<'_>,
    precond: P,
    label: &str,
) -> KrylovSolveModule<P> {
    let fgmres = FgmresWorkspace::new_from_system(
        device,
        n,
        num_cells,
        max_restart,
        system,
        precond_bindings,
        label,
    );

    KrylovSolveModule::new(fgmres, precond)
}

/// Identity preconditioner (no preconditioning).
///
/// Note: A true identity preconditioner would simply copy input to output.
/// For FGMRES, this is not typically useful, but we provide it for completeness.
/// In practice, use at least Jacobi preconditioning.
pub struct IdentityPreconditioner {
    copy_pipeline: Option<wgpu::ComputePipeline>,
}

impl Default for IdentityPreconditioner {
    fn default() -> Self {
        Self {
            copy_pipeline: None,
        }
    }
}

impl IdentityPreconditioner {
    pub fn new() -> Self {
        Self::default()
    }
}

impl FgmresPreconditionerModule for IdentityPreconditioner {
    fn encode_apply(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        fgmres: &FgmresWorkspace,
        input: wgpu::BindingResource<'_>,
        output: &wgpu::Buffer,
        dispatch: DispatchGrids,
    ) {
        let _ = &self.copy_pipeline;

        let vector_bg = fgmres.create_vector_bind_group(
            device,
            input,
            output.as_entire_binding(),
            output.as_entire_binding(),
            "Identity preconditioner copy BG",
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Identity preconditioner copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(fgmres.pipeline_copy());
            pass.set_bind_group(0, &vector_bg, &[]);
            pass.set_bind_group(1, fgmres.matrix_bg(), &[]);
            pass.set_bind_group(2, fgmres.precond_bg(), &[]);
            pass.set_bind_group(3, fgmres.params_bg(), &[]);
            pass.dispatch_workgroups(dispatch.dofs.0, dispatch.dofs.1, 1);
        }
    }
}

impl PreconditionerWithBuffers for IdentityPreconditioner {
    type Buffers = ();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_spec() {
        let spec = LinearSolverSpec {
            solver_type: LinearSolverType::Fgmres { max_restart: 50 },
            preconditioner: crate::solver::gpu::structs::PreconditionerType::Jacobi,
            max_iters: 100,
            tolerance: 1e-8,
            tolerance_abs: 1e-12,
        };

        let config = GenericLinearSolverConfig::from(&spec);
        assert_eq!(config.max_restart, 50);
        assert!((config.tol - 1e-8).abs() < 1e-10);
        assert!((config.tol_abs - 1e-12).abs() < 1e-14);
    }
}
