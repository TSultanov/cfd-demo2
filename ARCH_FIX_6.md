# ARCH_FIX_6: Decouple Preconditioner Trait from FGMRES

**Status: ✅ COMPLETE (Phases 1–6)**  
All preconditioners now use the solver-agnostic `PreconditionerModule` trait and
`PrecondContext` struct.  The old `FgmresPreconditionerModule` is deprecated and
has zero remaining callers.

## Problem (Architecture Review Issue #6)

The `FgmresPreconditionerModule` trait (in `krylov_precond.rs`) is the sole abstraction for
plugging preconditioners into the linear solver, but it is deeply coupled to FGMRES internals:

```rust
pub trait FgmresPreconditionerModule {
    fn encode_prepare(&mut self, ..., fgmres: &FgmresWorkspace, ...);
    fn encode_apply(&mut self, ..., fgmres: &FgmresWorkspace, ...);
}
```

Every preconditioner — including `IdentityPreconditioner`, `RuntimePreconditionerModule`,
`CoupledSchurModule`, and `GenericCoupledSchurPreconditioner` — takes an `&FgmresWorkspace`
even when it only needs raw buffer bindings (input/output vectors).

Consequences:
- `IdentityPreconditioner` calls `fgmres.pipeline_copy()` and `fgmres.create_vector_bind_group()`
  just to copy input → output. It could be a simple buffer copy.
- `RuntimePreconditionerModule` (Jacobi, Block-Jacobi, AMG) uses `fgmres.matrix_bg()`,
  `fgmres.precond_bg()`, `fgmres.params_bg()`, and `fgmres.indirect_args_buffer()` — these are
  FGMRES-specific resource accessors that leak workspace internals into preconditioner code.
- `CoupledSchurModule` and `GenericCoupledSchurPreconditioner` use `fgmres.scalars_buffer()`,
  `fgmres.matrix_bg()`, and `fgmres.indirect_args_buffer()` — again, deeply coupled.
- `ScalarCgModule` exists as a completely separate, duplicated linear solver that cannot reuse any
  preconditioner. CG and FGMRES share no abstraction boundary.
- Adding a new solver (e.g., BiCGSTAB, direct solver, multigrid-as-solver) would require either
  implementing a fake `FgmresWorkspace` or duplicating the entire preconditioner infrastructure.

## Design Goal

Introduce a solver-agnostic preconditioner trait that operates on raw buffer bindings (input
vector, output vector) and solver-agnostic resource descriptors. The FGMRES-specific coupling
should be confined to an adapter layer inside the FGMRES solver, not leaked into every
preconditioner implementation.

## Constraints

- **Pure refactor** — no behavioral changes; all tests and OpenFOAM metrics must remain identical.
- **Incremental** — each phase should compile and pass all tests independently.
- **GPU pipeline compatibility** — preconditioners encode into `wgpu::CommandEncoder`; the trait
  must remain GPU-aware. The abstraction boundary is at the level of buffer bindings, not at the
  level of raw matrices/vectors (which would require host-side data).
- **Performance-neutral** — no additional GPU submissions, buffer copies, or bind group creations
  beyond what exists today.

## Inventory of FgmresWorkspace Usage in Preconditioners

Before designing the new trait, we must catalog exactly what each preconditioner uses from
`FgmresWorkspace`:

### IdentityPreconditioner
- `fgmres.create_vector_bind_group()` — to bind input/output
- `fgmres.pipeline_copy()` — to dispatch a copy kernel
- `fgmres.matrix_bg()`, `fgmres.precond_bg()`, `fgmres.params_bg()` — bind groups required by
  the copy pipeline's layout (even though they're unused by the shader)
- `fgmres.indirect_args_buffer()` + `indirect_dispatch_dofs_offset()` — for indirect dispatch

### RuntimePreconditionerModule (Jacobi / Block-Jacobi / AMG)
- **prepare**: `fgmres.create_vector_bind_group()`, `fgmres.matrix_bg()`,
  `fgmres.precond_bg()`, `fgmres.params_bg()` (Jacobi diag-inv extract);
  `fgmres.w_buffer()`, `fgmres.temp_buffer()`, `fgmres.z_binding(0)` (Block-Jacobi build)
- **apply (Jacobi)**: `fgmres.create_vector_bind_group()`, `fgmres.matrix_bg()`,
  `fgmres.precond_bg()`, `fgmres.params_bg()`, `fgmres.indirect_args_buffer()`
- **apply (Block-Jacobi)**: same as Jacobi plus `fgmres.temp_buffer()`
- **apply (AMG)**: `fgmres.scalars_buffer()`, AMG override bind group

### CoupledSchurModule
- `fgmres.create_vector_bind_group()` (via schur BG creation)
- `fgmres.matrix_bg()`, `fgmres.indirect_args_buffer()`, `fgmres.scalars_buffer()`
- `fgmres.temp_buffer()` (as auxiliary workspace)

### GenericCoupledSchurPreconditioner
- Delegates to `CoupledSchurModule` for apply
- `fgmres` passed through for setup dispatch

## Key Insight

The preconditioners use `FgmresWorkspace` for two distinct purposes:

1. **Solver-agnostic resources**: matrix bind group, params bind group, precond bind group,
   indirect dispatch buffer, temp buffers — these are properties of the *linear system* and
   *dispatch configuration*, not of FGMRES specifically.

2. **FGMRES-specific helpers**: `create_vector_bind_group()`, `pipeline_copy()`,
   `basis_binding()`, `w_buffer()`, `z_binding()` — these are FGMRES workspace internals.

The fix is to split these two concerns: extract a `LinearSolverContext` (or similar) that
carries the solver-agnostic resources, and have the preconditioner trait depend on that instead
of `FgmresWorkspace`.

## Phased Plan

### Phase 1: Extract `PreconditionerContext` struct ✅ DONE

Created `PrecondContext<'a>` in `krylov_precond.rs` with solver-agnostic fields:
- `matrix_bg`, `precond_bg`, `params_bg` — bind groups (groups 1–3)
- `indirect_args` — indirect dispatch buffer
- `scalars_buffer` — solver control scalars
- `scratch_a`, `scratch_b` — full-size scratch buffers (mapped to FGMRES `w` and `temp`)
- `scratch_c` — scratch binding resource (mapped to FGMRES `z_binding(0)`)
- `dispatch: DispatchGrids` — workgroup dimensions
- `num_dofs: u32` — system size

Added `INDIRECT_DISPATCH_DOFS_OFFSET` and `INDIRECT_DISPATCH_CELLS_OFFSET` constants.

Added `FgmresWorkspace::precond_context(dispatch)` method that constructs a
`PrecondContext` from the workspace's internal buffers and bind groups.

**Files**: `krylov_precond.rs`, `fgmres.rs`

### Phase 2: Introduce `PreconditionerModule` trait ✅ DONE

Defined `PreconditionerModule` trait in `krylov_precond.rs`:
- `encode_prepare(device, queue, encoder, ctx, rhs)` — default no-op
- `encode_apply(device, encoder, ctx, input, output)` — required

Created `PrecondAdapter<P: PreconditionerModule>` wrapper struct that implements
`FgmresPreconditionerModule` by constructing a `PrecondContext` from the
`FgmresWorkspace` and delegating to the wrapped `PreconditionerModule`.

**Files**: `krylov_precond.rs`

### Phase 3: Migrate `IdentityPreconditioner` ✅ DONE

Converted `IdentityPreconditioner` to implement `PreconditionerModule`:
- Uses `encoder.copy_buffer_to_buffer()` instead of dispatching a compute pipeline
- No longer needs `pipeline_copy`, `create_vector_bind_group`, or FGMRES bind groups
- Removed dead `copy_pipeline: Option<wgpu::ComputePipeline>` field

Kept `impl FgmresPreconditionerModule for IdentityPreconditioner` as a thin
delegation layer that creates a `PrecondContext` and calls the
`PreconditionerModule` implementation. All existing callers continue to work
unchanged.

**Files**: `generic_linear_solver.rs`

### Phase 4: Migrate `RuntimePreconditionerModule` ✅ DONE

Converted Jacobi, Block-Jacobi, and AMG preconditioner code to use `PrecondContext`:
- Refactored `encode_refresh_jacobi_diag_inv`, `encode_build_block_jacobi`,
  and `encode_prepare_impl` to take `&PrecondContext<'_>` instead of `&FgmresWorkspace`
- `fgmres.create_vector_bind_group()` → `ctx.create_vector_bind_group()`
- `fgmres.matrix_bg()` → `ctx.matrix_bg`, etc.
- `fgmres.w_buffer()` → `ctx.scratch_a`, `fgmres.temp_buffer()` → `ctx.scratch_b`,
  `fgmres.z_binding(0)` → `ctx.scratch_c`
- `FgmresWorkspace::indirect_dispatch_{dofs,cells}_offset()` →
  `PrecondContext::INDIRECT_DISPATCH_{DOFS,CELLS}_OFFSET`
- Identity fallbacks now use `PreconditionerModule::encode_apply` on the identity field
- `FgmresPreconditionerModule` impl is a thin delegation layer (creates context + delegates)

Extended `PrecondContext` with:
- `vectors_layout: &'a wgpu::BindGroupLayout` — group-0 vector BGL
- `vector_bindings: &'static [WgslBindingDesc]` — group-0 binding descriptors
- `create_vector_bind_group()` helper method (replaces `FgmresWorkspace::create_vector_bind_group`)

**Files**: `krylov_precond.rs`, `fgmres.rs`, `runtime_preconditioner.rs`

### Phase 5: Migrate Schur preconditioners ✅ DONE

Converted `CoupledSchurModule` and `GenericCoupledSchurPreconditioner` to use `PrecondContext`:
- `dispatch_schur` refactored: `fgmres.matrix_bg()` → `ctx.matrix_bg`,
  `fgmres.indirect_args_buffer()` → `ctx.indirect_args`
- `fgmres.temp_buffer()` → `ctx.scratch_b` (used as Chebyshev swap buffer / `p_prev`)
- `fgmres.scalars_buffer()` → `ctx.scalars_buffer` (AMG sync)
- `GenericCoupledSchurPreconditioner` delegates prepare/apply through `PreconditionerModule`

**Files**: `coupled_schur.rs`, `generic_coupled_schur.rs`

### Phase 6: Deprecate `FgmresPreconditionerModule` ✅ DONE

Completed full migration — `FgmresPreconditionerModule` is no longer used:
- `KrylovSolveModule<P>` bound changed from `FgmresPreconditionerModule` to `PreconditionerModule`
- `KrylovSolveModule` now creates `PrecondContext` from its own `FgmresWorkspace` when calling
  preconditioner methods
- All generic function bounds in `linear_solver.rs` changed:
  `solve_fgmres<P: PreconditionerModule>`,
  `encode_solve_fgmres_fixed_iterations<P: PreconditionerModule>`,
  `submit_solve_fgmres_fixed_iterations_chunked<P: PreconditionerModule>`
- `GenericLinearSolverModule<P>` bound changed to `PreconditionerModule`
- `PreconditionerWithBuffers` supertrait changed to `PreconditionerModule`
- `PreconditionerFactory<P>` bound changed to `PreconditionerModule`
- Removed all `FgmresPreconditionerModule` impls from preconditioner types
  (RuntimePreconditionerModule, CoupledSchurModule, GenericCoupledSchurPreconditioner,
  IdentityPreconditioner) — no longer needed since callers use `PreconditionerModule`
- `FgmresPreconditionerModule` trait and `PrecondAdapter` struct marked
  `#[deprecated]` — retained only for external backward compatibility
- `coupled_schur.rs` no longer imports `FgmresWorkspace` at all

**Files**: `krylov_precond.rs`, `krylov_solve.rs`, `linear_solver.rs`,
`generic_linear_solver.rs`, `runtime_preconditioner.rs`, `coupled_schur.rs`,
`generic_coupled_schur.rs`

### Phase 7 (Optional / Deferred): Unify CG solver path

`ScalarCgModule` is a fully self-contained unpreconditioned CG solver with its own
pipelines, bind group layouts, and buffer management.  It does not share any infrastructure
with the FGMRES solver or the preconditioner trait.

With `PreconditionerModule` now solver-agnostic, it would be possible to add a
`preconditioner: Option<Box<dyn PreconditionerModule>>` to the CG solver, enabling
preconditioned CG.  However this is a feature addition, not a refactor, and the CG
solver's pipeline layout and iteration structure are sufficiently different from FGMRES
that the integration would require new WGSL kernels and bind groups.

**Status**: Deferred — not required for the core ARCH_FIX_6 objective.

## Risks

- **Bind group layout compatibility**: The current preconditioner pipelines are compiled against
  `FgmresWorkspace`'s bind group layouts. If `PrecondContext` provides bind groups with different
  layouts, pipeline creation will fail. The mitigation is to keep the same buffer bindings and
  layouts — only the *source* of the bind groups changes (from `FgmresWorkspace` methods to
  `PrecondContext` fields).

- **`create_vector_bind_group` coupling**: This `FgmresWorkspace` helper creates a bind group
  matching the FGMRES pipeline's group-0 layout. Preconditioner pipelines share this layout
  (they're compiled from the same WGSL source with the same bind group indices). The
  `PrecondContext` may need to carry a bind-group-creation helper or the preconditioners
  must own their own bind group layout references.

- **Scope creep**: The Schur preconditioners are architecturally complex (Chebyshev iterations,
  AMG V-cycles, pressure sub-problems). Migrating them safely requires careful testing. It may
  be better to defer Phase 5 and focus on Phases 1–4.

## Verification

Each phase must pass:
- `cargo build` — clean compilation
- `cargo test` — all lib, codegen, and integration tests pass (only pre-existing failures allowed)
- OpenFOAM reference tests — zero metric drift
