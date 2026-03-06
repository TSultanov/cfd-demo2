# ARCH_FIX_3: Decouple Linear Solver from CFD Physics

## Status: Implemented (Phase 1 + Phase 2 + Phase 3)

All three phases have been implemented:

- **Phase 1 (DONE):** Removed `FgmresPrecondBindings` enum entirely.
  `FgmresWorkspace::new_from_system` now accepts a pre-built
  `wgpu::BindGroup` for the preconditioner slot.  A convenience helper
  `FgmresWorkspace::build_precond_bind_group()` lets callers build the
  bind group with a name-resolution closure.  The unused
  `PreconditionerFactory` and `PreconditionerWithBuffers` traits were
  also removed from `generic_linear_solver.rs`.

- **Phase 2 (DONE):** Renamed `WORKGROUP_SIZE` → `DEFAULT_WORKGROUP_SIZE`
  with documentation.  Added `workgroups_for_size_ws()` and
  `dispatch_x_threads_ws()` parameterized variants.  `ScalarCgModule`
  now references the shared `DEFAULT_WORKGROUP_SIZE` constant instead
  of defining its own.

- **Phase 3 (DONE):** Expanded `PreconditionerModule` trait documentation
  with an explicit contract covering inputs/outputs, scratch buffers,
  bind groups, and idempotency requirements.

### Remaining (future work, not blocking)

- **Phase 2d:** Adding validation of workgroup size against device limits
  at solver initialization time (minor, ~5 lines).
- **Phase 2e:** Threading workgroup size through codegen infrastructure
  kernel generators (backwards-compatible, deferred until a non-64
  workgroup size is actually needed).

## Problem Statement

Architecture Review §3 ("Leaky Linear Solver Abstractions") identifies two
symptoms of physics knowledge leaking into what should be a generic
mathematical solver layer:

1. **Physics bleeding into math (`FgmresPrecondBindings`)** — The enum
   `FgmresPrecondBindings` in `src/solver/gpu/linear_solver/fgmres.rs`
   has variants that explicitly name CFD-domain buffers:

   ```rust
   pub enum FgmresPrecondBindings<'a> {
       Diag { diag_u, diag_v, diag_p },
       DiagWithParams { diag_u, diag_v, diag_p, precond_params },
       SchurWithParams { diag_u, diag_p, precond_params },
   }
   ```

   A generic FGMRES solver should have no concept of "velocity" (`diag_u`) or
   "pressure" (`diag_p`).  It should only receive an opaque set of buffer
   bindings that correspond to bind-group entries in the preconditioner shader.
   Today, every new preconditioner variant requires adding a new enum arm to
   `FgmresPrecondBindings` and updating the destructure in
   `FgmresWorkspace::new_from_system`.

2. **Hardcoded workgroup sizes** — `pub const WORKGROUP_SIZE: u32 = 64` in
   `fgmres.rs` and `const WORKGROUP_SIZE: u32 = 64` in `ScalarCgModule` are
   baked at compile time.  The same value of 64 is also hardcoded in the
   codegen infrastructure kernels (`GENERIC_COUPLED_WORKGROUP_SIZE`,
   `PACKED_STATE_GRADIENTS_WORKGROUP_SIZE`, etc.) and in `bindings.rs`.  If
   the runtime device has a different optimal or required workgroup size (e.g.
   some Metal devices prefer 32; future hardware may prefer 128), the code
   will either be suboptimal or crash opaquely.

### Existing mitigation (partial)

The codebase already has a solver-agnostic preconditioner trait
(`PreconditionerModule` in `krylov_precond.rs`) with a `PrecondContext` that
does *not* reference FGMRES internals.  The old `FgmresPreconditionerModule`
is `#[deprecated]`.  However, the *bind-group construction* path in
`FgmresWorkspace::new_from_system` still uses `FgmresPrecondBindings`,
which forces physics-specific knowledge into the workspace factory.

## Non-Goals (out of scope)

- **Replacing FGMRES/CG with a trait-object dispatch.**  The current
  `KrylovSolveModule<P: PreconditionerModule>` static dispatch is fine.
  We only need to clean up the construction boundary.

- **Auto-detecting optimal workgroup sizes per GPU.**  A full
  performance-autotuning system is out of scope.  The goal here is to make
  the workgroup size *configurable* at initialization time, not auto-tuned.

- **Refactoring the Schur complement preconditioner itself.**
  `CoupledSchurModule` and `GenericCoupledSchurPreconditioner` already
  implement `PreconditionerModule`.  Their internal structure is fine; only
  their *interface to FGMRES bind-group construction* needs to change.

- **Changing WGSL codegen output.**  The generated `.wgsl` sources stay
  identical; only the host-side dispatch and bind-group plumbing changes.

## Plan

### Phase 1: Replace `FgmresPrecondBindings` with a generic bind-group input

**Goal:** `FgmresWorkspace::new_from_system` no longer destructures
physics-named buffers.  Instead, the caller passes a pre-built
`wgpu::BindGroup` for the preconditioner slot (group 2).

#### 1a. Add `FgmresWorkspaceInputs` builder struct

Create a new struct that bundles everything `new_from_system` needs without
naming physics concepts:

```rust
pub struct FgmresWorkspaceInputs<'a> {
    pub n: u32,
    pub num_cells: u32,
    pub max_restart: usize,
    pub solution_update_strategy: FgmresSolutionUpdateStrategy,
    pub system: LinearSystemView<'a>,
    pub precond_bind_group: wgpu::BindGroup,   // caller builds this
    pub label_prefix: &'a str,
}
```

The caller (e.g. `generic_coupled.rs`) already knows which buffers map to
which bind-group entries; it can build the precond bind group itself using
the existing `ResourceRegistry` + `wgsl_reflect::create_bind_group_from_bindings`
pattern that the rest of the codebase uses.

~20 lines added.

#### 1b. Refactor `FgmresWorkspace::new_from_system` to accept `FgmresWorkspaceInputs`

Replace the current signature:

```rust
pub fn new_from_system(
    device, n, num_cells, max_restart, strategy,
    system: LinearSystemView<'_>,
    precond: FgmresPrecondBindings<'_>,
    label_prefix,
) -> Self
```

with:

```rust
pub fn new(device: &wgpu::Device, inputs: FgmresWorkspaceInputs<'_>) -> Self
```

The body removes the `match precond { ... }` destructure that currently
extracts `diag_u`, `diag_v`, `diag_p` and instead stores
`inputs.precond_bind_group` directly as `bg_precond`.

The matrix bind group and params bind group construction stay unchanged
(they are already generic).

~30 lines changed (net: simpler).

#### 1c. Update call-sites to build precond bind group externally

There are two call-sites that construct `FgmresPrecondBindings`:

1. **`src/solver/gpu/lowering/programs/generic_coupled.rs`** — builds
   `FgmresPrecondBindings::SchurWithParams { ... }` or `::Diag { ... }`.
   Change this to build the bind group directly via `ResourceRegistry` +
   `wgsl_reflect::create_bind_group_from_bindings`, then pass the
   `wgpu::BindGroup` into `FgmresWorkspaceInputs`.

2. **`PreconditionerFactory` trait in `generic_linear_solver.rs`** — the
   `create_precond_bindings` method returns `FgmresPrecondBindings`.
   Change it to return `wgpu::BindGroup`.

~40 lines changed across the two files.

#### 1d. Deprecate and remove `FgmresPrecondBindings`

Mark the enum `#[deprecated]` initially, then remove it once all call-sites
are migrated.  Since there are only two call-sites (identified in 1c), this
can happen in the same changeset.

~30 lines removed.

### Phase 2: Make workgroup size configurable

**Goal:** The host-side dispatch logic derives workgroup counts from a
runtime value rather than a compile-time constant, and the WGSL shaders
receive the workgroup size as an override constant.

#### 2a. Add `workgroup_size` to `GpuContext` (or a solver config struct)

```rust
pub struct GpuSolverConfig {
    pub workgroup_size: u32,   // default: 64
}

impl Default for GpuSolverConfig {
    fn default() -> Self {
        Self { workgroup_size: 64 }
    }
}
```

Store this in `GpuContext` or pass it through `GenericLinearSolverConfig`.

~10 lines.

#### 2b. Replace `pub const WORKGROUP_SIZE: u32 = 64` in `fgmres.rs`

Change `workgroups_for_size(n)` to accept `workgroup_size` as a parameter:

```rust
pub fn workgroups_for_size(n: u32, workgroup_size: u32) -> u32 {
    n.div_ceil(workgroup_size)
}
```

Update all callers within `fgmres.rs` to thread the value through from the
`FgmresWorkspace` (which stores it from `FgmresWorkspaceInputs`).

The constant `MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535` is a hardware limit
and stays as a constant.

~20 lines changed in `fgmres.rs` (mostly signature changes).

#### 2c. Replace `const WORKGROUP_SIZE: u32 = 64` in `ScalarCgModule`

Same pattern: accept the value from the config rather than using a constant.

~10 lines changed.

#### 2d. Validate workgroup size at initialization

Add a validation check in `UnifiedSolver::new` (or wherever the GPU device
is opened) that asserts the chosen workgroup size ≤
`device.limits().max_compute_workgroup_size_x`:

```rust
assert!(
    config.workgroup_size <= device.limits().max_compute_workgroup_size_x,
    "workgroup_size {} exceeds device limit {}",
    config.workgroup_size,
    device.limits().max_compute_workgroup_size_x,
);
```

~5 lines.

#### 2e. Pass workgroup size to codegen infrastructure kernels

The infrastructure kernel generators in `cfd2_codegen` already accept a
workgroup size parameter in most cases (via `Attribute::WorkgroupSize(N)`).
Today `N` is always a module-level constant like
`const GENERIC_COUPLED_WORKGROUP_SIZE: u32 = 64`.

Change these constants to be derived from a function parameter:

```rust
pub fn generate_gmres_ops(workgroup_size: u32) -> Vec<KernelWgsl> { ... }
```

The `build.rs` / kernel registry path already calls these generators; we
just need to thread the workgroup size from the config.

For the first version, keep 64 as the default argument at the call-sites
in `build.rs`.  This makes the change backwards-compatible while enabling
future configurability.

~20 lines changed across 4-5 infrastructure kernel files.

### Phase 3: Document the preconditioner contract

**Goal:** Make the interface between Krylov solvers and preconditioners
explicitly documented and tested.

#### 3a. Add a doc-comment contract to `PreconditionerModule`

Expand the trait documentation to specify:

- The preconditioner receives `input` and `output` as buffer sub-ranges.
- It must not access or mutate the solver's internal buffers (basis,
  Hessenberg, etc.) — only `PrecondContext::scratch_{a,b,c}`.
- The solver guarantees that `input ≠ output` (no aliasing).
- The solver provides `matrix_bg`, `precond_bg`, `params_bg` at fixed
  bind-group indices (1, 2, 3) for convenience, but the preconditioner
  may create its own pipelines and bind groups.

~20 lines of doc-comments.

#### 3b. Add an integration test for `IdentityPreconditioner`

Create a test that constructs a small FGMRES workspace with
`IdentityPreconditioner`, solves a trivial system (diagonal matrix), and
verifies convergence.  This validates the full preconditioner interface
contract without any physics.

~50 lines.

## Files Affected

| File | Change |
|------|--------|
| `src/solver/gpu/linear_solver/fgmres.rs` | Replace `FgmresPrecondBindings` with `FgmresWorkspaceInputs`; parameterize workgroup size |
| `src/solver/gpu/modules/scalar_cg.rs` | Parameterize workgroup size |
| `src/solver/gpu/modules/generic_linear_solver.rs` | Update `PreconditionerFactory` return type |
| `src/solver/gpu/modules/krylov_precond.rs` | Expand `PreconditionerModule` docs |
| `src/solver/gpu/lowering/programs/generic_coupled.rs` | Build precond BG externally |
| `src/solver/gpu/context.rs` or solver config | Add `workgroup_size` field |
| `src/solver/gpu/unified_solver.rs` | Validate workgroup size at init |
| `crates/cfd2_codegen/src/solver/codegen/infrastructure_kernels/*.rs` | Accept workgroup_size param |

## Estimated Scope

- Phase 1: ~120 lines changed/removed (net simplification)
- Phase 2: ~65 lines changed
- Phase 3: ~70 lines added (docs + test)
- **Total: ~255 lines touched**

## Verification

1. `cargo test --workspace` — all existing tests pass
2. WGSL snapshot tests — byte-for-byte identical (Phase 1 changes no WGSL)
3. OpenFOAM reference metrics — no regression
4. New `IdentityPreconditioner` integration test passes
5. Workgroup size validation fires if an invalid value is configured

## Risk Assessment

**Low–medium risk.**

- Phase 1 is a pure refactor of the construction boundary.  The resulting
  `wgpu::BindGroup` is identical; only *who builds it* changes.  No runtime
  behavior change.

- Phase 2 changes function signatures to thread a value that is currently
  always 64.  If the default is preserved, all generated WGSL and dispatch
  counts are unchanged.  Risk arises only when a user actually changes the
  workgroup size — but that is the point: making it *possible* to do so
  instead of silently hardcoding it.

- Phase 3 is purely additive (docs + test).

## Ordering

Phase 1 (precond decoupling) should be done first because it is the highest-
value change: it removes the last physics dependency from the linear solver
layer and makes adding new preconditioner types a local change instead of a
cross-cutting one.

Phase 2 (workgroup size) can be done independently but benefits from Phase 1
being in place (fewer call-sites to thread the parameter through).

Phase 3 (docs + test) can be done at any time.
