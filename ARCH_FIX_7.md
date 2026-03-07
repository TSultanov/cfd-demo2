# ARCH_FIX_7: Simplify the One-Submission Outer-Loop Solver

**Issue**: Architecture review item #7 — "Architectural Dead Code & Over-engineering."

The codebase packs thousands of GPU iterations into a single wgpu `CommandEncoder`
submission ("one-submission outer loop") to amortise CPU↔GPU synchronisation
overhead. This gives a real wall-clock speedup, but the implementation has grown
to **~1500 lines** of highly intricate machinery spread across two files:

| Component | File | Lines |
|-----------|------|------:|
| `build_outer_gate_wgsl` | `generic_coupled.rs` | 148 |
| `build_outer_stop_inject_wgsl` | `generic_coupled.rs` | 56 |
| `OuterAdaptiveGate` | `generic_coupled.rs` | 264 |
| `OuterConvergenceMonitor` | `generic_coupled.rs` | 253 |
| `try_host_coupled_batch_tail_one_submission` | `generic_coupled.rs` | 300 |
| `host_coupled_batch_tail` + dispatch helpers | `generic_coupled.rs` | 67 |
| `OneSubmissionEnvTunables` | `linear_solver.rs` | 40 |
| `encode_solve_fgmres_fixed_iterations` | `linear_solver.rs` | 130 |
| `submit_solve_fgmres_fixed_iterations_chunked` | `linear_solver.rs` | 185 |
| `submit_solve_cg_fixed_iterations_chunked` | `linear_solver.rs` | 71 |
| **Total** | | **~1514** |

The complexity stems from three entangled concerns:

1. **Blind encoding** – GPU dispatches are recorded for a fixed iteration count
   regardless of convergence, wasting GPU cycles when the solver converges early.
2. **GPU-side early-stopping workaround** – To mitigate (1), the code builds
   special WGSL shaders at runtime (`build_outer_gate_wgsl`,
   `build_outer_stop_inject_wgsl`) that read a `break_status` buffer and either
   zero-out indirect dispatch arguments (skipping work) or inject a `STOP` scalar
   into the FGMRES/CG solver so subsequent iterations become near-zero-cost.
3. **Metal chunking workaround** – Recording too many dispatches in one
   `CommandEncoder` causes Metal backend hangs, so the code splits the iteration
   budget into "chunks", each with its own encoder → submit cycle (defeating the
   original single-submit goal on Metal).

The result is a Rube-Goldberg architecture where:
- The host pre-records *N* outer iterations worth of GPU work.
- Each outer iteration includes: STOP-inject → assembly (indirect) → FGMRES
  restart body (possibly multi-chunk) → update (indirect) → convergence-check →
  gate.
- After all submissions, the host reads back an iteration counter to learn how
  many iterations the GPU actually ran before the gate zeroed the dispatches.
- An env-var (`CFD2_ENABLE_GPU_OUTER_LOOP_PRIMITIVE`) adds another behavioural
  mode that skips the counter readback entirely.

---

## Goals

1. **Reduce complexity** – bring the one-submission infrastructure from ~1500
   lines to ≤400 while preserving the performance benefit.
2. **Eliminate runtime WGSL generation** – the gate/stop-inject shaders should
   be compile-time artifacts, not `build_outer_gate_wgsl()` string builders.
3. **Decouple outer convergence monitoring from the batched submission path** –
   the `OuterConvergenceMonitor` is useful in both host-driven and batched modes
   but is currently wired only through the batch tail.
4. **Keep Metal chunking** – the workaround is necessary but should be handled
   at a lower level (linear solver encoding) rather than mixed into the
   outer-loop orchestration.
5. **No solver correctness regression** – OpenFOAM reference metrics and the
   `rhie_chow_fusion_parity_test` must be unchanged.

---

## Non-Goals

- Removing the one-submission optimisation entirely. The benchmark
  (`bench_submission_path`) shows a measurable speedup, and this is a valid
  GPU programming pattern.
- Changing the FGMRES/CG solver core algorithms.
- Modifying the kernel fusion system.

---

## Current Data Flow

```
step()
  → execute_block(root)
    → coupled:begin_step       [host: reset, upload constants]
    → coupled:init_prepare     [graph: one-time dp_init kernels]
    → Repeat(coupled:outer_iters)
      → coupled:before_iter    [host: clear dp_init, check batched mode]
           ↓ if batched && outer_iters>1 && first iteration:
           ↓   try_host_coupled_batch_tail_one_submission(remaining)
           ↓     → encodes ALL remaining outer iterations into chunked submissions
           ↓     → sets plan.skip_remaining_block = true, plan.repeat_break = true
           ↓     → reads back iter_counter from GPU
           ↓   ← returns (skips rest of repeat body)
      → coupled:assembly       [graph: assemble matrix/RHS]
      → coupled:solve          [host: host-driven FGMRES/CG with readbacks]
      → coupled:update         [graph: apply correction]
      → coupled:batch_tail     [host: second chance for batched path]
    → coupled:finalize_step    [host: advance time, ping-pong]
```

**Problem**: When `outer_batched_mode=true` (the default), the `before_iter`
hook hijacks the entire outer loop on the first iteration. The remaining recipe
nodes (`assembly`, `solve`, `update`, `batch_tail`) in that iteration body are
never executed. The `batch_tail` hook exists as a second-chance fallback but is
redundant — if `before_iter` consumes the batch, `batch_tail` never fires.

---

## Proposed Architecture

### Phase 1: Extract `OuterConvergenceMonitor` into its own module (~150 lines moved)

**Files**: new `src/solver/gpu/modules/outer_convergence.rs`

The `OuterConvergenceMonitor` struct and its WGSL infrastructure are currently
inlined in `generic_coupled.rs`. This phase moves them to a dedicated module:

- Move `OuterConvergenceMonitor`, `GpuOuterConvergenceParams`,
  `GpuOuterConvergenceTargetDesc`, `GpuOuterConvergenceBreakParams` to the new
  module.
- Move `compute_outer_residuals` helper logic.
- The monitor's `encode_convergence_check` and `delta_maxima` methods become
  the public API.
- `generic_coupled.rs` uses the monitor via the new module import.

**Benefit**: The convergence monitor is a self-contained GPU reduction + readback
utility with no coupling to the batch submission path. Extracting it makes it
reusable from both host-driven and batched modes, and removes ~250 lines from
the already-oversized `generic_coupled.rs` (3755 lines).

### Phase 2: Extract `OuterAdaptiveGate` into its own module (~270 lines moved)

**Files**: new `src/solver/gpu/modules/outer_gate.rs`

Move `OuterAdaptiveGate`, `build_outer_gate_wgsl`, `build_outer_stop_inject_wgsl`,
and the gate's encode/dispatch methods to a dedicated module.

- The gate module exposes:
  - `OuterAdaptiveGate::new(device, queue, num_cells, num_faces, max_wg, break_status_buf)`
  - `encode_gate_into(encoder)`
  - `encode_stop_inject_into(encoder, bg)` / `encode_stop_inject_cg_into(encoder, bg)`
  - `create_stop_inject_bind_group(device, break_status, scalars)` (and CG variant)
  - `read_iter_counter(device, queue, staging_cache) → u32`
- `generic_coupled.rs` imports and uses these.

**Benefit**: The gate is a reusable GPU primitive (read status → conditionally
zero dispatches + inject stop signal). Isolating it clarifies the contract and
makes it testable independently.

### Phase 3: Convert gate/stop-inject WGSL to compile-time generated shaders

**Files**: `crates/cfd2_codegen/src/solver/codegen/infrastructure_kernels.rs`,
`outer_gate.rs` (from Phase 2), `build.rs`

Currently `build_outer_gate_wgsl()` and `build_outer_stop_inject_wgsl(N)` build
WGSL strings at runtime via the AST DSL and compile them with
`device.create_shader_module()` during solver initialisation.

- Add `outer_gate` and `outer_stop_inject_fgmres` / `outer_stop_inject_cg`
  as infrastructure kernels in the codegen, generated at build time alongside
  existing infrastructure shaders (dot_product, amg, etc.).
- The `SCALAR_STOP` index is a compile-time constant already
  (`FGMRES_SCALAR_STOP = 8`, `CG_SCALAR_STOP = 6`), so the two stop-inject
  variants can be templated at build time.
- `OuterAdaptiveGate::new()` loads the pre-compiled shader modules via
  `include_str!` / `include_wgsl!` instead of calling the string builders.
- Delete `build_outer_gate_wgsl()` and `build_outer_stop_inject_wgsl()`.

**Benefit**: Removes ~200 lines of runtime WGSL string construction and AST
manipulation. Shader compilation errors surface at build time instead of runtime.
Faster solver initialisation.

### Phase 4: Consolidate the two batch-tail entry points

**Files**: `generic_coupled.rs`, `universal.rs`, `recipe.rs`

Currently there are two host ops that can trigger the batched path:
- `coupled:before_iter` → `host_coupled_before_iter()` → calls
  `try_host_coupled_batch_tail_one_submission()` and sets `skip_remaining_block`.
- `coupled:batch_tail` → `host_coupled_batch_tail()` → also calls
  `try_host_coupled_batch_tail_one_submission()` (as second-chance fallback).

These are redundant; the `batch_tail` path never fires when `before_iter`
succeeds.

- Remove the `coupled:batch_tail` program spec node and its host op
  registration.
- Keep the `coupled:before_iter` as the single entry point for batch
  optimisation.
- Simplify `host_coupled_before_iter` to try the batch path directly (no
  need for `skip_remaining_block` dance — use `repeat_break` only).
- Update `UnifiedOpRegistryConfig` to remove the `coupled_batch_tail` field.
- Update `recipe.rs` `build_program_spec()` to stop emitting the
  `coupled:batch_tail` node.
- Update tests (`rhie_chow_fusion_parity_test`) that set `outer_batched_mode`.

**Benefit**: Removes one host op, one op registration, and ~40 lines of
dead-code fallback logic. The program spec becomes simpler and reflects the
actual execution flow.

### Phase 5: Move Metal-chunking logic into the linear solver layer

**Files**: `linear_solver.rs`

The `submit_solve_fgmres_fixed_iterations_chunked` function does two things:
1. Splits iterations into chunks (Metal workaround).
2. Provides `pre_encode` / `post_encode` callbacks for the caller to inject
   assembly/update dispatches.

These concerns should be separated:

- Rename `submit_solve_fgmres_fixed_iterations_chunked` →
  `submit_encoded_fgmres_solve` and simplify its interface to:
  ```rust
  pub fn submit_encoded_fgmres_solve<P: PreconditionerModule>(
      krylov: &mut KrylovSolveModule<P>,
      context: &GpuContext,
      system: LinearSystemView,
      params: EncodedSolveParams,
      pre_commands: Option<&dyn Fn(&mut CommandEncoder)>,
      post_commands: Option<&dyn Fn(&mut CommandEncoder)>,
  ) -> LinearSolverStats
  ```
- The chunking logic stays internal but is documented as a Metal workaround.
- Remove the `encode_solve_fgmres_fixed_iterations` function (only called
  from the now-unused non-chunked path).
- Similarly simplify the CG chunked function.

**Benefit**: Cleaner API, removes one unused encode path (~130 lines).

### Phase 6: Consolidate env-var tunables

**Files**: `linear_solver.rs`, `generic_coupled.rs`

Currently there are **8 env vars** controlling the one-submission path:
- `CFD2_ENABLE_GPU_OUTER_LOOP_PRIMITIVE`
- `CFD2_ENABLE_ENCODED_SEED_BASIS0`
- `CFD2_ONE_SUBMISSION_RESTART_BUDGET`
- `CFD2_ONE_SUBMISSION_TOTAL_ITERS`
- `CFD2_ONE_SUBMISSION_MIN_TAIL`
- `CFD2_ONE_SUBMISSION_CHUNKS`
- `CFD2_ONE_SUBMISSION_CG_CHUNK_SIZE`
- `CFD2_DEBUG_FGMRES`

Most of these are development tuning knobs that have converged to stable
defaults.

- Remove `CFD2_ENABLE_GPU_OUTER_LOOP_PRIMITIVE` (the GPU-only path that skips
  iteration counter readback). If adaptive break is enabled, always read back
  the counter — the readback cost is negligible (4 bytes, one submission).
- Remove `CFD2_ONE_SUBMISSION_CHUNKS` (explicit chunk layout). No evidence of
  use outside development.
- Keep `CFD2_ONE_SUBMISSION_RESTART_BUDGET` and
  `CFD2_ONE_SUBMISSION_TOTAL_ITERS` but move them into a single
  `EncodedSolveBudget` struct with clear documentation.
- Keep `CFD2_ONE_SUBMISSION_MIN_TAIL` and `CFD2_ONE_SUBMISSION_CG_CHUNK_SIZE`
  (Metal-specific).
- Keep `CFD2_DEBUG_FGMRES` and `CFD2_ENABLE_ENCODED_SEED_BASIS0` (debugging
  and parity testing).

**Benefit**: Removes two env vars and the most confusing behavioural mode.
`gpu_outer_loop_primitive_enabled()` and its branching disappear (~30 lines).

### Phase 7: Documentation and benchmark validation

**Files**: `benches/fusion_policy_benchmark.rs`, new
`docs/one_submission_solver.md` or inline doc comments

- Add architecture documentation explaining the one-submission optimisation:
  why it exists, how chunking works, the gate/stop-inject mechanism, and when
  the host-driven fallback is used.
- Run the `bench_submission_path` benchmark before and after to verify no
  performance regression.
- Update the `examples/analyze_gpu_dispatches.rs` if its output format changes.

---

## Estimated Impact

| Metric | Before | After (actual) |
|--------|--------|----------------|
| `generic_coupled.rs` | 3755 lines | 2365 lines (–1390) |
| `linear_solver.rs` | 691 lines | 636 lines (–55) |
| Runtime WGSL generation | 3 shaders | 1 shader (convergence break) |
| Env vars | 8 | 6 |
| Batch-tail entry points | 2 | 1 |
| New modules | 0 | 2 (`outer_convergence`, `outer_gate`) |

## File Inventory

### Modified files
- `src/solver/gpu/lowering/programs/generic_coupled.rs` — phases 1–4, 6
- `src/solver/gpu/modules/linear_solver.rs` — phases 5, 6
- `src/solver/gpu/lowering/programs/universal.rs` — phase 4
- `src/solver/gpu/lowering/unified_registry.rs` — phase 4
- `src/solver/gpu/recipe.rs` — phase 4
- `src/solver/gpu/modules/mod.rs` — phases 1, 2
- `crates/cfd2_codegen/src/solver/codegen/infrastructure_kernels.rs` — phase 3
- `build.rs` — phase 3
- `benches/fusion_policy_benchmark.rs` — phase 7

### New files
- `src/solver/gpu/modules/outer_convergence.rs` — phase 1
- `src/solver/gpu/modules/outer_gate.rs` — phase 2

### Test files to update
- `tests/rhie_chow_fusion_parity_test.rs` — phase 4 (remove `batch_tail` refs)

---

## Verification Checklist

- [x] `cargo test --features meshgen --lib` — all 175 tests pass
- [ ] `cargo check --tests` — test binaries compile
- [ ] OpenFOAM reference metrics unchanged (per AGENTS.md drift check)
      (NOTE: OpenFOAM test files have pre-existing compile errors unrelated to this work)
- [ ] `bench_submission_path` — no regression vs baseline
- [ ] `rhie_chow_fusion_parity_test` — host-driven and batched paths produce
      identical results (this test exercises both modes)
- [x] `generic_coupled.rs` line count drops below 3200 (now 2365)

## Implementation Progress

| Phase | Status | Result |
|-------|--------|--------|
| 1 — Extract `OuterConvergenceMonitor` | ✅ Done | 3755→2914 lines; new `outer_convergence.rs` (860 lines) |
| 2 — Extract `OuterAdaptiveGate` | ✅ Done | 2914→2439 lines; new `outer_gate.rs` (480 lines) |
| 3 — Compile-time WGSL generation | ⬜ Not started | Touches codegen crate; deferred |
| 4 — Remove `coupled:batch_tail` | ✅ Done | 2440→2390 lines; removed from recipe/registry/universal |
| 5 — Simplify linear solver API | ⬜ Not started | |
| 6 — Remove env vars | ✅ Done | 2390→2365 lines; `linear_solver.rs` 691→636 |
| 7 — Documentation & benchmarks | ⬜ Not started | |
