# ARCH_FIX_4: Eliminate Hazard-Blind Aggressive Kernel Fusion

## Status: Complete

## Problem Statement

Architecture Review §4 ("Dangerous 'Aggressive' Kernel Fusion") identifies two
risks in the kernel fusion system (`crates/cfd2_codegen/src/solver/codegen/fusion.rs`):

1. **Hazard-blind fusion under `FusionSafetyPolicy::Aggressive`** — The
   `ensure_safe_composition` function detects RAW, WAR, WAW, and
   barrier/atomics hazards across a sequence of `KernelProgram`s. Under `Safe`
   policy any hazard is a hard rejection. Under `Aggressive` policy hazards are
   collected into a `Vec<HazardReport>` but **do not block synthesis**. The
   fused program proceeds, meaning kernels with genuine data dependencies may
   be merged into a single dispatch without barriers, producing
   non-deterministic GPU data races.

   Today this works by accident: the 5 aggressive-only Rhie-Chow rules in
   `src/solver/model/modules/rhie_chow.rs` produce bitwise-identical results
   (verified by `rhie_chow_fusion_parity_test.rs` with `rel_tol = 0.0`). But
   nothing structurally prevents a future rule from silently introducing a real
   race condition. The `fusion_coverage` tool reports hazards under Aggressive
   policy in `FUSION_COVERAGE.md`, but this is advisory — no CI gate enforces
   that reported hazards have been audited.

2. **Fragile AST-based cleanup transforms** — The function
   `apply_aggressive_cleanup` (called only under `Aggressive` policy) runs two
   AST passes:
   - `apply_ast_load_after_store_forwarding`: local store→load forwarding
     within the flat body statement list.
   - `apply_ast_noop_self_assign_cleanup`: removes `ident = ident` statements.

   The review calls this "regex parsing HTML," but upon inspection the
   implementation operates on a typed AST (`cfd2_ir::ast`), uses conservative
   invalidation (clears forwarding map on any control flow, does not forward
   when the stored value itself reads from memory), and has 5 targeted unit
   tests. The actual risk is **not fragility of the current code** but rather
   **coupling cleanup transforms to the Aggressive hazard bypass**: any future
   cleanup pass added to `apply_aggressive_cleanup` inherits the blanket hazard
   waiver.

### Root cause

`FusionSafetyPolicy` conflates two orthogonal concerns:

- **Hazard policy**: should synthesis be blocked when data hazards exist?
- **Cleanup policy**: should post-synthesis AST optimization passes run?

Because both are gated on a single `Aggressive` enum variant, opting into
cleanup transforms (which are safe and beneficial) requires also opting into
hazard blindness (which is dangerous).

### Existing mitigation (partial)

- `detect_hazards()` is a standalone public function that returns hazard reports
  without blocking. `synthesize_fused_program_with_report` returns hazards
  alongside the fused program. The `fusion_coverage` tool logs aggressive
  hazards to FUSION_COVERAGE.md.
- Parity tests verify bitwise-identical output for all current aggressive rules.
- The `FusionGuard::MinPolicy(Aggressive)` mechanism correctly gates which
  rules are allowed at which policy level.

## Non-Goals (out of scope)

- **Removing kernel fusion entirely.** The fusion system is well-engineered and
  delivers real dispatch-count reductions (verified by tests). Only the hazard
  bypass path needs structural enforcement.

- **Rewriting the AST cleanup passes.** The current `apply_ast_load_after_store_forwarding`
  and `apply_ast_noop_self_assign_cleanup` are correct and well-tested. They
  should be preserved and decoupled from the hazard policy.

- **Adding automatic barrier insertion.** Automatically inserting
  `storageBarrier()` or `workgroupBarrier()` into fused kernels to resolve
  detected hazards is a large and risky change. Out of scope.

- **Changing the `KernelFusionPolicy` enum in the model layer.** The
  `Off`/`Safe`/`Aggressive` user-facing policy knob in `KernelFusionPolicy`
  can remain as-is for backward compatibility. What changes is how the codegen
  layer interprets the internal `FusionSafetyPolicy`.

## Plan

### Phase 1: Require explicit hazard whitelisting for Aggressive rules

**Goal:** Aggressive fusion rules that trigger hazards must explicitly declare
which hazards they expect. If synthesis detects an undeclared hazard, it fails
even under Aggressive policy.

#### 1a. Add `expected_hazards` field to `ModelKernelFusionRule`

```rust
// In src/solver/model/kernel.rs
pub struct ModelKernelFusionRule {
    // ... existing fields ...

    /// Hazard whitelist for aggressive rules.  Each entry is a
    /// `(HazardKind, resource_description)` pair that the rule author has
    /// audited and confirmed is safe (e.g. because the kernels operate on
    /// disjoint index ranges, or the dependency is a false positive from
    /// conservative side-effect tracking).
    ///
    /// Under `Safe` policy this field is ignored (all hazards reject).
    /// Under `Aggressive` policy, only whitelisted hazards are tolerated;
    /// any non-whitelisted hazard still causes a hard rejection.
    pub expected_hazards: Vec<ExpectedHazard>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedHazard {
    pub kind: HazardKind,
    pub kernel_id: &'static str,
    /// Human-readable justification for why this hazard is safe.
    pub justification: &'static str,
}
```

~25 lines added.

#### 1b. Thread `expected_hazards` into `synthesize_fused_program`

Add an `expected_hazards` parameter to `synthesize_fused_program_remapped`
(or a new variant). In `ensure_safe_composition`, under `Aggressive` policy,
instead of unconditionally accepting all hazards:

```rust
// Before (current):
if policy == FusionSafetyPolicy::Safe {
    if let Some(h) = hazards.first() {
        return Err(...);
    }
}
// Under Aggressive: hazards silently returned.

// After (proposed):
match policy {
    FusionSafetyPolicy::Safe => {
        if let Some(h) = hazards.first() {
            return Err(...);
        }
    }
    FusionSafetyPolicy::Aggressive => {
        for h in &hazards {
            if !expected_hazards.iter().any(|e| e.matches(h)) {
                return Err(format!(
                    "fusion rejected: unexpected {} hazard at kernel '{}' \
                     (not in expected_hazards whitelist)",
                    h.kind, h.kernel_id
                ));
            }
        }
    }
}
```

~20 lines changed in `fusion.rs`.

#### 1c. Populate `expected_hazards` for existing Rhie-Chow rules

Audit the 5 aggressive-only rules in `rhie_chow.rs` against their
`detect_hazards` output. For each rule, record the exact set of expected
hazards with a justification string.

The `fusion_coverage` tool already reports hazards — use its output as the
source of truth:

```rust
ModelKernelFusionRule {
    name: "rhie_chow:dp_init_dp_update_store_grad_p_...",
    // ...
    expected_hazards: vec![
        ExpectedHazard {
            kind: HazardKind::WAW,
            kernel_id: "dp_update_from_diag",
            justification: "dp_update writes d_p[idx] which dp_init also writes; \
                            dp_update is sequentially after dp_init in body, \
                            last-writer-wins is correct",
        },
        // ...
    ],
}
```

~40 lines across 5 rules (variable based on actual hazard count).

#### 1d. Add tests verifying whitelist enforcement

- Test that an Aggressive rule with no `expected_hazards` and a detected
  hazard now **fails** synthesis (regression test for the old behavior).
- Test that an Aggressive rule with correct `expected_hazards` succeeds.
- Test that an Aggressive rule with a *subset* of expected hazards fails
  on the non-whitelisted hazard.

~50 lines of tests.

### Phase 2: Decouple cleanup transforms from hazard policy

**Goal:** The AST cleanup passes (`load_after_store_forwarding`,
`noop_self_assign_cleanup`) run based on a separate opt-in flag, not the
hazard policy.

#### 2a. Add `FusionCleanupPolicy` enum

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionCleanupPolicy {
    /// No post-synthesis AST cleanup (default for Safe policy).
    None,
    /// Run conservative AST cleanup passes (store-load forwarding,
    /// noop removal). These passes do not change semantics.
    Standard,
}
```

~10 lines.

#### 2b. Thread `FusionCleanupPolicy` through synthesis

Replace the current `if policy == Aggressive { apply_aggressive_cleanup(...) }`
with `if cleanup_policy == Standard { apply_cleanup(...) }`.

The mapping from `KernelFusionPolicy` → `(FusionSafetyPolicy, FusionCleanupPolicy)`
becomes:

| `KernelFusionPolicy` | `FusionSafetyPolicy` | `FusionCleanupPolicy` |
|---|---|---|
| `Off`        | n/a (fusion disabled) | n/a |
| `Safe`       | `Safe`                | `None` |
| `Aggressive` | `Aggressive`          | `Standard` |

This preserves current behavior while making the cleanup orthogonal.

~15 lines changed.

#### 2c. Rename `apply_aggressive_cleanup` to `apply_fusion_cleanup`

Rename the function and remove the word "aggressive" from documentation,
since cleanup is no longer aggressive-specific. Update all references.

~5 lines renamed.

#### 2d. Allow Safe rules to opt into cleanup (future enablement)

With the cleanup policy decoupled, a future change can enable
`FusionCleanupPolicy::Standard` for Safe rules too, since the cleanup
passes are semantics-preserving. This is not done in this changeset but
the architecture now supports it.

No code changes — this is a documentation note.

### Phase 3: Add CI-level hazard audit gate

**Goal:** Ensure that any new fusion rule with hazards must have
`expected_hazards` populated, and that existing whitelists stay in sync
with actual hazard detection.

#### 3a. Add `#[test]` that validates all rule whitelists against `detect_hazards`

For each model × rule combination, synthesize the rule's programs, run
`detect_hazards`, and verify:
- Every detected hazard has a matching `expected_hazards` entry.
- Every `expected_hazards` entry matches a detected hazard (no stale
  whitelist entries).

This catches both directions: new hazards introduced by code changes AND
stale whitelist entries from refactored kernels.

```rust
#[test]
fn all_aggressive_fusion_rules_have_accurate_hazard_whitelists() {
    for model in all_models() {
        for rule in model.fusion_rules() {
            if !rule.requires_aggressive_policy() { continue; }
            let programs = generate_programs_for_rule(&model, &rule);
            let hazards = detect_hazards(&programs);
            assert_whitelists_match(&rule.expected_hazards, &hazards, rule.name);
        }
    }
}
```

~40 lines.

#### 3b. Extend `fusion_coverage` tool to flag unwhitelisted hazards

Update `src/bin/fusion_coverage.rs` to emit a ❌ marker next to any
aggressive rule whose detected hazards are not fully covered by
`expected_hazards`. This makes the FUSION_COVERAGE.md report actionable.

~15 lines.

### Phase 4: Document the fusion safety contract

**Goal:** Make the fusion system's safety model explicitly documented.

#### 4a. Add module-level documentation to `fusion.rs`

Document:
- The hazard detection model (conservative side-effect sets, not true
  data-flow analysis).
- Why false positives occur (e.g. two kernels write to the same buffer
  at different indices — detected as WAW but actually safe).
- The whitelist mechanism and when to use it.
- The cleanup passes and their correctness invariants.

~40 lines of doc-comments.

#### 4b. Add a `FUSION.md` architecture decision record section

Add a section to the existing FUSION.md (or create one if it doesn't exist)
that records the decision to require explicit hazard whitelisting and the
rationale.

~20 lines.

## Files Affected

| File | Change |
|------|--------|
| `crates/cfd2_codegen/src/solver/codegen/fusion.rs` | Add `expected_hazards` parameter; enforce whitelist under Aggressive; rename `apply_aggressive_cleanup`; add `FusionCleanupPolicy` |
| `src/solver/model/kernel.rs` | Add `ExpectedHazard` struct and `expected_hazards` field to `ModelKernelFusionRule`; update `fusion_safety_policy_for_rule` |
| `src/solver/model/modules/rhie_chow.rs` | Populate `expected_hazards` for all 5 aggressive rules |
| `src/solver/model/modules/generic_coupled.rs` | Add empty `expected_hazards` to Safe rule (no-op) |
| `build.rs` | Thread expected_hazards through synthesis call |
| `src/bin/fusion_coverage.rs` | Extend reporting for whitelist coverage |
| `src/solver/gpu/lowering/fusion_schedule_registry.rs` | No changes expected |

## Estimated Scope

- Phase 1: ~135 lines (struct + enforcement + per-rule whitelists + tests)
- Phase 2: ~30 lines (enum + threading + rename)
- Phase 3: ~55 lines (CI test + coverage tool update)
- Phase 4: ~60 lines (documentation)
- **Total: ~280 lines touched**

## Verification

1. `cargo test --workspace` — all existing tests pass (including all
   `rhie_chow_fusion_parity_test`, `compressible_fusion_parity_test`,
   `generic_diffusion_fusion_parity_test` tests).
2. New whitelist enforcement test passes.
3. New whitelist completeness test passes (no stale / missing entries).
4. `cargo run --bin fusion_coverage -- --write FUSION_COVERAGE.md` runs
   cleanly and shows ✅ for all whitelisted hazards.
5. OpenFOAM reference metrics — no regression.
6. Manually verify: removing an `expected_hazards` entry from a Rhie-Chow
   rule causes the whitelist completeness test to fail.

## Risk Assessment

**Low risk.**

- Phase 1 is **strictly more restrictive** than the current behavior: rules
  that previously synthesized under Aggressive with unreported hazards will
  now fail unless explicitly whitelisted. Since we populate the whitelists
  from the known-correct current hazard reports, no behavior changes for
  existing rules.

- Phase 2 is a pure refactoring of the cleanup dispatch path. The same
  transforms run on the same programs; only the control-flow gating changes.

- Phase 3 is purely additive (new tests).

- Phase 4 is purely additive (documentation).

The only regression risk is if the hazard detection itself is non-deterministic
(e.g. depends on `HashMap` iteration order). The existing implementation uses
`BTreeSet` and `BTreeMap` throughout, so hazard reports are deterministic.

## Ordering

Phase 1 (hazard whitelisting) should be done first — it is the highest-value
change because it converts a silent hazard bypass into an explicit, audited
opt-in. This directly addresses the architecture review's primary concern.

Phase 2 (cleanup decoupling) can be done independently but benefits from
Phase 1 being in place (cleaner separation of concerns).

Phase 3 (CI gate) should be done immediately after Phase 1 to lock in the
whitelist invariant.

Phase 4 (documentation) can be done at any time.
