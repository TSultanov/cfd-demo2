# Codegen Plan (OpenFOAM-like Abstractions)

## Goal
Introduce a high-level, math-like model definition (PDE-style) and derive the numerical
discretization automatically, with modular swapping of schemes and boundary conditions.

## Core Architecture
- **Model DSL (AST):** Typed expressions for `ddt`, `div`, `grad`, `laplacian`, `source`.
- **Equation System:** `eqn(field)` that tracks implicit vs explicit terms (fvm vs fvc).
- **Discretization IR:** Lowered form of terms into per-face / per-cell contributions,
  including matrix blocks and RHS contributions.
- **Scheme Registry:** Per-term, per-field scheme selection (e.g., `div(phi, U)` uses
  QUICK, `grad(p)` uses linear, `ddt` uses BDF2).
- **BC Layer:** Boundary-condition objects applied during IR lowering for each term.
- **Codegen Backend:** Emit WGSL kernels from IR and feed into `wgsl_bindgen`.

## Integration Points in Current Code
- Replace hand-coded assembly in `src/solver/gpu/shaders/coupled_assembly_merged.wgsl`
  and `src/solver/gpu/shaders/pressure_assembly.wgsl` with generated kernels.
- Use existing `build.rs` to run codegen prior to `wgsl_bindgen` output.
- Extend `src/solver/scheme.rs` into a per-term scheme registry.

## Phased Plan
1. **AST + IR Skeleton (incompressible only)**
   - Define DSL nodes and a minimal IR with flux + diffusion + source terms.
   - Emit a no-op or debug WGSL to prove toolchain wiring.
2. **Single-Term Codegen (convection)**
   - Generate `div(phi, U)` for a single scalar and compare output with current shader.
   - Add scheme switch (Upwind/SOU/QUICK) within codegen.
3. **Full Incompressible Kernel**
   - Add `ddt`, `laplacian`, `grad(p)` terms.
   - Match current `coupled_assembly_merged.wgsl` output for U and V blocks.
4. **BC + Pressure Equation**
   - Encode BCs in IR lowering (Dirichlet/Neumann/Wall/Outlet).
   - Generate the pressure assembly kernel.
5. **Scheme Modularity + Config**
   - Implement `fvSchemes`-like config mapping terms to schemes.
   - Wire config to UI or test harness.
6. **Compressible Extension (optional)**
   - Add conservative variables and Riemann solver hooks in IR.
   - Add all-speed/preconditioning parameters as term modifiers.

## Deliverables / Milestones
- **M1:** DSL + IR compiles and codegen generates WGSL files in `src/solver/gpu/shaders/generated/`.
- **M2:** Convection-only generated kernel matches current results for a simple test.
- **M3:** Full incompressible generated assembly replaces hand-written shader.
- **M4:** BC and pressure assembly generated.
- **M5:** Scheme registry and config-driven selection.

## Progress
- **2025-02-10:** Added codegen module skeleton (AST, scheme registry, IR lowering, WGSL stub)
  with unit tests covering constructors, scheme selection, lowering, and WGSL emission. (`src/solver/codegen/*.rs`)
- **2025-02-10:** Expanded unit tests to cover string formatting helpers, defaults, and optional
  metadata handling in codegen output.
- **2025-02-10:** Fixed test coverage gaps in codegen unit tests (ownership handling, unused imports).
- **2025-02-10:** Added codegen file emitter with test coverage for filesystem output (`src/solver/codegen/emit.rs`).
- **2025-02-10:** Added math-like term composition (`Term + Term` -> `TermSum`) and `eqn()` helpers.
- **2025-02-10:** Added a reusable incompressible momentum model builder for codegen tests/examples.
- **2025-02-10:** Added scheme string parsing helpers to prep for config-driven scheme selection.
- **2025-02-10:** Added default/generated WGSL output path helpers and tests for generated folder emission.
- **2025-02-10:** Added a codegen emitter for an incompressible momentum WGSL file and a CLI
  entrypoint to generate it (`src/bin/codegen.rs`).
- **2025-02-10:** Added per-term WGSL stub function emission with sanitized identifiers and tests.
- **2025-02-10:** Removed the `fvSchemes`-style parser to keep configuration code-driven only (GUI-driven solver).
- **2025-02-10:** WGSL stubs now emit scheme IDs for convection scaffolding (per-term scheme selection).
- **2025-02-10:** Added scheme-aware WGSL emission for the momentum model (registry-driven output).
- **2025-02-10:** Cleaned up emit module imports to keep tests warning-free.
- **2025-02-10:** WGSL generator now emits per-equation assembly functions and a `main` stub.
- **2025-02-10:** WGSL stubs now expose discretization IDs for explicit/implicit terms.
- **2025-02-10:** Fixed WGSL stub discretization helper imports after compile error.
- **2025-02-10:** Added build-script codegen hook to emit generated WGSL before bindings are built.
- **2025-02-10:** Emit step now skips writing unchanged WGSL to avoid rebuild churn.
- **2025-02-10:** Build script now scans nested WGSL files so generated shaders are bound.
- **2025-02-10:** Generated `coupled_assembly_merged` shader now mirrors the base shader plus codegen
  library, and the solver uses the generated path for layout/pipeline setup.
- **2025-02-10:** First generated injection: convection coefficient block now routes through a
  codegen helper in the generated shader.
- **2025-02-10:** Added diffusion coefficient helper injection in the generated coupled assembly shader.
- **2025-02-10:** Codegen now uses a dedicated template (not the runtime shader) to generate
  `coupled_assembly_merged` from scratch.
- **2025-02-10:** Removed legacy base-shader generation path to enforce template-only codegen.
- **2025-02-10:** Coupled-assembly WGSL is now emitted directly in code (no templates or text replacements).
- **2025-02-10:** Removed template shader file and replacement helpers; coupled assembly now emits from code only.
- **2025-02-10:** Added a WGSL AST with expression parser and renderer to move shader generation off raw string templates (`src/solver/codegen/wgsl_ast.rs`).
- **2025-02-10:** Added a WGSL DSL helper layer for building AST blocks and statements from expressions (`src/solver/codegen/wgsl_dsl.rs`).
- **2025-02-10:** Refactored WGSL stub/library generation to build AST items and render via the new WGSL renderer (`src/solver/codegen/wgsl.rs`).
- **2025-02-10:** Added a state layout definition for mapping field names to packed array offsets (`src/solver/codegen/state_layout.rs`).
- **2025-02-10:** Added a model spec that includes an explicit state layout for the incompressible solver (`src/solver/codegen/model.rs`).
- **2025-02-10:** Rebuilt coupled-assembly generation using the WGSL AST/DSL with state stored as a packed `array<f32>` and compile-time field offsets (`src/solver/codegen/coupled_assembly.rs`).
- **2025-02-10:** Added shared packed-state access helpers for WGSL expression generation (`src/solver/codegen/state_access.rs`).
- **2025-02-10:** Extended packed-state helpers with component-level addressing for writes (`src/solver/codegen/state_access.rs`).
- **2025-02-10:** Added prepare-coupled WGSL generator using packed state arrays and AST/DSL (`src/solver/codegen/prepare_coupled.rs`).
- **2025-02-10:** Added pressure-assembly WGSL generator using packed state arrays and AST/DSL (`src/solver/codegen/pressure_assembly.rs`).
- **2025-02-10:** Extended WGSL AST storage classes to support workgroup globals used by update kernels (`src/solver/codegen/wgsl_ast.rs`).
- **2025-02-10:** Added update-fields-from-coupled WGSL generator using packed state arrays and AST/DSL (`src/solver/codegen/update_fields_from_coupled.rs`).
- **2025-02-10:** Switched physics pipeline bindings to generated prepare/pressure/update shaders for packed state buffers (`src/solver/gpu/init/physics.rs`).
- **2025-02-10:** Began deriving coupled-assembly term blocks from `DiscreteSystem` (term flags + scheme id) and added tests (`src/solver/codegen/coupled_assembly.rs`).
- **2025-02-10:** Removed unused coupled-assembly layout parameter in codegen base-item builder to keep codegen warning-free (`src/solver/codegen/coupled_assembly.rs`).
- **2025-02-10:** Avoided unsupported bitshift in codegen WGSL DSL by switching update-kernel stride reduction to `/ 2u` (`src/solver/codegen/update_fields_from_coupled.rs`).
- **2025-02-10:** Verified codegen unit tests after generating packed-state prepare/pressure/update shaders via build hooks (`cargo test codegen --lib`).
- **2025-02-10:** Added codegen for Rhie-Chow flux kernel with packed state arrays and wired generated shader into physics pipelines (`src/solver/codegen/flux_rhie_chow.rs`, `src/solver/gpu/init/physics.rs`).
- **2025-02-10:** Verified codegen unit tests after adding flux kernel generator (`cargo test codegen --lib`).
- **2025-02-10:** Added `ShaderVariant` plumbing so GPU init can switch between manual and generated kernels; added a solver-level comparison test (`src/solver/gpu/init/*.rs`, `tests/gpu_codegen_matches_manual_test.rs`).
- **2025-02-10:** Verified manual vs generated solver comparison test (`cargo test --test gpu_codegen_matches_manual_test`).
- **2025-02-10:** Added a high-level pressure equation to the model definition so the system includes momentum + pressure (`src/solver/codegen/model.rs`).
- **2025-02-10:** Adjusted model term construction to reuse the pressure field without moving it (`src/solver/codegen/model.rs`).
- **2025-02-10:** Verified codegen unit tests after adding the high-level pressure equation (`cargo test codegen --lib`).
