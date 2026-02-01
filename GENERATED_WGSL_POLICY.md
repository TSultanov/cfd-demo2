# Generated WGSL Commit Policy

## Summary

This repository follows a **commit generated WGSL** policy. All WGSL shader files in `src/solver/gpu/shaders/generated/` are deterministically generated at build time and **must be committed** to version control.

## Rationale

- **Stable builds**: Users can build the project without requiring the full codegen pipeline to run successfully
- **Code review**: Generated shaders are visible in PRs for review
- **Debugging**: Generated code is available for inspection without rebuilding
- **Determinism**: The generated output is stable and reproducible

## Generation Pipeline

### Build-time Generation (build.rs)

The `build.rs` script generates WGSL files during the build process:

1. **Model-specific kernel generation** (`emit_model_kernels_wgsl`):
   - Generates kernels based on `ModelSpec` definitions
   - Uses the model's module list (`ModelSpec.modules`) and per-module WGSL generators
   - Outputs to `src/solver/gpu/shaders/generated/`

2. **Binding generation** (`wgsl_bindgen`):
   - Parses all WGSL files (handwritten + generated)
   - Generates Rust bindings in `src/solver/gpu/bindings.rs`
   - This file embeds shader source as string constants (`SHADER_STRING`)

3. **Metadata generation** (build-time outputs to `OUT_DIR`):
   - `generic_coupled_registry.rs` - model-specific kernel lookup tables
   - `kernel_registry_map.rs` - kernel ID to pipeline constructor mapping
   - `wgsl_binding_meta.rs` - binding metadata for all shaders

### Key Files

| File | Purpose | Generated? | Committed? |
|------|---------|------------|------------|
| `src/solver/gpu/shaders/generated/*.wgsl` | Generated shader code | Yes (build.rs) | **YES** ✓ |
| `src/solver/gpu/bindings.rs` | Rust bindings + embedded source | Yes (wgsl_bindgen) | **YES** ✓ |
| `OUT_DIR/generic_coupled_registry.rs` | Runtime kernel registry | Yes (build.rs) | No (ephemeral) |
| `OUT_DIR/kernel_registry_map.rs` | Kernel lookup tables | Yes (build.rs) | No (ephemeral) |
| `OUT_DIR/wgsl_binding_meta.rs` | Binding metadata | Yes (build.rs) | No (ephemeral) |

## No Runtime Generation

**Critical invariant**: There is **no runtime WGSL generation**. All shaders are:
1. Generated at build time by `build.rs`
2. Embedded into the binary via `bindings.rs`
3. Loaded at runtime from embedded constants

The runtime accesses shaders via:
- `bindings::generated::{module}::SHADER_STRING` (embedded source)
- `create_main_pipeline_embed_source(device)` (pipeline constructors)
- Registry lookups in `kernel_registry.rs` (using build-time generated tables)

## Module Manifests (Model Contract)

Models are composed from **modules**. Each module carries a small manifest that acts as the contract between:
- model definitions (`src/solver/model/definitions/*`)
- build-time WGSL emission (build.rs)
- runtime lowering/recipes

Today, module manifests can contribute:
- **Method selection** (exactly one module must define a method)
- **Flux module configuration** (0 or 1 module may define a flux module)
- **Named parameter keys** (the set of `plan.set_named_param()` keys accepted)
- **Typed invariants** (state-layout / equation-system requirements validated early)
- **PortManifest** (IR-safe port declarations for fields, params, buffers, and pre-resolved metadata)

Conflicts are **explicit errors** (e.g. multiple modules defining a method or flux module).

### PortManifest and PortRegistry

Modules declare their ports via `PortManifest` attached to `KernelBundleModule.port_manifest`:

- **Fields**: Declare required state fields with expected kind (Scalar/Vector2/Vector3) and physical dimension
- **Params**: Declare uniform parameters with WGSL type and dimension
- **Buffers**: Declare required buffer bindings with type and access mode
- **Pre-resolved metadata**: Store `gradient_targets`, `resolved_state_slots`, etc. for WGSL generation

At runtime (and build-time), `PortRegistry` consumes the `PortManifest` to:
- Validate declared fields exist in `StateLayout` with correct kind/dimension
- Register ports idempotently (conflicting specs return errors)
- Provide typed access to offsets, strides, and component indices
- Support the `AnyDimension` escape hatch for genuinely dynamic fields

This enables **module WGSL generators** to consume pre-resolved slot/target metadata without probing `StateLayout` directly.

### EOS-implied low-mach policy

Low-mach resources and named params are currently **EOS-implied** (not module-declared):
- When the model EOS is `IdealGas` or `LinearCompressibility`, the runtime allocates low-mach resources.
- The named params `low_mach.*` are considered valid only when those resources exist.

This keeps the short-term contract simple while we migrate more resource requirements into manifests.

## Verification

### Manual Check

```bash
# Regenerate WGSL files
cargo build

# Check for uncommitted changes
git status src/solver/gpu/shaders/generated/
```

If files are modified, they need to be committed.

### Automated Check

Run the verification script:

```bash
./scripts/check_generated_wgsl.sh
```

This script:
1. Saves current generated files
2. Runs `cargo build` to regenerate
3. Diffs the results
4. Exits with error if any differences found

### CI Integration (Recommended)

Add to your CI workflow (e.g., `.github/workflows/ci.yml`):

```yaml
- name: Check generated WGSL is up to date
  run: ./scripts/check_generated_wgsl.sh
```

## Developer Workflow

### When modifying codegen logic:

1. Make changes to codegen in `src/solver/codegen/`
2. Run `cargo build` to regenerate WGSL
3. Review the diff in `src/solver/gpu/shaders/generated/`
4. Commit both your changes AND the regenerated files

### When adding a new model:

1. Add model definition in `src/solver/model/definitions.rs`
2. Run `cargo build` to generate model-specific kernels
3. New files will appear in `src/solver/gpu/shaders/generated/`
4. Commit the new generated files

### Common Issues

**Build fails with missing generated files:**
- Just run `cargo build` - it will create them

**Generated files have uncommitted changes:**
- Review the diff to ensure it matches your intent
- Commit the changes along with your code changes

**Merge conflicts in generated files:**
- Resolve conflicts in source code first
- Run `cargo build` to regenerate
- Commit the regenerated version

## Guardrails

### Existing Protection

✓ **Deterministic generation**: `build.rs` uses `write_if_changed()` to avoid unnecessary updates

✓ **Build-time validation**: Codegen errors cause build to fail (no silent divergence)

✓ **Type safety**: `wgsl_bindgen` generates typed bindings checked at compile time

### Recommended Additions

1. **CI check** (see script above): Prevents merging stale generated files

2. **Pre-commit hook** (optional):
   ```bash
   # .git/hooks/pre-commit
   ./scripts/check_generated_wgsl.sh
   ```

3. **Documentation**: This file serves as the policy reference

## Current Status

As of 2026-01-12:

- ✓ All generated WGSL files are committed
- ✓ No runtime generation exists
- ✓ Build.rs generates deterministically
- ✓ Bindings embed shader source
- ✓ Verification script available
- ⚠ No CI enforcement yet (recommended)
- ⚠ No pre-commit hook (optional)

## Related Documentation

- `CODEGEN_PLAN.md` - Overall codegen architecture and future plans
- `build.rs` - Build-time generation logic
- `src/solver/model/module.rs` - Module + manifest definitions
