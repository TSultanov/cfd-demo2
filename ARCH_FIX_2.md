# ARCH_FIX_2: Eliminate the build.rs Text-Parser Boundary Linter

## Problem Statement

Architecture Review §2 ("Broken Boundaries: The build.rs Linter") identifies
three symptoms of a leaky abstraction boundary between codegen and runtime:

1. **`enforce_codegen_ir_boundary`** — a grep-based linter in `build.rs`
   (lines 356-420) that scans every `.rs` file in `cfd2_codegen/` for strings
   like `"crate::solver::model"` and panics if found.  This is a textual
   firewall, not a structural one.

2. **`parse_wgsl_bindings`** — a second text parser in `build.rs`
   (lines 1149-1190) that re-reads the generated `.wgsl` files to extract
   `@group(N) @binding(M) var<…> NAME` triples.  The binding metadata was
   *already present* in the codegen AST (`Module` → `GlobalVar` →
   `Attribute::Group / Binding`), but `KernelWgsl::new(module)` flattens it
   to a string, discarding it.

3. **Stringly-typed resource resolution** — at runtime, `ResourceRegistry::resolve(name: &str)`
   and `MeshResources::buffer_for_binding_name(name: &str)` match GPU buffers
   to shader variables by comparing raw strings.  If a WGSL variable name
   changes in codegen, the mismatch is only detected at GPU bind-group
   creation time (or worse, silently returns `None`).

The root cause is that binding metadata is generated structurally in the AST,
destroyed by string-flattening, then recovered by text parsing, and finally
matched at runtime by string equality.  The fix is to keep the structured
metadata alongside the WGSL source all the way to runtime.

## Non-Goals (out of scope)

- Removing the `enforce_codegen_ir_boundary` linter entirely.  The linter
  guards a *real* crate-boundary invariant (codegen must not import model
  types).  Removing it requires promoting `cfd2_codegen` to its own crate
  boundary with `pub`/`pub(crate)` enforcement — a larger refactor.  Instead,
  we keep the linter but *acknowledge* it as a temporary measure and add
  a TODO comment explaining what would replace it.

- Introducing a full typed descriptor / pipeline-layout system.  That would
  touch every pipeline creation site across the linear solver, AMG, Schur
  complement, etc.  Worth doing later, but too large for one changeset.

- Changing the `cfd2_codegen → cfd2_ir` dependency direction or creating a
  new `cfd2_core` crate.  That is a structural change best done after the
  binding metadata flows correctly.

## Plan

### Phase 1: Extract binding metadata from the AST (not from WGSL text)

**Goal:** `KernelWgsl` carries both the WGSL source string *and* a structured
binding list, eliminating `parse_wgsl_bindings` in `build.rs`.

#### 1a. Add `Module::bindings()` method

In `crates/cfd2_codegen/src/solver/codegen/wgsl_ast.rs`, add a public method:

```rust
impl Module {
    /// Extract all `(group, binding, var_name)` triples from GlobalVar items.
    pub fn bindings(&self) -> Vec<(u32, u32, String)> {
        let mut out = Vec::new();
        for item in &self.items {
            if let Item::GlobalVar(gv) = item {
                let mut group = None;
                let mut binding = None;
                for attr in &gv.attributes {
                    match attr {
                        Attribute::Group(g) => group = Some(*g),
                        Attribute::Binding(b) => binding = Some(*b),
                        _ => {}
                    }
                }
                if let (Some(g), Some(b)) = (group, binding) {
                    out.push((g, b, gv.name.clone()));
                }
            }
        }
        out.sort_by_key(|&(g, b, _)| (g, b));
        out
    }
}
```

~15 lines.  Purely additive, no existing API changes.

#### 1b. Extend `KernelWgsl` to carry binding metadata

In `crates/cfd2_codegen/src/solver/codegen/kernel_wgsl.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingDesc {
    pub group: u32,
    pub binding: u32,
    pub name: String,
}

pub struct KernelWgsl {
    source: String,
    bindings: Vec<BindingDesc>,
}

impl KernelWgsl {
    pub fn new(module: Module) -> Self {
        let bindings = module.bindings()
            .into_iter()
            .map(|(g, b, n)| BindingDesc { group: g, binding: b, name: n })
            .collect();
        Self {
            source: module.to_wgsl(),
            bindings,
        }
    }

    pub fn from_source(source: impl Into<String>) -> Self {
        // Legacy path: no structured metadata, parse from text
        let src = source.into();
        let bindings = parse_wgsl_bindings_from_text(&src);
        Self { source: src, bindings }
    }

    pub fn bindings(&self) -> &[BindingDesc] {
        &self.bindings
    }

    pub fn to_wgsl(&self) -> String { self.source.clone() }
}
```

The `from_source` constructor is kept for the few places that pass raw WGSL
strings.  It calls a small helper (moved from `build.rs`) that does the
text-based extraction as a fallback.

~40 lines changed in `kernel_wgsl.rs`.

#### 1c. Move `parse_wgsl_bindings` from `build.rs` into `cfd2_codegen`

Move the existing `parse_wgsl_bindings` + `parse_attr_u32` + `parse_var_name`
helpers from `build.rs` into `cfd2_codegen::solver::codegen::kernel_wgsl` as
`pub(crate) fn parse_wgsl_bindings_from_text(…)`.  This lets
`KernelWgsl::from_source` use it, and `build.rs` can stop re-implementing
the parser.

~60 lines moved (net zero).

#### 1d. Update `build.rs` to use `KernelWgsl::bindings()` instead of text parsing

In `generate_kernel_registry_map`, instead of:
```rust
let src = fs::read_to_string(&path)?;
let bindings = parse_wgsl_bindings(&src);
```

Do:
```rust
// For model/shared kernels generated via Module → KernelWgsl,
// we already have the WGSL on disk.  Re-read it into a KernelWgsl::from_source
// which extracts bindings structurally.
let src = fs::read_to_string(&path)?;
let kw = KernelWgsl::from_source(src);
let bindings: Vec<(u32, u32, String)> = kw.bindings()
    .iter()
    .map(|b| (b.group, b.binding, b.name.clone()))
    .collect();
```

Better yet, for kernels that are produced by `kernel_generators()` (the
build-time codegen path), pass the `KernelWgsl` directly instead of
writing to disk and re-reading.  That eliminates the round-trip for
model/shared kernels entirely.

This is the largest sub-step.  The `generate_kernel_registry_map` function
currently reads `.wgsl` files from disk.  We can change it to accept
`(KernelId, KernelWgsl)` pairs produced earlier in `main()` and extract
bindings via `kw.bindings()`.  The infrastructure kernels path already
calls `generator()` which returns `KernelWgsl`, so we just keep the
result instead of discarding it after writing.

~30 lines changed in `build.rs`.

### Phase 2: Add a compile-time cross-check for binding names

**Goal:** Detect binding name mismatches at build time, not at GPU runtime.

#### 2a. Add a `#[cfg(test)]` exhaustive binding-name test

Create a new test in `crates/cfd2_codegen/src/solver/codegen/` (or `tests/`)
that:

1. Iterates every `infrastructure_kernels::all_infrastructure_kernels()` generator
2. Calls `generator()` to get a `KernelWgsl`
3. Asserts `kw.bindings()` is non-empty
4. Checks every binding name against a known allowlist

This ensures that if a binding name changes in codegen, the test fails
immediately with a clear message like:

```
binding 'state_current' not in allowlist; did you mean 'state'?
```

~40 lines.  The known-good list can be derived from the existing
`buffer_for_binding_name` match arms.

#### 2b. Annotate `enforce_codegen_ir_boundary` with an explanatory comment

Add a doc-comment explaining that the grep-based linter is a stopgap:

```rust
/// Stopgap boundary enforcement: scans `cfd2_codegen/` source files for
/// accidental imports of `crate::solver::model`.  This will become
/// unnecessary once `cfd2_codegen` is extracted into a workspace crate with
/// its own `pub` visibility boundary (see ARCHITECTURE_REVIEW.md §2).
fn enforce_codegen_ir_boundary(manifest_dir: &str) { … }
```

No functional change.  ~5 lines.

### Phase 3: Improve `ResourceRegistry` error reporting

**Goal:** When a binding name is missing at runtime, produce a clear error
instead of returning `None` and silently failing.

#### 3a. Add `resolve_or_panic` method

```rust
impl ResourceRegistry<'_> {
    pub fn resolve_or_err(&self, name: &str, kernel: &str) -> Result<wgpu::BindingResource<'_>, String> {
        self.resolve(name).ok_or_else(|| {
            format!(
                "ResourceRegistry: no buffer for binding '{name}' required by kernel '{kernel}'. \
                 Available: {:?}",
                self.available_names()
            )
        })
    }

    fn available_names(&self) -> Vec<&str> { … }
}
```

Then update the call-sites in `create_bind_group_from_bindings` (already has
`ok_or_else` but doesn't list available names).

~25 lines.

## Files Affected

| File | Change |
|------|--------|
| `crates/cfd2_codegen/src/solver/codegen/wgsl_ast.rs` | Add `Module::bindings()` |
| `crates/cfd2_codegen/src/solver/codegen/kernel_wgsl.rs` | Add `BindingDesc`, carry bindings |
| `build.rs` | Remove `parse_wgsl_bindings` (moved), use `KernelWgsl::bindings()` |
| `src/solver/gpu/wgsl_reflect.rs` | Possible simplification |
| `src/solver/gpu/modules/resource_registry.rs` | Add `available_names` for diagnostics |
| `tests/` or inline `#[cfg(test)]` | Binding-name cross-check test |

## Estimated Scope

- ~15 lines added to `wgsl_ast.rs`
- ~40 lines changed in `kernel_wgsl.rs`
- ~60 lines moved from `build.rs` to `kernel_wgsl.rs` (net zero)
- ~30 lines changed in `build.rs`
- ~40 lines for the cross-check test
- ~25 lines for error reporting improvement
- **Total: ~150 new/changed lines, ~60 lines moved**

## Verification

1. `cargo test --workspace` — all existing tests pass
2. WGSL snapshot tests (`flux_module_wgsl_matches_committed_*`) — byte-for-byte
3. OpenFOAM reference metrics — no regression
4. New binding-name test passes
5. `enforce_codegen_ir_boundary` still runs and passes

## Risk Assessment

**Low risk.** Phase 1 is purely additive: the existing text-parsing path
remains as a fallback in `from_source`.  The only behavioral change is that
the binding metadata now *also* lives in `KernelWgsl` as structured data.
No generated WGSL changes.  No runtime behavior changes.
