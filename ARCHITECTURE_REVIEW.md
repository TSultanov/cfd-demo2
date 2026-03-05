### 1. The "Façade" of Type Safety (Ports & Dimensions)

The codebase implements a complex type-level physical dimension system in cfd2_ir::dimensions
(using const generics for rational exponents like Dim<M_NUM, M_DEN...>) to enforce unit correctness
at compile time. However, this system clashes heavily with the runtime architecture.

- Type Erasure and AnyDimension Escape Hatch: At the boundary between the model definition and GPU
execution, the types are erased into stringly-typed metadata (ResolvedStateSlotSpec). Because the
strict typing becomes too painful, the codebase introduces AnyDimension
(crates/cfd2_ir/src/ports/dimensions.rs), which explicitly bypasses the checks.
- Runtime Panics for Compile-Time Concepts: The #[derive(PortSet)] macro generates code that
registers ports at runtime. If a dimension or type mismatches, it results in a PortRegistryError
that bubbles up as a runtime panic during solver initialization.
- Redundant Expression Trees: There are two distinct expression trees. PrimitiveExpr
(cfd2_ir/src/flux.rs) is used for thermodynamics (EOS), while Expr (cfd2_ir/src/ast/expr.rs) is
used for the codegen AST. They do exactly the same thing but require separate lowering passes
(lower_primitive_expr_dyn).

Recommendation: If the graph and port layout must be resolved at runtime (which is standard for GPU
compute graphs), abandon the const-generic type-level dimension system. Store units as runtime
enums/structs (UnitDim) and validate the graph once during the UnifiedSolver::new phase.

### 2. Broken Boundaries: The build.rs Linter

A clear sign of a broken architectural boundary is when you have to write a custom text-parser to
enforce it.

- The Grep-based Boundary: In build.rs (lines ~136-168), there is a function called
enforce_codegen_ir_boundary. It opens every .rs file in cfd2_codegen, reads it line-by-line, and
panics if it finds strings like "crate::solver::model".
- Why this happens: cfd2_codegen is supposed to be agnostic to the physics models, but because the
AST, Codegen, and Models are so tightly intertwined, developers kept accidentally importing model
definitions into the codegen.
- Stringly-Typed WGSL Bindings: In src/solver/gpu/modules/resource_registry.rs, resources are bound
to the shader using hardcoded strings ("matrix_values", "diag_u", "constants"). If a WGSL variable
name is changed in the cfd2_codegen string formatting, the Rust host code will silently fail to
bind the resource at runtime.

Recommendation: Create a strict cfd2_core crate that only contains the AST and WGSL builder. Move
the physical models entirely into the host code. Use a typed descriptor struct to map GPU buffers
to bind groups rather than matching on magic strings.

### 3. Leaky Linear Solver Abstractions

The GPU linear solver module (src/solver/gpu/modules/generic_linear_solver.rs) claims to be a
"unified interface for linear solvers that can be used across all solver families." However, it is
heavily coupled to specific CFD physics.

- Physics Bleeding into Math: In src/solver/gpu/linear_solver/fgmres.rs, the FgmresPrecondBindings
enum explicitly requires SchurWithParams, knowing exactly about diag_u and diag_p. A generic FGMRES
solver should only know about a generic M^{-1}x preconditioner trait/closure, it should not know
about "velocity" (diag_u) and "pressure" (diag_p).
- Hardcoded Workgroup Sizes: Throughout the linear solver and WGSL generation (e.g., fgmres.rs,
scalar_cg.rs), workgroup sizes are hardcoded to 64 (pub const WORKGROUP_SIZE: u32 = 64;). If
deployed on hardware where 64 is suboptimal (or invalid for certain Metal/WebGPU limits), this will
cause opaque crashes.

### 4. Dangerous "Aggressive" Kernel Fusion

The codebase includes a highly sophisticated AST-based kernel fusion system
(cfd2_codegen/src/solver/codegen/fusion.rs) to combine multiple WGSL dispatches into one.

- Ignoring Data Hazards: The KernelFusionPolicy::Aggressive explicitly skips the
ensure_safe_composition check, which detects Read-After-Write (RAW), WAR, and WAW hazards. If
kernels with data hazards are fused into a single dispatch without appropriate workgroupBarrier()
or storageBarrier() calls, the GPU will experience data races. This will lead to non-deterministic
divergence across different GPU vendors.
- Brittle AST Manipulation: The function apply_ast_load_after_store_forwarding attempts to perform
Common Subexpression Elimination (CSE) and register forwarding by manually traversing the custom
AST looking for ExprNode::Index and ExprNode::Ident. This is a classic "regex parsing HTML"
problem; it is incredibly fragile and assumes a specific formatting of the generated WGSL.

### 5. Computational Geometry Anti-Patterns in Mesh Generation

The mesh generation code contains several dangerous anti-patterns that will fail at differing
spatial scales.

- Integer Quantization for Vertex Deduplication: In src/meshgen/cut_cell.rs, to deduplicate
vertices, the code does this:
let quantize = |v: f64| (v * 100000.0).round() as i64;
This assumes all geometry features are cleanly resolvable at a $10^{-5}$ scale. If the user
simulates a micro-fluidic channel ($10^{-6}$ m), all vertices will collapse into a single point.
- Epsilon Hardcoding: Throughout delaunay.rs and cut_cell.rs, hardcoded epsilons are used (e.g.,
1e-6, 1e-10, 1e-12) for point-in-polygon tests and circumcircle calculations. These should be
scaled relative to the bounding box of the local cell (min_cell_size).
- SoA Mesh Structure: The Mesh struct is a massive Struct of Arrays (SoA) containing over 20 flat
Vecs (vx, vy, face_owner, face_nx, etc.). While good for GPU upload, it makes the CPU-side mesh
generation extremely difficult to mutate. For example, fix_concave_cells in voronoi.rs has to
manually reconstruct the topology by keeping 10 different arrays perfectly synchronized.

### 6. Defensive Programming vs. Panicking

There is a severe inconsistency in how errors are handled.

- Setup Path Panics: In src/solver/gpu/unified_solver.rs and the init modules, almost every wgpu
setup step utilizes .unwrap() or .expect(). While acceptable in a CLI app, this codebase has a ui
feature (egui). If the user selects an incompatible mesh/solver combo, the entire UI application
will instantly crash to desktop.
- Silent Failures in Abstractions: In src/solver/gpu/modules/state.rs, the ping_pong_indices just
wraps around silently. In flux_module_wgsl.rs, missing fields fallback to returning 0.0 or
substituting the owner's index silently instead of failing clearly.

### 7. Architectural Dead Code & Over-engineering

- One-Submission Outer Loop: The code implements a massive workaround
(CFD2_ENABLE_GPU_OUTER_LOOP_PRIMITIVE, submit_solve_fgmres_fixed_iterations_chunked) to pack
thousands of iterations into a single wgpu CommandEncoder to avoid CPU/GPU syncs. However, doing
this blindly forces the GPU to execute iterations even if the linear solver converged early. It
includes complex "STOP-inject WGSL" to allow the GPU to skip work, but this requires compiling
separate WGSL shaders just to read a status buffer and write to an indirect dispatch buffer
(build_outer_gate_wgsl).

### Summary Recommendations

1. Remove the Type-Level Dimension System: The codebase spends thousands of lines managing
const-generic UnitDimension types, only to erase them at the GPU boundary. Standardize on f32/f64
and validate physical units via a runtime JSON/Struct schema upon initialization.
2. Invert the Codegen Dependency: cfd2_codegen should not know about specific CFD models. It should
accept a pure DAG of mathematical operations. The Models (compressible, incompressible) should live
in the main crate and build the DAG. This will eliminate the need for build.rs to police
boundaries.
3. Fix Mesh Quantization: Replace (v * 100000.0) as i64 with an R-Tree, Quad-Tree, or Spatial Hash
map using an epsilon relative to min_cell_size.
4. Remove "Aggressive" Fusion: Remove kernel fusions that intentionally bypass hazard checks. Rely
on standard Safe fusion and optimize the mathematical algorithms (like the preconditioners) rather
than risking GPU data races.
