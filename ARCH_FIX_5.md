# ARCH_FIX_5: Fix Computational Geometry Anti-Patterns in Mesh Generation

## Status: All Phases Implemented

### Implementation Notes

**Phase 1** (tolerances struct + threading): ✅ Done
- `src/meshgen/tolerances.rs` created with `MeshgenTolerances` struct (15 fields)
- `src/meshgen/cut_cell.rs` — replaced quantize closure + all 14 hardcoded epsilons
- `src/meshgen/delaunay.rs` — replaced quantize closure + all epsilons; threaded `&tol`
- `src/meshgen/voronoi.rs` — replaced all 7 epsilons; threaded `&tol`
- `src/meshgen/meshgen_utils.rs` — added `&MeshgenTolerances` param to both functions
- `src/meshgen/meshgen_ext.rs` — added `smooth_with_tolerances()`, auto-derives tolerances from mesh bounds
- `src/meshgen/mod.rs` — added `pub(crate) mod tolerances` and re-export

**Phase 2** (MeshBuilder): ✅ Done
- `src/meshgen/mesh_builder.rs` created with `MeshBuilder`, `VertexId`, `CellId`, `FaceId`
- `src/meshgen/delaunay.rs` — `generate_delaunay_mesh` rewritten to use `MeshBuilder`
- `src/meshgen/cut_cell.rs` — finalization section rewritten to use `MeshBuilder`
- `src/meshgen/voronoi.rs` — `fix_concave_cells` rewritten to use `MeshBuilder`
  (reduced from ~230 lines of manual array synchronization to ~160 lines)
- 4 unit tests for MeshBuilder (single quad, two triangles, set_face_neighbor, normals)

**Phase 3** (scale-invariance tests): ✅ Done — 6 new tests in `tests.rs`
- `quantize_does_not_collapse_distinct_vertices_at_small_scale`
- `quantize_merges_nearby_vertices_at_large_scale`
- `quantize_adapts_to_microfluidic_scale`
- `classify_boundary_scales_with_domain`
- `cut_cell_mesh_scale_invariance_topology`
- `tolerances_are_self_consistent`

**Phase 4** (boundary detection helper): ✅ Done — `classify_boundary()` method on `MeshgenTolerances`, called from all 5 sites.

**Verification:**
- All 175 lib tests pass (including 11 meshgen + 4 MeshBuilder tests)
- OpenFOAM reference metrics: identical before/after (zero diff)
- No new clippy warnings in meshgen

---

## Problem Statement

Architecture Review §5 ("Computational Geometry Anti-Patterns in Mesh
Generation") identifies three issues in the `src/meshgen/` module:

1. **Integer Quantization for Vertex Deduplication** — Both `cut_cell.rs`
   (line 26) and `delaunay.rs` (line 139) use a hardcoded quantization
   function to deduplicate vertices:

   ```rust
   let quantize = |v: f64| (v * 100000.0).round() as i64;
   ```

   This maps continuous coordinates to a fixed grid with spacing 10⁻⁵.
   If the user simulates geometry at a different spatial scale — e.g. a
   micro-fluidic channel at 10⁻⁶ m where `min_cell_size` is ~10⁻⁷ — all
   vertices collapse to the same integer key, producing a degenerate mesh.
   Conversely, for very large domains (e.g. 10³ m), vertices that should
   be merged may receive distinct keys, creating duplicate vertices and
   non-manifold topology.

2. **Hardcoded Epsilons** — Throughout the meshgen module, ~38 hardcoded
   epsilon values are used for geometric predicates (point-in-circumcircle,
   collinearity, intersection, boundary detection, edge degeneracy). A
   complete inventory:

   | File | Epsilon | Purpose |
   |------|---------|---------|
   | `cut_cell.rs:126` | `1e-12` | Line intersection denominator check |
   | `cut_cell.rs:262` | `1e-12` | Degenerate segment check |
   | `cut_cell.rs:276` | `1e-10` | SIMD intersection epsilon |
   | `cut_cell.rs:277-278` | `1e-6` | Parametric t clamping (near endpoints) |
   | `cut_cell.rs:319,344` | `epsilon` (1e-10) | SIMD endpoint/closeness |
   | `cut_cell.rs:364` | `1e-10` | Scalar endpoint distance |
   | `cut_cell.rs:371,373` | `1e-6`, `1e-10` | Scalar t range / projection |
   | `cut_cell.rs:422` | `1e-9` | Cell area degeneracy |
   | `cut_cell.rs:442` | `1e-9` | Edge length degeneracy |
   | `cut_cell.rs:457,459` | `1e-6` | Boundary detection (face position) |
   | `cut_cell.rs:78` | `1e-9` | SDF tolerance for intersection |
   | `delaunay.rs:71` | `1e-12` | Circumcircle denominator |
   | `delaunay.rs:117` | `1e-10` | In-circumcircle determinant |
   | `delaunay.rs:395,411` | `1e-6` | Smoothing weight denominator clamp |
   | `delaunay.rs:547,553,559` | `1e-10` | Triangle containment cross product |
   | `delaunay.rs:801,803` | `1e-6` | Boundary detection (face position) |
   | `voronoi.rs:174,176,199,201` | `1e-6` | Boundary detection (face position) |
   | `voronoi.rs:431` | `1e-6` | Generator-vertex coincidence |
   | `voronoi.rs:747,773` | `1e-12` | Convexity cross product |
   | `meshgen_utils.rs:5` | `1e-6` | SDF gradient finite difference |
   | `meshgen_utils.rs:18` | `1e-6` | Line intersection det threshold |
   | `meshgen_ext.rs:40,102` | `1e-6` | Box boundary detection, SDF gradient |
   | `meshgen_ext.rs:120` | `1e-8` | Edge collapse detection (squared) |
   | `meshgen_ext.rs:159` | `1e-12` | Normal degeneracy |

   These fall into several categories with different scaling requirements:
   - **Degeneracy guards** (denominators near zero): scale-invariant, should
     use machine epsilon (e.g. `f64::EPSILON * scale_factor`).
   - **Geometric proximity** (edge collapse, vertex merge, endpoint detection):
     must scale with `min_cell_size`.
   - **Boundary detection** (face at domain edge): must scale with
     `domain_size`.
   - **Finite differences** (SDF gradient): must scale with local feature size.

3. **SoA Mesh Structure** — The `Mesh` struct (`src/solver/mesh/structs.rs`)
   is a Struct of Arrays with 16 parallel `Vec`s. While this is efficient for
   GPU upload, it makes CPU-side topological mutations (e.g. `fix_concave_cells`
   in `voronoi.rs`) extremely error-prone: the function manually synchronizes
   ~10 arrays across cell splits, face creation, and connectivity updates
   (~380 lines of index arithmetic).

### Root cause

The meshgen module was written for a single fixed-scale geometry
(channel with obstacle, domain size ~O(1) m, min_cell_size ~O(10⁻²) m)
and the magic constants happened to work. No tolerance scaling infrastructure
exists.

## Non-Goals (out of scope)

- **Replacing the SoA `Mesh` struct with an AoS or half-edge data structure.**
  The `Mesh` struct is used throughout the solver, GPU upload, and I/O paths.
  Replacing it would touch hundreds of files and all GPU buffer layouts. The
  SoA layout is correct for the GPU pipeline. Instead, we introduce an
  intermediate CPU-side mesh builder that encapsulates the index bookkeeping,
  then flattens to `Mesh` at the end.

- **Implementing exact arithmetic predicates** (e.g. Shewchuk's robust
  predicates). While ideal for Delaunay triangulation, the current
  float-based predicates with properly scaled epsilons are sufficient for
  the mesh quality requirements of this CFD solver. This can be revisited
  later if numerical robustness issues persist.

- **Changing the quadtree/Poisson-disk point generation algorithm.** The
  point generation is not the source of the anti-patterns. Only the
  deduplication and geometric predicate tolerances need fixing.

- **Rewriting the boundary detection logic.** The boundary detection pattern
  (`face_center.x < eps → Inlet`) is specific to the rectangular domain
  assumption. A general boundary detection system is a separate feature.
  We only fix the hardcoded `1e-6` to scale with `domain_size`.

## Plan

### Phase 1: Introduce `MeshgenTolerances` and scale-aware quantization ✅

**Goal:** Replace all hardcoded epsilons and the `(v * 100000.0)` quantizer
with a single tolerance struct computed from `min_cell_size` and `domain_size`.

#### 1a. Define `MeshgenTolerances` struct ✅

```rust
// In src/meshgen/tolerances.rs (new file)

/// Tolerances for mesh generation, scaled to the problem geometry.
///
/// All tolerances are derived from two inputs:
/// - `min_cell_size`: the smallest cell dimension in the mesh
/// - `domain_size`: the bounding box of the computational domain
#[derive(Debug, Clone)]
pub struct MeshgenTolerances {
    /// Vertex deduplication: two vertices closer than this are merged.
    /// Set to `min_cell_size * 1e-6` (relative precision within a cell).
    pub vertex_merge: f64,

    /// Quantization grid spacing for vertex hash keys.
    /// Set to `vertex_merge` so that vertices within merge distance
    /// can differ by at most 1 in each quantized coordinate.
    pub quantize_inv: f64,

    /// Degeneracy guard for determinants and denominators.
    /// Scale-independent: `min_cell_size^2 * 1e-12`.
    pub determinant_eps: f64,

    /// Parametric tolerance for line-segment intersection.
    /// A point within `t_eps` of an endpoint (t=0 or t=1) is treated
    /// as coincident with the endpoint. Set to `vertex_merge / min_cell_size`
    /// (typically ~1e-6).
    pub t_eps: f64,

    /// Geometric proximity tolerance for "is this point on the boundary?"
    /// Set to `domain_bbox_diag * 1e-6`.
    pub boundary_eps: f64,

    /// Finite difference step for SDF gradient computation.
    /// Set to `min_cell_size * 1e-4`.
    pub sdf_grad_eps: f64,

    /// Edge collapse guard for smoothing: edges shorter than this squared
    /// distance trigger move rejection.
    /// Set to `(min_cell_size * 1e-3)^2`.
    pub edge_collapse_sq: f64,

    /// Circumcircle / in-circle predicate tolerance.
    /// Set to `min_cell_size^2 * 1e-10`.
    pub circumcircle_eps: f64,

    /// Cross product tolerance for convexity / orientation checks.
    /// Set to `min_cell_size^2 * 1e-10`.
    pub cross_eps: f64,

    /// Cell area degeneracy threshold.
    /// Set to `min_cell_size^2 * 1e-6`.
    pub area_eps: f64,

    /// Edge length degeneracy threshold.
    /// Set to `min_cell_size * 1e-6`.
    pub edge_len_eps: f64,
}

impl MeshgenTolerances {
    pub fn from_geometry(min_cell_size: f64, domain_size: nalgebra::Vector2<f64>) -> Self {
        let diag = (domain_size.x * domain_size.x + domain_size.y * domain_size.y).sqrt();
        let mcs = min_cell_size;
        Self {
            vertex_merge: mcs * 1e-6,
            quantize_inv: 1.0 / (mcs * 1e-6),
            determinant_eps: mcs * mcs * 1e-12,
            t_eps: 1e-6,  // dimensionless parametric tolerance
            boundary_eps: diag * 1e-6,
            sdf_grad_eps: mcs * 1e-4,
            edge_collapse_sq: (mcs * 1e-3) * (mcs * 1e-3),
            circumcircle_eps: mcs * mcs * 1e-10,
            cross_eps: mcs * mcs * 1e-10,
            area_eps: mcs * mcs * 1e-6,
            edge_len_eps: mcs * 1e-6,
        }
    }

    /// Quantize a coordinate to an integer grid key for vertex deduplication.
    #[inline]
    pub fn quantize(&self, v: f64) -> i64 {
        (v * self.quantize_inv).round() as i64
    }

    /// Quantize a 2D point to a grid key.
    #[inline]
    pub fn quantize_point(&self, x: f64, y: f64) -> (i64, i64) {
        (self.quantize(x), self.quantize(y))
    }
}
```

~90 lines in new file.

#### 1b. Thread `MeshgenTolerances` through all mesh generators ✅

Each top-level mesh generation function (`generate_cut_cell_mesh`,
`generate_voronoi_mesh`, `triangulate`) already receives `min_cell_size`
and `domain_size`. At the top of each function, construct the tolerances:

```rust
let tol = MeshgenTolerances::from_geometry(min_cell_size, domain_size);
```

Then pass `&tol` to all internal helper functions and closures.

**Changes per file:**

- `cut_cell.rs`: Replace `let quantize = |v: f64| (v * 100000.0).round() as i64`
  with `tol.quantize(v)`. Replace all 14 hardcoded epsilons with the
  appropriate `tol.*` field. ~20 lines changed.

- `delaunay.rs`: Replace `quantize` closure (line 139) with `tol.quantize_point`.
  Replace epsilons in `in_circumcircle` (line 117), `calculate_circumcircle`
  (line 71), `smooth_generators` (lines 395, 411), triangle containment
  (lines 547–559), boundary detection (lines 801–803). ~15 lines changed.
  Pass `&tol` to `generate_poisson_points`, `smooth_generators`,
  `compute_triangulation`. ~5 signature changes.

- `voronoi.rs`: Replace boundary detection epsilons (lines 174–201),
  generator-vertex coincidence (line 431), convexity checks (lines 747, 773).
  Pass `&tol` to `fix_concave_cells`. ~10 lines changed.

- `meshgen_utils.rs`: Replace `1e-6` in `compute_normal` and
  `intersect_lines` with `tol.sdf_grad_eps` and `tol.determinant_eps`.
  Add `&tol` parameter. ~5 lines changed.

- `meshgen_ext.rs`: Replace `1e-6` in `smooth` and `calculate_max_skewness`
  with `tol.boundary_eps`, `tol.sdf_grad_eps`, `tol.edge_collapse_sq`.
  Note: `smooth` and `calculate_max_skewness` are methods on `Mesh` — they
  don't have access to `min_cell_size`. Add `min_cell_size` and `domain_size`
  parameters (or accept `&MeshgenTolerances` directly). ~10 lines changed.

#### 1c. Update `Triangle::in_circumcircle` and `Triangle::calculate_circumcircle` ✅

These are methods on `Triangle` in `delaunay.rs`. They currently use
hardcoded `1e-12` and `1e-10`. Two options:

- **Option A:** Add `eps` parameter to both methods.
- **Option B:** Make them free functions that accept `&MeshgenTolerances`.

Prefer **Option A** for minimal disruption:

```rust
pub fn in_circumcircle(&self, p: Point2<f64>, points: &[Point2<f64>], eps: f64) -> bool {
    // ... existing code ...
    det > eps
}
```

~4 lines changed across 2 methods; ~10 call sites updated.

### Phase 2: Introduce `MeshBuilder` for safe topological mutations ✅

**Goal:** Encapsulate the error-prone parallel-array bookkeeping in a
builder struct that enforces consistency, then flatten to `Mesh` at the end.

#### 2a. Define `MeshBuilder` struct

```rust
// In src/meshgen/mesh_builder.rs (new file)

use crate::solver::mesh::{BoundaryType, Mesh};

/// A vertex handle returned by MeshBuilder::add_vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexId(pub usize);

/// A cell handle returned by MeshBuilder::add_cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId(pub usize);

/// A face handle returned by MeshBuilder::add_face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FaceId(pub usize);

pub struct MeshBuilder {
    vx: Vec<f64>,
    vy: Vec<f64>,
    v_fixed: Vec<bool>,

    cells: Vec<CellData>,
    faces: Vec<FaceData>,
}

struct CellData {
    vertices: Vec<VertexId>,
    faces: Vec<FaceId>,
}

struct FaceData {
    v1: VertexId,
    v2: VertexId,
    owner: CellId,
    neighbor: Option<CellId>,
    boundary: Option<BoundaryType>,
}

impl MeshBuilder {
    pub fn new() -> Self { /* ... */ }

    pub fn add_vertex(&mut self, x: f64, y: f64, fixed: bool) -> VertexId {
        let id = VertexId(self.vx.len());
        self.vx.push(x);
        self.vy.push(y);
        self.v_fixed.push(fixed);
        id
    }

    pub fn add_cell(&mut self, vertices: Vec<VertexId>) -> CellId {
        let id = CellId(self.cells.len());
        self.cells.push(CellData { vertices, faces: Vec::new() });
        id
    }

    pub fn add_face(
        &mut self,
        v1: VertexId,
        v2: VertexId,
        owner: CellId,
        neighbor: Option<CellId>,
        boundary: Option<BoundaryType>,
    ) -> FaceId {
        let id = FaceId(self.faces.len());
        self.faces.push(FaceData { v1, v2, owner, neighbor, boundary });
        self.cells[owner.0].faces.push(id);
        if let Some(n) = neighbor {
            self.cells[n.0].faces.push(id);
        }
        id
    }

    pub fn set_face_neighbor(&mut self, face: FaceId, neighbor: CellId) {
        let f = &mut self.faces[face.0];
        f.neighbor = Some(neighbor);
        f.boundary = None;
        self.cells[neighbor.0].faces.push(face);
    }

    pub fn vertex_pos(&self, v: VertexId) -> (f64, f64) {
        (self.vx[v.0], self.vy[v.0])
    }

    pub fn num_cells(&self) -> usize { self.cells.len() }
    pub fn num_vertices(&self) -> usize { self.vx.len() }

    /// Flatten into the SoA Mesh struct, computing all geometry.
    pub fn build(self) -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vx = self.vx;
        mesh.vy = self.vy;
        mesh.v_fixed = self.v_fixed;

        // Flatten faces
        for f in &self.faces {
            mesh.face_v1.push(f.v1.0);
            mesh.face_v2.push(f.v2.0);
            mesh.face_owner.push(f.owner.0);
            mesh.face_neighbor.push(f.neighbor.map(|c| c.0));
            mesh.face_boundary.push(f.boundary);
            // Placeholders — recalculate_geometry fills these
            mesh.face_cx.push(0.0);
            mesh.face_cy.push(0.0);
            mesh.face_nx.push(0.0);
            mesh.face_ny.push(0.0);
            mesh.face_area.push(0.0);
        }

        // Flatten cells
        mesh.cell_face_offsets.push(0);
        mesh.cell_vertex_offsets.push(0);
        for cell in &self.cells {
            mesh.cell_cx.push(0.0);
            mesh.cell_cy.push(0.0);
            mesh.cell_vol.push(0.0);
            for &fid in &cell.faces {
                mesh.cell_faces.push(fid.0);
            }
            mesh.cell_face_offsets.push(mesh.cell_faces.len());
            for &vid in &cell.vertices {
                mesh.cell_vertices.push(vid.0);
            }
            mesh.cell_vertex_offsets.push(mesh.cell_vertices.len());
        }

        // Compute face normals with correct orientation, then cell geometry
        mesh.recalculate_geometry();
        // Fix normal orientation (owner → neighbor)
        for i in 0..mesh.num_faces() {
            let owner = mesh.face_owner[i];
            let c_owner = nalgebra::Point2::new(mesh.cell_cx[owner], mesh.cell_cy[owner]);
            let f_center = nalgebra::Point2::new(mesh.face_cx[i], mesh.face_cy[i]);
            let normal = nalgebra::Vector2::new(mesh.face_nx[i], mesh.face_ny[i]);
            if (f_center - c_owner).dot(&normal) < 0.0 {
                mesh.face_nx[i] = -normal.x;
                mesh.face_ny[i] = -normal.y;
            }
        }
        mesh.recalculate_geometry();

        mesh
    }
}
```

~130 lines in new file.

#### 2b. Rewrite `fix_concave_cells` using `MeshBuilder`

The current `fix_concave_cells` in `voronoi.rs` is ~380 lines of manual
array synchronization. Rewrite it to:

1. Build a `MeshBuilder` from the input `Mesh` (copy vertices, iterate
   cells and faces to populate the builder).
2. For each concave cell, use `MeshBuilder::add_cell` and `add_face` to
   create sub-cells, instead of manually pushing to 10+ parallel arrays.
3. Call `builder.build()` to produce the final `Mesh`.

This will reduce the function to ~150 lines and eliminate the entire class
of "forgot to push to one of the arrays" bugs.

~230 lines removed, ~150 lines added (net -80 lines).

#### 2c. Migrate `cut_cell.rs` cell construction to `MeshBuilder`

The bottom half of `generate_cut_cell_mesh` (lines ~400–510) manually
pushes to `mesh.face_v1`, `mesh.face_cx`, etc. Replace with `MeshBuilder`
calls.

~40 lines changed.

#### 2d. Migrate `delaunay.rs` `build_delaunay_mesh` to `MeshBuilder`

The mesh construction at the bottom of `delaunay.rs` (~lines 750–850)
manually builds the `Mesh`. Replace with `MeshBuilder`.

~40 lines changed.

### Phase 3: Add scale-invariance tests ✅

**Goal:** Verify that mesh generation produces topologically identical
results at different spatial scales.

#### 3a. Scale-invariance test for cut-cell mesh ✅

```rust
#[test]
fn cut_cell_mesh_scale_invariance_topology() {
    // Generates rectangular channel meshes at 1× and 10⁻³× scale
    // Verifies same cell/face count and volume ratio = scale²
}
```

#### 3b. Quantization and tolerance tests ✅

```rust
#[test]
fn quantize_does_not_collapse_distinct_vertices_at_small_scale()
fn quantize_merges_nearby_vertices_at_large_scale()
fn quantize_adapts_to_microfluidic_scale()
fn classify_boundary_scales_with_domain()
fn tolerances_are_self_consistent()
```

### Phase 4: Migrate boundary detection to use `boundary_eps` ✅

**Goal:** Fix the boundary detection pattern that uses hardcoded `1e-6`
across all three mesh generators.

#### 4a. Extract boundary detection helper ✅

```rust
// In src/meshgen/tolerances.rs

impl MeshgenTolerances {
    /// Determine boundary type from face center position.
    /// Assumes rectangular domain [0, domain_x] × [0, domain_y].
    pub fn classify_boundary(
        &self,
        face_center_x: f64,
        face_center_y: f64,
        domain_x: f64,
        domain_y: f64,
    ) -> Option<BoundaryType> {
        if face_center_x < self.boundary_eps {
            Some(BoundaryType::Inlet)
        } else if (face_center_x - domain_x).abs() < self.boundary_eps {
            Some(BoundaryType::Outlet)
        } else if face_center_y < self.boundary_eps {
            Some(BoundaryType::Wall)
        } else if (face_center_y - domain_y).abs() < self.boundary_eps {
            Some(BoundaryType::Wall)
        } else {
            None
        }
    }
}
```

#### 4b. Replace all boundary detection call sites ✅

Replaced inline `if face_center.x < 1e-6 { Inlet } else if ...`
pattern in all 5 sites:
- `cut_cell.rs` (1 site)
- `delaunay.rs` (1 site)
- `voronoi.rs` (2 sites)
- `meshgen_ext.rs` (1 site — uses `is_on_box` closure with `tol.boundary_eps`)

## Files Affected

| File | Change | Status |
|------|--------|--------|
| `src/meshgen/tolerances.rs` | **New file**: `MeshgenTolerances` struct + `classify_boundary` | ✅ |
| `src/meshgen/mod.rs` | Add `pub(crate) mod tolerances;` + re-export | ✅ |
| `src/meshgen/cut_cell.rs` | Replace quantize closure + 14 hardcoded epsilons; use `tol.classify_boundary` | ✅ |
| `src/meshgen/delaunay.rs` | Replace quantize closure + 10 hardcoded epsilons; thread `&tol` through helpers | ✅ |
| `src/meshgen/voronoi.rs` | Replace 7 hardcoded epsilons; use `tol.classify_boundary` | ✅ |
| `src/meshgen/meshgen_utils.rs` | Add `&MeshgenTolerances` param to `compute_normal`, `intersect_lines` | ✅ |
| `src/meshgen/meshgen_ext.rs` | Add `smooth_with_tolerances`; auto-derive tolerances from mesh bounds | ✅ |
| `src/meshgen/tests.rs` | Add 6 scale-invariance and quantization tests | ✅ |
| `src/meshgen/mesh_builder.rs` | **New file**: `MeshBuilder`, `VertexId`, `CellId`, `FaceId` | ✅ |

## Estimated Scope

- Phase 1: ~160 lines (new struct + threading through all files) ✅
- Phase 2: ~270 lines (new MeshBuilder + rewrites of 3 construction sites) ✅
- Phase 3: ~80 lines (tests) ✅
- Phase 4: ~35 lines (boundary helper + call site updates) ✅
- **Total: ~545 lines touched** (~220 new, ~325 modified) — all complete

## Verification

1. ✅ `cargo test --features meshgen --lib` — all 175 tests pass (including 11 meshgen + 4 MeshBuilder).
2. ✅ New scale-invariance tests pass at scales 1.0 and 1e-3.
3. ✅ New quantization edge-case tests pass at microfluidic (1e-6) and large (10.0) scales.
4. ✅ OpenFOAM reference metrics — zero diff before/after.
5. ✅ No new clippy warnings in meshgen.

## Risk Assessment

**Medium risk** — this touches the foundational mesh generation code and
any bug could produce invalid meshes that cause solver divergence.

Mitigations:
- **Phase 1 is value-preserving at default scale.** The tolerance values are
  chosen so that `MeshgenTolerances::from_geometry(0.01, Vector2::new(2.0, 1.0))`
  produces epsilons numerically close to the current hardcoded values. At
  default scale, the mesh output should be bitwise identical (or very nearly
  so, due to floating-point order-of-operations differences in quantization).

- **Phase 2 (`MeshBuilder`) does not change topology.** The builder is a
  bookkeeping wrapper; the same vertices, cells, and faces are created in the
  same order. The only difference is that `recalculate_geometry` is called
  once at the end instead of inline, which is already the pattern in
  `fix_concave_cells`.

- **Phase 3 adds tests before Phase 1/2 go live.** Write the
  scale-invariance tests first (they will fail with the old code at extreme
  scales), then fix the code to make them pass.

- **Phase 4 is a pure refactoring** of existing conditional logic into a
  helper function.

## Ordering

**Phase 1** (tolerances) is the highest-priority fix because it directly
addresses the failure mode described in the architecture review (mesh
collapse at non-standard scales). It should be done first. ✅

**Phase 3** (tests) should be written in parallel with Phase 1 to validate
the tolerance scaling. ✅

**Phase 4** (boundary detection) is a quick follow-up that depends on Phase 1
(needs `MeshgenTolerances`). ✅

**Phase 2** (MeshBuilder) is independent and lower priority — it improves
code quality and maintainability but does not fix a correctness issue. It
can be done at any time. ✅
