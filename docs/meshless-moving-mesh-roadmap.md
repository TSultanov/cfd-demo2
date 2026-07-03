# Meshless Voronoi + Arbitrary Moving Mesh — Evolution Roadmap

**Goal:** evolve cfd2 toward (1) a *meshless* mesh format — the mesh is a set of seed points;
Voronoi cell geometry is computed per-cell, independently and in parallel, per
Ray/Sokolov/Lefebvre/Lévy, "Meshless Voronoi on the GPU" (ACM TOG 2018) — and (2) *completely
arbitrary moving-mesh* (ALE) support: seeds move with the flow or with prescribed boundary
motion, the diagram is regenerated every step, the solver runs ALE on it. **Both CPU and GPU.**

**Hard constraint:** the existing static-mesh support remains. Everything here is additive and
opt-in; the static path stays the default; existing defaults stay byte-identical; the old
generators (structured, cut-cell, Bowyer–Watson Delaunay/Voronoi) are untouched.

**Provenance:** synthesized from 4 parallel architecture designs + 4 adversarial code-grounded
reviews (all verdicts SOUND-WITH-FIXES; fixes incorporated below). Every `file:line` cited here
was verified against the working tree as of 2026-07-03 (branch `codegen`).

---

## 1. The paper's method, adapted to 2D

- **Stage 1 — kNN:** seeds binned in a uniform grid (~4/bin), CSR layout (counting sort by bin
  id); per seed, visit bins in concentric rings keeping k candidates; stop when the next ring
  is provably farther than the k-th distance.
- **Stage 2 — per-cell clipping:** cell i = intersection of half-planes (bisectors of seed i
  vs. neighbors), clipped in increasing-distance order starting from the domain box.
  **Radius-of-security stop:** once d(seed, next) > 2R (R = max distance seed→cell vertex),
  the cell is final. In 2D: convex polygon, Sutherland–Hodgman, shoelace integrals; the 3D
  paper's shared-memory dual structure collapses to per-thread arrays (~600 B).
- **Statuses, not failures:** security-radius-not-reached / vertex overflow / needs-exact —
  flagged cells get recomputed exactly (CPU f64 fallback = the paper's "certified" variant).
- **Applications:** Lloyd/CVT relaxation (seed ← cell centroid) and Voronoi moving-mesh fluids.

## 2. Governing principles

1. **Static path untouched.** Meshless/moving is opt-in (new generators, new `*_ale` model
   variants, new driver wrapper). Gate: generated WGSL for static models byte-identical
   (snapshot test — the existing `check-generated-wgsl.yml` CI only checks *freshness*, not
   invariance, so a dedicated hash-pinned snapshot test is required).
2. **Seed i == cell i, forever.** Index identity is the load-bearing invariant: all
   cell-indexed solver state (state ×3 history, warm-start x, gradients) survives every
   regeneration untouched. Fixed seed count in v1 (no insertion/deletion). Note: the *current*
   Voronoi generator does NOT have this property (dead-generator culling voronoi.rs:244-251 +
   `fix_concave_cells` splitting voronoi.rs:324) — the meshless engine establishes it by
   construction and removes the need for concave fixing.
3. **Topology authority is CPU/certified.** GPU f32 accelerates, never redefines: the CPU
   backend always uses the CPU f64 engine; the GPU-resident loop feeds only the GPU solver
   (already f32 end-to-end). Flagged/uncertain cells are recomputed in f64 **on the f32-rounded
   seed values**, and face reciprocity is enforced in release builds (non-reciprocal ⇒ extend
   the flagged set and re-resolve).
4. **SCL by construction.** Mesh face fluxes come from swept geometry (never w·S from
   velocities); Σ_f swept = ΔV per cell is exact in f64 (telescoping shoelace identity) and
   closed after f32 rounding by a deterministic spanning-tree sweep over the cell dual graph
   (per-cell fix-up of shared faces is NOT well-posed — coupled system).
5. **Determinism everywhere.** Per-cell functional computation into disjoint padded slots ⇒
   byte-identical across thread counts by construction; kNN ties broken by id; GPU clip order
   pinned by in-register id-sort per bin; the only atomic is the flag-list append (sorted on
   CPU before patching).

## 3. Resolved cross-cutting decisions

| Decision | Resolution |
|---|---|
| Boundary handling | Clip against **discrete boundary-segment lines** (exact parity with the current wall polyline). **Curved/embedded walls: midpoint-seeded** — boundary seeds at segment midpoints, so every polyline vertex is equidistant from its flanking seeds and reproduced exactly by their mutual bisector; no seed sits at a fluid-angle>π vertex (review-verified blocker: vertex-seeded clipping produces phantom walls + orphaned slivers at every obstacle-circle seed — the very cells `fix_concave_cells` patches today). GPU may realize the same lines via ghost seeds reflected across the *discrete segment lines* (bisector of seed and its line-reflection IS that line — conforming by construction; never reflect across the SDF). |
| Interior cells & walls | Boundary-seed layer *shields* interior cells (Poisson spacing guarantees it at init); safety status check = ring-vertex distance to boundary loops (not SDF — the discrete boundary is the chord polyline). |
| Sparsity format | Classic CSR rebuilt on topology change, with **capacity-reserved device buffers + sized bindings** (`BufferBinding{size: logical}` instead of `as_entire_buffer_binding`) so reallocations are rare and `arrayLength` guards stay correct. Padded k-slot stencil **rejected for v1** (codegen fork, 1.7× SpMV bandwidth, duplicate-column hazards in AMG) but GPU engine outputs are already shaped for it as the M5-v2 escape hatch. |
| ALE term placement | **Assembly-side**: `phi_rel = phi − ρ_f·mesh_flux[face]` at the point of convective use (keeps stored `fluxes` = absolute mass flux; propagates automatically to rhs_only + fused kernel variants). Separate `*_ale` generated kernels — no runtime branch in static kernels. `bounded` correction accumulates from `phi_rel`. |
| Moving-volume ddt | BDF1: `diag += V^{n+1}c/dt`, `rhs += V^n c/dt·φⁿ`; BDF2 extends the existing r=dt/dt_old form with V^{n-1}. Two new cell buffers `cell_vols_old{,_old}`, rotated **by the refresh seam** (single owner — NOT `host_prepare_step`, which runs after the new vols are already uploaded). BDF2 → Euler fallback on the first step after a topology change (reuse existing startup mechanism). |
| Continuity on moving mesh | Incompressible pressure row has no ddt ⇒ add explicit ALE volume source `+ρ(V^{n+1}−V^n)/dt`, exact by SCL construction. |
| dt handshake | `MovingMeshDriver` pulls the driver's `compute_next_dt()`, applies the mesh-motion CFL cap (~0.2·h/|w|), **pins dt**, then advects seeds → regen → swept fluxes → refresh → step. (Letting `SolverDriver::step` re-run adaptive dt after the swept volumes are computed breaks GCL.) |
| v1 scope guards | Fixed seed count; ALE for `incompressible_momentum` only (allmach needs a persisted upwinded ρ_f face buffer — designed follow-up); no periodic domains under motion; SRD unsupported with moving mesh; compressible retry/rollback paths need mesh rollback (deferred — note `RollbackAccept` also fires, not just retry); renderer must re-tessellate + grow buffers per refresh (fatal wgpu overrun otherwise — commit ea2c421 precedent). |
| AMG under motion | Per-backend: CPU Schur AMG re-Galerkins values every solve (only aggregation quality staleness) → rebuild aggregation every K steps. GPU Schur pressure AMG has **no reset hook today** (coarse ops built once at 2nd prepare) → add `reset()`; v1 alternative: Chebyshev fallback under motion (existing precedent). This is a real budget item, absent from naive overhead math. |
| Test tier reality | "Always-on" tier = `cargo test --features meshgen,cpu` (crate default features are **empty**; plain `cargo test` compiles none of the needed suites). |

## 4. Milestones

Two parallel tracks that join at M4. **Track B (M2+M3) needs neither M0 nor M1** — ALE is fully
testable on structured meshes with prescribed vertex motion + `recalculate_geometry` + refresh.
This is the fastest way to de-risk the physics and should start alongside M0.

```
Track A (mesh engine):   M0 ──► M1 ─────────┐
                                            ├──► M4 ──► M5 ──► M6
Track B (solver/ALE):    M2 ──► M3 ─────────┘
```

### M0 — Meshless Voronoi engine (CPU, f64) + Lloyd/CVT  [Track A]

New `src/meshgen/meshless/` (mod, seed_grid, clip, boundary, assemble, lloyd).

- **API:** `build_diagram(&MeshlessInput) -> MeshlessDiagram` (padded SoA: ring_xy/ring_plane/
  ring_len stride MAX_CLIP_VERTS=32, centroid, area, status; `PlaneTag::{Bisector(j),
  Boundary(seg), Box(side)}` — the neighbor list and face geometry are the *same* array);
  `assemble_mesh -> Mesh` (classic static pipeline); `generate_meshless_voronoi_mesh`,
  `generate_cvt_mesh` entry points; `compute_cell()` per-cell entry for the M1 fallback.
- **SeedGrid:** CSR uniform grid (counting sort, ~4 seeds/bin), Chebyshev-ring search, bounded
  max-heap k=16 (escalate ×2 to k_max=64, then exhaustive — CPU never fails), id tie-breaks.
- **Clip kernel:** seed-relative f64, unnormalized bisector `s(x)=x·q−½|q|²`, stable
  `t=s_u/(s_u−s_v)` intersections, security-radius early-out `d²>4·r2`, ~1 KB stack polygon,
  Vec spill on overflow (status). Degeneracy: |s|≤eps ⇒ inside (zero-length faces kept
  per-cell; merged once globally at assembly via the existing DisjointSet policy).
- **Boundary:** new `Geometry::get_boundary_loops()` (default-impl'd — four external
  implementors must not break; existing `get_boundary_points` impls retained **verbatim**,
  order feeds Poisson RNG). Midpoint seeding on curved walls per §3; two flanking seeds at
  reflex corners (step). BoundaryType tags = `classify_boundary` parity + Wall fallback.
- **Assembly:** tag-canonical vertex re-evaluation (circumcenter from sorted ids ⇒ bit-identical
  from all incident cells), quantized dedup, (i<j)-owner face emission matching the incumbent
  normal convention, first-use face ordering (Morton-friendly; seeds stay Morton-sorted).
  `fix_concave_cells`, `Mesh::smooth`, `close_untagged_boundary_faces` all **not called**.
- **Lloyd/CVT:** seed ← density-weighted centroid (ρ=h⁻⁴ for graded), boundary seeds fixed
  (v1), rebuild diagram per iter (no assembly), convergence max-disp/h < 0.01. Replaces
  generator smoothing entirely in the CVT path; vertex smoothing must NOT run after (it would
  move Voronoi vertices off bisectors).
- **Perf targets (16T, acceptance):** diagram ≤100 ms @300k; Lloyd iter ≤120 ms; assembly
  ≤250 ms; CVT total ≤ incumbent Voronoi wall time. Lloyd iteration ≈10–30× cheaper than
  triangulation-based regeneration — this ratio is what makes M4/M5 plausible; **>2× miss =
  roadmap red flag** (after re-checking the estimate-error items flagged in review).
- **Gates (review-corrected):** equivalence vs incumbent on identical seeds compared
  **interior-only by generator coordinate** (incumbent culls/splits cells; index comparison
  invalid), eps_face aligned to the existing `vertex_merge = 1e-6·h` quantization (not 1e-9),
  boundary via wall-polyline/per-type **length sums** (midpoint seeding legitimately changes
  face counts); full `validate_mesh` battery incl. Σ A·n closure; face reciprocity; convexity
  (interior); seed-in-cell (strict for interior, on-boundary-tolerant for boundary seeds);
  byte-identical across thread counts; fuzz classes incl. exact-cocircular lattice (statuses,
  never panics); skewness gates on **interior faces** (max-skew is boundary-dominated);
  solver-level: Ghia on CVT mesh ≤ incumbent-Voronoi error ×1.05 — needs a **new unstructured
  centerline sampler** (`tensor_grid` breaks on Voronoi vertices — verified).
- **Standalone value / off-ramp:** even if the roadmap stops here: CVT mesh quality (near-zero
  interior skewness), a faster + more robust + index-stable Voronoi generator, GUI `VoronoiCvt`
  mesh option.

### M1 — GPU engine port (WGSL f32 + statuses + CPU fallback)  [Track A]

New `src/solver/gpu/voronoi/` cloned from the srd.rs seam (WGSL literal, manual layouts,
per-call bind groups, own submissions).

- **Kernel:** one thread/seed, workgroup 64, `dispatch_2d` idiom; **streaming clip** (ring
  search + clip directly, no k-array) — 2D makes kNN-then-clip unnecessary; MAX_VERTS=24
  private arrays (precedent exists: block_precond.wgsl carries 2 KB/thread function-scope
  arrays with dynamic indexing — portability already proven in-repo). CPU engine implements
  the same traversal order so parity triage is tractable.
- **Grid:** CPU counting-sort build in v1 (2-4 ms @300k, deterministic, needed anyway for the
  fallback path); GPU counting sort + own ~120-line scan primitive in v2. **Graded meshes are
  a real problem** (h ratio 5× ⇒ 25× density ratio; uniform h_mean bins mass-overflow): size
  bins by r_min with distance-based ring termination, or two-level grid — re-derive costs on
  the nozzle-graded set before committing.
- **Robustness:** conservative epsilon filter (near-parallel intersections, orientation signs,
  short faces) → NEEDS_EXACT status → tiny readback → CPU f64 recompute **on f32-rounded
  seeds** → patch upload; expected flag rate 1e-4…1e-3 (budget 2e-3, re-measured *after* Lloyd
  — hex-like configs are more cocircular); bitwise-symmetric plane construction from
  (min,max) id order so threads i,j agree on shared faces; reciprocity enforced in release.
- **Equivalence criterion (explicit):** GPU f32 never bit-matches CPU f64. Topology identical
  post eps_face filter; geometry within 1e-5 rel; flagged cells resolved to exact. Cross-
  backend physics compared on observables, not bits.
- **Lloyd on GPU:** trivial update kernel + new 40-line max-reduce (existing reductions are
  sum-typed and bindgen-bound — not reusable).
- **Perf:** ~2-5 ms full regen @300k discrete GPU (~10-40 ms integrated).
- **Standalone value:** GPU mesh generation for large meshes; GPU Lloyd for CVT.

### M2 — Solver mesh-refresh seam  [Track B — start immediately]

`UnifiedSolver::refresh_mesh(&Mesh, MeshRefreshLevel::{Geometry,Topology})` + driver
passthrough. The core deliverable is the **complete inventory** of mesh-derived state (verified;
see the design docs): GPU G1-G12 (15 mesh buffers, host CSR mirrors, num_faces, block-CSR +
nnz-sized matrix_values, fluxes, bc tables + boundary_faces, bind groups, FGMRES workspace,
AMG, Schur host CSR copies + p_matrix_values, SRD, monitor buffers) and CPU C1-C6 (Buffers
entries, diag-first CSR + struct-field duplicates, **num_faces is Tier B** — it drives
face-kernel dispatch), plus `SolverDriver.min_cell_size`.

- **Tier A (geometry-only):** add `COPY_DST` to geometry buffers (flag-only change, step 0);
  `queue.write_buffer` f32 casts from a shared `mesh_geometry_f32()` used by init and refresh
  (CPU consumes the same arrays — bit-identical geometry across backends); recompute
  min_cell_size. Bind groups stay valid.
- **Tier B (topology):** rebuild both CSR layouts (GPU sorted-adjacency vs CPU diagonal-first —
  they differ and cannot share a builder; factor both out of the init paths); reallocate
  (capacity-reserved) face/nnz-sized buffers; rebuild bind groups only (pipelines unchanged —
  requires making `ResourceRegistry` reconstructable from the resources struct, incl. the
  `"y"`→rhs quirk); re-scatter bc tables + re-apply runtime BC overrides via callback;
  invalidate AMG (incl. **new `reset()` on the GPU Schur preconditioner**) and resize
  FGMRES/Schur. Note block-CSR expansion dominates uploads (~9× scalar nnz ≈ 100-120 MB
  @300k) — capacity reserve avoids the allocation churn even when the upload remains.
- **Gates:** no-op Geometry and Topology refresh **byte-identical** (CPU compared at native
  precision, not an f32 mirror; GPU exact-per-device with env waiver); refresh-to-perturbed-
  mesh ≡ fresh build + injected state (requires new full snapshot/restore incl. BDF2 history
  and face fluxes — flagged API prerequisite).
- **Standalone value:** the seam any future adaptivity (AMR, r-refinement) needs.

### M3 — ALE physics in the EDSL/codegen  [Track B]

- **`mesh_fluxes` face buffer** (swept volume/dt, owner-signed): new codegen storage item in
  `base_assembly_items`/flux items for ALE variants; zero-filled always (empty-means-zero,
  `face_wrap_shift` precedent); CPU `insert_f32` mirror; three touch points (codegen item,
  `buffer_for_binding_name`, `binding_names()` list).
- **EDSL:** `Term.relative_to_mesh` flag (precedent: `linearize_pressure_flux`) +
  `ModelSpec.ale` gate; `incompressible_momentum_ale` model variant.
- **Physics:** assembly-side subtraction, moving-volume ddt, continuity volume source, SCL
  closure per §2/§3. Rhie–Chow: geometry inputs auto-refresh; the transient-consistency term
  is already absent on static meshes — leave the derivation, validate by MMS order, fix in
  `derive_rhie_chow` only if order degrades. (d_p for incompressible is `alpha_u·dt/rho` —
  geometry-free, motion-safe as-is.)
- **Gates:** static-model WGSL snapshot byte-identical; ALE-with-zero-fluxes vs static: target
  byte-identical (plausible: `x−0.0` is bitwise identity), downgrade to ≤1-ulp only if
  reassociation bites; **GCL gate** — uniform flow on prescribed sinusoidally-deforming
  structured mesh, 500 steps, Euler AND BDF2 (BDF2-only failure localizes the volume-history
  weighting), both backends, incl. one flip-inducing case; **prescribed-motion MMS** via the
  existing mms_support harness (spatial order ~2 preserved if SCL holds; temporal ratchets
  from ≥1.0 per repo convention); mass-conservation audit (pin tolerance after first
  measurement — 1e-12/step is optimistic under 1e-4 inexact-Picard; and `CFD2_LIN_TOL` is
  GPU-only, CPU needs its recipe tolerance set).
  *As-shipped deltas (July 2026, recorded honestly):* the GCL gate runs 220 steps (not 500 —
  2.2 motion periods with a late-window no-compounding split; per-step drift margin for a
  sign/weighting error is 10²–10⁴×, so extra periods add wall time, not power) and the
  flip-inducing case is DEFERRED to M2 Tier-B/M4 (topology refresh is not shipped; geometry-only
  seam cannot induce flips). Measured spatial order is 1.387, skew-limited (pre-existing
  non-orthogonal spatial band, localized by three probes); the gate pins ≥1.35 PLUS an in-test
  moving ≡ static-on-deformed-geometry equivalence assert (±5%), which is the actual
  ALE-correctness statement.
- **Standalone value:** deforming-domain ALE on structured meshes (prescribed motion) — a
  complete feature without any Voronoi work.

### M4 — Moving-mesh loop, CPU  [joins A+B]

New `MovingMeshDriver` wrapping `SolverDriver` (static users provably untouched):
dt handshake (§3) → seed velocities (cell velocity + AREPO-style centroid steering, χ-ramped)
→ advect seeds (f64) → regen via M0 engine → swept-quad mesh fluxes (f64, spanning-tree f32
closure) → `refresh_mesh` (owns vols rotation) → step.

- Topology-change protocol: swept quads where vertex correspondence exists (vertex ≡ seed
  triple); at flip events, close each cell's SCL defect onto its faces — conservative, locally
  first-order, rare under the motion-CFL cap; defect magnitude is the always-on diagnostic.
- Matrix freeze disabled when a step begins with refresh; AMG cadence-K rebuild policy
  measured here; boundary-type histogram asserted stable unless boundaries declared moving.
- **Gates:** frozen-seeds full loop ≡ static within f32 tolerance + skip-regen variant
  byte-identical (do-no-harm); GCL through the full Voronoi-regen path; Gresho-style vortex
  advection moving ≥ static − 2% retention (**the premise gate** — if moving can't beat static
  dissipation, the milestone's premise fails visibly; respecify on an inlet/outlet channel,
  periodic meshless is out of scope); vortex-street gates on moving mesh; quality soak
  (zero negative volumes ever, skew stationary under Lloyd steering); perf budget:
  regen+refresh ≤ 1× solver step @300k (report always).

### M5 — GPU-resident loop

M1 engine + derive_faces (count/scan/emit/gather → face-major `b_face_*` written in place —
9 of 15 mesh buffers GPU-writable; STORAGE suffices for kernel writes) + swept-flux kernel +
seed_advect/Lloyd kernels, orchestrated as separate submissions around the solver graph
(SRD precedent).

- **v1 keeps CPU legs:** seed readback (needed for fallback anyway), scalar-CSR + block-CSR +
  bc-table + Schur-CSR rebuild on CPU (~20-30 ms @300k), `write_buffer` uploads (needs
  COPY_DST added to `cell_face_matrix_indices`/`diagonal_indices`) — the full M2 inventory
  applies. Host CSR mirrors must be updated with the buffers.
- **AMG is the elephant:** once-built hierarchy incompatible with per-step topology change;
  v1 = stale-hierarchy-for-K-steps or Chebyshev-under-motion; budget honestly (worst case
  20-30%+ overhead on integrated GPUs; scope the overhead gate to discrete adapters or to the
  v2 config).
- **v2 escape hatches (only if measured walls demand):** GPU counting-sort grid; padded-stencil
  assembly format retiring the CSR readback entirely (engine outputs already shaped for it).
- **Gates:** no-op regen ≡ static within f32 tol; GCL through GPU path; overhead gate;
  1000-step soak with zero unresolved statuses and bounded capacity growth.

### M6 — Moving boundaries + applications

Boundary-bound seeds moving rigidly with a prescribed boundary motion (oscillating cylinder
in channel): regenerate boundary loops per step, `MovingWall` BC velocity = boundary motion.
Gates: obstacle-contour faces exist and tagged every step; no-penetration
(U_f−w_f)·n < 1e-3·U_max on moving walls; bounded solution over 2 forcing periods; near-wall
quality instrument (skew/min-vol within 3 layers). Renderer: capacity/resize regression gate
(per-cell vertex counts drift every step; ea2c421 crash precedent).

## 5. Cost model (300k cells, honest)

| Component | Estimate |
|---|---|
| CPU step today (baseline) | 0.26–0.35 s warm (16T); GPU 0.30–0.59 s |
| M0 diagram regen (16T CPU) | 40–100 ms (budget-gated; >2× miss = red flag) |
| Swept fluxes (f64 host) | 5–10 ms |
| CSR rebuild both layouts + block expansion | 15–30 ms + O(19M) expansion pass |
| Uploads (Tier B, capacity-reserved) | ~100–120 MB ≈ 10–30 ms PCIe |
| Bind-group rebuild | <2 ms |
| AMG re-setup (when triggered) | 50–200 ms → amortize every K steps or Chebyshev |
| GPU regen (M5, discrete adapter) | 2–5 ms (+CPU legs 20–30 ms in v1) |

Bottom line: **moving-mesh CPU step ≈ 1.2–1.5× static step** if M0 hits budget and AMG is
amortized — viable. The wall candidates, in order: M0 engine regen, AMG policy, CSR bridge.

## 6. Risk register (top ranked; detection gate in parentheses)

1. **SCL/GCL violation** → spurious sources (GCL uniform-flow gate, Euler+BDF2 variants;
   mass audit). Mitigation: swept-geometry fluxes + spanning-tree closure; off-ramp: Euler-only
   for moving cases until BDF2 proven by temporal MMS.
2. **Curved-wall boundary clipping** (the caught blocker) → phantom walls/orphaned slivers
   (equivalence + watertightness gates). Mitigation: midpoint seeding, designed in from M0.
3. **f32 GPU topology corruption unflagged** (M1 zero-tolerance topology parity clause).
   Mitigation: conservative certified filter + f64 fallback on f32-rounded inputs +
   release-mode reciprocity; off-ramp: GPU kNN + CPU clipping.
4. **Performance wall** — regen+CSR+AMG dwarfs the solve (perf budget arms, visible from M2).
   Off-ramps: topology refresh every N steps with geometry-only between; padded-stencil v2.
5. **Topology flips inject solution noise / implicit-solver inconsistency** (flip-inducing GCL
   case; refresh≡fresh-build gate pins newborn-face semantics). Mitigation: eps_face hysteresis
   — flips pass through a zero-area state.
6. **Lloyd steering vs. flow advection** — tangling or lost benefit (soak zero-negative-volume
   hard assert; Gresho premise gate). Mitigation: displacement cap + quality-adaptive blend;
   off-ramp: near-Lagrangian with mild steering.
7. **Rhie–Chow on moving mesh** → checkerboarding (ALE-MMS pressure order; GCL via p-U
   coupling). Off-ramp: add the mesh-flux term inside `derive_rhie_chow`.
8. **Determinism loss** (M0 thread-count byte gate; M2 byte gates). Designed out up front.
9. **Stale mesh-derived caches** (refresh≡fresh-build catches any by construction — the reason
   that gate exists). Complete inventory is the M2 deliverable.
10. **Near-moving-wall cell quality** (M6 near-wall instrument; M4 soak precursors).
    Mitigation: rigid boundary seed layers + wall-aware Lloyd weights.

## 7. Validation program (tiers)

- **Tier 1, always-on under `--features meshgen,cpu`** (≤ ~2 min added): engine equivalence +
  invariants + determinism + default fuzz; GPU parity default; all four M2 byte gates; M3 GCL +
  conservation; M4 static-limit. *Budget honesty (July 2026): the M3 gates blew this budget —
  zero-flux ~2 min, GCL ~5 min, conservation ~3 min (GPU pipeline compilation for the ALE
  models + multi-hundred-step GPU runs dominate; the CPU legs alone fit the original budget
  and skip cleanly without an adapter). See AGENTS.md for the measured per-suite numbers.*
- **Tier 2, dev-tests** (AGENTS.md-mandated for mesh/ALE/codegen changes): Ghia-on-CVT,
  status-rate @300k, ALE-MMS orders, vortex gates, M6 demo.
- **Nightly/env-gated:** fuzz at scale, soaks, perf budgets.
- New shared `tests/mesh_support/` module (validate_mesh extraction, reciprocity/convexity/
  seed-identity checks, diagram comparison, `AleAudit`, state fingerprints).

## 8. Open questions (carried forward)

1. FGMRES warm-start state and byte-equality of the *next* step after snapshot/restore (gates
   the M2 identity definition) — needs a code check.
2. BDF2 SCL-consistent weighting choice → whether temporal MMS targets 2.0 immediately.
3. AMG cadence K under continuous motion — measure in M4.
4. Whether `mesh_fluxes` lives in `MeshResources` (proposed) or field resources — revisit at M5.
5. Boundary seeds relaxing tangentially along loops (better wall spacing) — M0 or M4.
6. Periodic domains under motion (ghost images across the seam) — explicitly out of v1.

## 9. Critical files

Engine: `src/meshgen/{voronoi.rs,delaunay.rs,geometry.rs,tolerances.rs}`, `src/solver/mesh/structs.rs`
Solver seam: `src/solver/gpu/init/mesh.rs`, `src/solver/cpu/solver.rs`,
`src/solver/gpu/lowering/programs/generic_coupled.rs`, `src/sim/driver.rs`,
`src/solver/gpu/modules/generic_coupled_schur.rs`, `src/solver/gpu/program/generic_coupled_backend.rs`
Codegen: `crates/cfd2_codegen/src/solver/codegen/{time_integration.rs,unified_assembly.rs,coupled_common.rs}`
GPU kernel seam: `src/solver/gpu/srd.rs` (template), `src/solver/gpu/{buffers.rs,readback.rs,context.rs}`
Tests: `tests/{meshgen_validation.rs,mms_support/,cpu_gui_parity.rs,ghia_lid_cavity_test.rs,cpu_obstacle_bench.rs}`
