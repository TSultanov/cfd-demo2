# Meshless Voronoi + Arbitrary Moving Mesh — Evolution Roadmap

**Goal:** evolve cfd2 toward (1) a *meshless* mesh format — the mesh is a set of seed points;
Voronoi cell geometry is computed per-cell, independently and in parallel, per
Ray/Sokolov/Lefebvre/Lévy, "Meshless Voronoi on the GPU" (ACM TOG 2018) — and (2) *completely
arbitrary moving-mesh* (ALE) support: seeds move with the flow or with prescribed boundary
motion, the diagram is regenerated every step, the solver runs ALE on it. **Both CPU and GPU.**

**Hard constraint:** the existing static-mesh support remains. Everything here is additive and
opt-in; the static path stays the default; existing defaults stay byte-identical; the old
generators (structured, cut-cell, Bowyer–Watson Delaunay/Voronoi) are untouched.

---

## Roadmap status — COMPLETE (M0–M6), 2026-07

**End-to-end, on both CPU and GPU:** a meshless CVT-Voronoi mesh whose seeds advect with the flow
(or with prescribed rigid boundary motion), regenerated every step, solved with a conservative ALE
(moving-volume ddt + SCL-closed swept fluxes) incompressible model, surgically refreshed across
the topology change, and rendered live in the GUI.

- **M0** — CPU meshless Voronoi + Lloyd/CVT (Ghia 40% better than the old generator).
- **M1** — GPU engine port (WGSL f32, zero-tolerance parity, regen ~12 ms @264k).
- **M2** — solver mesh-refresh seam (byte-identical no-op refresh, both backends).
- **M3** — ALE physics in the EDSL/codegen (GCL ~1e-6, temporal order 2.15, conservation 6e-15/step).
- **M4** — moving-mesh loop (CPU): dt handshake → advect → regen → swept flux → refresh → step.
- **M5** — GPU-resident loop: **core shipped** — GPU surgical topology refresh (recompile
  eliminated via a per-device pipeline cache; refresh 20k −68%, GCL cold-restart artifact gone),
  the full GPU moving loop with surgical BDF2 history (CPU↔GPU field RMS 3e-6), and the GUI GPU
  backend. **GPU-resident regen: foundation shipped** (the two-level exclusive scan primitive +
  the `derive_faces` count→scan half, both bit-exact on GPU); the face-major emit + CSR bridge
  stay CPU (blocker documented in §M5 — CPU face geometry is a post-vertex-merge product of
  `assemble_mesh`; that merge port + the padded-stencil CSR is the remaining v2 optimization).
- **M6** — moving boundaries (oscillating cylinder, MovingWall BC), CPU-authored, GPU-enabled by M5.

The regen still runs on the CPU M0 engine feeding the GPU solver via the CSR bridge (as scoped);
per-step GPU moving is now viable because the refresh is surgical, not a recompile-cold-restart.

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

**M1 as-shipped notes (stage-5 review record):**

- *Traversal-order deviation:* the "CPU engine implements the same traversal order" clause
  above was NOT kept. The kernel streams Chebyshev grid rings (CPU `for_each_ring_bin` order,
  in-bin ids ascending) while M0 clips in kNN ascending-(d², id) order. Rationale
  (`src/solver/gpu/voronoi/mod.rs`): the clipped polygon is order-independent up to f32
  rounding and every parity gate compares eps_face-filtered sets, never bits, so matching the
  CPU order buys nothing for triage while costing a k-array + register sort + a "k too small"
  failure mode. Decision made at stage 1 and kept for all M1 stages.
- *Flag budget carve-out:* the ≤2e-3 pre-Lloyd budget is enforced on the BULK interior;
  the wall strip (boundary seeds + their interior neighbors) runs under a defensive 35% cap
  because M0 `boundary_seeds` emits same-segment reflex-guard pairs 1e-3·h…1e-2·h apart —
  knife-edge twins by construction. Measured strip rates at h=0.05: rect 1.8%, obstacle 1.9%,
  backstep 2.5%, nozzle 27.7%, graded nozzle 28.7%; at design scale (obstacle h=0.0046,
  ~89k seeds) the strip rate drops to 0 and the global rate is 1.2e-4. Follow-up lever (M0
  arc): collapse/equalize same-segment guard pairs in `boundary_seeds`, then ratchet the cap.
- *Perf baseline (Apple-Silicon integrated adapter, Metal; discrete-GPU gate-5 numbers not
  measurable on this machine — the design estimate is ×4-8 between the classes):*

  | phase | ~88k seeds | ~264k seeds |
  |---|---|---|
  | SeedGrid::build (CPU) | 0.22 ms | 0.65 ms |
  | upload_case (grid+canon+writes) | 2.6 ms | 9.0 ms |
  | regen kernel (submit→poll, best) | 4.6 ms | 12.1 ms |
  | read_cells full validation readback | 24 ms | 43 ms |
  | resolve_flagged (f64 + reciprocity readback) | 39 ms (7 flagged) | 113 ms (38 flagged) |
  | Lloyd chained, per iteration | 5.5 ms | 17.2 ms |

  Conservation (gate 4) is asserted ≤1e-4 in the bench (measured ~1.9e-8). The stage-2 commit
  message's "30k regen 1.5 ms" is superseded: after stages 3-5 (boundary clipping, slack
  read, diagnostics writes, derated security stop) the 30k regen is ~2.7 ms best.
- *Assembler note:* the mutual-orphan endpoint reconciliation added for M1 also fires on pure
  f64 M0 inputs (4 sites on the obstacle circle, gaps ~1.7e-8 — sub-dedup-pitch quantize-bin
  straddles, welding only what the dedup already targets). Instrumented + gated CPU-only by
  `tests/meshless_orphan_cpu_test.rs`.

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

**M2 as-shipped notes (Tier A + Tier B, July 2026 — SHIPPED both backends):**

- *Scope delivered.* Tier A (geometry-only, byte-invisible) and Tier B (topology: face
  set / adjacency / nnz change at an INVARIANT cell count) both land on GPU and CPU.
  `refresh_mesh(Topology)` rebuilds both CSR layouts (factored
  `build_{sorted,diag_first}_scalar_csr`, gated byte-equal to the historical inline
  builders), reallocates every face/nnz buffer (capacity-reserved; the scalar-CSR + face
  mesh buffers bind as sized ranges, block-CSR named buffers bind entire under an EXACT
  guard — see *Capacity policy* below),
  re-scatters the bc tables + `boundary_faces`, and rebuilds bind groups. The M3 ALE
  rotation is FUSED with the topology rebuild via `begin_ale_step_topology` (rotate volume
  history → topology rebuild → geometry → mesh-flux upload, in that order); plain
  `refresh_mesh` stays REJECTED on ALE models; the M3 sequencing guards (adaptive-dt Err,
  double-arm Err, SRD Err) extend to the topology arm. **Zero codegen change** — the
  generated WGSL is untouched (host-side only), `static_wgsl_snapshot` byte-identical.

- *Gates (all green).* no-op Geometry+Topology refresh byte-identical both backends;
  **refresh(A→B) ≡ fresh-build(B)+restore** (CPU byte-exact, GPU f32-exact — THE
  stale-cache detector); 20-cycle alternating A↔B stable + deterministic-rebuild + bounded
  logical sizes; CSR-rebuild byte-equals a fresh build (CPU solver-level + both-layout
  builder determinism); ALE topology-seam GCL (below). `tests/mesh_refresh_topology_test.rs`.

- *Stale-cache finding (the refresh≡fresh gate earned its keep).* The equivalence gate
  first failed at EXACTLY `schur_amg_active`: on the 243-cell meshless CUT-CELL mesh the CPU
  adaptive Jacobi→AMG flip fires within a few steps, so the snapshot carries
  `schur_amg_active=true`, while the topology refresh correctly RESETS it (F8
  stale-aggregation — the AMG aggregation is invalid on the new sparsity). That is an
  adaptive-solver-MODE difference, orthogonal to mesh-rebuild correctness; the equivalence
  gate pins `CFD2_CPU_SCHUR_AMG=0` to isolate the rebuild. Every buffer + CSR struct field
  is otherwise byte-identical between the refreshed and fresh legs (verified by
  fingerprinting during the hunt) — the topology rebuild leaks nothing.

- *GPU LA-stack reconstruction (stage-2 deviation, cost now measured).* The GPU topology
  refresh RECONSTRUCTS the linear-algebra modules (scalar CG, FGMRES, Schur/AMG, monitors)
  rather than surgically rebinding, which recompiles their STATIC hand-written pipelines
  (the GENERATED model kernels take the in-place bind-group rebuild — no codegen recompile,
  byte-identity preserved). This re-zeroes the warm-start `x`. Consequence for the ALE
  topology-seam GCL: on GPU each step restarts the coupled solve COLD → a bounded, SATURATED
  ~1.5e-3 free-stream residual (non-compounding, asserted as late-quarter ≤ 1.5× an EARLY
  post-cold-start window — a convergence artifact NOT a GCL violation), vs the CPU's surgical
  refresh (preserves `x`) holding the M3 ~1e-6 scale.
  Warm-start preservation across the GPU rebuild needs the `x`-readback/upload plumbing
  stage-3 deferred (GPU exposes only `read_state_bytes`) → M4.

- *Measured topology-refresh cost* (median-of-5; a NO-OP topology refresh does the FULL
  Tier B rebuild, so the cost is faithful; Apple-Silicon integrated Metal / f64 CPU, via
  `bench_topology_refresh_cost` `#[ignore]`):

  | cells / faces | GPU | CPU |
  |---|---|---|
  | ~20k / 40k | 21.7 ms | 0.40 ms |
  | ~300k / 600k | 98.3 ms | 7.9 ms |

  GPU is dominated by the LA-pipeline recompile (fixed cost — visible as the 20k number) +
  block-CSR re-expansion (~9× scalar nnz) + uploads; CPU is the surgical CSR rebuild +
  buffer reallocation only. The M4 surgical-GPU-refresh optimization (threaded pipeline
  cache + in-place LA rebind + warm-start preservation) targets closing the GPU→CPU gap for
  the per-step motion loop.

- *Capacity policy.* `CapacityPlan::EXACT` (headroom 1.0) is the byte-neutral default: each
  refresh reallocates at the new exact size. The **scalar-CSR + face mesh buffers** (the ones
  carrying `arrayLength` guards) resolve through `MeshResources::binding_resource_for` as
  **sized ranges** (`BufferBinding{offset:0, size:logical}`), so under headroom>1 their
  `arrayLength` sees the logical face/nnz count, not the padded allocation — the
  `ResourceRegistry::resolve` mesh arm is wired to this (stage-5 review fix; byte-neutral at
  EXACT, proven by the no-op byte gates). The **block-CSR named buffers** (`matrix_values`/
  `col_indices`, bound ENTIRE via `with_buffer` across FGMRES/AMG/Schur) are NOT yet sized;
  `init_matrix` therefore ASSERTS `headroom == 1.0` — a loud guard against the silent
  arrayLength-into-the-tail corruption until those named buffers get a sized-binding path.
  Headroom>1 + reuse-on-fit ("reallocate only on overflow" ⇒ reallocations stop after the
  first growth to max-seen) is wired at the buffer helper (`capacity.rs`) but NOT yet
  enabled — the block-CSR sized bindings + the reuse loop are the M4 per-step optimization.

- *Flip deferral (crisp — what is missing).* The ALE topology seam is validated as the
  no-op-TOPOLOGY case (structured mesh, fixed face set, real vertex motion): the full rebuild
  machinery runs EVERY step and preserves the GCL free-stream. A genuine Voronoi FLIP under
  motion regenerates the mesh with a NEW vertex/face set, for which
  `swept_mesh_fluxes_closed` cannot produce fluxes — it needs a persistent face↔swept-quad
  correspondence (a born face has no old counterpart, a dead face no new one). Supplying
  SCL-consistent fluxes across a re-tessellation (a conservative-remap / generalized
  swept-volume accounting that still telescopes to per-cell ΔV) is **M4** (the mesh-motion
  loop). The topology-refresh machinery this milestone ships is exactly what such a flux
  path will drive; only the flux construction is deferred.

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
  sign/weighting error is 10²–10⁴×, so extra periods add wall time, not power). The
  flip-inducing case is STILL deferred to M4, but its precondition — the Tier B topology
  refresh — is now SHIPPED (M2): the ALE topology seam (`begin_ale_step_topology`) is gated
  by a GCL free-stream driven through a per-step FULL topology rebuild
  (`gcl_topology_seam_preserves_uniform_flow_{cpu_euler,cpu_bdf2,gpu}`), which proves the
  rebuild+rotation machinery preserves the GCL on the no-op-topology (fixed-face-set) case.
  What a real flip still needs is the swept-flux correspondence across a re-tessellation
  (born/dead faces have no swept quad) — a conservative remap, which is M4, not the refresh
  machinery. Measured spatial order is 1.387, skew-limited (pre-existing
  non-orthogonal spatial band, localized by three probes); the gate pins ≥1.35 PLUS an in-test
  moving ≡ static-on-deformed-geometry equivalence assert (±5%), which is the actual
  ALE-correctness statement.
- **Standalone value:** deforming-domain ALE on structured meshes (prescribed motion) — a
  complete feature without any Voronoi work.

### M4 — Moving-mesh loop, CPU  [joins A+B]  — **SHIPPED (CPU)**

`MovingMeshDriver` (src/sim/moving_mesh_driver.rs) wraps `SolverDriver` (static users provably
untouched — a purely additive wrapper): dt handshake (§3) → seed velocities → advect seeds
(f64) → regen via M0 engine → swept-quad mesh fluxes (f64 telescoping, spanning-**forest** f32
closure) → surgical geometry/topology seam (owns vols rotation) → step. `MeshMotionSpec` =
`Frozen` (M4.1), `Prescribed(fn)` (M4.2), `FlowCoupled { regularization }` (M4.4).

- **dt handshake (F2):** the driver PINS a fixed dt (`requested_dt` capped by the mesh-motion
  CFL `dt ≤ cfl_mesh·min_h/max|w|`, default `cfl_mesh = 0.2`) via `set_requested_dt` BEFORE the
  swept fluxes are closed, so the flux dt == the step dt exactly; `begin_ale_step*` hard-rejects
  `adaptive_dt`. Consequence: a `FlowCoupled` (near-Lagrangian) mesh runs at `dt ≈ 0.2h/|U|`.
- **Topology flips (M4.3):** born faces carry zero swept contribution; each cell's residual is
  closed onto its slack faces by the M3 spanning forest, so `Σ_f σ·flux = ΔV_i/dt` stays EXACT
  per cell (GCL survives the flip) — conservative, per-face locally first-order, `flip_defect`
  the always-on diagnostic. **Per-face accuracy note:** on/around a flip the slack face soaks
  the whole per-cell residual, so the individual face flux is O(motion·h/V) first-order; the
  per-cell SUM (what the GCL and the moving-volume ddt see) is exact. Died faces carry no
  face-to-face flux; their mass is absorbed into the exact per-cell balance of the survivors.
- **FlowCoupled (M4.4):** seed velocity = readback cell velocity `U_i` + AREPO-style
  distortion-ramped centroid steering `χ_i·(centroid_i − seed_i)` (χ ramps 0→`regularization`
  once `|s_i−c_i| > η·R_i`, η = 0.25, so well-shaped cells advect PURELY with the flow), clamped
  per step to `flow_disp_cap·R_i` (default 0.25) as a hard anti-tangling guard. Boundary seeds
  fixed (v1). A skew-triggered Lloyd escalation (reusing `lloyd_relax`) runs 1–2 blended sweeps
  when the regenerated mesh's max skew rises above a target — this is what keeps the regen from
  ever producing a sliver face the swept-flux path would reject.

- **Gates (tests/moving_mesh_{loop,gcl,flip,flow}_test.rs, CPU):**
  - **Frozen do-no-harm (M4.1):** the full frozen regen loop is BYTE-IDENTICAL to a static
    `incompressible_momentum_ale` run; skip-regen variant is a pure passthrough.
  - **GCL through the full Voronoi regen (M4.2):** uniform flow held at `max|U−U0| ≈ 1.9e-6`,
    `max|p| ≈ 2.3e-5` (the M3 structured-mesh scale); per-step f64 telescoping identity 4e-14,
    f32 SCL defect ~1e-9. Closed-box conservation Σρ·V vs area 8e-16.
  - **Topology-flip GCL (M4.3):** 60/140 steps flip (peak 4.8% cells/step), `flip_defect` O(1)
    at 2.8e-2 — yet uniform flow holds at `max|U−U0| = 1.9e-6`, non-compounding.
  - **PREMISE gate (M4.4) — the decisive milestone test:** a compact Gaussian vortex advected by
    a uniform free stream in an inlet/outlet slip channel (periodic meshless is out of scope —
    review #12), static CVT vs FlowCoupled. **Static retention 89.3%, moving 99.9% (+10.6 pts,
    gate ≥ static − 2 pts).** The flow-following mesh nearly eliminates the advective dissipation
    of the vortex peak — the milestone premise, demonstrated, not assumed.
  - **Quality soak (`#[ignore]`, `CFD2_SOAK=1`):** zero negative/zero volumes ever, max skew
    bounded < 0.6 *under the soak's aggressive escalation (target 0.4, 2 Lloyd sweeps)*, mean-skew
    drift stationary, flip rate stationary under the Lloyd steering. (The bound is
    escalation-config-dependent: the premise gate's lighter default escalation — target 0.5, 1
    sweep — runs at max skew ≈ 0.62, which it tolerates because it asserts retention, not skew.)
  - **Perf budget (`CFD2_BENCH_MOVING=1`):** per-step {plan, regen, swept, refresh, solve} split
    (plan = the pre-regen velocity readback + quality-escalation probe); the moving-mesh overhead
    (plan+regen+swept+refresh) is ≪ 1× the CPU coupled solve (e.g. ~2 ms overhead vs a ~1 s solve
    at 844 cells → 0.00× — the CPU coupled solve dominates by orders of magnitude, so the M4 loop
    is never the bottleneck).
  - **Obstacle Kármán street on a FlowCoupled mesh (`#[ignore]`, documented finding):** the
    static meshless CVT mesh DOES shed a vigorous street (warm-up wake `u_y` std ≈ 0.48·U, ~22
    reversals). Handed off to a FlowCoupled moving mesh it stays BOUNDED/STABLE (max|u| ≈ 1.73·U,
    no divergence at mesh_cfl 0.2 — the soak-validated regime) and HOLDS the wake vortex at full
    strength (`|u_y|` frozen near 0.58·U, not decayed), but the fixed-probe `u_y` OSCILLATION is
    suppressed (std ≈ 0.008·U). This is not a defect: a near-Lagrangian flow-following mesh
    transports the shed vortices ALONG WITH IT, removing the advection of the pattern past a fixed
    point — the exact transport a fixed-probe time series measures, and the SAME mechanism that
    wins the premise gate. A fixed-probe Eulerian shedding signal is therefore conceptually
    incompatible with a Lagrangian mesh (needs a mesh-frame/vorticity observable); the gate is
    kept as a runnable experiment asserting the verifiable facts (static sheds; moving mesh stays
    bounded and preserves the vortex) and `#[ignore]`d for the incompatible original criterion.

- **GPU-resident moving loop deferred to M5:** a GPU `refresh_mesh` cold-restarts the LA stack
  (re-zeroes the warm-start `x`), so it cannot hold the warm-started GCL through a per-step flip;
  M4 is CPU-first by design, GPU per-step regen is M5.
- **Renderer (review #14):** a moving mesh invalidates build-time-sized renderer geometry buffers
  every step; a GUI capacity/resize gate is an M4/M5 GUI concern, deferred with the GPU loop
  (this stage is headless/CPU).

### M5 — GPU-resident loop — **SHIPPED (core); GPU-resident regen partial (honest)**

The milestone **core** — making per-step GPU moving-mesh viable — shipped in three stages.
The **stretch** (regen computed on the GPU feeding solver buffers without a CPU round-trip)
landed its foundation (the missing scan primitive + the count→scan half of `derive_faces`);
the face-major emit + the CSR bridge remain CPU, documented below with the exact blocker.

**Stage 1 — GPU surgical topology refresh (the core).** The GPU Tier-B refresh used to
*recompile* the entire hand-written LA stack (scalar-CG / FGMRES / Schur / AMG / block-precond,
~45 `create_compute_pipeline` sites) and re-zero warm-start `x` on **every** refresh — ~98 ms
@300k, recompile-dominated, plus a cold restart each step. Fix: a per-device `PipelineCache`
(`src/solver/gpu/pipeline_cache.rs`) keyed on `(model_id, KernelId)` that survives every refresh
(pipelines are keyed on constant shader source ⇒ reuse is bit-identical), and a buffer→buffer
blit carrying warm-start `x` across the reallocation (no readback). Reconstruction still rebuilds
bind groups; only the recompile is gone. **Measured: refresh 20k 17.7→5.6 ms (−68%), 300k
75.2→60.5 ms (−20%, residual = CSR host-build + block-CSR upload, *not* recompile). GPU
topology-seam GCL euler max|U−U0| 1.53e-3→5.5e-5 (late 1.31e-6, CPU ~1e-6 scale) — cold-restart
artifact gone, non-compounding.** Instrument: `CFD2_REFRESH_PROFILE=1`.

**Stage 2 — GPU moving loop end-to-end + surgical BDF2 history.** The GPU `MovingMeshDriver`
runs the full per-step cycle (dt handshake → CPU M0 regen → swept fluxes → surgical topology
refresh → GPU step). BDF2 history needs **no** snapshot readback: `state`/`state_old`/
`state_old_old` are cell-indexed buffers the refresh never reallocates, and the ALE volume
history is carried by swap — strictly cheaper than snapshot/restore. **Measured: GPU moving GCL
euler/bdf2 1.32e-5 / 2.14e-5 (late 8.3e-7 / 1.4e-6), CPU↔GPU field RMS 3.0e-6; per-step refresh
7.5 ms = 3% of a 292 ms debug step (23% of the 33 ms overhead) — the solve dominates, not the
refresh.** Gates in `tests/gpu_moving_mesh_test.rs`.

**Stage 3 — UI GPU backend + GPU-resident regen foundation.**
- **UI (shipped):** the moving-mesh toggle is now enabled on the GPU backend too (was CPU-only
  "GPU moving is M5"). `src/ui/app.rs` — the enable/gating + tooltip + backend-switch handler.
  Gate: `moving_mesh_gui_worker_gpu_backend` drives the real solver worker in moving mode on the
  GPU backend (CFD2_BACKEND path) and observes `MeshRefreshed`; the CPU worker path is unchanged;
  renderer capacity holds. Skips cleanly without an adapter.
- **GPU-resident regen (stretch — partial, foundation landed):**
  - **Scan primitive (`src/solver/gpu/voronoi/scan.rs`) — LANDED.** The design flagged that *no*
    prefix-sum existed anywhere in `src/solver/gpu/**` (only workgroup reductions). This is the
    standard two-level exclusive scan (256-thread workgroup, 4 elems/thread, Hillis–Steele block
    scan + block-sums level), deterministic and **bit-exact** vs a CPU reference across sub-block /
    exact-block / block+1 / 300k / random sizes (`tests/gpu_scan_test.rs`).
  - **`derive_faces` count→scan (`src/solver/gpu/voronoi/derive.rs`) — LANDED.** The
    `count_owned_faces` kernel counts each cell's owned faces (canonical `owner = min(i,j)` +
    boundary faces) from the M1 cell-major outputs, and the scan turns those into per-cell
    face-major **offsets** + the total `num_faces`, **all on the GPU**, in one encoder. Validated
    bit-exact vs a CPU reference on real Voronoi output (~3 faces/cell, single- and multi-block)
    in `tests/gpu_derive_faces_test.rs`. This is the scan's first real consumer.
  - **DEFERRED (the documented blocker):** the `emit`/`gather` passes that write the face-major
    *geometry* buffers (`b_face_owner`/`areas`/`normals`/`centers` + `cell_faces` CSR) and match
    them to the CPU `Mesh`, plus the §6.3 **CSR bridge** (`b_scalar_*`,
    `cell_face_matrix_indices`, `diagonal_indices`). Blocker: the CPU face-major geometry is **not**
    a direct projection of the cell-major clip slots — `assemble_mesh` produces it *after* a
    disjoint-set vertex merge on the quantization grid and a geometric shared-vertex-pair face
    resolution (chain/orphan handling). Matching it to f32 requires porting that merge + pairing
    to the GPU, then the CSR rebuild (CPU-readback in v1 by design; fully no-readback needs the
    padded-stencil format, out of scope). So the **face-major buffer readback is NOT yet
    eliminated** — only the count→offset addressing is now GPU-resident. That merge port + CSR
    bridge is the remaining M5-v2 optimization, on the count→scan foundation this stage provides.
- **AMG note (still the elephant, unchanged):** once-built hierarchy incompatible with per-step
  topology; v1 rebuilds it in the refresh (part of the 300k residual cost). Chebyshev-under-motion
  / stale-hierarchy-for-K-steps stays a v2 lever.

### M6 — Moving boundaries + applications — **SHIPPED (v1, CPU)**

Boundary-bound seeds moving rigidly with a prescribed boundary motion (oscillating cylinder
in channel): regenerate boundary loops per step, `MovingWall` BC velocity = boundary motion.
Gates: obstacle-contour faces exist and tagged every step; no-penetration at the moving wall
(enforced at the FACE by the Dirichlet `U_face = w_wall` + ALE mesh flux ⇒ zero relative flux —
structural; the near-wall OWNER-cell residual is inherently O(wall speed) and is instead gated by
an ON-vs-OFF control, review July 2026 stage-4); bounded solution over 2 forcing periods;
near-wall quality instrument (skew/min-vol within 3 layers). Renderer: capacity/resize regression
gate (per-cell vertex counts drift every step; ea2c421 crash precedent).

**Shipped (v1 scope): rigid PRESCRIBED boundary motion, FIXED seed count** — the obstacle
changes shape/position, not seed count; seed `i` ≡ cell `i` for the whole run. Built additively
on the M4 moving-mesh loop (static + M0–M4 + UI paths byte-identical / do-no-harm anchored).
Three stages, all on the CPU (`incompressible_momentum_ale` model):

- **Stage 1 — rigidly-moving boundary seeds + per-step moved-loop regen.** A new
  `BoundaryMotionSpec` (orthogonal to the interior `MeshMotionSpec`) moves one boundary loop's
  seeds rigidly from their t=0 labels (absolute sampling, no drift) and clips the regen against
  the moved loop. A rigid map preserves chord lengths ⇒ the loop's seed count / segment tags /
  watertightness are invariant. The M4 `align_old_vertices_by_seed_set` + swept-flux +
  born/dead-face closure path handles the now-moving boundary vertices unchanged.
- **Stage 2 — the fluid feels the wall (MovingWall ALE BC).** BC path (smallest correct):
  `MovingWall` (bc index 5) is ALREADY a per-face Dirichlet velocity in the
  `incompressible_momentum(_ale)` model — no codegen / WGSL change. Each regen re-tags the moving
  loop's open faces `MovingWall` (owner-is-moving-seed) and, after the ALE refresh, sets their
  per-face `bc_value = w_wall[owner] = (new−old)/dt` (re-applied every step; the topology seam
  resets per-face overrides). The convective half already sees the relative velocity
  `phi − ρ·mesh_flux` from the M3 ALE path; this is the Dirichlet half (no-slip + no-penetration
  at the wall's material velocity). Opt-in (`set_moving_wall_bc`); default off ⇒ static-velocity
  `Wall`.
- **Stage 3 — headline demo + near-wall instrument + GUI + docs.** A first-class
  `BoundaryMotionSpec::Oscillation { loop_index, amplitude, omega, axis }` variant (carries its
  own params so the GUI sliders can drive it — a bare `fn` pointer cannot). The GUI exposes an
  "Oscillating obstacle" toggle + amplitude/frequency in the Moving Mesh (ALE) panel
  (ChannelObstacle only), reusing the stage-1/2 driver path.

**Measured (CPU, meshless/dev-tests; `tests/moving_boundary_test.rs` + `tests/moving_mesh_gui_test.rs`):**

- *Oscillating-cylinder demo* (`oscillating_cylinder_demo_responds_to_forcing_cpu`): cross-stream
  forced cylinder, 100 steps = **2.5 forcing periods**, ~580 cells. Bounded `max|U| = 0.88 < 10·U_scale`,
  finite; obstacle contour tagged `MovingWall` + watertight every step; **SCL defect ≤ 3.4e-9**;
  **near-wall quality** (3 layers): worst skew **0.031 (< 0.7)**, min cell vol **1.2e-3 > 0**.
  **Flow response:** signed downstream wake-mean transverse velocity oscillates with peak-to-peak
  **6.2e-2** in the forced run vs **1.2e-3** in the static control (**≈50×** — a measurable forced
  signal; NOT a shedding lock-in claim: Re≈10 ⇒ the static case is steady + symmetric).
- *Free-stream preservation / GCL* (`moving_wall_freestream_preserved_cpu_{euler,bdf2}`): a
  rigidly-translating obstacle in a free stream = its own velocity (a CO-MOVING field, `U ≡ w_wall`)
  — the field stays on the free stream, drift `2.1e-7`, SCL `1.5e-9`; the wall-BC-OFF control drifts
  ~5 orders more (proves the wall drives the fluid), through adjacency flips. (This is
  free-stream preservation, NOT an independent no-penetration measure — see below.)
- *No-penetration, non-co-moving control* (`no_penetration_cross_stream_oscillation_cpu`, review
  stage-4 FINDING 2): a cross-stream-oscillating cylinder in a QUIESCENT closed box (`w_wall·n ≠ 0`,
  the only motion is wall-driven, `|U−w| = 0.35` = a genuinely disturbed field). Exact no-penetration
  is structural at the wall FACE; the OWNER-cell residual `|(U−w)·n|` is inherently O(wall speed), so
  the gate is a CONTROL: imposing `w_wall` (ON) pulls it to **0.157**, ≈33% below the zero-velocity-wall
  control (OFF **0.233** = the full wall normal speed A·ω). SCL `3.4e-9`. Also asserts the MovingWall
  tag set equals the geometric contour set every step (FINDING 5 — no static-Wall hole).
- *Area preservation + boundedness* (`rigid_obstacle_area_preserved_and_bounded_cpu`, review
  stage-4 FINDING 1): closed box + oscillating internal MovingWall obstacle, from rest. Σρ·V drift
  **2.25e-16** per step/total — a GEOMETRIC identity (a rigid polygon's area is invariant), NOT a
  statement about the solved field; the genuine per-cell **mass conservation under mesh motion** is
  the SCL defect `3.2e-9`. The from-rest wall-driven flow stays finite + bounded (`max|U| = 0.29`).
- *GUI worker path* (`moving_mesh_gui_test`, Part 4): the oscillating-obstacle driver through the
  real solver worker — 30 `MeshRefreshed` emitted, 614 cells fixed, SCL `1e-9`, obstacle
  demonstrably moved, and every emitted mesh replayed through the renderer capacity path (no
  overflow/truncation — the ea2c421 regression gate).

**v1 scope / limits (honest):** rigid PRESCRIBED motion only (no fluid-structure coupling — the
wall trajectory is analytic); pure TRANSLATION only — a rotating wall is NOT supported in v1
(the per-seed `w_wall`, uniform across the wall face, matches the per-vertex swept `mesh_flux`
only for translation; rotation needs a per-wall-face `w_wall`, deferred, review stage-4 FINDING 3);
fixed seed count (rigid motion, small amplitude < near-wall cell spacing so the frozen interior
seeds are never swallowed — an amplitude that swallows a seed is a hard `Err`, by design); runs on
both backends (M5 shipped the GPU moving loop; M6 was authored CPU-first); interior seeds are
`Frozen` in the demos (the obstacle deforms the near-wall cells, which the near-wall instrument
watches).
Headless / test-driven; the live GPU-render animation is the manual smoke below.

**Manual GUI smoke — oscillating obstacle** (needs a display; not covered by CI):

1. `cargo run --release --features "cpu ui"`.
2. Left panel → **Compute backend** → GPU or a CPU option (moving mesh runs on both as of M5).
3. **Geometry** → *Channel with obstacle* (the oscillating-obstacle controls appear only for it —
   the obstacle is loop 1 of its boundary spec).
4. **Moving Mesh (ALE)** group → tick **Enable Moving Mesh (ALE)** (auto-steers Mesh Type →
   Voronoi (CVT), model → incompressible ALE, fixed dt). Leave **Seed motion** on *Frozen* (or
   *Flow-coupled*) — the interior-seed law is orthogonal to the obstacle motion.
5. Tick **Oscillating obstacle**; set **Amplitude** (keep below the near-wall cell spacing, e.g.
   0.03–0.06) and **Frequency (Hz)**.
6. Click **Initialize / Reset**, then **Run**.

Expected: the cylinder visibly oscillates cross-stream, its boundary cells re-tessellate with it
(no flicker/crash/overflow), the fluid follows the moving wall (a transverse wake response), and
the ALE stats block shows a bounded SCL defect + healthy skew every step. A too-large amplitude
that swallows an interior seed surfaces as a worker error (fixed-seed v1), not a crash.

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
   that gate exists). Complete inventory is the M2 deliverable. **Retired (M2 shipped):** the
   refresh(A→B)≡fresh(B)+restore gate is green both backends; it caught one real confound
   (the adaptive-AMG `schur_amg_active` mode carried by the snapshot vs reset by the refresh —
   isolated, not a leak) and otherwise proves every buffer + CSR field byte-identical.
   *Coverage scope (honest):* the CPU leg exercises the FULL BDF2 history (5-step-evolved
   state, `state_old`/`state_old_old` weighting) — the strong stale-cache detector. The GPU
   snapshot is current-state-only (`has_history=false`, `step_count=0` — GPU exposes only
   `read_state_bytes`), so its post-refresh step runs the Euler startup fallback, not BDF2;
   it still catches gross topology-inventory staleness (CSR / bc / geometry / face-dispatch /
   boundary_faces) but not a stale buffer that surfaces ONLY through `state_old_old` weighting
   or warm-start `x` carry — those await the `x`/history readback (M4). A companion gate
   (`topology_refresh_face_count_change_matches_fresh_build_{cpu,gpu}`) additionally exercises
   a genuine face-count/nnz CHANGE (8×12↔4×24 structured, equal cell count), so the
   buffer-REALLOCATION-at-a-different-length path — not just the equal-size adjacency flip —
   is covered on both backends.
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
