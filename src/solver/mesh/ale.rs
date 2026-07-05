//! ALE swept-face mesh fluxes + the f32 SCL closure (M3.2 of the
//! meshless/moving-mesh roadmap).
//!
//! **SCL by construction**: the per-face volumetric mesh flux `V̇_f` is
//! computed GEOMETRICALLY from the old→new vertex positions (the signed area
//! of the quad each face sweeps over the step), never from a mesh velocity
//! dotted with a face normal. For a polygonal cell whose vertices move
//! linearly over the step, the swept areas telescope exactly:
//!
//! ```text
//!   Σ_f σ_{i,f} · A_swept(f)  =  V_i^{n+1} − V_i^n        (exact, f64)
//! ```
//!
//! (σ = +1 when cell i owns the face, −1 otherwise): both sides are sums of
//! the same vertex cross products — `shoelace(P^{n+1}) − shoelace(P^n)`
//! expands into per-edge quad areas whose `p_v^n × p_v^{n+1}` diagonal terms
//! cancel between edge-adjacent quads. The unit test
//! `swept_quads_telescope_to_volume_change` pins this identity to f64
//! roundoff.
//!
//! **The f32 closure** (review-solver-ale F1): the solver kernels see
//! f32-rounded volumes and f32 mesh fluxes, so after casting, the identity
//! only holds to independent-rounding noise. A per-cell "fix the largest
//! face" repair is ill-posed — interior faces are shared with opposite signs,
//! so closing cell A perturbs cell B (a coupled system `Divᵀδ = e`). Instead
//! we run the deterministic **spanning-forest sweep** over the cell dual
//! graph: roots are boundary-adjacent cells (their boundary face is the slack
//! unknown), every other cell's slack is the face toward its BFS parent;
//! processing children before parents lets each cell close its own defect
//! onto its slack face exactly once, never disturbing an already-closed cell.
//! O(N), sequential, index-ordered ⇒ deterministic across thread counts.
//!
//! Precision of the closure: the per-cell sum of f32 fluxes is evaluated in
//! f64 (exact for the ≲12 f32 terms of a cell), and the slack value is the
//! f32 nearest to the exact requirement (with a ±1-ulp refinement pick). An
//! EXACTLY zero defect is not generally representable — the requirement is an
//! f64 value squeezed into an f32 slot — so the guaranteed bound is ≤0.5 ulp
//! of the slack flux per cell, i.e. f32-roundoff scale. The
//! [`SweptMeshFluxes::max_defect_rel`] diagnostic reports the realized
//! per-cell maximum; the GCL gate asserts it every step.

use super::structs::Mesh;

/// Result of [`swept_mesh_fluxes_closed`].
#[derive(Debug)]
pub struct SweptMeshFluxes {
    /// Per-face volumetric swept rate `V̇_f = A_swept(f)/dt` (Volume/Time),
    /// f32, signed along the stored face normal (owner-outward convention —
    /// exactly the `mesh_fluxes` buffer semantics), SCL-closed per cell.
    pub fluxes: Vec<f32>,
    /// Post-closure per-cell SCL defect, relative:
    /// `max_i |Σ_f σ·flux_f·dt − (V32_i^{n+1} − V32_i^n)| / V_i^{n+1}`
    /// where `V32` are the f32-cast volumes the kernels consume and the sums
    /// run in f64 over the f32 values. f32-roundoff scale by construction.
    pub max_defect_rel: f64,
    /// Pre-closure f64 telescoping-identity residual, relative:
    /// `max_i |Σ_f σ·A_swept − (V_i^{n+1} − V_i^n)| / V_i^{n+1}` — a
    /// diagnostic that the swept-quad geometry itself is consistent
    /// (f64-roundoff scale on any linear vertex motion). On a persistent
    /// topology this is roundoff and hard-asserted; on a FLIP step (via
    /// [`swept_mesh_fluxes_closed_flip`]) it is instead the O(motion·h/V)
    /// **flip defect** the closure repairs onto the slack faces — NOT an error.
    pub max_identity_err_rel: f64,
    /// Number of faces whose swept contribution was FORCED to zero because they
    /// are newborn (a flip's new adjacency, no t^n swept quad). Always 0 on the
    /// persistent path ([`swept_mesh_fluxes_closed`]).
    pub born_faces: usize,
}

/// Cell volumes from vertex positions, replicating the polygon-shoelace part
/// of [`Mesh::recalculate_geometry`] operation-for-operation (same
/// accumulation order), so the f32 casts of these volumes are bit-identical
/// to what a refresh after `recalculate_geometry` uploads.
fn cell_volumes_from(mesh: &Mesh, vx: &[f64], vy: &[f64]) -> Vec<f64> {
    let mut vols = vec![0.0f64; mesh.num_cells()];
    for i in 0..mesh.num_cells() {
        let start = mesh.cell_vertex_offsets[i];
        let end = mesh.cell_vertex_offsets[i + 1];
        let n = end - start;
        let mut signed_area = 0.0;
        for k in 0..n {
            let idx0 = mesh.cell_vertices[start + k];
            let idx1 = mesh.cell_vertices[start + (k + 1) % n];
            signed_area += vx[idx0] * vy[idx1] - vx[idx1] * vy[idx0];
        }
        vols[i] = (signed_area * 0.5).abs();
    }
    vols
}

/// Signed swept area of face `f` along its stored normal, f64.
///
/// The quad (v1_old, v2_old, v2_new, v1_new) has shoelace area
/// `½·cross(p2n − p1o, p1n − p2o)` (half the diagonal cross product — exact
/// for any quadrilateral). Its sign is tied to the (v1,v2) tangent ordering:
/// motion along the 90°-CCW rotation of the tangent gives positive shoelace.
/// The stored normal is one of the two tangent perpendiculars, so
/// `sign(cross(t, n̂))` maps the quad sign onto the owner-outward normal
/// convention (verified structurally by the telescoping unit test — a global
/// sign error flips the identity's sign).
fn swept_area_along_normal(
    mesh: &Mesh,
    face: usize,
    old_vx: &[f64],
    old_vy: &[f64],
    allow_degenerate: bool,
) -> Result<f64, String> {
    let v1 = mesh.face_v1[face];
    let v2 = mesh.face_v2[face];
    let (p1ox, p1oy) = (old_vx[v1], old_vy[v1]);
    let (p2ox, p2oy) = (old_vx[v2], old_vy[v2]);
    let (p1nx, p1ny) = (mesh.vx[v1], mesh.vy[v1]);
    let (p2nx, p2ny) = (mesh.vx[v2], mesh.vy[v2]);

    // ½ · cross(p2n − p1o, p1n − p2o)
    let dx1 = p2nx - p1ox;
    let dy1 = p2ny - p1oy;
    let dx2 = p1nx - p2ox;
    let dy2 = p1ny - p2oy;
    let quad = 0.5 * (dx1 * dy2 - dy1 * dx2);

    // Orient onto the stored (owner-outward) normal via the OLD tangent.
    let tx = p2ox - p1ox;
    let ty = p2oy - p1oy;
    let cross_tn = tx * mesh.face_ny[face] - ty * mesh.face_nx[face];
    if cross_tn == 0.0 {
        // A zero-length old face or a normal parallel to the old tangent — a
        // collapsing (near-flip) face with no reliable swept quad. In the flip
        // path the closure absorbs a zero swept contribution onto the slack face
        // (per-cell conservation preserved), so tolerate it there; the
        // persistent path must still hard-fail (its telescoping identity is the
        // load-bearing correctness check).
        if allow_degenerate {
            return Ok(0.0);
        }
        return Err(format!(
            "swept_mesh_fluxes: face {face} has a degenerate tangent/normal pair \
             (zero-length face or normal parallel to tangent)"
        ));
    }
    Ok(if cross_tn > 0.0 { quad } else { -quad })
}

/// Compute the per-face ALE mesh fluxes for one step of prescribed vertex
/// motion, f64 swept-quad geometry + deterministic f32 SCL closure.
///
/// * `mesh` — the mesh at its NEW (t^{n+1}) vertex positions, with
///   [`Mesh::recalculate_geometry`] already run (volumes/normals current).
/// * `old_vx`/`old_vy` — the vertex positions at t^n (before the move).
/// * `dt` — the step the solver will take with these fluxes (the exact value
///   passed to `set_dt`, f64-widened).
///
/// The closure targets the f32-cast volumes — `(f32(V^{n+1}) − f32(V^n))/dt`
/// per cell, evaluated in f64 — because those are the values the kernels'
/// `cell_vols` / `cell_vols_old` buffers hold (the old volumes are recomputed
/// here with the exact `recalculate_geometry` shoelace, so their f32 casts
/// match the previously-uploaded buffer bit-for-bit).
///
/// Errors on meshes with a boundary-free connected component (a closure slack
/// face cannot be chosen — periodic domains are out of ALE v1 scope).
///
/// PERSISTENT-topology entry: the swept-quad geometry must be self-consistent,
/// so the f64 telescoping identity is HARD-asserted (>1e-9 ⇒ Err). For a
/// Voronoi FLIP (born/dead faces have no swept quad), use
/// [`swept_mesh_fluxes_closed_flip`] instead.
pub fn swept_mesh_fluxes_closed(
    mesh: &Mesh,
    old_vx: &[f64],
    old_vy: &[f64],
    dt: f64,
) -> Result<SweptMeshFluxes, String> {
    swept_closed_impl(mesh, old_vx, old_vy, None, dt)
}

/// Flip-aware swept-flux construction (roadmap M4 / review-solver-ale R2): the
/// conservative-remap flux path across a Voronoi RE-TESSELLATION.
///
/// `born_mask[f]` (length `mesh.num_faces()`) marks each NEW-mesh face whose
/// `(i,j)` adjacency did not exist at t^n — a **born** face with no swept quad.
/// Born faces are given **zero** swept contribution; every persistent face gets
/// its swept quad from the aligned old vertex positions (born vertices carry
/// their new position ⇒ locally partial sweep). The residual this leaves per
/// cell — `e_i = ΔV_i/dt − Σ_persistent σ·swept` — is exactly what the
/// spanning-forest closure distributes onto each cell's slack face, so
/// `Σ_f σ·mesh_flux_f = ΔV_i/dt` holds **per cell** (⇒ GCL / free-stream
/// preservation survives the flip) and **globally** (interior slacks cancel;
/// only true boundary faces feed the total — mass conserved). DIED faces have
/// no new-mesh representative, so their old flux mass is not carried face-to-face;
/// it is absorbed into the exact per-cell balance of the surviving new faces.
///
/// The per-FACE flux on and around a flip is only locally first-order accurate
/// (the slack face soaks the whole `e_i`); this is the roadmap's accepted flip
/// cost. The pre-closure `max_identity_err_rel` IS that per-cell flip defect —
/// reported as the always-on diagnostic, NOT hard-asserted (born faces are
/// EXPECTED to break the telescoping identity). The post-closure
/// `max_defect_rel` is still f32-roundoff by construction: the closure is the
/// correctness guarantee.
///
/// Preconditions: same as [`swept_mesh_fluxes_closed`] (non-periodic, valid dt,
/// boundary-connected dual graph). A born face is a perfectly valid slack edge
/// for a BFS parent — the forest closure is unchanged.
///
/// `old_cell_vol` are the ACTUAL t^n cell volumes (the previous mesh's
/// `cell_vol`, == what the kernel's `cell_vols_old` buffer holds after the M2
/// rotation). This is load-bearing: on a flip the new ring's aligned old
/// positions do NOT reconstruct the old cell's polygon (the born vertices carry
/// their new position), so the ring-reconstructed old volume would spuriously
/// equal the NEW volume and the closure would target a zero ΔV — missing the
/// real volume change. Passing the actual old volumes makes the per-cell target
/// `(V_i^{n+1} − V_i^n)/dt` consistent with the moving-volume ddt, which is
/// exactly the GCL condition. Cell `i` == cell `i` across the regen (fixed
/// seeds), so the volumes are directly comparable.
///
/// `hard_assert_exclude` (review July 2026, F1): when `Some(exclude)`, the caller
/// asserts this step is NOT a genuine adjacency flip — the only faces forced to
/// zero are geometrically DEGENERATE (sliver) faces — so the load-bearing >1e-9
/// telescoping-identity check must stay LIVE on every cell NOT incident to such a
/// face (`exclude[i] == false`). This prevents a single sliver face from disabling
/// the whole-step identity guard (which would let the f32 closure launder an
/// arbitrarily-wrong per-face swept flux into a correct per-cell sum). `None`
/// (a genuine flip, or the windmill unit test) keeps the relaxed check: born faces
/// are EXPECTED to break the identity and the closure alone is the guarantee.
pub fn swept_mesh_fluxes_closed_flip(
    mesh: &Mesh,
    old_vx: &[f64],
    old_vy: &[f64],
    old_cell_vol: &[f64],
    born_mask: &[bool],
    hard_assert_exclude: Option<&[bool]>,
    dt: f64,
) -> Result<SweptMeshFluxes, String> {
    if born_mask.len() != mesh.num_faces() {
        return Err(format!(
            "swept_mesh_fluxes_closed_flip: born_mask length {} != num_faces {}",
            born_mask.len(),
            mesh.num_faces()
        ));
    }
    if old_cell_vol.len() != mesh.num_cells() {
        return Err(format!(
            "swept_mesh_fluxes_closed_flip: old_cell_vol length {} != num_cells {}",
            old_cell_vol.len(),
            mesh.num_cells()
        ));
    }
    if let Some(ex) = hard_assert_exclude {
        if ex.len() != mesh.num_cells() {
            return Err(format!(
                "swept_mesh_fluxes_closed_flip: hard_assert_exclude length {} != num_cells {}",
                ex.len(),
                mesh.num_cells()
            ));
        }
    }
    swept_closed_impl(
        mesh,
        old_vx,
        old_vy,
        Some((born_mask, old_cell_vol, hard_assert_exclude)),
        dt,
    )
}

/// Shared core for the persistent ([`swept_mesh_fluxes_closed`]) and flip
/// ([`swept_mesh_fluxes_closed_flip`]) paths. `born_mask = None` is the
/// persistent path (hard identity check, no forced-zero faces); `Some(mask)`
/// forces the marked faces' swept area to zero and RELAXES the identity check
/// (the flip defect is the diagnostic, the forest closure the guarantee).
#[allow(clippy::type_complexity)]
fn swept_closed_impl(
    mesh: &Mesh,
    old_vx: &[f64],
    old_vy: &[f64],
    flip: Option<(&[bool], &[f64], Option<&[bool]>)>,
    dt: f64,
) -> Result<SweptMeshFluxes, String> {
    let born_mask: Option<&[bool]> = flip.map(|(m, _, _)| m);
    // Some(exclude): a degeneracy-only step — hard-assert the telescoping identity
    // on every cell NOT incident to a forced-degenerate face (review July 2026 F1).
    let hard_assert_exclude: Option<&[bool]> = flip.and_then(|(_, _, ex)| ex);
    let num_cells = mesh.num_cells();
    let num_faces = mesh.num_faces();
    if old_vx.len() != mesh.vx.len() || old_vy.len() != mesh.vy.len() {
        return Err(format!(
            "swept_mesh_fluxes: old vertex arrays ({}, {}) do not match the mesh ({} vertices)",
            old_vx.len(),
            old_vy.len(),
            mesh.vx.len()
        ));
    }
    if !mesh.face_wrap_shift.is_empty() {
        return Err("swept_mesh_fluxes: periodic meshes are unsupported under mesh motion".into());
    }
    if dt <= 0.0 || !dt.is_finite() {
        return Err(format!("swept_mesh_fluxes: invalid dt {dt}"));
    }

    // ── 1. f64 swept areas + telescoping-identity diagnostic ──────────────
    // Born faces (flip path) carry NO swept quad — their contribution is zero
    // and the per-cell residual they leave is closed onto the slack face below.
    // (Skipping the geometry also avoids a spurious degenerate-tangent error on
    // a born face whose t^n endpoints collapsed to the born vertex position.)
    let mut swept = vec![0.0f64; num_faces];
    for f in 0..num_faces {
        if let Some(mask) = born_mask {
            if mask[f] {
                continue;
            }
        }
        swept[f] = swept_area_along_normal(mesh, f, old_vx, old_vy, born_mask.is_some())?;
    }

    // Old cell volumes for the closure target + identity diagnostic. Persistent
    // path: reconstruct from the aligned old ring (byte-identical to the M3
    // behaviour, and == the actual old volume because the topology is the same).
    // Flip path: the caller-supplied ACTUAL old volumes (the new ring's aligned
    // old positions do NOT reproduce the old polygon across a flip — see
    // `swept_mesh_fluxes_closed_flip`).
    let recon_old_vols = cell_volumes_from(mesh, old_vx, old_vy);
    let old_vols: &[f64] = match flip {
        Some((_, actual, _)) => actual,
        None => &recon_old_vols,
    };
    let sign = |cell: usize, face: usize| -> f64 {
        if mesh.face_owner[face] == cell {
            1.0
        } else {
            -1.0
        }
    };

    let mut max_identity_err_rel = 0.0f64;
    // On a degeneracy-only step, the max identity residual over cells NOT incident
    // to a forced-degenerate face — the subset on which the hard check stays live.
    let mut max_hard_err_rel = 0.0f64;
    for i in 0..num_cells {
        let mut sum = 0.0f64;
        for k in mesh.cell_face_offsets[i]..mesh.cell_face_offsets[i + 1] {
            let f = mesh.cell_faces[k];
            sum += sign(i, f) * swept[f];
        }
        let dv = mesh.cell_vol[i] - old_vols[i];
        let err = (sum - dv).abs() / mesh.cell_vol[i].max(f64::MIN_POSITIVE);
        max_identity_err_rel = max_identity_err_rel.max(err);
        if let Some(exclude) = hard_assert_exclude {
            if !exclude[i] {
                max_hard_err_rel = max_hard_err_rel.max(err);
            }
        }
    }

    // The identity is f64-roundoff exact (≲1e-13 measured) whenever the
    // inputs are consistent: same topology, `recalculate_geometry` volumes,
    // linear vertex motion. A violation means the swept-quad geometry does
    // NOT describe the actual volume change (wrong old positions, stale
    // volumes, inverted/degenerate cells — `cell_volumes_from` takes |·|, so
    // an inversion shows up HERE, not in the volumes). Failing is load-bearing
    // (adversarial review, July 2026): the f32 closure downstream would
    // otherwise silently "repair" arbitrarily wrong fluxes to the per-cell
    // sums — free-stream preservation only senses those sums, so the GCL gate
    // would stay green while the per-face flux distribution is garbage. The
    // 1e-9 threshold is ~3-4 orders looser than roundoff and ~orders tighter
    // than any real defect.
    //
    // FLIP path (born_mask = Some): the born faces deliberately zero out their
    // swept quads, so the identity is EXPECTED to be violated by exactly the
    // flip defect e_i — that IS the diagnostic (returned as
    // `max_identity_err_rel`), not an error. The forest closure below still
    // makes the per-cell sums exact, so GCL is preserved; a truly garbage
    // input (an inverted cell) instead shows up as a non-finite defect, which
    // the closure would propagate to a non-finite flux — guarded here.
    if born_mask.is_none() && max_identity_err_rel > 1e-9 {
        return Err(format!(
            "swept_mesh_fluxes: telescoping identity violated (max rel residual {:.3e} > 1e-9): \
             the swept-quad areas do not sum to the per-cell volume changes. The old/new vertex \
             positions are inconsistent with the mesh geometry (stale recalculate_geometry, \
             wrong old positions, or inverted/degenerate cells)",
            max_identity_err_rel
        ));
    }
    // Degeneracy-only step (F1): the only forced-zero faces are slivers, so the
    // identity is still load-bearing on every cell away from them. Keep the hard
    // check there — a sliver must not silently disable it for the whole step.
    if hard_assert_exclude.is_some() && max_hard_err_rel > 1e-9 {
        return Err(format!(
            "swept_mesh_fluxes: telescoping identity violated on a non-degenerate cell (max rel \
             residual {:.3e} > 1e-9 away from the forced-degenerate faces): a degenerate face \
             was forced born, but the swept-quad geometry is inconsistent on cells that are NOT \
             incident to one — the sliver did not cause it (stale recalculate_geometry, wrong \
             old positions, or an inverted cell)",
            max_hard_err_rel
        ));
    }
    if born_mask.is_some() && !max_identity_err_rel.is_finite() {
        return Err(format!(
            "swept_mesh_fluxes_closed_flip: non-finite pre-closure flip defect ({}): a cell is \
             inverted or degenerate at the flip",
            max_identity_err_rel
        ));
    }

    // ── 2. f32 cast ────────────────────────────────────────────────────────
    let mut fluxes: Vec<f32> = swept.iter().map(|&a| (a / dt) as f32).collect();

    // Per-cell closure target: the f32 volumes the kernels consume, in f64.
    let target: Vec<f64> = (0..num_cells)
        .map(|i| ((mesh.cell_vol[i] as f32) as f64 - (old_vols[i] as f32) as f64) / dt)
        .collect();

    // ── 3. deterministic spanning-forest closure ──────────────────────────
    // Roots: boundary-adjacent cells; slack = their first boundary face (in
    // cell_faces order). Everyone else: slack = the face toward the BFS
    // parent. BFS explores cells in index order, faces in cell_faces order.
    const UNSET: usize = usize::MAX;
    let mut slack_face = vec![UNSET; num_cells];
    let mut visited = vec![false; num_cells];
    let mut order: Vec<usize> = Vec::with_capacity(num_cells);
    let mut queue = std::collections::VecDeque::new();

    for i in 0..num_cells {
        for k in mesh.cell_face_offsets[i]..mesh.cell_face_offsets[i + 1] {
            let f = mesh.cell_faces[k];
            if mesh.face_neighbor[f].is_none() {
                slack_face[i] = f;
                visited[i] = true;
                queue.push_back(i);
                break;
            }
        }
    }
    while let Some(i) = queue.pop_front() {
        order.push(i);
        for k in mesh.cell_face_offsets[i]..mesh.cell_face_offsets[i + 1] {
            let f = mesh.cell_faces[k];
            let other = match mesh.face_neighbor[f] {
                Some(n) => {
                    if mesh.face_owner[f] == i {
                        n
                    } else {
                        mesh.face_owner[f]
                    }
                }
                None => continue,
            };
            if !visited[other] {
                visited[other] = true;
                slack_face[other] = f;
                queue.push_back(other);
            }
        }
    }
    if order.len() != num_cells {
        return Err(format!(
            "swept_mesh_fluxes: {} cells are not connected to a boundary face — the SCL \
             closure has no slack there (periodic/closed components are out of ALE v1 scope)",
            num_cells - order.len()
        ));
    }

    // Children before parents: reverse BFS discovery order. Each cell's slack
    // face is adjusted exactly once; a child's slack faces are never touched
    // by later (parent/root) adjustments.
    for &i in order.iter().rev() {
        let adj = slack_face[i];
        let s_adj = sign(i, adj);
        let mut others = 0.0f64;
        for k in mesh.cell_face_offsets[i]..mesh.cell_face_offsets[i + 1] {
            let f = mesh.cell_faces[k];
            if f != adj {
                others += sign(i, f) * fluxes[f] as f64;
            }
        }
        let needed = (target[i] - others) * s_adj;
        let v0 = needed as f32;
        // ±1-ulp refinement: pick the candidate minimizing the exact defect.
        let mut best = v0;
        let mut best_err = f64::INFINITY;
        for cand in [
            v0,
            f32::from_bits(v0.to_bits().wrapping_add(1)),
            f32::from_bits(v0.to_bits().wrapping_sub(1)),
        ] {
            if !cand.is_finite() {
                continue;
            }
            let err = (others + s_adj * cand as f64 - target[i]).abs();
            if err < best_err {
                best_err = err;
                best = cand;
            }
        }
        fluxes[adj] = best;
    }

    // ── 4. post-closure defect diagnostic ──────────────────────────────────
    let mut max_defect_rel = 0.0f64;
    for i in 0..num_cells {
        let mut sum = 0.0f64;
        for k in mesh.cell_face_offsets[i]..mesh.cell_face_offsets[i + 1] {
            let f = mesh.cell_faces[k];
            sum += sign(i, f) * fluxes[f] as f64;
        }
        let defect = ((sum - target[i]) * dt).abs() / mesh.cell_vol[i].max(f64::MIN_POSITIVE);
        max_defect_rel = max_defect_rel.max(defect);
    }

    let born_faces = born_mask
        .map(|m| m.iter().filter(|&&b| b).count())
        .unwrap_or(0);
    Ok(SweptMeshFluxes {
        fluxes,
        max_defect_rel,
        max_identity_err_rel,
        born_faces,
    })
}

/// Topology-flip report between two same-cell-count meshes (roadmap M4 flip
/// detection): which NEW-mesh faces were BORN (an `(i,j)` adjacency absent at
/// t^n), how many DIED (an old adjacency with no new face), and which cells a
/// flip touched. Adjacency is keyed by the incident cell/seed pair — interior
/// faces by the sorted `(owner,neighbor)` pair, boundary faces by
/// `(owner, BOUNDARY)`; seed `i` == cell `i` across the regen, so these pairs
/// are directly comparable.
#[derive(Debug, Clone)]
pub struct FlipReport {
    /// `born_face_mask[f]` — the NEW-mesh face `f`'s adjacency did not exist at
    /// t^n. Feeds [`swept_mesh_fluxes_closed_flip`] verbatim.
    pub born_face_mask: Vec<bool>,
    /// Number of born faces (`= born_face_mask.iter().filter(..).count()`).
    pub born_faces: usize,
    /// Number of old adjacencies with no counterpart in the new mesh.
    pub died_faces: usize,
    /// Number of NEW-mesh cells incident to at least one born face
    /// (`= cell_flipped.iter().filter(..).count()`).
    pub flipped_cells: usize,
    /// Per-cell flag: cell `c` is incident to at least one born face — the cells
    /// whose telescoping identity a flip (or a forced-degenerate face) legitimately
    /// breaks. Consumed by the driver both as the flip-rate telemetry source and as
    /// the hard-assert EXCLUDE set on a degeneracy-only step (the identity stays
    /// enforced on every cell NOT in this set).
    pub cell_flipped: Vec<bool>,
}

impl FlipReport {
    /// Whether the topology actually flipped (any born or died face). `false`
    /// ⇒ the persistent-topology swept path applies (adjacency unchanged; a
    /// pure face-array REORDER is not a flip).
    pub fn is_flip(&self) -> bool {
        self.born_faces != 0 || self.died_faces != 0
    }
}

/// The adjacency key of a face: `(min(o,n), max(o,n))` for an interior face,
/// `(owner, BOUNDARY_KEY)` for a boundary face. Seed `i` == cell `i`, so the
/// key is stable across a regen.
const BOUNDARY_KEY: usize = usize::MAX;
fn face_adjacency_key(mesh: &Mesh, f: usize) -> (usize, usize) {
    let o = mesh.face_owner[f];
    match mesh.face_neighbor[f] {
        Some(nb) => (o.min(nb), o.max(nb)),
        None => (o, BOUNDARY_KEY),
    }
}

/// Detect Voronoi topology flips between the t^n mesh (`old`) and the
/// regenerated t^{n+1} mesh (`new`), producing the born-face mask
/// [`swept_mesh_fluxes_closed_flip`] consumes plus the flip diagnostics. O(faces).
///
/// A flip is an ADJACENCY change: `new` gained an `(i,j)` face pair that `old`
/// lacked (born) and/or lost one `old` had (died). A pure reordering of the
/// deterministic face emission — same adjacency set — is NOT a flip (empty
/// born mask), so the persistent swept path stays byte-stable through reorders.
pub fn detect_flips(old_mesh: &Mesh, new_mesh: &Mesh) -> Result<FlipReport, String> {
    if old_mesh.num_cells() != new_mesh.num_cells() {
        return Err(format!(
            "detect_flips: cell counts differ ({} old vs {} new) — v1 ALE is fixed-seed",
            old_mesh.num_cells(),
            new_mesh.num_cells()
        ));
    }
    let old_keys: std::collections::HashSet<(usize, usize)> = (0..old_mesh.num_faces())
        .map(|f| face_adjacency_key(old_mesh, f))
        .collect();
    let new_keys: std::collections::HashSet<(usize, usize)> = (0..new_mesh.num_faces())
        .map(|f| face_adjacency_key(new_mesh, f))
        .collect();

    let mut born_face_mask = vec![false; new_mesh.num_faces()];
    let mut born_faces = 0usize;
    let mut cell_flipped = vec![false; new_mesh.num_cells()];
    for f in 0..new_mesh.num_faces() {
        if !old_keys.contains(&face_adjacency_key(new_mesh, f)) {
            born_face_mask[f] = true;
            born_faces += 1;
            cell_flipped[new_mesh.face_owner[f]] = true;
            if let Some(nb) = new_mesh.face_neighbor[f] {
                cell_flipped[nb] = true;
            }
        }
    }
    let died_faces = old_keys.iter().filter(|k| !new_keys.contains(k)).count();
    let flipped_cells = cell_flipped.iter().filter(|&&b| b).count();

    Ok(FlipReport {
        born_face_mask,
        born_faces,
        died_faces,
        flipped_cells,
        cell_flipped,
    })
}

/// The old→new vertex correspondence across a Voronoi **regeneration**, keyed
/// by the incident-seed SET (roadmap R2: a Voronoi vertex ≡ the set of seeds
/// meeting at it — a triple point for three cells, more at degeneracies). Seed
/// `i` == cell `i`, so a vertex's incident-seed set is exactly the set of cells
/// whose `cell_vertices` ring contains it.
///
/// Returns arrays sized to `new_mesh.num_vertices()`: `old_vx_aligned[nv]` /
/// `old_vy_aligned[nv]` are the position **at t^n** of the vertex `new_mesh`
/// vertex `nv` corresponds to, ready to hand straight to
/// [`swept_mesh_fluxes_closed`] as its `old_vx`/`old_vy` (that function indexes
/// the old arrays by the NEW mesh's vertex ids, so this is precisely the shape
/// it needs). `unmatched` counts new vertices whose incident-seed set did NOT
/// exist in `old_mesh` — a topology flip (a born vertex with no t^n
/// counterpart); it is 0 for a persistent topology.
///
/// **Why this makes the telescoping identity hold** ([`swept_mesh_fluxes_closed`]
/// asserts it): with a per-new-vertex old position, every vertex shared around a
/// cell's ring carries ONE old position, so the swept quads tile the annulus
/// between the cell's old and new polygons exactly. For a persistent topology
/// the cyclic neighbour order around each cell is preserved, so the new ring's
/// old positions reproduce the old polygon and the per-cell swept areas sum to
/// the true ΔV. A mismatch (bad correspondence / an undetected flip) shows up as
/// a blown telescoping residual there — the load-bearing check.
///
/// Determinism: new vertices are processed in index order; among old vertices
/// sharing a seed set (a degenerate collision — two triple points on the same
/// three seeds) the nearest to the new position wins, ties broken by the lower
/// old index. No map iteration feeds the output.
pub fn align_old_vertices_by_seed_set(
    old_mesh: &Mesh,
    new_mesh: &Mesh,
) -> Result<(Vec<f64>, Vec<f64>, usize), String> {
    if old_mesh.num_cells() != new_mesh.num_cells() {
        return Err(format!(
            "align_old_vertices: cell counts differ ({} old vs {} new) — v1 ALE is fixed-seed",
            old_mesh.num_cells(),
            new_mesh.num_cells()
        ));
    }
    let old_sets = vertex_incident_cells(old_mesh);
    let new_sets = vertex_incident_cells(new_mesh);

    // seed set → old vertex ids (ascending; insertion order = old vertex order).
    let mut old_by_set: std::collections::HashMap<Vec<usize>, Vec<usize>> =
        std::collections::HashMap::with_capacity(old_mesh.num_vertices());
    for (ov, set) in old_sets.iter().enumerate() {
        old_by_set.entry(set.clone()).or_default().push(ov);
    }

    let nv = new_mesh.num_vertices();
    let mut ovx = vec![0.0f64; nv];
    let mut ovy = vec![0.0f64; nv];
    let mut unmatched = 0usize;
    for v in 0..nv {
        let (nx, ny) = (new_mesh.vx[v], new_mesh.vy[v]);
        match old_by_set.get(&new_sets[v]) {
            None => {
                // A born vertex (flip): no t^n counterpart. Leave the old
                // position at the new one (zero local sweep) and report it —
                // the caller treats any unmatched as a deferred flip; if it
                // proceeds, the telescoping check will reject the inconsistency.
                ovx[v] = nx;
                ovy[v] = ny;
                unmatched += 1;
            }
            Some(cands) => {
                let mut best = cands[0];
                let mut best_d2 = f64::INFINITY;
                for &ov in cands {
                    let dx = old_mesh.vx[ov] - nx;
                    let dy = old_mesh.vy[ov] - ny;
                    let d2 = dx * dx + dy * dy;
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = ov;
                    }
                }
                ovx[v] = old_mesh.vx[best];
                ovy[v] = old_mesh.vy[best];
            }
        }
    }
    Ok((ovx, ovy, unmatched))
}

/// Per-vertex sorted list of the cells (== seeds) incident to it — the set that
/// keys the [`align_old_vertices_by_seed_set`] correspondence.
fn vertex_incident_cells(mesh: &Mesh) -> Vec<Vec<usize>> {
    let mut sets = vec![Vec::<usize>::new(); mesh.num_vertices()];
    for c in 0..mesh.num_cells() {
        for k in mesh.cell_vertex_offsets[c]..mesh.cell_vertex_offsets[c + 1] {
            sets[mesh.cell_vertices[k]].push(c);
        }
    }
    for s in &mut sets {
        s.sort_unstable();
        s.dedup();
    }
    sets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType};

    fn test_mesh() -> Mesh {
        generate_structured_rect_mesh(
            24,
            16,
            1.5,
            1.0,
            BoundarySides {
                left: BoundaryType::Inlet,
                right: BoundaryType::Outlet,
                bottom: BoundaryType::Wall,
                top: BoundaryType::Wall,
            },
        )
    }

    /// Smooth interior bump displacement, zero on the domain boundary.
    fn displace(mesh: &mut Mesh, amp: f64, lx: f64, ly: f64) -> (Vec<f64>, Vec<f64>) {
        let old_vx = mesh.vx.clone();
        let old_vy = mesh.vy.clone();
        for v in 0..mesh.num_vertices() {
            let (x, y) = (old_vx[v], old_vy[v]);
            let bump = (std::f64::consts::PI * x / lx).sin().powi(2)
                * (std::f64::consts::PI * y / ly).sin().powi(2);
            mesh.vx[v] = x + amp * bump;
            mesh.vy[v] = y - 0.6 * amp * bump;
        }
        mesh.recalculate_geometry();
        (old_vx, old_vy)
    }

    /// The load-bearing f64 identity: swept quads telescope to the exact
    /// per-cell volume change (this also pins the owner-outward sign
    /// convention — a global sign flip would give −ΔV).
    #[test]
    fn swept_quads_telescope_to_volume_change() {
        let mut mesh = test_mesh();
        let h = 1.5 / 24.0;
        let (ovx, ovy) = displace(&mut mesh, 0.2 * h, 1.5, 1.0);
        let out = swept_mesh_fluxes_closed(&mesh, &ovx, &ovy, 1e-2).expect("swept fluxes");
        println!(
            "[ale-scl] f64 identity residual (rel) = {:.3e}",
            out.max_identity_err_rel
        );
        assert!(
            out.max_identity_err_rel < 1e-12,
            "telescoping identity violated: {:.3e}",
            out.max_identity_err_rel
        );
    }

    /// The f32 closure: per-cell defect at f32-roundoff scale (measured
    /// ~1e-11..1e-10 relative; asserted with margin), deterministic.
    #[test]
    fn f32_closure_defect_at_roundoff_scale() {
        let mut mesh = test_mesh();
        let h = 1.5 / 24.0;
        let dt = 5e-3;
        let (ovx, ovy) = displace(&mut mesh, 0.15 * h, 1.5, 1.0);
        let out = swept_mesh_fluxes_closed(&mesh, &ovx, &ovy, dt).expect("swept fluxes");
        println!(
            "[ale-scl] post-closure max defect (rel) = {:.3e}",
            out.max_defect_rel
        );
        // Scale: defect ≤ ~0.5 ulp of the slack face's flux × dt / V. This
        // test moves by the full 0.15·h amplitude in one step, so
        // A_swept ~ 0.1·V and the ulp bound is ~0.1·V·2⁻²⁴ ≈ 6e-9·V.
        // Measured 4.8e-9 (July 2026); asserted at ~4× measured.
        assert!(
            out.max_defect_rel < 2e-8,
            "f32 SCL closure defect too large: {:.3e}",
            out.max_defect_rel
        );

        // Determinism: same inputs, same bits.
        let out2 = swept_mesh_fluxes_closed(&mesh, &ovx, &ovy, dt).expect("swept fluxes");
        assert_eq!(
            out.fluxes.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            out2.fluxes.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        );
    }

    /// Inconsistent inputs are REJECTED, not laundered (adversarial review,
    /// July 2026): stale `cell_vol` (geometry not recalculated after the
    /// move) violates the telescoping identity, and the function must error
    /// instead of letting the f32 closure silently repair the per-cell sums.
    /// Note perturbing the OLD positions alone does not violate the identity
    /// (both sides derive from the same old/new vertex sets — the geometry is
    /// self-consistent, just a different motion); staleness of the mesh's own
    /// volumes is the observable inconsistency.
    #[test]
    fn stale_volumes_are_rejected() {
        let mut mesh = test_mesh();
        let h = 1.5 / 24.0;
        let (ovx, ovy) = displace(&mut mesh, 0.2 * h, 1.5, 1.0);
        mesh.cell_vol[10] *= 1.001; // simulate a stale/corrupt volume
        let err = swept_mesh_fluxes_closed(&mesh, &ovx, &ovy, 1e-2)
            .expect_err("stale volumes must be rejected");
        assert!(
            err.contains("telescoping identity violated"),
            "unexpected error: {err}"
        );
    }

    /// The degeneracy-only escape hatch must NOT disable the load-bearing
    /// telescoping-identity assert for the whole step (review July 2026, F1):
    /// when the only forced-zero faces are slivers (no genuine adjacency flip),
    /// the flip path still hard-asserts the identity on every cell NOT incident
    /// to a forced face. So an inconsistency (a wrong old volume) on a FAR cell
    /// is still rejected, while a matching inconsistency on an EXCLUDED
    /// (degen-incident) cell is tolerated.
    #[test]
    fn degenerate_only_step_keeps_identity_assert_on_far_cells() {
        let mut mesh = test_mesh();
        let h = 1.5 / 24.0;
        let (ovx, ovy) = displace(&mut mesh, 0.15 * h, 1.5, 1.0);
        // Persistent topology ⇒ the reconstructed old ring volumes ARE the actual
        // old cell volumes (the closure target the driver would pass).
        let old_vols = cell_volumes_from(&mesh, &ovx, &ovy);

        // Force ONE interior face "degenerate" (born) and exclude its two cells —
        // exactly what the driver does on a sliver step with no genuine flip.
        let f = (0..mesh.num_faces())
            .find(|&f| mesh.face_neighbor[f].is_some())
            .expect("an interior face");
        let mut born = vec![false; mesh.num_faces()];
        born[f] = true;
        let mut exclude = vec![false; mesh.num_cells()];
        exclude[mesh.face_owner[f]] = true;
        exclude[mesh.face_neighbor[f].unwrap()] = true;

        // (a) Consistent geometry ⇒ Ok: the sliver's zeroing is absorbed on its
        //     own cells; every far cell satisfies the identity.
        swept_mesh_fluxes_closed_flip(&mesh, &ovx, &ovy, &old_vols, &born, Some(&exclude), 1e-2)
            .expect("degen-only step with consistent geometry must pass");

        // (b) Corrupt a FAR (non-excluded) cell's old volume ⇒ the hard assert on
        //     the non-degenerate cells fires — the sliver did NOT disable it.
        let far = (0..mesh.num_cells())
            .find(|&c| !exclude[c])
            .expect("a far cell");
        let mut bad_far = old_vols.clone();
        bad_far[far] *= 1.001;
        let err =
            swept_mesh_fluxes_closed_flip(&mesh, &ovx, &ovy, &bad_far, &born, Some(&exclude), 1e-2)
                .expect_err("a far-cell inconsistency must still be rejected");
        assert!(
            err.contains("non-degenerate cell"),
            "unexpected error: {err}"
        );

        // (c) The SAME corruption on an excluded (degen-incident) cell is tolerated
        //     — the identity is legitimately relaxed there.
        let mut bad_near = old_vols.clone();
        bad_near[mesh.face_owner[f]] *= 1.001;
        swept_mesh_fluxes_closed_flip(&mesh, &ovx, &ovy, &bad_near, &born, Some(&exclude), 1e-2)
            .expect("an excluded-cell inconsistency is tolerated (relaxed there)");
    }

    /// No motion ⇒ all-zero fluxes and zero defect (the closure must not
    /// invent fluxes: the slack adjustment of a zero-defect cell is 0.0).
    #[test]
    fn zero_motion_yields_zero_fluxes() {
        let mut mesh = test_mesh();
        let ovx = mesh.vx.clone();
        let ovy = mesh.vy.clone();
        mesh.recalculate_geometry();
        let out = swept_mesh_fluxes_closed(&mesh, &ovx, &ovy, 1e-2).expect("swept fluxes");
        assert!(out.fluxes.iter().all(|&f| f == 0.0), "nonzero flux on a static mesh");
        assert_eq!(out.max_defect_rel, 0.0);
        assert_eq!(out.max_identity_err_rel, 0.0);
    }

    // ── Hand-constructed 2-cell Voronoi flip (deliverable 4) ─────────────────

    /// A specced face of the hand-built windmill mesh.
    struct FaceSpec {
        v1: usize,
        v2: usize,
        owner: usize,
        neighbor: Option<usize>,
    }

    /// Assemble a valid FV `Mesh` from an explicit vertex list, per-cell CCW
    /// vertex rings, and face specs (shared-edge endpoints + owner/neighbor).
    /// Boundary faces (`neighbor = None`) are tagged `Wall`. Face normals are
    /// seeded owner-outward so `recalculate_geometry` keeps the sign.
    fn assemble_windmill(
        verts: &[(f64, f64)],
        cell_rings: &[Vec<usize>],
        faces: &[FaceSpec],
    ) -> Mesh {
        let mut m = Mesh::new();
        m.vx = verts.iter().map(|v| v.0).collect();
        m.vy = verts.iter().map(|v| v.1).collect();
        m.v_fixed = vec![false; verts.len()];
        let nf = faces.len();
        for fs in faces {
            m.face_v1.push(fs.v1);
            m.face_v2.push(fs.v2);
            m.face_owner.push(fs.owner);
            m.face_neighbor.push(fs.neighbor);
            m.face_boundary.push(if fs.neighbor.is_none() {
                Some(BoundaryType::Wall)
            } else {
                None
            });
        }
        m.face_nx = vec![0.0; nf];
        m.face_ny = vec![0.0; nf];
        m.face_area = vec![0.0; nf];
        m.face_cx = vec![0.0; nf];
        m.face_cy = vec![0.0; nf];
        let nc = cell_rings.len();
        m.cell_cx = vec![0.0; nc];
        m.cell_cy = vec![0.0; nc];
        m.cell_vol = vec![0.0; nc];
        for ring in cell_rings {
            m.cell_vertex_offsets.push(m.cell_vertices.len());
            m.cell_vertices.extend_from_slice(ring);
        }
        m.cell_vertex_offsets.push(m.cell_vertices.len());
        for c in 0..nc {
            m.cell_face_offsets.push(m.cell_faces.len());
            for (fi, fs) in faces.iter().enumerate() {
                if fs.owner == c || fs.neighbor == Some(c) {
                    m.cell_faces.push(fi);
                }
            }
        }
        m.cell_face_offsets.push(m.cell_faces.len());
        // Rough ring-average centroids to orient the seed normals outward.
        let cent: Vec<(f64, f64)> = cell_rings
            .iter()
            .map(|ring| {
                let n = ring.len() as f64;
                let sx: f64 = ring.iter().map(|&v| verts[v].0).sum();
                let sy: f64 = ring.iter().map(|&v| verts[v].1).sum();
                (sx / n, sy / n)
            })
            .collect();
        for (fi, fs) in faces.iter().enumerate() {
            let (x1, y1) = verts[fs.v1];
            let (x2, y2) = verts[fs.v2];
            let (tx, ty) = (x2 - x1, y2 - y1);
            let (mut nx, mut ny) = (ty, -tx);
            let (ox, oy) = cent[fs.owner];
            let (mx, my) = ((x1 + x2) * 0.5, (y1 + y2) * 0.5);
            if nx * (mx - ox) + ny * (my - oy) < 0.0 {
                nx = -nx;
                ny = -ny;
            }
            m.face_nx[fi] = nx;
            m.face_ny[fi] = ny;
        }
        m.recalculate_geometry();
        m
    }

    /// The unit box [0,1]² tessellated into 4 cells (W=0, E=1, S=2, N=3) meeting
    /// near the centre, with the central adjacency oriented one of two ways —
    /// exactly the 4-seed cocircular Voronoi flip. `horizontal=true` ⇒ N–S share
    /// the central face (W,E not adjacent); `false` ⇒ W–E share it (N,S not
    /// adjacent). `d` is the half-separation of the two central triple points.
    fn windmill(horizontal: bool, d: f64) -> Mesh {
        // Corners: bl=0, br=1, tr=2, tl=3.
        let bl = 0;
        let br = 1;
        let tr = 2;
        let tl = 3;
        if horizontal {
            // Central pair on the horizontal midline: P1=(.5-d,.5)=(W,N,S),
            // P2=(.5+d,.5)=(E,N,S). Shared central face is (N,S).
            let p1 = 4;
            let p2 = 5;
            let verts = vec![
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.5 - d, 0.5),
                (0.5 + d, 0.5),
            ];
            let rings = vec![
                vec![bl, p1, tl],      // W (0): triangle
                vec![br, tr, p2],      // E (1): triangle
                vec![bl, br, p2, p1],  // S (2): quad
                vec![p1, p2, tr, tl],  // N (3): quad
            ];
            let faces = vec![
                FaceSpec { v1: bl, v2: p1, owner: 0, neighbor: Some(2) }, // W-S
                FaceSpec { v1: p1, v2: tl, owner: 0, neighbor: Some(3) }, // W-N
                FaceSpec { v1: br, v2: p2, owner: 1, neighbor: Some(2) }, // E-S
                FaceSpec { v1: tr, v2: p2, owner: 1, neighbor: Some(3) }, // E-N
                FaceSpec { v1: p1, v2: p2, owner: 2, neighbor: Some(3) }, // N-S (central)
                FaceSpec { v1: tl, v2: bl, owner: 0, neighbor: None },    // left
                FaceSpec { v1: bl, v2: br, owner: 2, neighbor: None },    // bottom
                FaceSpec { v1: br, v2: tr, owner: 1, neighbor: None },    // right
                FaceSpec { v1: tr, v2: tl, owner: 3, neighbor: None },    // top
            ];
            assemble_windmill(&verts, &rings, &faces)
        } else {
            // Central pair on the vertical midline: Q1=(.5,.5-d)=(W,E,S),
            // Q2=(.5,.5+d)=(W,E,N). Shared central face is (W,E).
            let q1 = 4;
            let q2 = 5;
            let verts = vec![
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.5, 0.5 - d),
                (0.5, 0.5 + d),
            ];
            let rings = vec![
                vec![bl, q1, q2, tl],  // W (0): quad
                vec![br, tr, q2, q1],  // E (1): quad
                vec![bl, br, q1],      // S (2): triangle
                vec![tl, q2, tr],      // N (3): triangle
            ];
            let faces = vec![
                FaceSpec { v1: bl, v2: q1, owner: 0, neighbor: Some(2) }, // W-S
                FaceSpec { v1: q1, v2: q2, owner: 0, neighbor: Some(1) }, // W-E (central)
                FaceSpec { v1: q2, v2: tl, owner: 0, neighbor: Some(3) }, // W-N
                FaceSpec { v1: br, v2: q1, owner: 1, neighbor: Some(2) }, // E-S
                FaceSpec { v1: tr, v2: q2, owner: 1, neighbor: Some(3) }, // E-N
                FaceSpec { v1: tl, v2: bl, owner: 0, neighbor: None },    // left
                FaceSpec { v1: bl, v2: br, owner: 2, neighbor: None },    // bottom
                FaceSpec { v1: br, v2: tr, owner: 1, neighbor: None },    // right
                FaceSpec { v1: tr, v2: tl, owner: 3, neighbor: None },    // top
            ];
            assemble_windmill(&verts, &rings, &faces)
        }
    }

    /// THE flip unit test (deliverable 4): a known 4-seed cocircular
    /// reconfiguration (N–S adjacency → W–E adjacency). One face is BORN, one
    /// DIES; the flip-aware swept flux must close each cell's defect onto its
    /// slack faces so `Σ_f σ·flux·dt = ΔV_i` holds per cell to f32-closure
    /// roundoff, and globally (Σ boundary flux = 0 ⇒ mass conserved).
    #[test]
    fn flip_windmill_per_cell_closure() {
        let d = 0.15;
        let dt = 1e-2;
        let old_mesh = windmill(true, d); // N–S adjacent
        let new_mesh = windmill(false, d); // W–E adjacent (flipped)

        // Areas are exact: box conserved cell-by-cell up to the flip.
        let area: f64 = new_mesh.cell_vol.iter().sum();
        assert!((area - 1.0).abs() < 1e-12, "box area drifted: {area}");

        // Flip detection: exactly one born (W-E) and one died (N-S) adjacency;
        // the born face touches W and E only.
        let flip = detect_flips(&old_mesh, &new_mesh).expect("detect_flips");
        assert_eq!(flip.born_faces, 1, "expected exactly one born face");
        assert_eq!(flip.died_faces, 1, "expected exactly one died face");
        assert_eq!(flip.flipped_cells, 2, "born face should touch W and E only");
        assert!(flip.is_flip());

        // No-flip control: a mesh against itself is flip-free.
        let self_flip = detect_flips(&new_mesh, &new_mesh).expect("detect_flips self");
        assert_eq!(self_flip.born_faces, 0);
        assert_eq!(self_flip.died_faces, 0);
        assert!(!self_flip.is_flip());

        // Seed-set correspondence: the two central triple points are born.
        let (ovx, ovy, unmatched) =
            align_old_vertices_by_seed_set(&old_mesh, &new_mesh).expect("align");
        assert_eq!(unmatched, 2, "both new central vertices should be born");

        // The persistent path SILENTLY MISSES the flip: with born vertices
        // aligned to their new positions, the new ring's reconstructed old
        // volume equals the new volume, so it reports a zero defect and all-zero
        // fluxes — exactly why the flip needs the actual old volumes.
        let persistent = swept_mesh_fluxes_closed(&new_mesh, &ovx, &ovy, dt).expect("persistent");
        assert!(
            persistent.fluxes.iter().all(|&f| f == 0.0),
            "persistent path should (wrongly) see no motion across a flip"
        );

        // The flip-aware path: actual old volumes as the target, born faces
        // zeroed, defect closed onto slack faces.
        let out = swept_mesh_fluxes_closed_flip(
            &new_mesh,
            &ovx,
            &ovy,
            &old_mesh.cell_vol,
            &flip.born_face_mask,
            None, // genuine flip: relaxed identity check
            dt,
        )
        .expect("flip swept fluxes");
        assert_eq!(out.born_faces, 1);
        println!(
            "[ale-flip] pre-closure flip defect (rel) = {:.3e}, post-closure per-cell defect \
             (rel) = {:.3e}",
            out.max_identity_err_rel, out.max_defect_rel
        );
        // The flip genuinely perturbs the per-cell volumes (else the test is
        // vacuous): the pre-closure defect is O(1)-relative, not roundoff.
        assert!(
            out.max_identity_err_rel > 1e-3,
            "flip defect suspiciously small ({:.3e}) — the flip is not exercised",
            out.max_identity_err_rel
        );

        // Per-cell closure to f32 roundoff: Σ_f σ·flux·dt == ΔV_i exactly, where
        // ΔV_i is the ACTUAL config-A→config-B volume change (each computed on
        // its own topology).
        let mut max_cell_rel = 0.0f64;
        for i in 0..new_mesh.num_cells() {
            let mut sum = 0.0f64;
            for k in new_mesh.cell_face_offsets[i]..new_mesh.cell_face_offsets[i + 1] {
                let f = new_mesh.cell_faces[k];
                let s = if new_mesh.face_owner[f] == i { 1.0 } else { -1.0 };
                sum += s * out.fluxes[f] as f64 * dt;
            }
            let dv = (new_mesh.cell_vol[i] as f32) as f64 - (old_mesh.cell_vol[i] as f32) as f64;
            let rel = (sum - dv).abs() / new_mesh.cell_vol[i];
            max_cell_rel = max_cell_rel.max(rel);
        }
        assert!(
            max_cell_rel < 1e-6,
            "per-cell closure defect {:.3e} above f32 roundoff",
            max_cell_rel
        );
        assert!(out.max_defect_rel < 1e-6, "reported defect {:.3e}", out.max_defect_rel);

        // Global mass conservation: interior fluxes cancel in the total, so the
        // net across the (static) box boundary must be zero to roundoff.
        let mut boundary_sum = 0.0f64;
        for f in 0..new_mesh.num_faces() {
            if new_mesh.face_neighbor[f].is_none() {
                boundary_sum += out.fluxes[f] as f64 * dt;
            }
        }
        assert!(
            boundary_sum.abs() < 1e-6,
            "net boundary swept flux {:.3e} != 0 (mass not conserved)",
            boundary_sum
        );

        // Determinism: identical bits on a re-run.
        let out2 = swept_mesh_fluxes_closed_flip(
            &new_mesh,
            &ovx,
            &ovy,
            &old_mesh.cell_vol,
            &flip.born_face_mask,
            None,
            dt,
        )
        .expect("flip swept fluxes 2");
        assert_eq!(
            out.fluxes.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            out2.fluxes.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        );
    }
}
