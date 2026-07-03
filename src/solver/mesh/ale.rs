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
    /// (f64-roundoff scale on any linear vertex motion).
    pub max_identity_err_rel: f64,
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
pub fn swept_mesh_fluxes_closed(
    mesh: &Mesh,
    old_vx: &[f64],
    old_vy: &[f64],
    dt: f64,
) -> Result<SweptMeshFluxes, String> {
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
    let mut swept = vec![0.0f64; num_faces];
    for f in 0..num_faces {
        swept[f] = swept_area_along_normal(mesh, f, old_vx, old_vy)?;
    }

    let old_vols = cell_volumes_from(mesh, old_vx, old_vy);
    let sign = |cell: usize, face: usize| -> f64 {
        if mesh.face_owner[face] == cell {
            1.0
        } else {
            -1.0
        }
    };

    let mut max_identity_err_rel = 0.0f64;
    for i in 0..num_cells {
        let mut sum = 0.0f64;
        for k in mesh.cell_face_offsets[i]..mesh.cell_face_offsets[i + 1] {
            let f = mesh.cell_faces[k];
            sum += sign(i, f) * swept[f];
        }
        let dv = mesh.cell_vol[i] - old_vols[i];
        let err = (sum - dv).abs() / mesh.cell_vol[i].max(f64::MIN_POSITIVE);
        max_identity_err_rel = max_identity_err_rel.max(err);
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
    if max_identity_err_rel > 1e-9 {
        return Err(format!(
            "swept_mesh_fluxes: telescoping identity violated (max rel residual {:.3e} > 1e-9): \
             the swept-quad areas do not sum to the per-cell volume changes. The old/new vertex \
             positions are inconsistent with the mesh geometry (stale recalculate_geometry, \
             wrong old positions, or inverted/degenerate cells)",
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

    Ok(SweptMeshFluxes {
        fluxes,
        max_defect_rel,
        max_identity_err_rel,
    })
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
}
