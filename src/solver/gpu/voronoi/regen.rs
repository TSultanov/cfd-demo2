//! On-device mesh regeneration orchestrator (Phase C stage D4): bundles the GPU
//! Voronoi engine + the four device passes (emit → csr → cell_geom → swept) and
//! rebuilds a solver-ready [`Mesh`] + ALE mesh fluxes from a seed set ENTIRELY on
//! device, with no CPU `assemble_meshless_from_seeds` and no CPU swept path.
//!
//! This is the library form of the loop the `gpu_moving_gcl_test` proves: one
//! [`GpuMeshRegen::regen`] call per moving step yields the vertex-less solver
//! mesh (see [`assemble_solver_mesh`]) with boundary faces tagged by the same
//! `PlaneTag → BoundaryType` rule as the CPU assembler, and the per-face
//! `mesh_fluxes` the solver consumes. The caller (e.g. `MovingMeshDriver`) may
//! re-tag boundaries (`boundary_retag`), feeds the result to
//! `begin_ale_step_topology`, and MUST `reapply_boundary_conditions` after
//! (the topology refresh drops runtime BC overrides).
//!
//! Non-flip only: the swept quads assume a persistent adjacency (a Voronoi flip
//! that births/kills faces has no valid device swept). The caller compares the
//! returned mesh's adjacency across steps and falls back to the CPU path on a
//! flip; `needs_cpu` additionally counts resolve-unresolved cells, tolerated
//! sub-eps one-sided faces, and untaggable open faces (see
//! [`GpuRegenResult::needs_cpu`]).

use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::readback::StagingBufferCache;
use crate::solver::mesh::Mesh;
use crate::meshgen::meshless::{tag_boundary_type, BoundarySpec, PlaneTag, SeedKind};
use nalgebra::Vector2;

use super::cell_geom::CellGeometry;
use super::csr_gpu::GpuCsr;
use super::emit::EmitFaces;
use super::engine::GpuVoronoiEngine;
use super::solver_mesh::assemble_solver_mesh;
use super::swept_gpu::SweptFluxGeometry;
use crate::meshgen::MeshgenTolerances;

/// The result of one on-device regen.
pub struct GpuRegenResult {
    /// Vertex-less solver mesh. Boundary faces carry the engine's tags
    /// (`face_bc` → [`tag_boundary_type`], byte-identical to the CPU
    /// assembler's rule); the caller's `boundary_retag` may overwrite them.
    pub mesh: Mesh,
    /// Per-face ALE mesh flux (`swept_area / dt`), in the mesh's face order.
    pub mesh_fluxes: Vec<f32>,
    /// Count of conditions the device build could not certify — `> 0` ⇒ the
    /// caller must use the CPU path for this step. Sums: swept faces the
    /// kernel could not close (ill-conditioned brackets / unknown edge tags),
    /// resolve-`unresolved` cells (f64 ring beyond the padded GPU layout —
    /// paper §3.4 overflow class), tolerated sub-eps one-sided faces (they
    /// break the reciprocity precondition of the CSR builder's rank scan),
    /// FACELESS cells (post-resolve EMPTY_CELL — coalesced duplicates or an
    /// empty f64 recompute), and open faces with no boundary tag.
    pub needs_cpu: usize,
}

/// Bundles the GPU Voronoi engine + the device mesh-build passes for repeated
/// (per-step) regeneration at a fixed seed count.
pub struct GpuMeshRegen {
    engine: GpuVoronoiEngine,
    emit: EmitFaces,
    csr: GpuCsr,
    cellgeom: CellGeometry,
    swept: SweptFluxGeometry,
    cache: StagingBufferCache,
    flags: Vec<u32>,
}

impl GpuMeshRegen {
    /// Build the engine + passes for `n` seeds over `domain` at tolerance `tol`.
    pub fn new(device: &wgpu::Device, n: usize, domain: Vector2<f64>, tol: &MeshgenTolerances) -> Self {
        Self {
            engine: GpuVoronoiEngine::new(device, n as u32, domain, tol),
            emit: EmitFaces::new(device),
            csr: GpuCsr::new(device),
            cellgeom: CellGeometry::new(device),
            swept: SweptFluxGeometry::new(device),
            cache: StagingBufferCache::default(),
            flags: vec![0u32; n],
        }
    }

    /// Regenerate the solver mesh + ALE fluxes at `new_seeds` (interleaved f32
    /// x/y), with `old_seeds` (the previous step's positions) for the swept
    /// fluxes. `dt` is the step the fluxes are closed against. `kinds`/`spec`
    /// classify the seeds + boundary loops at t^{n+1} (as `upload_case`
    /// expects); `old_spec` is the boundary at t^n — the swept pass resolves
    /// polyline segment faces against BOTH tables (identical under a static
    /// boundary), so a moving wall sweeps its faces correctly.
    pub fn regen(
        &mut self,
        ctx: &GpuContext,
        new_seeds: &[f32],
        old_seeds: &[f32],
        kinds: &[SeedKind],
        spec: &BoundarySpec,
        old_spec: &BoundarySpec,
        dt: f32,
    ) -> GpuRegenResult {
        self.engine.upload_case(&ctx.device, &ctx.queue, new_seeds, &self.flags, kinds, spec);
        let idx = self.engine.run_regen(&ctx.device, &ctx.queue);
        let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
        // The Appendix-A hybrid: kernel-flagged cells are recomputed in f64 and
        // patched into the device outputs BEFORE the build passes read them.
        // The report's failure classes have no valid device representation, so
        // they must reject the device mesh (paper §3.4: overflow/boundary
        // failures route to the CPU), not silently ship:
        // * `unresolved`  — f64 ring beyond K_FACE_MAX/MAX_CLIP_VERTS ⇒ the
        //   cell's slots are EMPTY; emit/csr would build a faceless
        //   zero-volume cell.
        // * `sub_eps_asymmetries` — tolerated one-sided sliver faces ⇒ the
        //   CSR rank scan's reciprocity precondition is broken; the scan
        //   would fabricate an out-of-range/aliased global face index.
        let report = self.engine.resolve_flagged(ctx, &self.cache);

        let faces = self.emit.emit(ctx, &self.cache, &self.engine);
        let csr = self.csr.build_csr(ctx, &self.cache, &self.engine);
        let cells = self.cellgeom.build(ctx, &self.cache, &self.engine);
        let sw = self.swept.compute(ctx, &self.cache, &self.engine, old_seeds, old_spec);
        let mut needs_cpu = sw.needs_cpu.iter().filter(|&&x| x != 0).count()
            + report.unresolved.len()
            + report.sub_eps_asymmetries;

        let mut mesh = assemble_solver_mesh(&faces, &csr, &cells, &sw)
            .expect("assemble on-device solver mesh");
        // Faceless cells: a legitimate 2D Voronoi cell has >= 3 faces. A cell
        // whose CSR row is EMPTY is a post-resolve EMPTY_CELL (a coalesced
        // duplicate — unflagged by design — or an f64 recompute that came back
        // empty, which resolve PATCHES rather than reports unresolved); every
        // build pass silently skips it, so it reaches the mesh as a
        // zero-volume, zero-face cell that no other needs_cpu bucket counts.
        // It must not reach the solver.
        for i in 0..mesh.cell_face_offsets.len().saturating_sub(1) {
            if mesh.cell_face_offsets[i + 1] == mesh.cell_face_offsets[i] {
                needs_cpu += 1;
            }
        }
        // Tag boundary faces from the device tags — the same
        // `PlaneTag → BoundaryType` rule the CPU assembler applies
        // (`tag_boundary_type`), so `boundary_retag: None` keeps its
        // "keep the engine's tags" meaning on the device path too. An open
        // face with no tag has no CPU analogue ("nothing is ever left
        // untagged") — reject the mesh rather than let it scatter to bc
        // row 0 with a zeroed kind.
        for f in 0..mesh.face_boundary.len() {
            if mesh.face_neighbor[f].is_some() {
                continue;
            }
            let bc = faces.face_bc[f];
            let tag = if bc < 4 {
                Some(PlaneTag::Box(bc as u8))
            } else if bc != super::BC_NONE && (bc & super::BC_SEG_FLAG) != 0 {
                Some(PlaneTag::Boundary(bc & !super::BC_SEG_FLAG))
            } else {
                None
            };
            match tag.and_then(|t| tag_boundary_type(t, spec)) {
                Some(bt) => mesh.face_boundary[f] = Some(bt),
                None => needs_cpu += 1,
            }
        }
        let mesh_fluxes: Vec<f32> = sw.swept.iter().map(|&s| s / dt).collect();
        GpuRegenResult { mesh, mesh_fluxes, needs_cpu }
    }
}
