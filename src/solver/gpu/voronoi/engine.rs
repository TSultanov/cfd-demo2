//! `GpuVoronoiEngine`: buffers, pipeline, CPU grid build/upload, regen
//! encoding and validation readback (design §3.3, cloned from the srd.rs
//! seam: manual layouts, own encoder + submit).

use std::collections::HashMap;

use nalgebra::{Point2, Vector2};

use crate::meshgen::meshless::SeedGrid;
use crate::meshgen::MeshgenTolerances;
use crate::solver::gpu::buffers::{create_buffer, create_buffer_init};
use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::linear_solver::fgmres::dispatch_2d;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::{read_buffer_cached, StagingBufferCache};

use super::{wgsl, K_FACE_MAX, MAX_VERTS};

const WORKGROUP_SIZE: u32 = 64;

/// Uniform parameter block — must match the WGSL `Params` struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_seeds: u32,
    gw: u32,
    gh: u32,
    flag_cap: u32,
    cell_size: f32,
    edge_len_eps: f32,
    domain_x: f32,
    domain_y: f32,
}

/// Per-cell padded outputs (design §3.2), allocated at seed capacity.
/// `centroid` and `face mid` are SEED-RELATIVE (see module docs).
pub struct VoronoiCellOutputs {
    pub b_nbr_ids: wgpu::Buffer,
    pub b_face_bc: wgpu::Buffer,
    pub b_face_geom: wgpu::Buffer,
    pub b_face_mid: wgpu::Buffer,
    pub b_cell_centroid: wgpu::Buffer,
    pub b_cell_area: wgpu::Buffer,
    pub b_cell_nfaces: wgpu::Buffer,
    pub b_status: wgpu::Buffer,
    /// `[0]` = atomic count, `[1..=capacity]` = flagged cell ids.
    pub b_flagged: wgpu::Buffer,
}

/// CPU-side snapshot of the outputs (validation/parity path).
#[derive(Clone, Debug)]
pub struct GpuVoronoiCells {
    pub n: usize,
    pub status: Vec<u32>,
    /// `n * K_FACE_MAX`; `NBR_NONE` = boundary face or unused slot.
    pub nbr_ids: Vec<u32>,
    /// `n * K_FACE_MAX`; bbox side id for boundary faces, `BC_NONE` else.
    pub face_bc: Vec<u32>,
    /// `n * K_FACE_MAX` of `[nx, ny, len, 0]`.
    pub face_geom: Vec<[f32; 4]>,
    /// `n * K_FACE_MAX`, seed-relative.
    pub face_mid: Vec<[f32; 2]>,
    /// Seed-relative (absolute = seed + centroid_rel).
    pub centroid_rel: Vec<[f32; 2]>,
    pub area: Vec<f32>,
    pub nfaces: Vec<u32>,
    /// Flagged cell ids, sorted ascending (the raw list order is the one
    /// atomic-nondeterministic output).
    pub flagged: Vec<u32>,
}

pub struct GpuVoronoiEngine {
    capacity: u32,
    n_seeds: u32,
    domain: Vector2<f64>,
    tol: MeshgenTolerances,

    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    /// Rebuilt on every `upload_seeds` (grid buffers are recreated there);
    /// all other bindings are engine-owned capacity buffers.
    bind_group: Option<wgpu::BindGroup>,

    b_params: wgpu::Buffer,
    b_seeds: wgpu::Buffer,
    /// FIXED bit / boundary-segment id per seed — carried for later stages
    /// (Lloyd, boundary); not bound by the stage-1 kernel.
    #[allow(dead_code)]
    b_seed_flags: wgpu::Buffer,
    b_seed_canon: wgpu::Buffer,
    b_grid_offsets: Option<wgpu::Buffer>,
    b_grid_ids: Option<wgpu::Buffer>,

    pub outputs: VoronoiCellOutputs,
}

impl GpuVoronoiEngine {
    /// Create pipelines + capacity-sized buffers. `domain` is the clip bbox
    /// `[0, x] × [0, y]`; `tol` supplies the coalescing quantization and the
    /// clip epsilon (`edge_len_eps`), exactly as the M0 engine uses them.
    pub fn new(
        device: &wgpu::Device,
        capacity_seeds: u32,
        domain: Vector2<f64>,
        tol: &MeshgenTolerances,
    ) -> Self {
        use wgpu::BufferUsages as U;
        assert!(capacity_seeds > 0, "capacity must be positive");
        let cap = capacity_seeds as u64;
        let k = K_FACE_MAX as u64;

        let b_params = create_buffer(
            device,
            "voronoi:params",
            std::mem::size_of::<Params>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        let b_seeds = create_buffer(device, "voronoi:seeds", cap * 8, U::STORAGE | U::COPY_DST);
        let b_seed_flags =
            create_buffer(device, "voronoi:seed_flags", cap * 4, U::STORAGE | U::COPY_DST);
        let b_seed_canon =
            create_buffer(device, "voronoi:seed_canon", cap * 4, U::STORAGE | U::COPY_DST);

        // Outputs get COPY_DST too: the stage-3 fallback patches flagged
        // cells' slots with `queue.write_buffer`.
        let out = U::STORAGE | U::COPY_SRC | U::COPY_DST;
        let outputs = VoronoiCellOutputs {
            b_nbr_ids: create_buffer(device, "voronoi:nbr_ids", cap * k * 4, out),
            b_face_bc: create_buffer(device, "voronoi:face_bc", cap * k * 4, out),
            b_face_geom: create_buffer(device, "voronoi:face_geom", cap * k * 16, out),
            b_face_mid: create_buffer(device, "voronoi:face_mid", cap * k * 8, out),
            b_cell_centroid: create_buffer(device, "voronoi:cell_centroid", cap * 8, out),
            b_cell_area: create_buffer(device, "voronoi:cell_area", cap * 4, out),
            b_cell_nfaces: create_buffer(device, "voronoi:cell_nfaces", cap * 4, out),
            b_status: create_buffer(device, "voronoi:status", cap * 4, out),
            b_flagged: create_buffer(device, "voronoi:flagged", (1 + cap) * 4, out),
        };

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voronoi:bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),   // seeds
                storage(2, true),   // canon
                storage(3, true),   // grid_offsets
                storage(4, true),   // grid_ids
                storage(5, false),  // nbr_ids
                storage(6, false),  // face_bc
                storage(7, false),  // face_geom
                storage(8, false),  // face_mid
                storage(9, false),  // cell_centroid
                storage(10, false), // cell_area
                storage(11, false), // cell_nfaces
                storage(12, false), // status
                storage(13, false), // flagged
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voronoi:pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voronoi:cell_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl::voronoi_cell_shader().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("voronoi:cell_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("voronoi_cell"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            capacity: capacity_seeds,
            n_seeds: 0,
            domain,
            tol: tol.clone(),
            pipeline,
            bgl,
            bind_group: None,
            b_params,
            b_seeds,
            b_seed_flags,
            b_seed_canon,
            b_grid_offsets: None,
            b_grid_ids: None,
            outputs,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn n_seeds(&self) -> u32 {
        self.n_seeds
    }

    /// Upload seeds (`seeds_xy` = interleaved f32 x/y pairs, `flags` = one
    /// u32 per seed) and build + upload the CPU `SeedGrid` and the
    /// coalescing table. LOAD-BEARING: the grid and the coalescing keys are
    /// computed from the SAME f32 values the kernel sees (widened to f64),
    /// so grid membership, coalescing verdicts and kernel arithmetic agree.
    pub fn upload_seeds(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        seeds_xy: &[f32],
        flags: &[u32],
    ) {
        assert_eq!(seeds_xy.len() % 2, 0, "seeds_xy must be x/y pairs");
        let n = seeds_xy.len() / 2;
        assert!(n > 0, "need at least one seed");
        assert!(n as u32 <= self.capacity, "seed count exceeds capacity");
        assert_eq!(flags.len(), n, "one flag word per seed");

        // Widen the f32-rounded coordinates back to f64 for the grid build
        // and coalescing keys (f32 -> f64 is exact).
        let pts: Vec<Point2<f64>> = (0..n)
            .map(|i| Point2::new(seeds_xy[2 * i] as f64, seeds_xy[2 * i + 1] as f64))
            .collect();

        let grid = SeedGrid::build(&pts, self.domain);

        // Coalescing table: canon[i] = lowest seed index in i's quantize
        // bin (the M0 EmptyCell rule; same-bin planes are skipped by the
        // kernel via canon[j] == canon[i]).
        let mut first: HashMap<(i64, i64), u32> = HashMap::with_capacity(n);
        let mut canon = vec![0u32; n];
        for (i, p) in pts.iter().enumerate() {
            let key = self.tol.quantize_point(p.x, p.y);
            let rep = *first.entry(key).or_insert(i as u32);
            canon[i] = rep;
        }

        queue.write_buffer(&self.b_seeds, 0, bytemuck::cast_slice(seeds_xy));
        queue.write_buffer(&self.b_seed_flags, 0, bytemuck::cast_slice(flags));
        queue.write_buffer(&self.b_seed_canon, 0, bytemuck::cast_slice(&canon));

        let b_grid_offsets = create_buffer_init(
            device,
            "voronoi:grid_offsets",
            &grid.offsets,
            wgpu::BufferUsages::STORAGE,
        );
        let b_grid_ids = create_buffer_init(
            device,
            "voronoi:grid_ids",
            &grid.ids,
            wgpu::BufferUsages::STORAGE,
        );

        let params = Params {
            n_seeds: n as u32,
            gw: grid.gw as u32,
            gh: grid.gh as u32,
            flag_cap: self.capacity,
            cell_size: grid.cell_size as f32,
            edge_len_eps: self.tol.edge_len_eps as f32,
            domain_x: self.domain.x as f32,
            domain_y: self.domain.y as f32,
        };
        queue.write_buffer(&self.b_params, 0, bytemuck::bytes_of(&params));

        fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            }
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voronoi:bg"),
            layout: &self.bgl,
            entries: &[
                entry(0, &self.b_params),
                entry(1, &self.b_seeds),
                entry(2, &self.b_seed_canon),
                entry(3, &b_grid_offsets),
                entry(4, &b_grid_ids),
                entry(5, &self.outputs.b_nbr_ids),
                entry(6, &self.outputs.b_face_bc),
                entry(7, &self.outputs.b_face_geom),
                entry(8, &self.outputs.b_face_mid),
                entry(9, &self.outputs.b_cell_centroid),
                entry(10, &self.outputs.b_cell_area),
                entry(11, &self.outputs.b_cell_nfaces),
                entry(12, &self.outputs.b_status),
                entry(13, &self.outputs.b_flagged),
            ],
        });

        self.b_grid_offsets = Some(b_grid_offsets);
        self.b_grid_ids = Some(b_grid_ids);
        self.bind_group = Some(bind_group);
        self.n_seeds = n as u32;
    }

    /// Encode one full regen (flag-list reset + `voronoi_cell` over all
    /// seeds) into `enc`. `upload_seeds` must have run.
    pub fn encode_regen(&self, enc: &mut wgpu::CommandEncoder, n_seeds: u32) {
        assert_eq!(
            n_seeds, self.n_seeds,
            "encode_regen n_seeds must match the uploaded seed count"
        );
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("upload_seeds must be called before encode_regen");
        // Reset the flag-append cursor.
        enc.clear_buffer(&self.outputs.b_flagged, 0, Some(4));
        let workgroups = n_seeds.div_ceil(WORKGROUP_SIZE).max(1);
        let (dx, dy) = dispatch_2d(workgroups);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voronoi:cell"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(dx, dy, 1);
    }

    /// Own encoder + submit (srd.rs style). Returns the submission index so
    /// callers can `device.poll` on it.
    pub fn run_regen(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> wgpu::SubmissionIndex {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voronoi:regen_encoder"),
        });
        self.encode_regen(&mut enc, self.n_seeds);
        queue.submit(Some(enc.finish()))
    }

    /// Validation-path readback of every output for the first `n` cells.
    /// The flag list is read in ONE bounded over-read (count word + full
    /// capacity id region — never two dependent round-trips) and sorted.
    pub fn read_cells(&self, ctx: &GpuContext, cache: &StagingBufferCache) -> GpuVoronoiCells {
        let n = self.n_seeds as usize;
        let k = K_FACE_MAX;
        let prof = ProfilingStats::new();
        let read = |buf: &wgpu::Buffer, size: u64, label: &'static str| -> Vec<u8> {
            pollster::block_on(read_buffer_cached(ctx, cache, &prof, buf, size, label))
        };

        let status = cast_vec::<u32>(read(&self.outputs.b_status, (n * 4) as u64, "voronoi:rb_status"));
        let nbr_ids = cast_vec::<u32>(read(&self.outputs.b_nbr_ids, (n * k * 4) as u64, "voronoi:rb_nbr"));
        let face_bc = cast_vec::<u32>(read(&self.outputs.b_face_bc, (n * k * 4) as u64, "voronoi:rb_bc"));
        let face_geom = cast_vec::<[f32; 4]>(read(
            &self.outputs.b_face_geom,
            (n * k * 16) as u64,
            "voronoi:rb_geom",
        ));
        let face_mid = cast_vec::<[f32; 2]>(read(
            &self.outputs.b_face_mid,
            (n * k * 8) as u64,
            "voronoi:rb_mid",
        ));
        let centroid_rel = cast_vec::<[f32; 2]>(read(
            &self.outputs.b_cell_centroid,
            (n * 8) as u64,
            "voronoi:rb_centroid",
        ));
        let area = cast_vec::<f32>(read(&self.outputs.b_cell_area, (n * 4) as u64, "voronoi:rb_area"));
        let nfaces = cast_vec::<u32>(read(
            &self.outputs.b_cell_nfaces,
            (n * 4) as u64,
            "voronoi:rb_nfaces",
        ));
        // Bounded over-read of the flag list (review F10).
        let flag_raw = cast_vec::<u32>(read(
            &self.outputs.b_flagged,
            ((1 + n) * 4) as u64,
            "voronoi:rb_flagged",
        ));
        let count = (flag_raw[0] as usize).min(n);
        let mut flagged: Vec<u32> = flag_raw[1..1 + count].to_vec();
        flagged.sort_unstable();

        GpuVoronoiCells {
            n,
            status,
            nbr_ids,
            face_bc,
            face_geom,
            face_mid,
            centroid_rel,
            area,
            nfaces,
            flagged,
        }
    }

    /// Raw bytes of all deterministic outputs (everything except the
    /// flag-list order) for the byte-stability gate.
    pub fn read_raw_outputs(&self, ctx: &GpuContext, cache: &StagingBufferCache) -> Vec<u8> {
        let n = self.n_seeds as u64;
        let k = K_FACE_MAX as u64;
        let prof = ProfilingStats::new();
        let mut all = Vec::new();
        for (buf, size, label) in [
            (&self.outputs.b_status, n * 4, "voronoi:raw_status"),
            (&self.outputs.b_nbr_ids, n * k * 4, "voronoi:raw_nbr"),
            (&self.outputs.b_face_bc, n * k * 4, "voronoi:raw_bc"),
            (&self.outputs.b_face_geom, n * k * 16, "voronoi:raw_geom"),
            (&self.outputs.b_face_mid, n * k * 8, "voronoi:raw_mid"),
            (&self.outputs.b_cell_centroid, n * 8, "voronoi:raw_centroid"),
            (&self.outputs.b_cell_area, n * 4, "voronoi:raw_area"),
            (&self.outputs.b_cell_nfaces, n * 4, "voronoi:raw_nfaces"),
        ] {
            let mut bytes =
                pollster::block_on(read_buffer_cached(ctx, cache, &prof, buf, size, label));
            all.append(&mut bytes);
        }
        all
    }
}

fn cast_vec<T: bytemuck::Pod>(bytes: Vec<u8>) -> Vec<T> {
    bytemuck::cast_slice(&bytes).to_vec()
}

// Compile-time guarantee that the WGSL constants (injected from mod.rs) and
// the private-array budget stay in sync with the design.
const _: () = assert!(MAX_VERTS == 24 && K_FACE_MAX == 16);
