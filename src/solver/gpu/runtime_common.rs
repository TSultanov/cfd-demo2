use crate::solver::gpu::context::GpuContext;
use crate::solver::gpu::init::mesh as mesh_init;
use crate::solver::gpu::profiling::ProfilingStats;
use crate::solver::gpu::readback::StagingBufferCache;
use crate::solver::mesh::Mesh;
use std::sync::Arc;

pub(crate) struct GpuRuntimeCommon {
    pub context: GpuContext,
    pub mesh: mesh_init::MeshResources,
    pub num_cells: u32,
    pub num_faces: u32,
    pub profiling_stats: Arc<ProfilingStats>,
    readback_cache: StagingBufferCache,
}

impl GpuRuntimeCommon {
    pub async fn new(
        mesh: &Mesh,
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
    ) -> Self {
        let context = GpuContext::new(device, queue).await;
        let num_cells = mesh.cell_cx.len() as u32;
        let num_faces = mesh.face_owner.len() as u32;

        let mesh_res = mesh_init::init_mesh(&context.device, mesh);

        Self {
            context,
            mesh: mesh_res,
            num_cells,
            num_faces,
            profiling_stats: Arc::new(ProfilingStats::new()),
            readback_cache: Default::default(),
        }
    }

    pub async fn read_buffer(
        &self,
        buffer: &wgpu::Buffer,
        size: u64,
        label: &'static str,
    ) -> Vec<u8> {
        crate::solver::gpu::readback::read_buffer_cached(
            &self.context,
            &self.readback_cache,
            &self.profiling_stats,
            buffer,
            size,
            label,
        )
        .await
    }
}
