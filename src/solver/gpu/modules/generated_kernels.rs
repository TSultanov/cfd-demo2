use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::graph::{DispatchKind, GpuComputeModule, RuntimeDims};
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::modules::unified_graph::UnifiedGraphModule;
use crate::solver::gpu::recipe::SolverRecipe;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::model::KernelId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

struct KernelBindGroups {
    groups_ping_pong: Vec<Vec<wgpu::BindGroup>>,
}

pub struct GeneratedKernelsModule {
    state_step_index: Arc<AtomicUsize>,
    max_compute_workgroups_per_dimension: u32,
    pipelines: HashMap<KernelId, wgpu::ComputePipeline>,
    bind_groups: HashMap<KernelId, KernelBindGroups>,
}

impl GeneratedKernelsModule {
    pub fn new_from_recipe(
        device: &wgpu::Device,
        model_id: &str,
        recipe: &SolverRecipe,
        registry: &ResourceRegistry<'_>,
        state_step_index: Arc<AtomicUsize>,
    ) -> Result<Self, String> {
        let total_start = Instant::now();
        let stage_prefix = format!("solver.new.kernels.{model_id}");

        let max_compute_workgroups_per_dimension = device.limits().max_compute_workgroups_per_dimension;
        let mut pipelines = HashMap::new();
        let mut bind_groups = HashMap::new();

        for kernel in &recipe.kernels {
            let id = kernel.id;
            let source = kernel_registry::kernel_source_by_id(model_id, id)?;
            let id_str = id.as_str();

            let pipeline_start = Instant::now();
            let pipeline = (source.create_pipeline)(device);
            crate::trace::record_init_event(
                format!("{stage_prefix}.pipeline.{id_str}"),
                pipeline_start.elapsed(),
                None,
            );

            let max_group = source
                .bindings
                .iter()
                .map(|b| b.group)
                .max()
                .unwrap_or(0);

            let bind_start = Instant::now();
            let mut groups_ping_pong = Vec::with_capacity((max_group as usize) + 1);
            let mut bound_group_count = 0usize;
            for group in 0..=max_group {
                if !source.bindings.iter().any(|b| b.group == group) {
                    groups_ping_pong.push(Vec::new());
                    continue;
                }

                bound_group_count += 1;
                let bgl = pipeline.get_bind_group_layout(group);
                let mut ping_pong = Vec::with_capacity(3);
                for phase in 0..3 {
                    let registry = registry.clone().at_ping_pong_phase(phase);
                    let label = format!("GeneratedKernels: {id:?} group {group} phase {phase}");
                    let bg = wgsl_reflect::create_bind_group_from_bindings(
                        device,
                        &label,
                        &bgl,
                        source.bindings,
                        group,
                        |name| registry.resolve(name),
                    )?;
                    ping_pong.push(bg);
                }
                groups_ping_pong.push(ping_pong);
            }
            crate::trace::record_init_event(
                format!("{stage_prefix}.bind_groups.{id_str}"),
                bind_start.elapsed(),
                Some(format!(
                    "groups={bound_group_count}/{} phases=3",
                    (max_group as usize) + 1
                )),
            );

            pipelines.insert(id, pipeline);
            bind_groups.insert(id, KernelBindGroups { groups_ping_pong });
        }

        crate::trace::record_init_event(
            format!("{stage_prefix}.total"),
            total_start.elapsed(),
            Some(format!("kernels={}", recipe.kernels.len())),
        );

        Ok(Self {
            state_step_index,
            max_compute_workgroups_per_dimension,
            pipelines,
            bind_groups,
        })
    }

    fn step_index(&self) -> usize {
        self.state_step_index.load(Ordering::Relaxed) % 3
    }

    fn dispatch_cells_or_faces(&self, items: u32) -> (u32, u32, u32) {
        const WORKGROUP_SIZE_X: u32 = 64;

        let groups = items.div_ceil(WORKGROUP_SIZE_X);
        if groups <= self.max_compute_workgroups_per_dimension {
            return (groups, 1, 1);
        }

        let x = self.max_compute_workgroups_per_dimension;
        let y = groups.div_ceil(x);
        assert!(
            y <= self.max_compute_workgroups_per_dimension,
            "dispatch requires more than 2D grid: groups={groups} max={x}"
        );
        (x, y, 1)
    }
}

impl GpuComputeModule for GeneratedKernelsModule {
    type PipelineKey = KernelId;
    type BindKey = KernelId;

    fn pipeline(&self, key: Self::PipelineKey) -> &wgpu::ComputePipeline {
        self.pipelines
            .get(&key)
            .unwrap_or_else(|| panic!("missing pipeline for {key:?}"))
    }

    fn bind(&self, key: Self::BindKey, pass: &mut wgpu::ComputePass) {
        let idx = self.step_index();
        let bind_groups = self
            .bind_groups
            .get(&key)
            .unwrap_or_else(|| panic!("missing bind groups for {key:?}"));

        for (group, ping_pong) in bind_groups.groups_ping_pong.iter().enumerate() {
            if ping_pong.is_empty() {
                continue;
            }
            pass.set_bind_group(group as u32, &ping_pong[idx], &[]);
        }
    }

    fn dispatch(&self, kind: DispatchKind, runtime: RuntimeDims) -> (u32, u32, u32) {
        match kind {
            DispatchKind::Cells => self.dispatch_cells_or_faces(runtime.num_cells),
            DispatchKind::Faces => self.dispatch_cells_or_faces(runtime.num_faces),
            DispatchKind::Custom { x, y, z } => (x, y, z),
        }
    }
}

impl UnifiedGraphModule for GeneratedKernelsModule {
    fn pipeline_for_kernel(&self, id: KernelId) -> Option<Self::PipelineKey> {
        self.pipelines.contains_key(&id).then_some(id)
    }

    fn bind_for_kernel(&self, id: KernelId) -> Option<Self::BindKey> {
        self.bind_groups.contains_key(&id).then_some(id)
    }
}
