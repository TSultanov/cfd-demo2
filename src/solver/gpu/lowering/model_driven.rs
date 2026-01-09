use crate::solver::gpu::plans::plan_instance::{PlanInitConfig, PlanParam, PlanParamValue};
use crate::solver::gpu::plans::program::GpuProgramPlan;
use crate::solver::mesh::Mesh;
use crate::solver::model::ModelSpec;
use std::collections::HashMap;

use super::models;
use super::templates::{build_program_spec, ProgramTemplateKind};
use super::types::{LoweredProgramParts, ModelGpuProgramSpecParts};

use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

macro_rules! create_bind_group {
    ($device:expr, $label:expr, $layout:expr, $entries:expr) => {{
        let entries = $entries.into_array();
        $device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some($label),
            layout: $layout,
            entries: &entries,
        })
    }};
}

macro_rules! with_generic_coupled_kernels {
    ($model_id:expr, |$gen_assembly:ident, $gen_update:ident| $body:block) => {{
        match $model_id {
            "generic_diffusion_demo" => {
                use crate::solver::gpu::bindings::generated::generic_coupled_assembly_generic_diffusion_demo as $gen_assembly;
                use crate::solver::gpu::bindings::generated::generic_coupled_update_generic_diffusion_demo as $gen_update;
                $body
            }
            "generic_diffusion_demo_neumann" => {
                use crate::solver::gpu::bindings::generated::generic_coupled_assembly_generic_diffusion_demo_neumann as $gen_assembly;
                use crate::solver::gpu::bindings::generated::generic_coupled_update_generic_diffusion_demo_neumann as $gen_update;
                $body
            }
            other => Err(format!(
                "GpuProgramPlan does not have generated generic-coupled kernels for model id '{other}'"
            )),
        }
    }};
}

pub(crate) async fn lower_program_model_driven(
    mesh: &Mesh,
    model: &ModelSpec,
    config: PlanInitConfig,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<GpuProgramPlan, String> {
    let template = ProgramTemplateKind::for_model(model)?;
    let parts = lower_parts_for_template(template, mesh, model, device, queue).await?;

    let program = build_program_spec(template);
    let spec = parts.spec.into_spec(program);
    let mut plan = GpuProgramPlan::new(
        parts.model,
        parts.context,
        parts.profiling_stats,
        parts.resources,
        spec,
    );

    plan.set_param(
        PlanParam::AdvectionScheme,
        PlanParamValue::Scheme(config.advection_scheme),
    )?;
    plan.set_param(
        PlanParam::TimeScheme,
        PlanParamValue::TimeScheme(config.time_scheme),
    )?;
    plan.set_param(
        PlanParam::Preconditioner,
        PlanParamValue::Preconditioner(config.preconditioner),
    )?;

    Ok(plan)
}

async fn lower_parts_for_template(
    template: ProgramTemplateKind,
    mesh: &Mesh,
    model: &ModelSpec,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
) -> Result<LoweredProgramParts, String> {
    match template {
        ProgramTemplateKind::Compressible => {
            let plan = crate::solver::gpu::plans::compressible::CompressiblePlanResources::new(
                mesh,
                model.clone(),
                device,
                queue,
            )
            .await?;

            let context = crate::solver::gpu::context::GpuContext {
                device: plan.common.context.device.clone(),
                queue: plan.common.context.queue.clone(),
            };
            let profiling_stats = std::sync::Arc::clone(&plan.common.profiling_stats);

            let mut resources = crate::solver::gpu::plans::program::ProgramResources::new();
            resources.insert(models::compressible::CompressibleProgramResources::new(
                plan,
            ));

            let ops = std::sync::Arc::new(models::compressible::CompressibleOpDispatcher);

            Ok(LoweredProgramParts {
                model: model.clone(),
                context,
                profiling_stats,
                resources,
                spec: ModelGpuProgramSpecParts {
                    ops,
                    num_cells: models::compressible::spec_num_cells,
                    time: models::compressible::spec_time,
                    dt: models::compressible::spec_dt,
                    state_buffer: models::compressible::spec_state_buffer,
                    write_state_bytes: models::compressible::spec_write_state_bytes,
                    initialize_history: Some(models::compressible::init_history),
                    params: HashMap::new(),
                    set_param_fallback: Some(models::compressible::set_param_fallback),
                    step_stats: Some(models::compressible::step_stats),
                    step_with_stats: Some(models::compressible::step_with_stats),
                    linear_debug: Some(models::compressible::linear_debug_provider),
                },
            })
        }
        ProgramTemplateKind::IncompressibleCoupled => {
            let plan =
                crate::solver::gpu::structs::GpuSolver::new(mesh, model.clone(), device, queue)
                    .await?;

            let context = crate::solver::gpu::context::GpuContext {
                device: plan.common.context.device.clone(),
                queue: plan.common.context.queue.clone(),
            };
            let profiling_stats = std::sync::Arc::clone(&plan.common.profiling_stats);

            let mut resources = crate::solver::gpu::plans::program::ProgramResources::new();
            resources.insert(models::incompressible::IncompressibleProgramResources::new(
                plan,
            ));

            let ops = std::sync::Arc::new(models::incompressible::IncompressibleOpDispatcher);

            Ok(LoweredProgramParts {
                model: model.clone(),
                context,
                profiling_stats,
                resources,
                spec: ModelGpuProgramSpecParts {
                    ops,
                    num_cells: models::incompressible::spec_num_cells,
                    time: models::incompressible::spec_time,
                    dt: models::incompressible::spec_dt,
                    state_buffer: models::incompressible::spec_state_buffer,
                    write_state_bytes: models::incompressible::spec_write_state_bytes,
                    initialize_history: Some(models::incompressible::init_history),
                    params: HashMap::new(),
                    set_param_fallback: Some(models::incompressible::set_param_fallback),
                    step_stats: Some(models::incompressible::step_stats),
                    step_with_stats: None,
                    linear_debug: Some(models::incompressible::linear_debug_provider),
                },
            })
        }
        ProgramTemplateKind::GenericCoupledScalar => {
            let coupled_stride = model.system.unknowns_per_cell();
            if coupled_stride != 1 {
                return Err(format!(
                    "generic coupled currently supports scalar systems only (unknowns_per_cell=1), got {}",
                    coupled_stride
                ));
            }

            let runtime =
                crate::solver::gpu::runtime::GpuScalarRuntime::new(mesh, device, queue).await;
            runtime.update_constants();

            let device = &runtime.common.context.device;

            with_generic_coupled_kernels!(model.id, |gen_assembly, gen_update| {
                let pipeline_assembly =
                    gen_assembly::compute::create_main_pipeline_embed_source(device);
                let pipeline_update =
                    gen_update::compute::create_main_pipeline_embed_source(device);

                let bg_mesh = {
                    let bgl = device
                        .create_bind_group_layout(&gen_assembly::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
                    create_bind_group!(
                        device,
                        "GenericCoupled: mesh bind group",
                        &bgl,
                        gen_assembly::WgpuBindGroup0Entries::new(
                            gen_assembly::WgpuBindGroup0EntriesParams {
                                face_owner: runtime
                                    .common
                                    .mesh
                                    .b_face_owner
                                    .as_entire_buffer_binding(),
                                face_neighbor: runtime
                                    .common
                                    .mesh
                                    .b_face_neighbor
                                    .as_entire_buffer_binding(),
                                face_areas: runtime
                                    .common
                                    .mesh
                                    .b_face_areas
                                    .as_entire_buffer_binding(),
                                face_normals: runtime
                                    .common
                                    .mesh
                                    .b_face_normals
                                    .as_entire_buffer_binding(),
                                face_centers: runtime
                                    .common
                                    .mesh
                                    .b_face_centers
                                    .as_entire_buffer_binding(),
                                cell_centers: runtime
                                    .common
                                    .mesh
                                    .b_cell_centers
                                    .as_entire_buffer_binding(),
                                cell_vols: runtime
                                    .common
                                    .mesh
                                    .b_cell_vols
                                    .as_entire_buffer_binding(),
                                cell_face_offsets: runtime
                                    .common
                                    .mesh
                                    .b_cell_face_offsets
                                    .as_entire_buffer_binding(),
                                cell_faces: runtime
                                    .common
                                    .mesh
                                    .b_cell_faces
                                    .as_entire_buffer_binding(),
                                cell_face_matrix_indices: runtime
                                    .common
                                    .mesh
                                    .b_cell_face_matrix_indices
                                    .as_entire_buffer_binding(),
                                diagonal_indices: runtime
                                    .common
                                    .mesh
                                    .b_diagonal_indices
                                    .as_entire_buffer_binding(),
                                face_boundary: runtime
                                    .common
                                    .mesh
                                    .b_face_boundary
                                    .as_entire_buffer_binding(),
                            }
                        )
                    )
                };

                let stride = model.state_layout.stride() as usize;
                let num_cells = runtime.common.num_cells as usize;
                let zero_state = vec![0.0f32; num_cells * stride];

                let state = crate::solver::gpu::modules::state::PingPongState::new([
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("GenericCoupled state buffer 0"),
                        contents: cast_slice(&zero_state),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    }),
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("GenericCoupled state buffer 1"),
                        contents: cast_slice(&zero_state),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    }),
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("GenericCoupled state buffer 2"),
                        contents: cast_slice(&zero_state),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    }),
                ]);

                let bg_fields_ping_pong = {
                    let bgl = device
                        .create_bind_group_layout(&gen_assembly::WgpuBindGroup1::LAYOUT_DESCRIPTOR);
                    let mut out = Vec::new();
                    for i in 0..3 {
                        let (idx_state, idx_old, idx_old_old) =
                            crate::solver::gpu::modules::state::ping_pong_indices(i);
                        out.push(create_bind_group!(
                            device,
                            &format!("GenericCoupled assembly fields bind group {i}"),
                            &bgl,
                            gen_assembly::WgpuBindGroup1Entries::new(
                                gen_assembly::WgpuBindGroup1EntriesParams {
                                    state: state.buffers()[idx_state].as_entire_buffer_binding(),
                                    state_old: state.buffers()[idx_old].as_entire_buffer_binding(),
                                    state_old_old: state.buffers()[idx_old_old]
                                        .as_entire_buffer_binding(),
                                    constants: runtime
                                        .constants
                                        .buffer()
                                        .as_entire_buffer_binding(),
                                }
                            )
                        ));
                    }
                    out
                };

                let bg_update_state_ping_pong = {
                    let bgl = device
                        .create_bind_group_layout(&gen_update::WgpuBindGroup0::LAYOUT_DESCRIPTOR);
                    let mut out = Vec::new();
                    for i in 0..3 {
                        let (idx_state, _, _) =
                            crate::solver::gpu::modules::state::ping_pong_indices(i);
                        out.push(create_bind_group!(
                            device,
                            &format!("GenericCoupled update state bind group {i}"),
                            &bgl,
                            gen_update::WgpuBindGroup0Entries::new(
                                gen_update::WgpuBindGroup0EntriesParams {
                                    state: state.buffers()[idx_state].as_entire_buffer_binding(),
                                    constants: runtime
                                        .constants
                                        .buffer()
                                        .as_entire_buffer_binding(),
                                }
                            )
                        ));
                    }
                    out
                };

                let bg_update_solution = {
                    let bgl = device
                        .create_bind_group_layout(&gen_update::WgpuBindGroup1::LAYOUT_DESCRIPTOR);
                    create_bind_group!(
                        device,
                        "GenericCoupled update solution bind group",
                        &bgl,
                        gen_update::WgpuBindGroup1Entries::new(
                            gen_update::WgpuBindGroup1EntriesParams {
                                x: runtime
                                    .linear_port_space
                                    .buffer(runtime.linear_ports.x)
                                    .as_entire_buffer_binding(),
                            }
                        )
                    )
                };

                let bg_solver = {
                    let bgl = device
                        .create_bind_group_layout(&gen_assembly::WgpuBindGroup2::LAYOUT_DESCRIPTOR);
                    create_bind_group!(
                        device,
                        "GenericCoupled assembly solver bind group",
                        &bgl,
                        gen_assembly::WgpuBindGroup2Entries::new(
                            gen_assembly::WgpuBindGroup2EntriesParams {
                                matrix_values: runtime
                                    .linear_port_space
                                    .buffer(runtime.linear_ports.values)
                                    .as_entire_buffer_binding(),
                                rhs: runtime
                                    .linear_port_space
                                    .buffer(runtime.linear_ports.rhs)
                                    .as_entire_buffer_binding(),
                                scalar_row_offsets: runtime
                                    .linear_port_space
                                    .buffer(runtime.linear_ports.row_offsets)
                                    .as_entire_buffer_binding(),
                            }
                        )
                    )
                };

                let (bc_kind, bc_value) = model
                    .boundaries
                    .to_gpu_tables(&model.system)
                    .map_err(|e| format!("failed to build BC tables: {e}"))?;

                let b_bc_kind = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("GenericCoupled bc_kind"),
                    contents: cast_slice(&bc_kind),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                let b_bc_value = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("GenericCoupled bc_value"),
                    contents: cast_slice(&bc_value),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

                let bg_bc = {
                    let bgl = device
                        .create_bind_group_layout(&gen_assembly::WgpuBindGroup3::LAYOUT_DESCRIPTOR);
                    create_bind_group!(
                        device,
                        "GenericCoupled BC bind group",
                        &bgl,
                        gen_assembly::WgpuBindGroup3Entries::new(
                            gen_assembly::WgpuBindGroup3EntriesParams {
                                bc_kind: b_bc_kind.as_entire_buffer_binding(),
                                bc_value: b_bc_value.as_entire_buffer_binding(),
                            }
                        )
                    )
                };

                let kernels = crate::solver::gpu::modules::generic_coupled_kernels::GenericCoupledKernelsModule::new(
                    state.step_handle(),
                    bg_mesh,
                    bg_fields_ping_pong,
                    bg_solver,
                    bg_bc,
                    bg_update_state_ping_pong,
                    bg_update_solution,
                    pipeline_assembly,
                    pipeline_update,
                );

                let context = crate::solver::gpu::context::GpuContext {
                    device: runtime.common.context.device.clone(),
                    queue: runtime.common.context.queue.clone(),
                };
                let profiling_stats = std::sync::Arc::clone(&runtime.common.profiling_stats);

                let mut resources = crate::solver::gpu::plans::program::ProgramResources::new();
                resources.insert(
                    models::generic_coupled::GenericCoupledProgramResources::new(
                        runtime, state, kernels, b_bc_kind, b_bc_value,
                    ),
                );

                let mut params = HashMap::new();
                params.insert(PlanParam::Dt, models::generic_coupled::param_dt as _);
                params.insert(
                    PlanParam::AdvectionScheme,
                    models::generic_coupled::param_advection_scheme as _,
                );
                params.insert(
                    PlanParam::TimeScheme,
                    models::generic_coupled::param_time_scheme as _,
                );
                params.insert(
                    PlanParam::Preconditioner,
                    models::generic_coupled::param_preconditioner as _,
                );
                params.insert(
                    PlanParam::DetailedProfilingEnabled,
                    models::generic_coupled::param_detailed_profiling as _,
                );

                let ops = std::sync::Arc::new(models::generic_coupled::GenericCoupledOpDispatcher);

                Ok(LoweredProgramParts {
                    model: model.clone(),
                    context,
                    profiling_stats,
                    resources,
                    spec: ModelGpuProgramSpecParts {
                        ops,
                        num_cells: models::generic_coupled::spec_num_cells,
                        time: models::generic_coupled::spec_time,
                        dt: models::generic_coupled::spec_dt,
                        state_buffer: models::generic_coupled::spec_state_buffer,
                        write_state_bytes: models::generic_coupled::spec_write_state_bytes,
                        initialize_history: None,
                        params,
                        set_param_fallback: None,
                        step_stats: None,
                        step_with_stats: None,
                        linear_debug: Some(models::generic_coupled::linear_debug_provider),
                    },
                })
            })
        }
    }
}
