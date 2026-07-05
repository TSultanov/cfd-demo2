use super::matrix::MatrixResources;
use super::state::StateResources;
use crate::solver::gpu::lowering::kernel_registry;
use crate::solver::gpu::modules::resource_registry::ResourceRegistry;
use crate::solver::gpu::pipeline_cache::PipelineCache;
use crate::solver::gpu::wgsl_reflect;
use crate::solver::model::KernelId;

pub struct PipelineResources {
    pub bg_linear_matrix: wgpu::BindGroup,
    pub bg_linear_state: wgpu::BindGroup,
    pub bg_dot_params: wgpu::BindGroup,
    pub bg_dot_p_v: wgpu::BindGroup,
    pub bg_dot_r_r: wgpu::BindGroup,
    pub pipeline_spmv_p_v: wgpu::ComputePipeline,
    pub pipeline_dot: wgpu::ComputePipeline,
    pub pipeline_dot_pair: wgpu::ComputePipeline,
    pub pipeline_cg_update_x_r: wgpu::ComputePipeline,
    pub pipeline_cg_update_p: wgpu::ComputePipeline,
}

pub fn init_pipelines(
    device: &wgpu::Device,
    cache: &PipelineCache,
    matrix: &MatrixResources,
    state: &StateResources,
) -> Result<PipelineResources, String> {
    // Pipelines (cached by KernelId so a topology refresh reuses the compiled
    // shader instead of recompiling it). `linear_src`/`dot_src` are still fetched
    // for their `.bindings` (used to build the bind groups below).
    let linear_src = kernel_registry::kernel_source_by_id("", KernelId::LINEAR_SOLVER_SPMV_P_V)
        .map_err(|e| format!("missing linear_solver/spmv_p_v kernel: {e}"))?;
    let pipeline_spmv_p_v = cache.pipeline(device, "", KernelId::LINEAR_SOLVER_SPMV_P_V)?;

    let dot_src = kernel_registry::kernel_source_by_id("", KernelId::DOT_PRODUCT)
        .map_err(|e| format!("missing dot_product kernel: {e}"))?;
    let pipeline_dot = cache.pipeline(device, "", KernelId::DOT_PRODUCT)?;

    let pipeline_dot_pair = cache.pipeline(device, "", KernelId::DOT_PRODUCT_PAIR)?;

    let pipeline_cg_update_x_r =
        cache.pipeline(device, "", KernelId::LINEAR_SOLVER_CG_UPDATE_X_R)?;
    let pipeline_cg_update_p = cache.pipeline(device, "", KernelId::LINEAR_SOLVER_CG_UPDATE_P)?;

    let bgl_linear_state = pipeline_spmv_p_v.get_bind_group_layout(0);
    let bgl_linear_matrix = pipeline_spmv_p_v.get_bind_group_layout(1);

    let bgl_dot_params = pipeline_dot.get_bind_group_layout(0);
    let bgl_dot_inputs = pipeline_dot.get_bind_group_layout(1);

    let linear_registry = ResourceRegistry::new()
        .with_buffer("x", &state.b_x)
        .with_buffer("r", &state.b_r)
        .with_buffer("p", &state.b_p_solver)
        .with_buffer("v", &state.b_v)
        .with_buffer("row_offsets", &matrix.b_row_offsets)
        .with_buffer("col_indices", &matrix.b_col_indices)
        .with_buffer("matrix_values", &matrix.b_matrix_values)
        .with_buffer("scalars", &state.b_scalars)
        .with_buffer("params", &state.b_solver_params);

    let bg_linear_state = wgsl_reflect::create_bind_group_from_bindings(
        device,
        "Linear State Bind Group",
        &bgl_linear_state,
        linear_src.bindings,
        0,
        |name| linear_registry.resolve(name),
    )
    .map_err(|e| format!("failed to create linear state bind group: {e}"))?;

    let bg_linear_matrix = wgsl_reflect::create_bind_group_from_bindings(
        device,
        "Linear Matrix Bind Group",
        &bgl_linear_matrix,
        linear_src.bindings,
        1,
        |name| linear_registry.resolve(name),
    )
    .map_err(|e| format!("failed to create linear matrix bind group: {e}"))?;

    let dot_params_registry = ResourceRegistry::new().with_buffer("params", &state.b_solver_params);
    let bg_dot_params = wgsl_reflect::create_bind_group_from_bindings(
        device,
        "Dot Params Bind Group",
        &bgl_dot_params,
        dot_src.bindings,
        0,
        |name| dot_params_registry.resolve(name),
    )
    .map_err(|e| format!("failed to create dot params bind group: {e}"))?;

    let bg_dot_p_v = {
        let registry = ResourceRegistry::new()
            .with_buffer("dot_result", &state.b_dot_result)
            .with_buffer("dot_a", &state.b_p_solver)
            .with_buffer("dot_b", &state.b_v);
        wgsl_reflect::create_bind_group_from_bindings(
            device,
            "Dot P V Bind Group",
            &bgl_dot_inputs,
            dot_src.bindings,
            1,
            |name| registry.resolve(name),
        )
        .map_err(|e| format!("failed to create dot p_v bind group: {e}"))?
    };

    let bg_dot_r_r = {
        let registry = ResourceRegistry::new()
            .with_buffer("dot_result", &state.b_dot_result)
            .with_buffer("dot_a", &state.b_r)
            .with_buffer("dot_b", &state.b_r);
        wgsl_reflect::create_bind_group_from_bindings(
            device,
            "Dot R R Bind Group",
            &bgl_dot_inputs,
            dot_src.bindings,
            1,
            |name| registry.resolve(name),
        )
        .map_err(|e| format!("failed to create dot r_r bind group: {e}"))?
    };

    Ok(PipelineResources {
        bg_linear_matrix,
        bg_linear_state,
        bg_dot_params,
        bg_dot_p_v,
        bg_dot_r_r,
        pipeline_spmv_p_v,
        pipeline_dot,
        pipeline_dot_pair,
        pipeline_cg_update_x_r,
        pipeline_cg_update_p,
    })
}
