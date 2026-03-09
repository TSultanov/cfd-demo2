use crate::solver::model::kernel::{
    DispatchKindId, KernelConditionId, KernelPhaseId, ModelKernelGeneratorSpec, ModelKernelSpec,
};
use crate::solver::model::module::KernelBundleModule;
use crate::solver::model::{FluxLayout, KernelId, ModelSpec};

use cfd2_codegen::solver::codegen::bc_table::BcTable;
use cfd2_codegen::solver::codegen::wgsl_ast::Expr;
use cfd2_codegen::solver::codegen::wgsl_dsl as dsl;
use cfd2_ir::kernel::{
    BindingAccess, DispatchDomain, EffectResource, KernelBinding, KernelProgram, LaunchSemantics,
};

const FIELD_RHO: &str = "rho";
const FIELD_RHO_U: &str = "rho_u";
const FIELD_RHO_E: &str = "rho_e";
const FIELD_P: &str = "p";
const FIELD_T: &str = "T";
const FIELD_U: &str = "u";

const BOUNDARY_INLET: u32 = 1;
const BOUNDARY_OUTLET: u32 = 2;

pub fn compressible_runtime_bc_module() -> KernelBundleModule {
    KernelBundleModule {
        name: "compressible_runtime_bc",
        kernels: vec![ModelKernelSpec {
            id: KernelId::COMPRESSIBLE_RUNTIME_BC_UPDATE,
            phase: KernelPhaseId::Preparation,
            dispatch: DispatchKindId::Faces,
            condition: KernelConditionId::Always,
        }],
        generators: vec![ModelKernelGeneratorSpec::new_dsl(
            KernelId::COMPRESSIBLE_RUNTIME_BC_UPDATE,
            generate_compressible_runtime_bc_kernel_program,
        )],
        ..Default::default()
    }
}

fn generate_compressible_runtime_bc_kernel_program(
    model: &ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<KernelProgram, String> {
    let rho = model
        .state_layout
        .field(FIELD_RHO)
        .ok_or_else(|| "compressible/runtime_bc_update: missing rho field".to_string())?;
    let rho_u = model
        .state_layout
        .field(FIELD_RHO_U)
        .ok_or_else(|| "compressible/runtime_bc_update: missing rho_u field".to_string())?;
    let rho_e = model
        .state_layout
        .field(FIELD_RHO_E)
        .ok_or_else(|| "compressible/runtime_bc_update: missing rho_e field".to_string())?;
    let p = model
        .state_layout
        .field(FIELD_P)
        .ok_or_else(|| "compressible/runtime_bc_update: missing p field".to_string())?;
    let t = model
        .state_layout
        .field(FIELD_T)
        .ok_or_else(|| "compressible/runtime_bc_update: missing T field".to_string())?;
    let u = model
        .state_layout
        .field(FIELD_U)
        .ok_or_else(|| "compressible/runtime_bc_update: missing u field".to_string())?;

    if rho.component_count() != 1
        || rho_u.component_count() != 2
        || rho_e.component_count() != 1
        || p.component_count() != 1
        || t.component_count() != 1
        || u.component_count() != 2
    {
        return Err(
            "compressible/runtime_bc_update: unexpected compressible field component layout"
                .to_string(),
        );
    }

    let state_stride = model.state_layout.stride();
    let flux_layout = FluxLayout::from_system(&model.system);
    let coupled_stride = model.system.unknowns_per_cell();
    let rho_bc_offset = flux_layout
        .offset_for(FIELD_RHO)
        .ok_or_else(|| "compressible/runtime_bc_update: missing rho unknown offset".to_string())?;
    let rho_u_x_bc_offset = flux_layout.offset_for("rho_u_x").ok_or_else(|| {
        "compressible/runtime_bc_update: missing rho_u_x unknown offset".to_string()
    })?;
    let rho_u_y_bc_offset = flux_layout.offset_for("rho_u_y").ok_or_else(|| {
        "compressible/runtime_bc_update: missing rho_u_y unknown offset".to_string()
    })?;
    let rho_e_bc_offset = flux_layout
        .offset_for(FIELD_RHO_E)
        .ok_or_else(|| "compressible/runtime_bc_update: missing rho_e unknown offset".to_string())?;
    let p_bc_offset = flux_layout
        .offset_for(FIELD_P)
        .ok_or_else(|| "compressible/runtime_bc_update: missing p unknown offset".to_string())?;
    let t_bc_offset = flux_layout
        .offset_for(FIELD_T)
        .ok_or_else(|| "compressible/runtime_bc_update: missing T unknown offset".to_string())?;
    let u_x_bc_offset = flux_layout
        .offset_for("u_x")
        .ok_or_else(|| "compressible/runtime_bc_update: missing u_x unknown offset".to_string())?;
    let u_y_bc_offset = flux_layout
        .offset_for("u_y")
        .ok_or_else(|| "compressible/runtime_bc_update: missing u_y unknown offset".to_string())?;

    let launch = LaunchSemantics::new(
        [64, 1, 1],
        "global_id.y * constants.stride_x + global_id.x",
        Some("idx >= arrayLength(&face_boundary)"),
    );
    let bindings = vec![
        KernelBinding::new(0, 0, "state", "array<f32>", BindingAccess::ReadOnlyStorage),
        KernelBinding::new(0, 1, "constants", "Constants", BindingAccess::Uniform),
        KernelBinding::new(0, 2, "bc_value", "array<f32>", BindingAccess::ReadWriteStorage),
        KernelBinding::new(
            1,
            0,
            "face_owner",
            "array<u32>",
            BindingAccess::ReadOnlyStorage,
        ),
        KernelBinding::new(
            1,
            1,
            "face_boundary",
            "array<u32>",
            BindingAccess::ReadOnlyStorage,
        ),
    ];

    let bc = BcTable::new(Expr::ident("idx"), coupled_stride);
    let base = Expr::ident("base");
    let owner = Expr::ident("owner");
    let r_safe = dsl::max(Expr::ident("constants").field("eos_r"), Expr::lit_f32(1.0e-12));
    let gm1 = Expr::ident("constants").field("eos_gm1");
    let gm1_safe = dsl::max(gm1.clone(), Expr::lit_f32(1.0e-6));

    // --- Shared: read interior owner-cell pressure ---
    let p_owner = dsl::max(
        dsl::array_access("state", base.clone() + p.offset()),
        Expr::lit_f32(1.0e-6),
    );

    // --- Inlet branch: refresh p, T, rho_e from interior pressure; keep prescribed rho, u ---
    let rho_target = bc.value(rho_bc_offset);
    let u_x_target = bc.value(u_x_bc_offset);
    let u_y_target = bc.value(u_y_bc_offset);
    let rho_safe_inlet = dsl::max(rho_target.clone(), Expr::lit_f32(1.0e-6));
    let ke_inlet = Expr::lit_f32(0.5)
        * rho_target.clone()
        * (u_x_target.clone() * u_x_target.clone() + u_y_target.clone() * u_y_target.clone());
    let rho_e_inlet = dsl::select(
        ke_inlet.clone(),
        p_owner.clone() / gm1_safe.clone() + ke_inlet.clone(),
        gm1.clone().gt(Expr::lit_f32(0.0)),
    );
    let t_inlet = p_owner.clone() / (rho_safe_inlet * r_safe.clone());

    let inlet_body = dsl::block(vec![
        dsl::assign_expr(bc.value(p_bc_offset), p_owner.clone()),
        dsl::assign_expr(bc.value(t_bc_offset), t_inlet),
        dsl::assign_expr(bc.value(rho_e_bc_offset), rho_e_inlet),
        dsl::assign_expr(bc.value(rho_u_x_bc_offset), rho_target.clone() * u_x_target.clone()),
        dsl::assign_expr(bc.value(rho_u_y_bc_offset), rho_target.clone() * u_y_target.clone()),
        dsl::assign_expr(bc.value(u_x_bc_offset), u_x_target),
        dsl::assign_expr(bc.value(u_y_bc_offset), u_y_target),
    ]);

    // --- Outlet branch: extrapolate non-pressure state from interior; keep Dirichlet pressure ---
    let rho_owner = dsl::max(
        dsl::array_access("state", base.clone() + rho.offset()),
        Expr::lit_f32(1.0e-6),
    );
    let u_x_owner = dsl::array_access("state", base.clone() + u.offset());
    let u_y_owner = dsl::array_access("state", base.clone() + u.offset() + 1u32);
    let p_outlet = bc.value(p_bc_offset);
    let ke_outlet = Expr::lit_f32(0.5)
        * rho_owner.clone()
        * (u_x_owner.clone() * u_x_owner.clone() + u_y_owner.clone() * u_y_owner.clone());
    let rho_e_outlet = dsl::select(
        ke_outlet.clone(),
        p_outlet.clone() / gm1_safe + ke_outlet.clone(),
        gm1.gt(Expr::lit_f32(0.0)),
    );
    let t_outlet = p_outlet / (rho_owner.clone() * r_safe);

    let outlet_body = dsl::block(vec![
        dsl::assign_expr(bc.value(rho_bc_offset), rho_owner.clone()),
        dsl::assign_expr(bc.value(u_x_bc_offset), u_x_owner.clone()),
        dsl::assign_expr(bc.value(u_y_bc_offset), u_y_owner.clone()),
        dsl::assign_expr(bc.value(rho_u_x_bc_offset), rho_owner.clone() * u_x_owner),
        dsl::assign_expr(bc.value(rho_u_y_bc_offset), rho_owner.clone() * u_y_owner),
        dsl::assign_expr(bc.value(t_bc_offset), t_outlet),
        dsl::assign_expr(bc.value(rho_e_bc_offset), rho_e_outlet),
    ]);

    let mut program = KernelProgram::new(
        KernelId::COMPRESSIBLE_RUNTIME_BC_UPDATE.as_str(),
        DispatchDomain::Faces,
        launch,
        bindings,
    );
    program.indexing = vec![
        dsl::let_expr("face_boundary_type", dsl::array_access("face_boundary", Expr::ident("idx"))),
        dsl::let_expr("is_inlet", Expr::ident("face_boundary_type").eq(Expr::from(BOUNDARY_INLET))),
        dsl::let_expr("is_outlet", Expr::ident("face_boundary_type").eq(Expr::from(BOUNDARY_OUTLET))),
        dsl::if_block_expr(
            !Expr::ident("is_inlet") & !Expr::ident("is_outlet"),
            dsl::block(vec![dsl::return_void()]),
            None,
        ),
        dsl::let_expr("owner", dsl::array_access("face_owner", Expr::ident("idx"))),
        dsl::let_expr("base", owner.clone() * state_stride),
    ];
    program.body = vec![
        dsl::if_block_expr(Expr::ident("is_inlet"), inlet_body, None),
        dsl::if_block_expr(Expr::ident("is_outlet"), outlet_body, None),
    ];
    program
        .side_effects
        .read_set
        .insert(EffectResource::binding(0, 0));
    program
        .side_effects
        .read_set
        .insert(EffectResource::binding(0, 1));
    program
        .side_effects
        .write_set
        .insert(EffectResource::binding(0, 2));
    Ok(program)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressible_runtime_bc_module_exposes_preparation_face_kernel() {
        let module = compressible_runtime_bc_module();
        assert_eq!(module.kernels.len(), 1);
        assert_eq!(module.kernels[0].id, KernelId::COMPRESSIBLE_RUNTIME_BC_UPDATE);
        assert_eq!(module.kernels[0].phase, KernelPhaseId::Preparation);
        assert_eq!(module.kernels[0].dispatch, DispatchKindId::Faces);
    }

    #[test]
    fn compressible_runtime_bc_kernel_writes_bc_values() {
        let model = crate::solver::model::compressible_model().expect("model");
        let program = generate_compressible_runtime_bc_kernel_program(
            &model,
            &crate::solver::ir::SchemeRegistry::default(),
        )
        .expect("runtime bc kernel");
        assert!(program
            .bindings
            .iter()
            .any(|binding| binding.name == "bc_value" && binding.access.allows_write()));
    }

    #[test]
    fn compressible_model_sets_outlet_pressure_dirichlet() {
        let model = crate::solver::model::compressible_model().expect("model");
        let (kind, _value) = model
            .boundaries
            .to_gpu_tables(&model.system)
            .expect("gpu tables");
        let flux_layout = crate::solver::model::FluxLayout::from_system(&model.system);
        let p_offset = flux_layout.offset_for(FIELD_P).expect("pressure offset") as usize;
        let stride = model.system.unknowns_per_cell() as usize;
        let outlet_index = BOUNDARY_OUTLET as usize * stride + p_offset;
        assert_eq!(
            kind[outlet_index],
            cfd2_ir::gpu_enums::GpuBcKind::Dirichlet as u32
        );
    }

    #[test]
    fn compressible_runtime_bc_kernel_handles_inlet_and_outlet() {
        let model = crate::solver::model::compressible_model().expect("model");
        let program = generate_compressible_runtime_bc_kernel_program(
            &model,
            &crate::solver::ir::SchemeRegistry::default(),
        )
        .expect("runtime bc kernel");
        // The kernel body should contain two conditional blocks (inlet and outlet).
        assert_eq!(
            program.body.len(),
            2,
            "expected exactly two conditional blocks in the runtime BC kernel body (inlet + outlet)"
        );
    }
}