// Generic boundary-expression refresh kernel.
//
// Lowers every expression-valued boundary condition declared in a model's
// `BoundarySpec` (see `BcValue::Expr` and `backend::boundary::BoundaryExpr`)
// into ONE Preparation-phase Faces kernel that rewrites the corresponding
// `bc_value` table entries from interior state, prescribed boundary values,
// and uniform params -- every outer iteration. This replaces hand-written
// runtime-BC kernels (e.g. the old compressible_runtime_bc module).
//
// Semantics implemented here (the contract `BoundaryExpr` documents):
// - all `bc(..)` reads are hoisted into `let` snapshots before any write,
//   so the expression set is order-independent and reads observe the
//   host-prescribed values, never sibling writes from the same refresh;
// - `interior(..)` reads the face owner's current state;
// - the kernel guards each boundary type with a body-wrapping `if`
//   (never an early `return`: fused downstream segments must keep running
//   for faces this kernel ignores).

use crate::solver::model::backend::boundary::BoundaryExpr;
use crate::solver::model::kernel::{
    DispatchKindId, KernelConditionId, KernelPhaseId, ModelKernelGeneratorSpec, ModelKernelSpec,
};
use crate::solver::model::module::KernelBundleModule;
use crate::solver::model::{FluxLayout, KernelId, ModelSpec};

use cfd2_codegen::solver::codegen::bc_table::BcTable;
use cfd2_codegen::solver::codegen::wgsl_ast::{Expr, Stmt};
use cfd2_codegen::solver::codegen::wgsl_dsl as dsl;
use cfd2_ir::kernel::{
    BindingAccess, DispatchDomain, EffectResource, KernelBinding, KernelProgram, LaunchSemantics,
};

pub fn bc_expr_module() -> KernelBundleModule {
    KernelBundleModule {
        name: "bc_expr",
        kernels: vec![ModelKernelSpec {
            id: KernelId::BC_EXPR_UPDATE,
            phase: KernelPhaseId::Preparation,
            dispatch: DispatchKindId::Faces,
            condition: KernelConditionId::Always,
        }],
        generators: vec![ModelKernelGeneratorSpec::new_dsl(
            KernelId::BC_EXPR_UPDATE,
            generate_bc_expr_kernel_program,
        )],
        ..Default::default()
    }
}

/// One expression-valued entry: boundary type, target unknown component,
/// and the declared expression.
struct ExprEntry {
    boundary: crate::solver::gpu::enums::GpuBoundaryType,
    target_offset: u32,
    expr: BoundaryExpr,
}

/// Collect expression-valued boundary conditions from the model in a
/// deterministic order (boundary type, then coupled-unknown order).
fn collect_expr_entries(model: &ModelSpec) -> Result<Vec<ExprEntry>, String> {
    use crate::solver::gpu::enums::GpuBoundaryType;

    let flux_layout = FluxLayout::from_system(&model.system);
    let boundary_types = [
        GpuBoundaryType::None,
        GpuBoundaryType::Inlet,
        GpuBoundaryType::Outlet,
        GpuBoundaryType::Wall,
        GpuBoundaryType::SlipWall,
        GpuBoundaryType::MovingWall,
    ];

    let mut entries = Vec::new();
    for &boundary in &boundary_types {
        for eqn in model.system.equations() {
            let field = eqn.target();
            let Some(spec) = model.boundaries.field(field.name()) else {
                continue;
            };
            let Some(conditions) = spec.by_boundary.get(&boundary) else {
                continue;
            };
            for (component, condition) in conditions.iter().enumerate() {
                let Some(expr) = condition.expr_value() else {
                    continue;
                };
                let target_offset =
                    unknown_offset(&flux_layout, field, component as u32).ok_or_else(|| {
                        format!(
                            "bc_expr: no coupled unknown offset for '{}' component {}",
                            field.name(),
                            component
                        )
                    })?;
                entries.push(ExprEntry {
                    boundary,
                    target_offset,
                    expr: expr.clone(),
                });
            }
        }
    }
    Ok(entries)
}

/// Coupled-unknown offset of a field component (FluxLayout names vector
/// components `<field>_x` / `<field>_y` / `<field>_z`).
fn unknown_offset(
    flux_layout: &FluxLayout,
    field: &crate::solver::model::backend::ast::FieldRef,
    component: u32,
) -> Option<u32> {
    if field.kind().component_count() == 1 {
        flux_layout.offset_for(field.name())
    } else {
        let suffix = component_suffix(component)?;
        flux_layout.offset_for(&format!("{}_{}", field.name(), suffix))
    }
}

fn component_suffix(component: u32) -> Option<&'static str> {
    match component {
        0 => Some("x"),
        1 => Some("y"),
        2 => Some("z"),
        _ => None,
    }
}

/// Snapshot-let identifier for an interior (owner-cell) read.
fn interior_ident(field: &str, component: u32) -> String {
    format!("in_{}_c{}", field, component)
}

/// Snapshot-let identifier for a prescribed boundary-value read.
fn bc_ident(field: &str, component: u32) -> String {
    format!("bcv_{}_c{}", field, component)
}

/// Lower a `BoundaryExpr` to WGSL, resolving atoms to snapshot lets.
fn lower_expr(expr: &BoundaryExpr) -> Expr {
    match expr {
        BoundaryExpr::Lit(v) => Expr::lit_f32(*v as f32),
        BoundaryExpr::Param(p) => Expr::ident("constants").field(p.name()),
        BoundaryExpr::Interior { field, component } => {
            Expr::ident(interior_ident(field.name(), *component))
        }
        BoundaryExpr::BcValue { field, component } => {
            Expr::ident(bc_ident(field.name(), *component))
        }
        BoundaryExpr::Add(a, b) => lower_expr(a) + lower_expr(b),
        BoundaryExpr::Sub(a, b) => lower_expr(a) - lower_expr(b),
        BoundaryExpr::Mul(a, b) => lower_expr(a) * lower_expr(b),
        BoundaryExpr::Div(a, b) => lower_expr(a) / lower_expr(b),
        BoundaryExpr::Neg(a) => -lower_expr(a),
        BoundaryExpr::Max(a, b) => dsl::max(lower_expr(a), lower_expr(b)),
        BoundaryExpr::Min(a, b) => dsl::min(lower_expr(a), lower_expr(b)),
        BoundaryExpr::SelectGt {
            lhs,
            rhs,
            on_true,
            on_false,
        } => dsl::select(
            lower_expr(on_false),
            lower_expr(on_true),
            lower_expr(lhs).gt(lower_expr(rhs)),
        ),
    }
}

fn generate_bc_expr_kernel_program(
    model: &ModelSpec,
    _schemes: &crate::solver::ir::SchemeRegistry,
) -> Result<KernelProgram, String> {
    let entries = collect_expr_entries(model)?;
    if entries.is_empty() {
        return Err(
            "bc_expr module attached but the model declares no expression-valued boundary \
             conditions"
                .to_string(),
        );
    }

    let flux_layout = FluxLayout::from_system(&model.system);
    let coupled_stride = model.system.unknowns_per_cell();
    let state_stride = model.state_layout.stride();

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

    // Group entries per boundary type, preserving collection order.
    let mut boundaries: Vec<crate::solver::gpu::enums::GpuBoundaryType> = Vec::new();
    for entry in &entries {
        if !boundaries.contains(&entry.boundary) {
            boundaries.push(entry.boundary);
        }
    }

    let mut body: Vec<Stmt> = Vec::new();
    for boundary in boundaries {
        let group: Vec<&ExprEntry> = entries
            .iter()
            .filter(|e| e.boundary == boundary)
            .collect();

        // Collect unique Interior/BcValue atoms across the group's
        // expressions, in first-reference order.
        let mut interior_reads: Vec<(String, u32, u32)> = Vec::new(); // (name, component, state offset)
        let mut bc_reads: Vec<(String, u32, u32)> = Vec::new(); // (name, component, unknown offset)
        for entry in &group {
            let mut err: Option<String> = None;
            entry.expr.visit(&mut |node| match node {
                BoundaryExpr::Interior { field, component } => {
                    let key = (field.name().to_string(), *component);
                    if !interior_reads.iter().any(|(n, c, _)| (n, c) == (&key.0, &key.1)) {
                        match model.state_layout.field(field.name()) {
                            Some(state_field) => interior_reads.push((
                                key.0,
                                key.1,
                                state_field.offset() + *component,
                            )),
                            None => {
                                err.get_or_insert(format!(
                                    "bc_expr: interior({}) is not a state field",
                                    field.name()
                                ));
                            }
                        }
                    }
                }
                BoundaryExpr::BcValue { field, component } => {
                    let key = (field.name().to_string(), *component);
                    if !bc_reads.iter().any(|(n, c, _)| (n, c) == (&key.0, &key.1)) {
                        match unknown_offset(&flux_layout, field, *component) {
                            Some(offset) => bc_reads.push((key.0, key.1, offset)),
                            None => {
                                err.get_or_insert(format!(
                                    "bc_expr: bc({}) is not a coupled unknown",
                                    field.name()
                                ));
                            }
                        }
                    }
                }
                _ => {}
            });
            if let Some(err) = err {
                return Err(err);
            }
        }

        let mut stmts: Vec<Stmt> = Vec::new();
        // Snapshot reads first (order-independent write semantics).
        for (name, component, state_offset) in &interior_reads {
            stmts.push(dsl::let_expr(
                &interior_ident(name, *component),
                dsl::array_access("state", base.clone() + *state_offset),
            ));
        }
        for (name, component, unknown_off) in &bc_reads {
            stmts.push(dsl::let_expr(
                &bc_ident(name, *component),
                bc.value(*unknown_off),
            ));
        }
        // Then the assignments, in declaration order.
        for entry in &group {
            stmts.push(dsl::assign_expr(
                bc.value(entry.target_offset),
                lower_expr(&entry.expr),
            ));
        }

        body.push(dsl::if_block_expr(
            Expr::ident("face_boundary_type").eq(Expr::from(boundary as u32)),
            dsl::block(stmts),
            None,
        ));
    }

    let mut program = KernelProgram::new(
        KernelId::BC_EXPR_UPDATE.as_str(),
        DispatchDomain::Faces,
        launch,
        bindings,
    );
    program.indexing = vec![
        dsl::let_expr(
            "face_boundary_type",
            dsl::array_access("face_boundary", Expr::ident("idx")),
        ),
        dsl::let_expr("owner", dsl::array_access("face_owner", Expr::ident("idx"))),
        dsl::let_expr("base", Expr::ident("owner") * state_stride),
    ];
    program.body = body;
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
    use crate::solver::gpu::enums::GpuBoundaryType;
    use crate::solver::model::backend::ast::FieldRef;

    fn diffusion_model_with_expr_outlet() -> ModelSpec {
        let mut model =
            crate::solver::model::generic_diffusion_demo_model().expect("model");
        let phi: FieldRef = *model.system.equations()[0].target();
        // Outlet phi := max(interior phi, 0) -- a synthetic expression BC.
        let expr = BoundaryExpr::interior(phi).max(BoundaryExpr::lit(0.0));
        let spec = crate::solver::model::definitions::FieldBoundarySpec::new().set_components(
            GpuBoundaryType::Outlet,
            vec![crate::solver::model::BoundaryCondition {
                kind: cfd2_ir::gpu_enums::GpuBcKind::Dirichlet,
                value: crate::solver::model::BcValue::Expr(expr),
                unit: phi.unit(),
            }],
        );
        model.boundaries.set_field(phi.name(), spec);
        model
    }

    #[test]
    fn generates_snapshot_then_assign_with_no_early_return() {
        let model = diffusion_model_with_expr_outlet();
        let program = generate_bc_expr_kernel_program(
            &model,
            &crate::solver::ir::SchemeRegistry::default(),
        )
        .expect("bc_expr kernel");

        // One guarded block for the outlet, no return statements anywhere.
        assert_eq!(program.body.len(), 1);
        let wgsl = format!("{program:?}");
        assert!(
            !wgsl.contains("Return"),
            "bc_expr kernel must not early-return (fusion safety)"
        );
        assert!(program
            .bindings
            .iter()
            .any(|b| b.name == "bc_value" && b.access.allows_write()));
    }

    #[test]
    fn rejects_models_without_expression_bcs() {
        let model = crate::solver::model::generic_diffusion_demo_model().expect("model");
        let err = generate_bc_expr_kernel_program(
            &model,
            &crate::solver::ir::SchemeRegistry::default(),
        )
        .unwrap_err();
        assert!(err.contains("no expression-valued"), "{err}");
    }

    #[test]
    fn compressible_generates_inlet_and_outlet_blocks() {
        let model = crate::solver::model::compressible_model().expect("model");
        let program = generate_bc_expr_kernel_program(
            &model,
            &crate::solver::ir::SchemeRegistry::default(),
        )
        .expect("bc_expr kernel");
        assert_eq!(
            program.body.len(),
            2,
            "expected exactly two guarded blocks (inlet + outlet)"
        );
        assert!(program
            .bindings
            .iter()
            .any(|b| b.name == "bc_value" && b.access.allows_write()));
    }

    #[test]
    fn compressible_model_sets_outlet_pressure_dirichlet() {
        let model = crate::solver::model::compressible_model().expect("model");
        let (kind, _value) = model
            .boundaries
            .to_gpu_tables(&model.system)
            .expect("gpu tables");
        let flux_layout = crate::solver::model::FluxLayout::from_system(&model.system);
        let p_offset = flux_layout.offset_for("p").expect("pressure offset") as usize;
        let stride = model.system.unknowns_per_cell() as usize;
        let outlet_index = GpuBoundaryType::Outlet as usize * stride + p_offset;
        assert_eq!(
            kind[outlet_index],
            cfd2_ir::gpu_enums::GpuBcKind::Dirichlet as u32
        );
    }
}
