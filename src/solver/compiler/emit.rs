use std::fs;
use std::path::{Path, PathBuf};

use crate::solver::codegen::coupled_assembly::generate_coupled_assembly_wgsl;
use crate::solver::codegen::flux_rhie_chow::generate_flux_rhie_chow_wgsl;
use crate::solver::codegen::generic_coupled_kernels::{
    generate_generic_coupled_apply_wgsl, generate_generic_coupled_update_wgsl,
};
use crate::solver::codegen::ir::{lower_system, DiscreteSystem};
use crate::solver::codegen::prepare_coupled::generate_prepare_coupled_wgsl;
use crate::solver::codegen::pressure_assembly::generate_pressure_assembly_wgsl;
use crate::solver::codegen::unified_assembly;
use crate::solver::codegen::update_fields_from_coupled::generate_update_fields_from_coupled_wgsl;
use crate::solver::codegen::wgsl::generate_wgsl;
use crate::solver::ir::{expand_schemes, EosSpec as IrEosSpec, FluxLayout, SchemeRegistry};
use crate::solver::model::incompressible_momentum_model;
use crate::solver::model::{KernelKind, ModelSpec};
use crate::solver::scheme::Scheme;
use crate::solver::units::si;

pub fn write_wgsl_file(
    system: &DiscreteSystem,
    output_path: impl AsRef<Path>,
) -> std::io::Result<PathBuf> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let content = generate_wgsl(system);
    if let Ok(existing) = fs::read_to_string(output_path) {
        if existing == content {
            return Ok(output_path.to_path_buf());
        }
    }
    fs::write(output_path, content)?;
    Ok(output_path.to_path_buf())
}

pub fn generated_dir_for(base_dir: impl AsRef<Path>) -> PathBuf {
    base_dir
        .as_ref()
        .join("src")
        .join("solver")
        .join("gpu")
        .join("shaders")
        .join("generated")
}

pub fn default_generated_dir() -> PathBuf {
    let base_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    generated_dir_for(base_dir)
}

pub fn write_wgsl_in_dir(
    system: &DiscreteSystem,
    base_dir: impl AsRef<Path>,
    filename: impl AsRef<Path>,
) -> std::io::Result<PathBuf> {
    let output_path = generated_dir_for(base_dir).join(filename);
    write_wgsl_file(system, output_path)
}

fn kernel_output_name(model: &ModelSpec, kind: KernelKind) -> String {
    match kind {
        KernelKind::PrepareCoupled => "prepare_coupled.wgsl".to_string(),
        KernelKind::CoupledAssembly => "coupled_assembly_merged.wgsl".to_string(),
        KernelKind::PressureAssembly => "pressure_assembly.wgsl".to_string(),
        KernelKind::UpdateFieldsFromCoupled => "update_fields_from_coupled.wgsl".to_string(),
        KernelKind::FluxRhieChow => "flux_rhie_chow.wgsl".to_string(),
        KernelKind::SystemMain => "system_main.wgsl".to_string(),

        // Transitional KT flux module bridge.
        KernelKind::KtGradients => "kt_gradients.wgsl".to_string(),
        KernelKind::FluxKt => "flux_kt.wgsl".to_string(),

        KernelKind::GenericCoupledAssembly => {
            format!("generic_coupled_assembly_{}.wgsl", model.id)
        }
        KernelKind::GenericCoupledApply => "generic_coupled_apply.wgsl".to_string(),
        KernelKind::GenericCoupledUpdate => format!("generic_coupled_update_{}.wgsl", model.id),
    }
}

fn generate_kernel_wgsl(
    model: &ModelSpec,
    schemes: &SchemeRegistry,
    kind: KernelKind,
) -> Result<String, String> {
    let discrete = lower_system(&model.system, schemes).map_err(|e| e.to_string())?;

    let wgsl = match kind {
        KernelKind::PrepareCoupled => {
            let fields = derive_coupled_incompressible_field_names(model, &discrete)?;
            generate_prepare_coupled_wgsl(
                &discrete,
                &model.state_layout,
                &fields.momentum,
                &fields.pressure,
                &fields.d_p,
                &fields.grad_p,
            )
        }
        KernelKind::CoupledAssembly => {
            let fields = derive_coupled_incompressible_field_names(model, &discrete)?;
            generate_coupled_assembly_wgsl(
                &discrete,
                &model.state_layout,
                &fields.momentum,
                &fields.pressure,
                &fields.d_p,
            )
        }
        KernelKind::PressureAssembly => {
            let fields = derive_coupled_incompressible_field_names(model, &discrete)?;
            generate_pressure_assembly_wgsl(
                &discrete,
                &model.state_layout,
                &fields.pressure,
                &fields.d_p,
                &fields.grad_p,
            )
        }
        KernelKind::UpdateFieldsFromCoupled => {
            let fields = derive_coupled_incompressible_field_names(model, &discrete)?;
            generate_update_fields_from_coupled_wgsl(
                &model.state_layout,
                &fields.momentum,
                &fields.pressure,
            )
        }
        KernelKind::FluxRhieChow => {
            let fields = derive_coupled_incompressible_field_names(model, &discrete)?;
            generate_flux_rhie_chow_wgsl(
                &discrete,
                &model.state_layout,
                &fields.momentum,
                &fields.pressure,
                &fields.d_p,
                &fields.grad_p,
            )
        }
        KernelKind::SystemMain => generate_wgsl(&discrete),

        KernelKind::KtGradients => {
            crate::solver::codegen::kt_gradients::generate_kt_gradients_wgsl(&model.state_layout)
        }
        KernelKind::FluxKt => {
            let flux_layout = FluxLayout::from_system(&model.system);
            let eos = ir_eos_from_model(model.eos);
            crate::solver::codegen::flux_kt::generate_flux_kt_wgsl(
                &model.state_layout,
                &flux_layout,
                &eos,
            )
        }

        KernelKind::GenericCoupledAssembly => {
            let needs_gradients = expand_schemes(&model.system, schemes)
                .map(|e| e.needs_gradients())
                .unwrap_or(false);
            let flux_stride = model.gpu.flux.map(|f| f.stride).unwrap_or(0);
            unified_assembly::generate_unified_assembly_wgsl(
                &discrete,
                &model.state_layout,
                flux_stride,
                needs_gradients,
            )
        }
        KernelKind::GenericCoupledApply => generate_generic_coupled_apply_wgsl(),
        KernelKind::GenericCoupledUpdate => {
            let prims = model
                .primitives
                .ordered()
                .map_err(|e| format!("primitive recovery ordering failed: {e}"))?;
            generate_generic_coupled_update_wgsl(&discrete, &model.state_layout, &prims)
        }
    };

    Ok(wgsl)
}

#[derive(Debug, Clone)]
struct CoupledIncompressibleFieldNames {
    momentum: String,
    pressure: String,
    d_p: String,
    grad_p: String,
}

fn derive_coupled_incompressible_field_names(
    model: &ModelSpec,
    system: &DiscreteSystem,
) -> Result<CoupledIncompressibleFieldNames, String> {
    let mut momentum_targets = Vec::new();
    for equation in &system.equations {
        if equation.target.kind() == crate::solver::ir::FieldKind::Vector2
            || equation.target.kind() == crate::solver::ir::FieldKind::Vector3
        {
            momentum_targets.push(equation.target);
        }
    }

    let momentum = match momentum_targets.as_slice() {
        [only] => only.name().to_string(),
        [] => return Err("missing momentum equation (no vector equation targets)".to_string()),
        many => {
            let velocity: Vec<_> = many
                .iter()
                .copied()
                .filter(|f| f.unit() == si::VELOCITY)
                .collect();
            match velocity.as_slice() {
                [only] => only.name().to_string(),
                _ => {
                    return Err(format!(
                        "ambiguous momentum equation target: [{}]",
                        many.iter()
                            .map(|f| f.name())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
        }
    };

    let mut pressure_targets = Vec::new();
    for equation in &system.equations {
        if equation.target.kind() == crate::solver::ir::FieldKind::Scalar
            && equation.target.unit() == si::PRESSURE
        {
            pressure_targets.push(equation.target);
        }
    }

    let pressure = if pressure_targets.len() == 1 {
        pressure_targets[0].name().to_string()
    } else {
        let momentum_equation = system
            .equations
            .iter()
            .find(|eq| eq.target.name() == momentum)
            .ok_or_else(|| format!("missing momentum equation for '{momentum}'"))?;

        let mut gradient_fields = Vec::new();
        for op in &momentum_equation.ops {
            if op.kind != crate::solver::codegen::DiscreteOpKind::Gradient {
                continue;
            }
            if op.field.kind() != crate::solver::ir::FieldKind::Scalar {
                continue;
            }
            gradient_fields.push(op.field);
        }

        let gradient_pressure: Vec<_> = gradient_fields
            .iter()
            .copied()
            .filter(|f| f.unit() == si::PRESSURE)
            .collect();
        match gradient_pressure.as_slice() {
            [only] => only.name().to_string(),
            _ => {
                return Err("missing/ambiguous pressure field for coupled-incompressible kernels".to_string());
            }
        }
    };

    let d_p = derive_unique_layout_field_by_unit(
        &model.state_layout,
        si::D_P,
        "d_p",
        Some(&format!("d_{pressure}")),
    )?;
    let grad_p = derive_unique_layout_field_by_unit(
        &model.state_layout,
        si::PRESSURE_GRADIENT,
        "grad_p",
        Some(&format!("grad_{pressure}")),
    )?;

    Ok(CoupledIncompressibleFieldNames {
        momentum,
        pressure,
        d_p,
        grad_p,
    })
}

fn derive_unique_layout_field_by_unit(
    layout: &crate::solver::ir::StateLayout,
    unit: crate::solver::units::UnitDim,
    label: &str,
    preferred_name: Option<&str>,
) -> Result<String, String> {
    if let Some(name) = preferred_name {
        if let Some(field) = layout.field(name) {
            if field.unit() == unit {
                return Ok(name.to_string());
            }
        }
    }

    let candidates: Vec<_> = layout
        .fields()
        .iter()
        .filter(|f| f.unit() == unit)
        .map(|f| f.name().to_string())
        .collect();

    match candidates.as_slice() {
        [] => Err(format!("missing required '{label}' state field (unit={unit})")),
        [only] => Ok(only.clone()),
        many => Err(format!(
            "ambiguous '{label}' state field for coupled-incompressible kernels: [{}]",
            many.join(", ")
        )),
    }
}

fn ir_eos_from_model(eos: crate::solver::model::EosSpec) -> IrEosSpec {
    match eos {
        crate::solver::model::EosSpec::IdealGas { gamma } => IrEosSpec::IdealGas { gamma },
        crate::solver::model::EosSpec::Constant => IrEosSpec::Constant,
    }
}

pub fn emit_model_kernel_wgsl(
    base_dir: impl AsRef<Path>,
    model: &ModelSpec,
    kind: KernelKind,
) -> std::io::Result<PathBuf> {
    let schemes = SchemeRegistry::new(Scheme::Upwind);
    emit_model_kernel_wgsl_with_schemes(base_dir, model, &schemes, kind)
}

pub fn emit_model_kernel_wgsl_with_schemes(
    base_dir: impl AsRef<Path>,
    model: &ModelSpec,
    schemes: &SchemeRegistry,
    kind: KernelKind,
) -> std::io::Result<PathBuf> {
    let base_dir = base_dir.as_ref();
    let wgsl = generate_kernel_wgsl(model, schemes, kind)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
    let output_path = generated_dir_for(base_dir).join(kernel_output_name(model, kind));
    if let Ok(existing) = fs::read_to_string(&output_path) {
        if existing == wgsl {
            return Ok(output_path);
        }
    }
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, wgsl)?;
    Ok(output_path)
}

pub fn emit_model_kernels_wgsl(
    base_dir: impl AsRef<Path>,
    model: &ModelSpec,
    schemes: &SchemeRegistry,
) -> std::io::Result<Vec<PathBuf>> {
    let mut outputs = Vec::new();
    let plan = model.kernel_plan();
    for kind in plan.kernels() {
        outputs.push(emit_model_kernel_wgsl_with_schemes(
            &base_dir, model, schemes, *kind,
        )?);
    }
    Ok(outputs)
}

pub fn emit_coupled_assembly_codegen_wgsl(base_dir: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let schemes = SchemeRegistry::new(Scheme::Upwind);
    emit_coupled_assembly_codegen_wgsl_with_schemes(base_dir, &schemes)
}

pub fn emit_coupled_assembly_codegen_wgsl_with_schemes(
    base_dir: impl AsRef<Path>,
    schemes: &SchemeRegistry,
) -> std::io::Result<PathBuf> {
    let model = incompressible_momentum_model();
    emit_model_kernel_wgsl_with_schemes(base_dir, &model, schemes, KernelKind::CoupledAssembly)
}

pub fn emit_prepare_coupled_codegen_wgsl(base_dir: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let model = incompressible_momentum_model();
    emit_model_kernel_wgsl(base_dir, &model, KernelKind::PrepareCoupled)
}

pub fn emit_pressure_assembly_codegen_wgsl(base_dir: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let model = incompressible_momentum_model();
    emit_model_kernel_wgsl(base_dir, &model, KernelKind::PressureAssembly)
}

pub fn emit_update_fields_from_coupled_codegen_wgsl(
    base_dir: impl AsRef<Path>,
) -> std::io::Result<PathBuf> {
    let model = incompressible_momentum_model();
    emit_model_kernel_wgsl(base_dir, &model, KernelKind::UpdateFieldsFromCoupled)
}

pub fn emit_flux_rhie_chow_codegen_wgsl(base_dir: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let model = incompressible_momentum_model();
    emit_model_kernel_wgsl(base_dir, &model, KernelKind::FluxRhieChow)
}

pub fn emit_system_main_wgsl(base_dir: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let schemes = SchemeRegistry::new(Scheme::Upwind);
    emit_system_main_wgsl_with_schemes(base_dir, &schemes)
}

pub fn emit_system_main_wgsl_with_schemes(
    base_dir: impl AsRef<Path>,
    schemes: &SchemeRegistry,
) -> std::io::Result<PathBuf> {
    let model = incompressible_momentum_model();
    emit_model_kernel_wgsl_with_schemes(
        base_dir,
        &model,
        schemes,
        KernelKind::SystemMain,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::codegen::ir::lower_system;
    use crate::solver::ir::{
        fvm, surface_scalar, vol_vector, Equation, EquationSystem, SchemeRegistry, TermOp,
    };
    use crate::solver::scheme::Scheme;
    use crate::solver::units::si;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static UNIQUE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn unique_suffix() -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{nanos}_{counter}")
    }

    fn temp_output_path() -> PathBuf {
        std::env::temp_dir().join(format!("cfd2_codegen_{}.wgsl", unique_suffix()))
    }

    fn temp_base_dir() -> PathBuf {
        std::env::temp_dir().join(format!("cfd2_codegen_base_{}", unique_suffix()))
    }

    #[test]
    fn emit_model_kernels_writes_expected_outputs() {
        let base_dir = temp_base_dir();
        let model = incompressible_momentum_model();
        let schemes = SchemeRegistry::new(Scheme::Upwind);
        let outputs = emit_model_kernels_wgsl(&base_dir, &model, &schemes).unwrap();

        let plan = model.kernel_plan();
        assert_eq!(outputs.len(), plan.kernels().len());
        for kind in plan.kernels() {
            let expected = generated_dir_for(&base_dir).join(kernel_output_name(&model, *kind));
            assert!(expected.exists());
            assert!(outputs.contains(&expected));
        }
    }

    #[test]
    fn emit_compressible_model_kernels_writes_expected_outputs() {
        let base_dir = temp_base_dir();
        let model = crate::solver::model::compressible_model();
        let schemes = SchemeRegistry::new(Scheme::Upwind);
        let outputs = emit_model_kernels_wgsl(&base_dir, &model, &schemes).unwrap();

        let plan = model.kernel_plan();
        assert_eq!(outputs.len(), plan.kernels().len());
        for kind in plan.kernels() {
            let expected = generated_dir_for(&base_dir).join(kernel_output_name(&model, *kind));
            assert!(expected.exists());
            assert!(outputs.contains(&expected));
        }
    }

    #[test]
    fn write_wgsl_file_creates_parent_and_writes_content() {
        let u = vol_vector("U", si::VELOCITY);
        let phi = surface_scalar("phi", si::MASS_FLUX);
        let eqn = Equation::new(u.clone()).with_term(fvm::div(phi, u));
        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let registry = SchemeRegistry::new(Scheme::Upwind);
        let discrete = lower_system(&system, &registry).unwrap();

        let output_path = temp_output_path();
        let output = write_wgsl_file(&discrete, &output_path).unwrap();

        let content = fs::read_to_string(&output).unwrap();
        assert!(content.contains("GENERATED BY CFD2 CODEGEN"));
        assert!(content.contains("term: div"));

        let _ = fs::remove_file(output);
    }

    #[test]
    fn write_wgsl_file_skips_write_when_unchanged() {
        let u = vol_vector("U", si::VELOCITY);
        let phi = surface_scalar("phi", si::MASS_FLUX);
        let eqn = Equation::new(u.clone()).with_term(fvm::div(phi, u));
        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let registry = SchemeRegistry::new(Scheme::Upwind);
        let discrete = lower_system(&system, &registry).unwrap();

        let output_path = temp_output_path();
        let first = write_wgsl_file(&discrete, &output_path).unwrap();
        let first_content = fs::read_to_string(&first).unwrap();

        let second = write_wgsl_file(&discrete, &output_path).unwrap();
        let second_content = fs::read_to_string(&second).unwrap();

        assert_eq!(first_content, second_content);

        let _ = fs::remove_file(first);
    }

    #[test]
    fn generated_dir_for_appends_expected_path() {
        let base = PathBuf::from("sandbox_root");
        let generated = generated_dir_for(&base);
        assert!(generated.ends_with("src/solver/gpu/shaders/generated"));
        assert!(generated.starts_with(&base));
    }

    #[test]
    fn write_wgsl_in_dir_writes_into_generated_folder() {
        let u = vol_vector("U", si::VELOCITY);
        let phi = surface_scalar("phi", si::MASS_FLUX);
        let eqn = Equation::new(u.clone()).with_term(fvm::div(phi, u));
        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let registry = SchemeRegistry::new(Scheme::Upwind);
        let discrete = lower_system(&system, &registry).unwrap();

        let base_dir = temp_base_dir();
        let output = write_wgsl_in_dir(&discrete, &base_dir, "debug.wgsl").unwrap();

        assert!(output.ends_with("src/solver/gpu/shaders/generated/debug.wgsl"));
        let content = fs::read_to_string(&output).unwrap();
        assert!(content.contains("GENERATED BY CFD2 CODEGEN"));

        let _ = fs::remove_file(&output);
        let _ = fs::remove_dir_all(base_dir.join("src"));
    }

    #[test]
    fn emit_system_main_wgsl_writes_expected_output() {
        let base_dir = temp_base_dir();
        let output = emit_system_main_wgsl(&base_dir).unwrap();

        assert!(output.ends_with("system_main.wgsl"));
        let content = fs::read_to_string(&output).unwrap();
        assert!(content.contains("// equation: U (vector2)"));

        let _ = fs::remove_file(&output);
        let _ = fs::remove_dir_all(base_dir.join("src"));
    }

    #[test]
    fn emit_system_main_wgsl_respects_scheme_registry() {
        let base_dir = temp_base_dir();
        let mut registry = SchemeRegistry::new(Scheme::Upwind);
        registry.set_for_term_names(TermOp::Div, Some("phi"), "U", Scheme::QUICK);

        let output = emit_system_main_wgsl_with_schemes(&base_dir, &registry).unwrap();
        let content = fs::read_to_string(&output).unwrap();

        assert!(content.contains("term_div_phi_U_quick"));

        let _ = fs::remove_file(&output);
        let _ = fs::remove_dir_all(base_dir.join("src"));
    }

    #[test]
    fn emit_coupled_assembly_codegen_wgsl_writes_generated_shader() {
        let base_dir = temp_base_dir();
        let output = emit_coupled_assembly_codegen_wgsl(&base_dir).unwrap();
        let content = fs::read_to_string(&output).unwrap();

        assert!(output.ends_with("generated/coupled_assembly_merged.wgsl"));
        assert!(content.contains("fn main("));
        assert!(content.contains("codegen_assemble_U"));

        let _ = fs::remove_dir_all(base_dir.join("src"));
    }

    #[test]
    fn emit_prepare_coupled_codegen_wgsl_writes_generated_shader() {
        let base_dir = temp_base_dir();
        let output = emit_prepare_coupled_codegen_wgsl(&base_dir).unwrap();
        let content = fs::read_to_string(&output).unwrap();

        assert!(output.ends_with("generated/prepare_coupled.wgsl"));
        assert!(content.contains("fn main("));
        assert!(content.contains("state: array<f32>"));

        let _ = fs::remove_dir_all(base_dir.join("src"));
    }

    #[test]
    fn emit_pressure_assembly_codegen_wgsl_writes_generated_shader() {
        let base_dir = temp_base_dir();
        let output = emit_pressure_assembly_codegen_wgsl(&base_dir).unwrap();
        let content = fs::read_to_string(&output).unwrap();

        assert!(output.ends_with("generated/pressure_assembly.wgsl"));
        assert!(content.contains("fn main("));
        assert!(content.contains("state: array<f32>"));

        let _ = fs::remove_dir_all(base_dir.join("src"));
    }

    #[test]
    fn emit_update_fields_from_coupled_codegen_wgsl_writes_generated_shader() {
        let base_dir = temp_base_dir();
        let output = emit_update_fields_from_coupled_codegen_wgsl(&base_dir).unwrap();
        let content = fs::read_to_string(&output).unwrap();

        assert!(output.ends_with("generated/update_fields_from_coupled.wgsl"));
        assert!(content.contains("fn main("));
        assert!(content.contains("state: array<f32>"));

        let _ = fs::remove_dir_all(base_dir.join("src"));
    }

    #[test]
    fn emit_flux_rhie_chow_codegen_wgsl_writes_generated_shader() {
        let base_dir = temp_base_dir();
        let output = emit_flux_rhie_chow_codegen_wgsl(&base_dir).unwrap();
        let content = fs::read_to_string(&output).unwrap();

        assert!(output.ends_with("generated/flux_rhie_chow.wgsl"));
        assert!(content.contains("fn main("));
        assert!(content.contains("state: array<f32>"));

        let _ = fs::remove_dir_all(base_dir.join("src"));
    }
}
