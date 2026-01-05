use std::fs;
use std::path::{Path, PathBuf};

use super::compressible_assembly::generate_compressible_assembly_wgsl;
use super::compressible_apply::generate_compressible_apply_wgsl;
use super::compressible_gradients::generate_compressible_gradients_wgsl;
use super::compressible_flux_kt::generate_compressible_flux_kt_wgsl;
use super::compressible_update::generate_compressible_update_wgsl;
use super::coupled_assembly::generate_coupled_assembly_wgsl;
use super::flux_rhie_chow::generate_flux_rhie_chow_wgsl;
use super::ir::{lower_system, DiscreteSystem};
use super::prepare_coupled::generate_prepare_coupled_wgsl;
use super::pressure_assembly::generate_pressure_assembly_wgsl;
use super::update_fields_from_coupled::generate_update_fields_from_coupled_wgsl;
use super::wgsl::generate_wgsl;
use crate::solver::model::{incompressible_momentum_model, KernelKind, ModelSpec};
use crate::solver::model::backend::SchemeRegistry;
use crate::solver::scheme::Scheme;

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
    let base_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    generated_dir_for(base_dir)
}

pub fn write_wgsl_in_dir(
    system: &DiscreteSystem,
    base_dir: impl AsRef<Path>,
    file_name: &str,
) -> std::io::Result<PathBuf> {
    let output_dir = generated_dir_for(base_dir);
    let output_path = output_dir.join(file_name);
    write_wgsl_file(system, output_path)
}

fn kernel_output_name(kind: KernelKind) -> &'static str {
    match kind {
        KernelKind::PrepareCoupled => "prepare_coupled.wgsl",
        KernelKind::CoupledAssembly => "coupled_assembly_merged.wgsl",
        KernelKind::PressureAssembly => "pressure_assembly.wgsl",
        KernelKind::UpdateFieldsFromCoupled => "update_fields_from_coupled.wgsl",
        KernelKind::FluxRhieChow => "flux_rhie_chow.wgsl",
        KernelKind::IncompressibleMomentum => "incompressible_momentum.wgsl",
        KernelKind::CompressibleAssembly => "compressible_assembly.wgsl",
        KernelKind::CompressibleApply => "compressible_apply.wgsl",
        KernelKind::CompressibleGradients => "compressible_gradients.wgsl",
        KernelKind::CompressibleUpdate => "compressible_update.wgsl",
        KernelKind::CompressibleFluxKt => "compressible_flux_kt.wgsl",
    }
}

fn generate_kernel_wgsl(
    model: &ModelSpec,
    schemes: &SchemeRegistry,
    kind: KernelKind,
) -> Result<String, String> {
    let discrete = lower_system(&model.system, schemes);
    let wgsl = match kind {
        KernelKind::PrepareCoupled => {
            let fields = model
                .fields
                .incompressible()
                .ok_or_else(|| "prepare_coupled requires incompressible fields".to_string())?;
            generate_prepare_coupled_wgsl(&discrete, &model.state_layout, fields)
        }
        KernelKind::CoupledAssembly => {
            let fields = model
                .fields
                .incompressible()
                .ok_or_else(|| "coupled_assembly requires incompressible fields".to_string())?;
            generate_coupled_assembly_wgsl(&discrete, &model.state_layout, fields)
        }
        KernelKind::PressureAssembly => {
            let fields = model
                .fields
                .incompressible()
                .ok_or_else(|| "pressure_assembly requires incompressible fields".to_string())?;
            generate_pressure_assembly_wgsl(&discrete, &model.state_layout, fields)
        }
        KernelKind::UpdateFieldsFromCoupled => {
            let fields = model
                .fields
                .incompressible()
                .ok_or_else(|| "update_fields requires incompressible fields".to_string())?;
            generate_update_fields_from_coupled_wgsl(&model.state_layout, fields)
        }
        KernelKind::FluxRhieChow => {
            let fields = model
                .fields
                .incompressible()
                .ok_or_else(|| "flux_rhie_chow requires incompressible fields".to_string())?;
            generate_flux_rhie_chow_wgsl(&discrete, &model.state_layout, fields)
        }
        KernelKind::IncompressibleMomentum => generate_wgsl(&discrete),
        KernelKind::CompressibleAssembly => {
            let fields = model
                .fields
                .compressible()
                .ok_or_else(|| "compressible_assembly requires compressible fields".to_string())?;
            generate_compressible_assembly_wgsl(&model.state_layout, fields)
        }
        KernelKind::CompressibleApply => {
            let fields = model
                .fields
                .compressible()
                .ok_or_else(|| "compressible_apply requires compressible fields".to_string())?;
            generate_compressible_apply_wgsl(&model.state_layout, fields)
        }
        KernelKind::CompressibleGradients => {
            let fields = model
                .fields
                .compressible()
                .ok_or_else(|| "compressible_gradients requires compressible fields".to_string())?;
            generate_compressible_gradients_wgsl(&model.state_layout, fields)
        }
        KernelKind::CompressibleUpdate => {
            let fields = model
                .fields
                .compressible()
                .ok_or_else(|| "compressible_update requires compressible fields".to_string())?;
            generate_compressible_update_wgsl(&model.state_layout, fields)
        }
        KernelKind::CompressibleFluxKt => {
            let fields = model
                .fields
                .compressible()
                .ok_or_else(|| "compressible_flux_kt requires compressible fields".to_string())?;
            generate_compressible_flux_kt_wgsl(&model.state_layout, fields)
        }
    };
    Ok(wgsl)
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
    let wgsl = generate_kernel_wgsl(model, schemes, kind).map_err(|err| {
        std::io::Error::new(std::io::ErrorKind::Other, err)
    })?;
    let output_path = generated_dir_for(base_dir).join(kernel_output_name(kind));
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
    for kind in model.kernel_plan.kernels() {
        outputs.push(emit_model_kernel_wgsl_with_schemes(
            &base_dir, model, schemes, *kind,
        )?);
    }
    Ok(outputs)
}

pub fn emit_coupled_assembly_codegen_wgsl(
    base_dir: impl AsRef<Path>,
) -> std::io::Result<PathBuf> {
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

pub fn emit_incompressible_momentum_wgsl(
    base_dir: impl AsRef<Path>,
) -> std::io::Result<PathBuf> {
    let schemes = SchemeRegistry::new(Scheme::Upwind);
    emit_incompressible_momentum_wgsl_with_schemes(base_dir, &schemes)
}

pub fn emit_incompressible_momentum_wgsl_with_schemes(
    base_dir: impl AsRef<Path>,
    schemes: &SchemeRegistry,
) -> std::io::Result<PathBuf> {
    let model = incompressible_momentum_model();
    emit_model_kernel_wgsl_with_schemes(
        base_dir,
        &model,
        schemes,
        KernelKind::IncompressibleMomentum,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::backend::ast::{fvm, surface_scalar, vol_vector, Equation, EquationSystem};
    use crate::solver::model::backend::ast::TermOp;
    use crate::solver::codegen::ir::lower_system;
    use crate::solver::model::backend::SchemeRegistry;
    use crate::solver::scheme::Scheme;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_output_path() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cfd2_codegen_{}.wgsl", nanos))
    }

    fn temp_base_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cfd2_codegen_base_{}", nanos))
    }

    #[test]
    fn emit_model_kernels_writes_expected_outputs() {
        let base_dir = temp_base_dir();
        let model = incompressible_momentum_model();
        let schemes = SchemeRegistry::new(Scheme::Upwind);
        let outputs = emit_model_kernels_wgsl(&base_dir, &model, &schemes).unwrap();

        assert_eq!(outputs.len(), model.kernel_plan.kernels().len());
        for kind in model.kernel_plan.kernels() {
            let expected = generated_dir_for(&base_dir).join(kernel_output_name(*kind));
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

        assert_eq!(outputs.len(), model.kernel_plan.kernels().len());
        for kind in model.kernel_plan.kernels() {
            let expected = generated_dir_for(&base_dir).join(kernel_output_name(*kind));
            assert!(expected.exists());
            assert!(outputs.contains(&expected));
        }
    }

    #[test]
    fn write_wgsl_file_creates_parent_and_writes_content() {
        let u = vol_vector("U");
        let phi = surface_scalar("phi");
        let eqn = Equation::new(u.clone()).with_term(fvm::div(phi, u));
        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let registry = SchemeRegistry::new(Scheme::Upwind);
        let discrete = lower_system(&system, &registry);

        let output_path = temp_output_path();
        let output = write_wgsl_file(&discrete, &output_path).unwrap();

        let content = fs::read_to_string(&output).unwrap();
        assert!(content.contains("GENERATED BY CFD2 CODEGEN"));
        assert!(content.contains("term: div"));

        let _ = fs::remove_file(output);
    }

    #[test]
    fn write_wgsl_file_skips_write_when_unchanged() {
        let u = vol_vector("U");
        let phi = surface_scalar("phi");
        let eqn = Equation::new(u.clone()).with_term(fvm::div(phi, u));
        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let registry = SchemeRegistry::new(Scheme::Upwind);
        let discrete = lower_system(&system, &registry);

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
        let u = vol_vector("U");
        let phi = surface_scalar("phi");
        let eqn = Equation::new(u.clone()).with_term(fvm::div(phi, u));
        let mut system = EquationSystem::new();
        system.add_equation(eqn);

        let registry = SchemeRegistry::new(Scheme::Upwind);
        let discrete = lower_system(&system, &registry);

        let base_dir = temp_base_dir();
        let output = write_wgsl_in_dir(&discrete, &base_dir, "debug.wgsl").unwrap();

        assert!(output.ends_with("src/solver/gpu/shaders/generated/debug.wgsl"));
        let content = fs::read_to_string(&output).unwrap();
        assert!(content.contains("GENERATED BY CFD2 CODEGEN"));

        let _ = fs::remove_file(&output);
        let _ = fs::remove_dir_all(base_dir.join("src"));
    }

    #[test]
    fn emit_incompressible_momentum_wgsl_writes_expected_output() {
        let base_dir = temp_base_dir();
        let output = emit_incompressible_momentum_wgsl(&base_dir).unwrap();

        assert!(output.ends_with("incompressible_momentum.wgsl"));
        let content = fs::read_to_string(&output).unwrap();
        assert!(content.contains("// equation: U (vector2)"));

        let _ = fs::remove_file(&output);
        let _ = fs::remove_dir_all(base_dir.join("src"));
    }

    #[test]
    fn emit_incompressible_momentum_wgsl_respects_scheme_registry() {
        let base_dir = temp_base_dir();
        let mut registry = SchemeRegistry::new(Scheme::Upwind);
        registry.set_for_term_names(TermOp::Div, Some("phi"), "U", Scheme::QUICK);

        let output = emit_incompressible_momentum_wgsl_with_schemes(&base_dir, &registry).unwrap();
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
