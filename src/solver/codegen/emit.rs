use std::fs;
use std::path::{Path, PathBuf};

use super::coupled_assembly::generate_coupled_assembly_wgsl;
use super::flux_rhie_chow::generate_flux_rhie_chow_wgsl;
use super::ir::{lower_system, DiscreteSystem};
use super::prepare_coupled::generate_prepare_coupled_wgsl;
use super::pressure_assembly::generate_pressure_assembly_wgsl;
use super::update_fields_from_coupled::generate_update_fields_from_coupled_wgsl;
use super::wgsl::generate_wgsl;
use crate::solver::model::{incompressible_momentum_model, incompressible_momentum_system};
use crate::solver::model::SchemeRegistry;
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
    let base_dir = base_dir.as_ref();
    let model = incompressible_momentum_model();
    let discrete = lower_system(&model.system, schemes);
    let combined = generate_coupled_assembly_wgsl(&discrete, &model.state_layout);

    let output_path = generated_dir_for(base_dir).join("coupled_assembly_merged.wgsl");
    if let Ok(existing) = fs::read_to_string(&output_path) {
        if existing == combined {
            return Ok(output_path);
        }
    }
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, combined)?;
    Ok(output_path)
}

pub fn emit_prepare_coupled_codegen_wgsl(base_dir: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let base_dir = base_dir.as_ref();
    let model = incompressible_momentum_model();
    let schemes = SchemeRegistry::new(Scheme::Upwind);
    let discrete = lower_system(&model.system, &schemes);
    let wgsl = generate_prepare_coupled_wgsl(&discrete, &model.state_layout);

    let output_path = generated_dir_for(base_dir).join("prepare_coupled.wgsl");
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

pub fn emit_pressure_assembly_codegen_wgsl(base_dir: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let base_dir = base_dir.as_ref();
    let model = incompressible_momentum_model();
    let schemes = SchemeRegistry::new(Scheme::Upwind);
    let discrete = lower_system(&model.system, &schemes);
    let wgsl = generate_pressure_assembly_wgsl(&discrete, &model.state_layout);

    let output_path = generated_dir_for(base_dir).join("pressure_assembly.wgsl");
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

pub fn emit_update_fields_from_coupled_codegen_wgsl(
    base_dir: impl AsRef<Path>,
) -> std::io::Result<PathBuf> {
    let base_dir = base_dir.as_ref();
    let model = incompressible_momentum_model();
    let wgsl = generate_update_fields_from_coupled_wgsl(&model.state_layout);

    let output_path = generated_dir_for(base_dir).join("update_fields_from_coupled.wgsl");
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

pub fn emit_flux_rhie_chow_codegen_wgsl(base_dir: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let base_dir = base_dir.as_ref();
    let model = incompressible_momentum_model();
    let schemes = SchemeRegistry::new(Scheme::Upwind);
    let discrete = lower_system(&model.system, &schemes);
    let wgsl = generate_flux_rhie_chow_wgsl(&discrete, &model.state_layout);

    let output_path = generated_dir_for(base_dir).join("flux_rhie_chow.wgsl");
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
    let system = incompressible_momentum_system();
    let discrete = lower_system(&system, schemes);
    write_wgsl_in_dir(&discrete, base_dir, "incompressible_momentum.wgsl")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::ast::{fvm, surface_scalar, vol_vector, Equation, EquationSystem};
    use crate::solver::model::TermOp;
    use crate::solver::codegen::ir::lower_system;
    use crate::solver::model::SchemeRegistry;
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
