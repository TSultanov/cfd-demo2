//! Enumerating a model's kernels as backend-agnostic `KernelProgram` IR.
//!
//! This mirrors the build-time enumeration in
//! `model::kernel::emit_model_kernels_wgsl_with_ids` (which walks every module's
//! `kernel_generators()` and emits WGSL), but stops at the typed artifact so the
//! CPU backend can consume the `KernelProgram` AST directly instead of WGSL.

use crate::solver::ir::KernelProgram;
use crate::solver::model::backend::SchemeRegistry;
use crate::solver::model::kernel::ModelKernelArtifact;
use crate::solver::model::module::ModelModule;
use crate::solver::model::{KernelId, ModelSpec};

/// All per-model kernel artifacts (in module/registration order), tagged with
/// `KernelId`. The generators are pure functions of `(ModelSpec, SchemeRegistry)`,
/// so the AST can be (re)built at runtime cheaply — the GPU runtime uses
/// pre-generated WGSL, but the IR itself is not GPU-bound.
pub fn model_kernel_artifacts(
    model: &ModelSpec,
    schemes: &SchemeRegistry,
) -> Vec<(KernelId, Result<ModelKernelArtifact, String>)> {
    let mut out = Vec::new();
    for module in &model.modules {
        let module: &dyn ModelModule = module;
        for spec in module.kernel_generators() {
            out.push((spec.id, (spec.generator)(model, schemes)));
        }
    }
    out
}

/// Partition a model's kernels into CPU-executable typed programs and the
/// `KernelId`s that are still WGSL-only (and therefore not yet CPU-runnable).
pub fn model_kernel_programs(
    model: &ModelSpec,
    schemes: &SchemeRegistry,
) -> Result<(Vec<(KernelId, KernelProgram)>, Vec<KernelId>), String> {
    let mut programs = Vec::new();
    let mut wgsl_only = Vec::new();
    for (id, art) in model_kernel_artifacts(model, schemes) {
        match art? {
            ModelKernelArtifact::DslProgram(p) => programs.push((id, p)),
            ModelKernelArtifact::Wgsl(_) => wgsl_only.push(id),
        }
    }
    Ok((programs, wgsl_only))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::scheme::Scheme;

    /// Not an assertion — a developer aid. Run with:
    /// `cargo test --features cpu --lib solver::cpu::lowering::tests::dump -- --nocapture`
    #[test]
    fn dump_scalar_transport_kernels() {
        let model = crate::solver::model::scalar_transport_model().expect("model");
        let schemes = SchemeRegistry::new(Scheme::Upwind);
        for (id, art) in model_kernel_artifacts(&model, &schemes) {
            println!("\n========== kernel `{}` ==========", id.as_str());
            match art {
                Ok(ModelKernelArtifact::DslProgram(p)) => {
                    println!(
                        "[DSL] dispatch={:?} idx=`{}` bounds={:?}",
                        p.dispatch, p.launch.invocation_index_expr, p.launch.bounds_check_expr
                    );
                    println!(
                        "indexing_stmts={} preamble_stmts={} body_stmts={}",
                        p.indexing.len(),
                        p.preamble.len(),
                        p.body.len()
                    );
                    for b in &p.bindings {
                        println!(
                            "  bind g{} b{} {:<28} {:<22} {:?}",
                            b.group, b.binding, b.name, b.wgsl_type, b.access
                        );
                    }
                    match cfd2_codegen::solver::codegen::fusion::lower_kernel_program_to_wgsl(&p) {
                        Ok(w) => println!("---- WGSL ----\n{}", w.to_wgsl()),
                        Err(e) => println!("(lower error: {e})"),
                    }
                }
                Ok(ModelKernelArtifact::Wgsl(w)) => {
                    println!("[WGSL-only]\n{}", w.to_wgsl());
                }
                Err(e) => println!("(generator error: {e})"),
            }
        }
    }
}
