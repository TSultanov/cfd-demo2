use crate::solver::gpu::wgsl_reflect::WgslBindingDesc;
use crate::solver::model::KernelId;

pub(crate) struct KernelSource {
    pub bindings: &'static [WgslBindingDesc],
    pub create_pipeline: fn(&wgpu::Device) -> wgpu::ComputePipeline,
}

mod generated {
    include!(concat!(env!("OUT_DIR"), "/kernel_registry_map.rs"));
}

pub(crate) fn kernel_source_by_id(
    model_id: &str,
    kernel_id: KernelId,
) -> Result<KernelSource, String> {
    if let Some((_shader, create_pipeline, bindings)) =
        generated::kernel_entry_by_id(model_id, kernel_id.as_str())
    {
        return Ok(KernelSource {
            bindings,
            create_pipeline,
        });
    }

    Err(format!(
        "KernelId '{}' does not have a generated kernel source entry",
        kernel_id.as_str()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesized_rhie_chow_fused_kernel_is_present_in_generated_registry() {
        let src = kernel_source_by_id(
            "incompressible_momentum",
            KernelId("rhie_chow/dp_update_store_grad_p_fused"),
        )
        .expect("missing synthesized fused rhie-chow kernel in generated registry");
        assert!(
            !src.bindings.is_empty(),
            "fused rhie-chow kernel registry entry should include reflected bindings"
        );
    }

    #[test]
    fn synthesized_rhie_chow_aggressive_fused_kernel_is_present_in_generated_registry() {
        let src = kernel_source_by_id(
            "incompressible_momentum",
            KernelId("rhie_chow/dp_update_store_grad_p_grad_p_update_fused"),
        )
        .expect("missing aggressive synthesized fused rhie-chow kernel in generated registry");
        assert!(
            !src.bindings.is_empty(),
            "aggressive fused rhie-chow kernel registry entry should include reflected bindings"
        );
    }

    #[test]
    fn synthesized_rhie_chow_aggressive_full_fused_kernel_is_present_in_generated_registry() {
        let src = kernel_source_by_id(
            "incompressible_momentum",
            KernelId("rhie_chow/dp_update_store_grad_p_grad_p_update_correct_velocity_delta_fused"),
        )
        .expect("missing aggressive full synthesized fused rhie-chow kernel in generated registry");
        assert!(
            !src.bindings.is_empty(),
            "aggressive full fused rhie-chow kernel registry entry should include reflected bindings"
        );
    }

    #[test]
    fn synthesized_rhie_chow_grad_p_update_correct_velocity_delta_fused_kernel_is_present_in_generated_registry(
    ) {
        // Test the standalone grad_p_update + correct_velocity_delta fused kernel
        // This is the aggressive-only pair fusion that does not include dp_update or store_grad_p.
        let src = kernel_source_by_id(
            "incompressible_momentum",
            KernelId("rhie_chow/grad_p_update_correct_velocity_delta_fused"),
        )
        .expect(
            "missing standalone grad_p_update_correct_velocity_delta fused kernel in generated registry"
        );
        assert!(
            !src.bindings.is_empty(),
            "standalone grad_p_update_correct_velocity_delta fused kernel registry entry should include reflected bindings"
        );
    }

    #[test]
    fn synthesized_rhie_chow_store_grad_p_grad_p_update_fused_kernel_is_present_in_generated_registry(
    ) {
        // Test the standalone store_grad_p + grad_p_update fused kernel
        // This is the aggressive-only pair fusion that does not include dp_update or correct_velocity_delta.
        let src = kernel_source_by_id(
            "incompressible_momentum",
            KernelId("rhie_chow/store_grad_p_grad_p_update_fused"),
        )
        .expect("missing standalone store_grad_p_grad_p_update fused kernel in generated registry");
        assert!(
            !src.bindings.is_empty(),
            "standalone store_grad_p_grad_p_update fused kernel registry entry should include reflected bindings"
        );
    }
}
