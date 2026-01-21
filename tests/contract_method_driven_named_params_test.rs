use cfd2::solver::model::{compressible_model, incompressible_momentum_model};

#[test]
fn contract_relaxation_named_params_follow_method_capabilities() {
    let compressible = compressible_model();
    let keys = compressible.named_param_keys();
    for key in ["alpha_u", "alpha_p", "nonconverged_relax"] {
        assert!(
            keys.contains(&key),
            "compressible must expose relaxation param '{key}' (dual-time stabilization)"
        );
    }

    let incompressible = incompressible_momentum_model();
    let keys = incompressible.named_param_keys();
    for key in ["alpha_u", "alpha_p", "nonconverged_relax"] {
        assert!(
            keys.contains(&key),
            "incompressible must expose relaxation param '{key}' (apply_relaxation_in_update=true)"
        );
    }
}
