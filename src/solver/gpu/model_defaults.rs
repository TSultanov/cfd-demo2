use crate::solver::model::{compressible_model, incompressible_momentum_model, ModelSpec};

pub fn default_incompressible_model() -> &'static ModelSpec {
    static MODEL: std::sync::OnceLock<ModelSpec> = std::sync::OnceLock::new();
    MODEL.get_or_init(incompressible_momentum_model)
}

pub fn default_compressible_model() -> &'static ModelSpec {
    static MODEL: std::sync::OnceLock<ModelSpec> = std::sync::OnceLock::new();
    MODEL.get_or_init(compressible_model)
}
