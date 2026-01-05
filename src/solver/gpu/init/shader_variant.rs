#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ShaderVariant {
    Manual,
    Generated,
}

impl Default for ShaderVariant {
    fn default() -> Self {
        ShaderVariant::Generated
    }
}
