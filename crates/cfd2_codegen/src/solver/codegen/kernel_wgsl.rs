use super::wgsl_ast::Module;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelWgsl {
    source: String,
}

impl KernelWgsl {
    pub fn new(module: Module) -> Self {
        Self {
            source: module.to_wgsl(),
        }
    }

    pub fn from_source(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
        }
    }

    pub fn to_wgsl(&self) -> String {
        self.source.clone()
    }
}

impl From<Module> for KernelWgsl {
    fn from(module: Module) -> Self {
        Self::new(module)
    }
}

impl From<String> for KernelWgsl {
    fn from(source: String) -> Self {
        Self::from_source(source)
    }
}
