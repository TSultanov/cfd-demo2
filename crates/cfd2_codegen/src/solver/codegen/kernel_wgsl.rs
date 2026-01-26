use super::wgsl_ast::Module;

#[derive(Debug, Clone, PartialEq)]
pub struct KernelWgsl {
    module: Module,
}

impl KernelWgsl {
    pub fn new(module: Module) -> Self {
        Self { module }
    }

    pub fn to_wgsl(&self) -> String {
        self.module.to_wgsl()
    }
}

impl From<Module> for KernelWgsl {
    fn from(module: Module) -> Self {
        Self::new(module)
    }
}

