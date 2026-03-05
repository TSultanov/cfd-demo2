use std::collections::BTreeSet;

use crate::ports::ParamSpec;

/// Kernel dispatch domain used for compatibility checks during fusion.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DispatchDomain {
    Cells,
    Faces,
    Custom(String),
}

/// Dispatch launch semantics preserved in IR for deterministic fusion validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchSemantics {
    pub workgroup_size: [u32; 3],
    /// Expression that computes the per-invocation logical index.
    pub invocation_index_expr: String,
    /// Optional early-return guard expression.
    pub bounds_check_expr: Option<String>,
}

impl LaunchSemantics {
    pub fn new(
        workgroup_size: [u32; 3],
        invocation_index_expr: impl Into<String>,
        bounds_check_expr: Option<impl Into<String>>,
    ) -> Self {
        Self {
            workgroup_size,
            invocation_index_expr: invocation_index_expr.into(),
            bounds_check_expr: bounds_check_expr.map(Into::into),
        }
    }
}

/// Access mode for a single bind entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BindingAccess {
    ReadOnlyStorage,
    ReadWriteStorage,
    Uniform,
}

impl BindingAccess {
    pub const fn allows_write(self) -> bool {
        matches!(self, BindingAccess::ReadWriteStorage)
    }
}

/// One bind entry in a kernel interface.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelBinding {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub wgsl_type: String,
    pub access: BindingAccess,
}

impl KernelBinding {
    pub fn new(
        group: u32,
        binding: u32,
        name: impl Into<String>,
        wgsl_type: impl Into<String>,
        access: BindingAccess,
    ) -> Self {
        Self {
            group,
            binding,
            name: name.into(),
            wgsl_type: wgsl_type.into(),
            access,
        }
    }
}

/// A resource touched by a kernel according to side-effect metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EffectResource {
    pub group: u32,
    pub binding: u32,
    pub component: Option<String>,
}

impl EffectResource {
    pub fn binding(group: u32, binding: u32) -> Self {
        Self {
            group,
            binding,
            component: None,
        }
    }

    pub fn component(group: u32, binding: u32, component: impl Into<String>) -> Self {
        Self {
            group,
            binding,
            component: Some(component.into()),
        }
    }
}

/// Side-effect description used for hazard checks in the fusion pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SideEffectMetadata {
    pub read_set: BTreeSet<EffectResource>,
    pub write_set: BTreeSet<EffectResource>,
    pub uses_barriers: bool,
    pub uses_atomics: bool,
}

/// IR representation of a fusion-capable kernel program.
///
/// All code sections (indexing, preamble, body) are represented as typed
/// `Vec<Stmt>` ASTs. String emission (`lower_kernel_program_to_wgsl`) is
/// the absolute final step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelProgram {
    pub id: String,
    pub dispatch: DispatchDomain,
    pub launch: LaunchSemantics,
    pub bindings: Vec<KernelBinding>,
    /// Module-level helper function definitions emitted before `fn main()`.
    ///
    /// Each entry is a complete WGSL function definition (including `fn` keyword and body).
    /// During fusion, helper functions from all programs are merged (deduplicated by content).
    pub helper_functions: Vec<String>,
    /// Typed AST preamble statements (local declarations, scheme guards, etc.).
    pub preamble: Vec<crate::ast::Stmt>,
    /// Typed AST indexing statements (base address computation, etc.).
    pub indexing: Vec<crate::ast::Stmt>,
    /// Typed AST body statements (the main kernel computation).
    pub body: Vec<crate::ast::Stmt>,
    pub side_effects: SideEffectMetadata,
    /// EOS parameters referenced by this kernel (structured declaration that replaces
    /// string-scan heuristic in `constants_extra_params_for_program`).
    pub eos_params: Vec<ParamSpec>,
}

impl KernelProgram {
    pub fn new(
        id: impl Into<String>,
        dispatch: DispatchDomain,
        launch: LaunchSemantics,
        bindings: Vec<KernelBinding>,
    ) -> Self {
        Self {
            id: id.into(),
            dispatch,
            launch,
            bindings,
            helper_functions: Vec::new(),
            preamble: Vec::new(),
            indexing: Vec::new(),
            body: Vec::new(),
            side_effects: SideEffectMetadata::default(),
            eos_params: Vec::new(),
        }
    }

    /// Collect local symbols by walking Let/Var declarations in preamble and body.
    ///
    /// Indexing symbols are excluded because the indexing section is shared across
    /// fused programs and never renamed.
    pub fn local_symbols(&self) -> Vec<String> {
        let stmts: Vec<crate::ast::Stmt> = self
            .preamble
            .iter()
            .chain(self.body.iter())
            .cloned()
            .collect();
        crate::ast::stmt::collect_local_symbols(&stmts)
    }

    pub fn sorted_bindings(&self) -> Vec<KernelBinding> {
        let mut sorted = self.bindings.clone();
        sorted.sort_by(|a, b| {
            a.group
                .cmp(&b.group)
                .then(a.binding.cmp(&b.binding))
                .then(a.name.cmp(&b.name))
        });
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_access_reports_writability() {
        assert!(!BindingAccess::ReadOnlyStorage.allows_write());
        assert!(BindingAccess::ReadWriteStorage.allows_write());
        assert!(!BindingAccess::Uniform.allows_write());
    }

    #[test]
    fn sorted_bindings_are_deterministic() {
        let launch = LaunchSemantics::new([64, 1, 1], "idx", Some("idx >= n"));
        let program = KernelProgram::new(
            "kernel/a",
            DispatchDomain::Cells,
            launch,
            vec![
                KernelBinding::new(1, 2, "b", "array<f32>", BindingAccess::ReadOnlyStorage),
                KernelBinding::new(0, 3, "a", "array<f32>", BindingAccess::ReadOnlyStorage),
                KernelBinding::new(1, 2, "a", "array<f32>", BindingAccess::ReadOnlyStorage),
            ],
        );

        let sorted = program.sorted_bindings();
        assert_eq!(sorted[0].group, 0);
        assert_eq!(sorted[0].binding, 3);
        assert_eq!(sorted[1].name, "a");
        assert_eq!(sorted[2].name, "b");
    }

    #[test]
    fn side_effect_metadata_defaults_empty() {
        let meta = SideEffectMetadata::default();
        assert!(meta.read_set.is_empty());
        assert!(meta.write_set.is_empty());
        assert!(!meta.uses_barriers);
        assert!(!meta.uses_atomics);
    }
}
