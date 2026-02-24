use std::collections::BTreeSet;

use super::ParamSpec;

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

/// A simple buffer access expression tracked in structured body IR metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelBufferAccess {
    pub base: String,
    pub index_expr: String,
}

impl KernelBufferAccess {
    pub fn new(base: impl Into<String>, index_expr: impl Into<String>) -> Self {
        Self {
            base: base.into(),
            index_expr: index_expr.into(),
        }
    }
}

/// A structured operation used by IR-only fusion cleanup passes.
///
/// `line_index` is 0-based within a kernel segment's executable section
/// (`preamble` followed by `body`), excluding indexing lines and fused segment markers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelBodyIrOp {
    /// `state[idx] = value;`-like write.
    Store {
        line_index: usize,
        access: KernelBufferAccess,
        value_expr: String,
        /// Memory reads that appear in `value_expr`.
        value_reads: Vec<KernelBufferAccess>,
    },
    /// `let x = state[idx];`-like direct load.
    LetLoad {
        line_index: usize,
        name: String,
        ty: Option<String>,
        access: KernelBufferAccess,
    },
    /// `x = x;`-like no-op assignment to a local symbol.
    NoopSelfAssign { line_index: usize },
    /// Conservative invalidation barrier (e.g. control flow boundary).
    Invalidate { line_index: usize },
}

impl KernelBodyIrOp {
    pub fn line_index(&self) -> usize {
        match self {
            KernelBodyIrOp::Store { line_index, .. }
            | KernelBodyIrOp::LetLoad { line_index, .. }
            | KernelBodyIrOp::NoopSelfAssign { line_index }
            | KernelBodyIrOp::Invalidate { line_index } => *line_index,
        }
    }
}

/// IR representation of a fusion-capable kernel program.
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
    pub preamble: Vec<String>,
    pub indexing: Vec<String>,
    pub body: Vec<String>,
    /// Structured per-segment operation metadata for IR-only cleanup passes.
    ///
    /// Operations are indexed over `preamble` + `body` for this kernel segment.
    pub body_ir_ops: Vec<KernelBodyIrOp>,
    /// Local symbols that may need deterministic renaming when composing kernels.
    pub local_symbols: Vec<String>,
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
            body_ir_ops: Vec::new(),
            local_symbols: Vec::new(),
            side_effects: SideEffectMetadata::default(),
            eos_params: Vec::new(),
        }
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

    #[test]
    fn kernel_program_initializes_ir_ops_empty() {
        let launch = LaunchSemantics::new([64, 1, 1], "idx", Some("idx >= n"));
        let program = KernelProgram::new(
            "kernel/a",
            DispatchDomain::Cells,
            launch,
            vec![KernelBinding::new(
                0,
                0,
                "state",
                "array<f32>",
                BindingAccess::ReadWriteStorage,
            )],
        );
        assert!(program.body_ir_ops.is_empty());
    }
}
