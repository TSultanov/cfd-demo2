use super::wgsl_ast::Module;

/// Structured description of a single WGSL `@group(G) @binding(B) var<…> NAME` declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingDesc {
    pub group: u32,
    pub binding: u32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelWgsl {
    source: String,
    bindings: Vec<BindingDesc>,
}

impl KernelWgsl {
    /// Create from an AST `Module`, extracting binding metadata structurally.
    pub fn new(module: Module) -> Self {
        let bindings = module
            .bindings()
            .into_iter()
            .map(|(g, b, n)| BindingDesc {
                group: g,
                binding: b,
                name: n,
            })
            .collect();
        Self {
            source: module.to_wgsl(),
            bindings,
        }
    }

    /// Create from raw WGSL source text (legacy path).
    ///
    /// Binding metadata is recovered by text-parsing the source as a fallback.
    pub fn from_source(source: impl Into<String>) -> Self {
        let src = source.into();
        let bindings = parse_wgsl_bindings_from_text(&src);
        Self {
            source: src,
            bindings,
        }
    }

    /// Structured binding metadata for this kernel.
    pub fn bindings(&self) -> &[BindingDesc] {
        &self.bindings
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

// ── Text-based binding extraction (fallback for raw WGSL strings) ──────

/// Parse `@group(G) @binding(B) var<…> NAME` triples from WGSL source text.
///
/// This is the fallback path used by `KernelWgsl::from_source` for kernels
/// that are not produced via `Module` (e.g. handwritten WGSL).  For kernels
/// built through the AST, `Module::bindings()` is authoritative.
pub fn parse_wgsl_bindings_from_text(shader: &str) -> Vec<BindingDesc> {
    let mut out = Vec::new();
    let mut pending: Option<(u32, u32)> = None;

    for raw in shader.lines() {
        let line = raw.trim();

        if pending.is_none() && line.contains("@group(") && line.contains("@binding(") {
            let group = parse_attr_u32(line, "@group(");
            let binding = parse_attr_u32(line, "@binding(");
            if let (Some(group), Some(binding)) = (group, binding) {
                pending = Some((group, binding));

                // Handle inline declarations like:
                // `@group(0) @binding(0) var<storage, read> foo: array<u32>;`
                if let Some(var_idx) = line.find("var") {
                    if let Some(name) = parse_var_name(line[var_idx..].trim_start()) {
                        out.push(BindingDesc {
                            group,
                            binding,
                            name,
                        });
                    }
                    pending = None;
                }
            }
            continue;
        }

        let Some((group, binding)) = pending else {
            continue;
        };

        if line.starts_with("var") {
            if let Some(name) = parse_var_name(line) {
                out.push(BindingDesc {
                    group,
                    binding,
                    name,
                });
            }
            pending = None;
        }
    }

    out.sort_by(|a, b| (a.group, a.binding).cmp(&(b.group, b.binding)));
    out.dedup_by(|a, b| a.group == b.group && a.binding == b.binding);
    out
}

fn parse_attr_u32(line: &str, prefix: &str) -> Option<u32> {
    let start = line.find(prefix)? + prefix.len();
    let rest = &line[start..];
    let end = rest.find(')')?;
    rest[..end].trim().parse().ok()
}

fn parse_var_name(line: &str) -> Option<String> {
    let after_var = line.strip_prefix("var")?.trim_start();
    let after_decl = if let Some(idx) = after_var.find('>') {
        after_var[idx + 1..].trim_start()
    } else {
        after_var
    };
    let name_end = after_decl.find(':')?;
    let name = after_decl[..name_end].trim();
    if name.is_empty() {
        return None;
    }
    Some(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::wgsl_ast::{
        AccessMode, Attribute, GlobalVar, Item, StorageClass, Type,
    };

    #[test]
    fn parse_bindings_inline_declaration() {
        let shader = r#"
@group(0) @binding(0) var<storage, read> face_owner: array<u32>;
@group(0) @binding(1) var<storage, read> face_neighbor: array<i32>;
"#;
        let bindings = parse_wgsl_bindings_from_text(shader);
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].group, 0);
        assert_eq!(bindings[0].binding, 0);
        assert_eq!(bindings[0].name, "face_owner");
        assert_eq!(bindings[1].binding, 1);
        assert_eq!(bindings[1].name, "face_neighbor");
    }

    #[test]
    fn parse_bindings_multiline_declaration() {
        let shader = r#"
@group(1) @binding(3)
var<uniform> constants: Constants;
"#;
        let bindings = parse_wgsl_bindings_from_text(shader);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].group, 1);
        assert_eq!(bindings[0].binding, 3);
        assert_eq!(bindings[0].name, "constants");
    }

    #[test]
    fn module_bindings_matches_text_parse() {
        use super::super::wgsl_ast::*;
        let mut module = Module::new();
        module.push(storage_var_item("face_owner", Type::array(Type::U32), 0, 0));
        module.push(storage_var_item("state", Type::array(Type::F32), 0, 1));
        module.push(uniform_var_item("constants", Type::Custom("Constants".into()), 1, 4));

        let kernel_from_module = KernelWgsl::new(module.clone());
        let kernel_from_text = KernelWgsl::from_source(module.to_wgsl());

        assert_eq!(kernel_from_module.bindings(), kernel_from_text.bindings());
    }

    fn storage_var_item(name: &str, ty: Type, group: u32, binding: u32) -> Item {
        Item::GlobalVar(GlobalVar::new(
            name,
            ty,
            StorageClass::Storage,
            Some(AccessMode::Read),
            vec![Attribute::Group(group), Attribute::Binding(binding)],
        ))
    }

    fn uniform_var_item(name: &str, ty: Type, group: u32, binding: u32) -> Item {
        Item::GlobalVar(GlobalVar::new(
            name,
            ty,
            StorageClass::Uniform,
            None,
            vec![Attribute::Group(group), Attribute::Binding(binding)],
        ))
    }
}
