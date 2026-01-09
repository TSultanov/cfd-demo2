#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WgslBindingDesc {
    pub group: u32,
    pub binding: u32,
    pub name: &'static str,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WgslBinding {
    pub group: u32,
    pub binding: u32,
    pub name: String,
}

pub(crate) trait WgslBindingLike {
    fn group(&self) -> u32;
    fn binding(&self) -> u32;
    fn name(&self) -> &str;
}

impl WgslBindingLike for WgslBinding {
    fn group(&self) -> u32 {
        self.group
    }

    fn binding(&self) -> u32 {
        self.binding
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl WgslBindingLike for WgslBindingDesc {
    fn group(&self) -> u32 {
        self.group
    }

    fn binding(&self) -> u32 {
        self.binding
    }

    fn name(&self) -> &str {
        self.name
    }
}

#[allow(dead_code)]
pub(crate) fn parse_bindings(shader: &str) -> Vec<WgslBinding> {
    let mut out = Vec::new();
    let mut pending: Option<(u32, u32)> = None;

    for raw in shader.lines() {
        let line = raw.trim();

        if pending.is_none() && line.contains("@group(") && line.contains("@binding(") {
            let group = parse_attr_u32(line, "@group(");
            let binding = parse_attr_u32(line, "@binding(");
            if let (Some(group), Some(binding)) = (group, binding) {
                pending = Some((group, binding));
            }
            continue;
        }

        let Some((group, binding)) = pending else {
            continue;
        };

        if line.starts_with("var") {
            if let Some(name) = parse_var_name(line) {
                out.push(WgslBinding {
                    group,
                    binding,
                    name,
                });
            }
            pending = None;
        }
    }

    out.sort_by(|a, b| (a.group, a.binding).cmp(&(b.group, b.binding)));
    out
}

#[allow(dead_code)]
fn parse_attr_u32(line: &str, prefix: &str) -> Option<u32> {
    let start = line.find(prefix)? + prefix.len();
    let rest = &line[start..];
    let end = rest.find(')')?;
    rest[..end].trim().parse().ok()
}

#[allow(dead_code)]
fn parse_var_name(line: &str) -> Option<String> {
    // Expected patterns:
    // - var<storage, read> face_owner: array<u32>;
    // - var<uniform> constants: Constants;
    // - var state: array<f32>;
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

pub(crate) fn create_bind_group_from_bindings<'a, B: WgslBindingLike>(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    bindings: &[B],
    group: u32,
    mut resolve: impl FnMut(&str) -> Option<wgpu::BindingResource<'a>>,
) -> Result<wgpu::BindGroup, String> {
    let mut entries = Vec::new();
    for binding in bindings.iter().filter(|b| b.group() == group) {
        let resource = resolve(binding.name()).ok_or_else(|| {
            format!(
                "missing bind resource for group {} binding {} ('{}')",
                binding.group(),
                binding.binding(),
                binding.name()
            )
        })?;
        entries.push(wgpu::BindGroupEntry {
            binding: binding.binding(),
            resource,
        });
    }
    entries.sort_by_key(|e| e.binding);
    Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    }))
}
