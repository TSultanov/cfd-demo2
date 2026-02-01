#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WgslBindingDesc {
    pub group: u32,
    pub binding: u32,
    pub name: &'static str,
}

pub trait WgslBindingLike {
    fn group(&self) -> u32;
    fn binding(&self) -> u32;
    fn name(&self) -> &str;
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
