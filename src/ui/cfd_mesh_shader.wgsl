// CFD Mesh Shader
// Renders triangulated CFD cells with colormap visualization

struct Uniforms {
    // Transform: scale_x, scale_y, translate_x, translate_y
    transform: vec4<f32>,
    // Viewport size in pixels
    viewport_size: vec2<f32>,
    // Value range [min, max]
    range: vec2<f32>,
    // Stride between elements
    stride: u32,
    // Offset to start reading
    offset: u32,
    // Mode: 0=value, 1=magnitude
    mode: u32,
    _padding: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> field_data: array<f32>;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) cell_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) field_value: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Apply transform: scale then translate
    let world_pos = in.position * uniforms.transform.xy + uniforms.transform.zw;
    
    // Convert to clip space (-1 to 1)
    // Assuming world coordinates are in a normalized range
    let clip_pos = world_pos * 2.0 - vec2<f32>(1.0, 1.0);
    
    out.clip_position = vec4<f32>(clip_pos.x, clip_pos.y, 0.0, 1.0);
    
    var val: f32;
    if (uniforms.mode == 1u) {
        // Magnitude mode: read vector at [offset, offset+1]
        let base_idx = in.cell_index * uniforms.stride + uniforms.offset;
        let vx = field_data[base_idx];
        let vy = field_data[base_idx + 1u];
        val = sqrt(vx * vx + vy * vy);
    } else {
        // Value mode: read single field at stride*cell + offset
        let idx = in.cell_index * uniforms.stride + uniforms.offset;
        val = field_data[idx];
    }

    let range = uniforms.range.y - uniforms.range.x;
    // Avoid division by zero
    let safe_range = select(range, 1.0, abs(range) < 1e-10);
    
    let normalized = clamp((val - uniforms.range.x) / safe_range, 0.0, 1.0);
    out.field_value = normalized;
    
    return out;
}

// Simple Rainbow colormap: blue -> green -> red
fn colormap(t: f32) -> vec4<f32> {
    let t_clamped = clamp(t, 0.0, 1.0);
    
    var r: f32;
    var g: f32;
    var b: f32;
    
    if (t_clamped < 0.5) {
        // Blue to Green (0.0 - 0.5)
        let s = t_clamped * 2.0;
        r = 0.0;
        g = s;
        b = 1.0 - s;
    } else {
        // Green to Red (0.5 - 1.0)
        let s = (t_clamped - 0.5) * 2.0;
        r = s;
        g = 1.0 - s;
        b = 0.0;
    }
    
    return vec4<f32>(r, g, b, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return colormap(in.field_value);
}

@fragment
fn fs_solid(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.3); // Semi-transparent black
}
