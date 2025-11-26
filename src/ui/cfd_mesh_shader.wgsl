// CFD Mesh Shader
// Renders triangulated CFD cells with colormap visualization

struct Uniforms {
    // Transform: scale_x, scale_y, translate_x, translate_y
    transform: vec4<f32>,
    // Viewport size in pixels
    viewport_size: vec2<f32>,
    _padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) field_value: f32,
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
    out.field_value = in.field_value;
    
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
