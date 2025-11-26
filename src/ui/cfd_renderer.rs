//! GPU-accelerated CFD mesh renderer using wgpu
//!
//! This module provides a shader-based renderer for CFD mesh data that significantly
//! improves rendering performance for very fine meshes (80k+ cells) compared to the
//! polygon-per-cell approach.
//!
//! The renderer:
//! 1. Uploads mesh geometry (triangulated cells) to GPU buffers
//! 2. Uploads field values as vertex attributes
//! 3. Renders all triangles in a single draw call with color interpolation

use wgpu::util::DeviceExt;

/// Vertex structure for the CFD mesh shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CfdVertex {
    /// Position in world coordinates
    pub position: [f32; 2],
    /// Normalized field value [0, 1] for color interpolation
    pub field_value: f32,
    /// Padding for alignment
    pub _padding: f32,
}

/// Uniform buffer for transform and rendering settings
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CfdUniforms {
    /// Transform matrix (2D affine: scale_x, scale_y, translate_x, translate_y)
    pub transform: [f32; 4],
    /// Viewport size in pixels
    pub viewport_size: [f32; 2],
    /// Reserved for future use
    pub _padding: [f32; 2],
}

/// GPU resources for CFD rendering
pub struct CfdRenderResources {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub num_vertices: u32,
}

impl CfdRenderResources {
    /// Create render resources for CFD visualization
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        max_vertices: usize,
    ) -> Self {
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CFD Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cfd_mesh_shader.wgsl").into()),
        });

        // Create bind group layout for uniforms
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CFD Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CFD Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CFD Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CfdVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: 8,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Don't cull any faces
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create vertex buffer with max capacity
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CFD Vertex Buffer"),
            size: (max_vertices * std::mem::size_of::<CfdVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CFD Uniform Buffer"),
            contents: bytemuck::bytes_of(&CfdUniforms {
                transform: [1.0, 1.0, 0.0, 0.0],
                viewport_size: [1.0, 1.0],
                _padding: [0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CFD Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            vertex_buffer,
            uniform_buffer,
            bind_group,
            num_vertices: 0,
        }
    }

    /// Update the vertex buffer with new mesh data
    pub fn update_mesh(
        &mut self,
        queue: &wgpu::Queue,
        vertices: &[CfdVertex],
    ) {
        self.num_vertices = vertices.len() as u32;
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
    }

    /// Update the uniform buffer with new transform
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &CfdUniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Render the CFD mesh
    /// Note: The render pass lifetime is generic to work with egui_wgpu's CallbackTrait
    pub fn paint<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.num_vertices == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.num_vertices, 0..1);
    }
}

/// Triangulate a polygon (fan triangulation from first vertex)
pub fn triangulate_polygon(vertices: &[[f64; 2]]) -> Vec<[usize; 3]> {
    if vertices.len() < 3 {
        return Vec::new();
    }
    
    let mut triangles = Vec::with_capacity(vertices.len() - 2);
    for i in 1..vertices.len() - 1 {
        triangles.push([0, i, i + 1]);
    }
    triangles
}

/// Build mesh vertices from cell data for GPU rendering
pub fn build_mesh_vertices(
    cells: &[Vec<[f64; 2]>],
    values: &[f64],
    min_val: f64,
    max_val: f64,
) -> Vec<CfdVertex> {
    let range = max_val - min_val;
    let range = if range.abs() < 1e-10 { 1.0 } else { range };
    
    let mut vertices = Vec::new();
    
    for (cell_idx, polygon) in cells.iter().enumerate() {
        if polygon.len() < 3 {
            continue;
        }
        
        let val = values.get(cell_idx).copied().unwrap_or(0.0);
        let normalized = ((val - min_val) / range).clamp(0.0, 1.0) as f32;
        
        // Triangulate the polygon using fan triangulation
        for i in 1..polygon.len() - 1 {
            // Triangle: polygon[0], polygon[i], polygon[i+1]
            vertices.push(CfdVertex {
                position: [polygon[0][0] as f32, polygon[0][1] as f32],
                field_value: normalized,
                _padding: 0.0,
            });
            vertices.push(CfdVertex {
                position: [polygon[i][0] as f32, polygon[i][1] as f32],
                field_value: normalized,
                _padding: 0.0,
            });
            vertices.push(CfdVertex {
                position: [polygon[i + 1][0] as f32, polygon[i + 1][1] as f32],
                field_value: normalized,
                _padding: 0.0,
            });
        }
    }
    
    vertices
}

/// Compute the bounding box of all cells
pub fn compute_bounds(cells: &[Vec<[f64; 2]>]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    
    for polygon in cells {
        for &[x, y] in polygon {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    
    (min_x, max_x, min_y, max_y)
}
