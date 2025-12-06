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
    /// Cell index for looking up field values
    pub cell_index: u32,
    /// Padding for alignment
    pub _padding: u32,
}

/// Uniform buffer for transform and rendering settings
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CfdUniforms {
    /// Transform matrix (2D affine: scale_x, scale_y, translate_x, translate_y)
    pub transform: [f32; 4],
    /// Viewport size in pixels
    pub viewport_size: [f32; 2],
    /// Value range for colormap [min, max]
    pub range: [f32; 2],
    /// Stride between elements (1 for scalar, 2 for vector)
    pub stride: u32,
    /// Offset to start reading (0 for scalar/x, 1 for y)
    pub offset: u32,
    /// Visualization mode (0: value, 1: magnitude)
    pub mode: u32,
    /// Padding
    pub _padding: u32,
}

/// GPU resources for CFD rendering
#[derive(Clone)]
pub struct CfdRenderResources {
    pub pipeline: wgpu::RenderPipeline,
    pub line_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub line_vertex_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub num_vertices: u32,
    pub num_line_vertices: u32,
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

        // Create bind group layout for uniforms and field data
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CFD Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
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
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
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
                            format: wgpu::VertexFormat::Uint32,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
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
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create line pipeline
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CFD Line Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
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
                            format: wgpu::VertexFormat::Uint32,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_solid"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create vertex buffer with max capacity
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CFD Vertex Buffer"),
            size: (max_vertices * std::mem::size_of::<CfdVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create line vertex buffer
        let line_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CFD Line Vertex Buffer"),
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
                range: [0.0, 1.0],
                stride: 1,
                offset: 0,
                mode: 0,
                _padding: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create dummy storage buffer for initial bind group
        let dummy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Field Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CFD Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            line_pipeline,
            vertex_buffer,
            line_vertex_buffer,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            num_vertices: 0,
            num_line_vertices: 0,
        }
    }

    /// Update the vertex buffer with new mesh data
    pub fn update_mesh(&mut self, queue: &wgpu::Queue, vertices: &[CfdVertex], line_vertices: &[CfdVertex]) {
        self.num_vertices = vertices.len() as u32;
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
        
        self.num_line_vertices = line_vertices.len() as u32;
        queue.write_buffer(&self.line_vertex_buffer, 0, bytemuck::cast_slice(line_vertices));
    }

    /// Update the uniform buffer with new transform
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &CfdUniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Update the bind group with a new field buffer
    pub fn update_bind_group(&mut self, device: &wgpu::Device, field_buffer: &wgpu::Buffer) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CFD Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: field_buffer.as_entire_binding(),
                },
            ],
        });
    }

    /// Render the CFD mesh
    /// Note: The render pass lifetime is generic to work with egui_wgpu's CallbackTrait
    pub fn paint<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, draw_lines: bool) {
        if self.num_vertices > 0 {
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }
        
        if draw_lines && self.num_line_vertices > 0 {
            render_pass.set_pipeline(&self.line_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.line_vertex_buffer.slice(..));
            render_pass.draw(0..self.num_line_vertices, 0..1);
        }
    }
}



/// Build mesh vertices from cell data for GPU rendering
pub fn build_mesh_vertices(
    cells: &[Vec<[f64; 2]>],
) -> Vec<CfdVertex> {
    let mut vertices = Vec::new();

    for (cell_idx, polygon) in cells.iter().enumerate() {
        if polygon.len() < 3 {
            continue;
        }

        // Triangulate the polygon using fan triangulation
        for i in 1..polygon.len() - 1 {
            // Triangle: polygon[0], polygon[i], polygon[i+1]
            vertices.push(CfdVertex {
                position: [polygon[0][0] as f32, polygon[0][1] as f32],
                cell_index: cell_idx as u32,
                _padding: 0,
            });
            vertices.push(CfdVertex {
                position: [polygon[i][0] as f32, polygon[i][1] as f32],
                cell_index: cell_idx as u32,
                _padding: 0,
            });
            vertices.push(CfdVertex {
                position: [polygon[i + 1][0] as f32, polygon[i + 1][1] as f32],
                cell_index: cell_idx as u32,
                _padding: 0,
            });
        }
    }

    vertices
}

/// Build line vertices from cell data for GPU rendering
pub fn build_line_vertices(
    cells: &[Vec<[f64; 2]>],
) -> Vec<CfdVertex> {
    let mut vertices = Vec::new();

    for (cell_idx, polygon) in cells.iter().enumerate() {
        if polygon.len() < 2 {
            continue;
        }

        for i in 0..polygon.len() {
            let p1 = polygon[i];
            let p2 = polygon[(i + 1) % polygon.len()];
            
            vertices.push(CfdVertex {
                position: [p1[0] as f32, p1[1] as f32],
                cell_index: cell_idx as u32,
                _padding: 0,
            });
            vertices.push(CfdVertex {
                position: [p2[0] as f32, p2[1] as f32],
                cell_index: cell_idx as u32,
                _padding: 0,
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
