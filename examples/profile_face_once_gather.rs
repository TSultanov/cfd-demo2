//! Isolated adversarial probe for canonical-face residuals plus signed CSR gather.
//!
//! The "incidence" path mirrors the current unstructured explicit schedule:
//! a face pass materializes the all-Mach mass flux, a cell pass revisits every
//! incidence and recomputes the spatial terms, and an RK pass consumes a cell
//! RHS.  The "face-once" path materializes the conservative four-row face
//! residual, reuses its pressure row as the corrected mass-flux channel, then
//! gathers and performs the RK update in one cell pass. Both paths traverse
//! each cell's CSR row in exactly the same order.
//!
//! Run a release build; the default cases include regular, skew/non-orthogonal,
//! and degree-32 high-valence meshes:
//!
//! ```text
//! cargo run --release --example profile_face_once_gather -- --n 512
//! ```

use bytemuck::{cast_slice, Pod, Zeroable};
use cfd2::solver::codegen::{lower_system_unchecked, DiscreteOpKind};
use cfd2::solver::gpu::context::GpuContext;
use cfd2::solver::model::{allmach_thermal_model, compressible_model, scalar_transport_model};
use cfd2::solver::scheme::Scheme;
use cfd2_ir::equation::SchemeRegistry;
use std::f32::consts::TAU;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const WG_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vec2([f32; 2]);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vec4([f32; 4]);

#[derive(Clone, Copy)]
struct Options {
    n: usize,
    repeats: usize,
    samples: usize,
    degree: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            n: 384,
            repeats: 32,
            samples: 9,
            degree: 32,
        }
    }
}

fn options() -> Options {
    let mut out = Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut integer = || {
            args.next()
                .unwrap_or_else(|| panic!("missing value after {flag}"))
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid integer after {flag}"))
        };
        match flag.as_str() {
            "--n" => out.n = integer(),
            "--repeats" => out.repeats = integer(),
            "--samples" => out.samples = integer(),
            "--degree" => out.degree = integer(),
            "--help" | "-h" => {
                println!(
                    "profile_face_once_gather [--n N] [--repeats N] [--samples N] [--degree N]"
                );
                std::process::exit(0);
            }
            _ => panic!("unknown option {flag}"),
        }
    }
    assert!(out.n >= 8 && out.repeats > 0 && out.samples >= 3);
    assert!(out.degree >= 4 && out.degree <= 256);
    out
}

#[derive(Clone, Copy)]
struct Face {
    owner: u32,
    neighbor: i32,
    area: f32,
    normal: [f32; 2],
    center: [f32; 2],
}

struct Mesh {
    label: &'static str,
    cell_centers: Vec<[f32; 2]>,
    volumes: Vec<f32>,
    faces: Vec<Face>,
    offsets: Vec<u32>,
    incidences: Vec<i32>,
    max_valence: usize,
}

fn finish_mesh(
    label: &'static str,
    cell_centers: Vec<[f32; 2]>,
    volumes: Vec<f32>,
    faces: Vec<Face>,
) -> Mesh {
    let cells = cell_centers.len();
    let mut counts = vec![0u32; cells];
    for face in &faces {
        counts[face.owner as usize] += 1;
        if face.neighbor >= 0 {
            counts[face.neighbor as usize] += 1;
        }
    }
    let max_valence = counts.iter().copied().max().unwrap_or(0) as usize;
    let mut offsets = vec![0u32; cells + 1];
    for c in 0..cells {
        offsets[c + 1] = offsets[c] + counts[c];
    }
    let mut cursor = offsets[..cells].to_vec();
    let mut incidences = vec![0i32; offsets[cells] as usize];
    for (f, face) in faces.iter().enumerate() {
        let owner_slot = &mut cursor[face.owner as usize];
        incidences[*owner_slot as usize] = f as i32;
        *owner_slot += 1;
        if face.neighbor >= 0 {
            let neighbor_slot = &mut cursor[face.neighbor as usize];
            incidences[*neighbor_slot as usize] = -(f as i32) - 1;
            *neighbor_slot += 1;
        }
    }
    Mesh {
        label,
        cell_centers,
        volumes,
        faces,
        offsets,
        incidences,
        max_valence,
    }
}

fn rect_mesh(n: usize, skew: bool) -> Mesh {
    let h = 1.0 / n as f32;
    let mut centers = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let mut x = (i as f32 + 0.5) * h;
            let mut y = (j as f32 + 0.5) * h;
            if skew {
                x += 0.22 * h * (TAU * y).sin() * (0.37 * i as f32).cos();
                y += 0.22 * h * (TAU * x).cos() * (0.31 * j as f32).sin();
            }
            centers.push([x, y]);
        }
    }
    let cell = |i: usize, j: usize| -> u32 { (j * n + i) as u32 };
    let mut faces = Vec::with_capacity(2 * n * (n + 1));

    for j in 0..n {
        faces.push(Face {
            owner: cell(0, j),
            neighbor: -1,
            area: h,
            normal: [-1.0, 0.0],
            center: [0.0, (j as f32 + 0.5) * h],
        });
        for i in 0..n - 1 {
            let owner = cell(i, j);
            let neighbor = cell(i + 1, j);
            let xo = centers[owner as usize];
            let xn = centers[neighbor as usize];
            let tangent_shift = if skew {
                0.38 * h * (0.71 * (i + 3 * j) as f32).sin()
            } else {
                0.0
            };
            faces.push(Face {
                owner,
                neighbor: neighbor as i32,
                area: h,
                normal: [1.0, 0.0],
                center: [0.5 * (xo[0] + xn[0]), 0.5 * (xo[1] + xn[1]) + tangent_shift],
            });
        }
        faces.push(Face {
            owner: cell(n - 1, j),
            neighbor: -1,
            area: h,
            normal: [1.0, 0.0],
            center: [1.0, (j as f32 + 0.5) * h],
        });
    }
    for i in 0..n {
        faces.push(Face {
            owner: cell(i, 0),
            neighbor: -1,
            area: h,
            normal: [0.0, -1.0],
            center: [(i as f32 + 0.5) * h, 0.0],
        });
        for j in 0..n - 1 {
            let owner = cell(i, j);
            let neighbor = cell(i, j + 1);
            let xo = centers[owner as usize];
            let xn = centers[neighbor as usize];
            let tangent_shift = if skew {
                0.38 * h * (0.63 * (j + 5 * i) as f32).cos()
            } else {
                0.0
            };
            faces.push(Face {
                owner,
                neighbor: neighbor as i32,
                area: h,
                normal: [0.0, 1.0],
                center: [0.5 * (xo[0] + xn[0]) + tangent_shift, 0.5 * (xo[1] + xn[1])],
            });
        }
        faces.push(Face {
            owner: cell(i, n - 1),
            neighbor: -1,
            area: h,
            normal: [0.0, 1.0],
            center: [(i as f32 + 0.5) * h, 1.0],
        });
    }
    finish_mesh(
        if skew { "skew" } else { "regular" },
        centers,
        vec![h * h; n * n],
        faces,
    )
}

fn high_valence_mesh(target_cells: usize, degree: usize) -> Mesh {
    // Each cluster has one degree-d hub, a ring of d leaves, d radial faces, and
    // d ring faces.  Thus H/N approaches four, like the rectangle, while the hub
    // supplies the adversarial long CSR row.
    let clusters = target_cells.div_ceil(degree + 1).max(1);
    let side = (clusters as f32).sqrt().ceil() as usize;
    let pitch = 1.0 / side as f32;
    let radius = 0.32 * pitch;
    let mut centers = Vec::with_capacity(clusters * (degree + 1));
    let mut volumes = Vec::with_capacity(clusters * (degree + 1));
    let mut faces = Vec::with_capacity(clusters * degree * 2);
    for cluster in 0..clusters {
        let gx = cluster % side;
        let gy = cluster / side;
        let origin = [(gx as f32 + 0.5) * pitch, (gy as f32 + 0.5) * pitch];
        let hub = centers.len() as u32;
        centers.push(origin);
        volumes.push(pitch * pitch / (degree + 1) as f32);
        let first_leaf = centers.len() as u32;
        for k in 0..degree {
            let theta = TAU * k as f32 / degree as f32;
            centers.push([
                origin[0] + radius * theta.cos(),
                origin[1] + radius * theta.sin(),
            ]);
            volumes.push(pitch * pitch / (degree + 1) as f32);
        }
        for k in 0..degree {
            let leaf = first_leaf + k as u32;
            let theta = TAU * k as f32 / degree as f32;
            let n = [theta.cos(), theta.sin()];
            let tangent = [-n[1], n[0]];
            faces.push(Face {
                owner: hub,
                neighbor: leaf as i32,
                area: pitch / degree as f32,
                normal: n,
                center: [
                    origin[0] + 0.5 * radius * n[0] + 0.17 * radius * tangent[0],
                    origin[1] + 0.5 * radius * n[1] + 0.17 * radius * tangent[1],
                ],
            });

            let next = first_leaf + ((k + 1) % degree) as u32;
            let a = centers[leaf as usize];
            let b = centers[next as usize];
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let length = (dx * dx + dy * dy).sqrt();
            let n_ring = [dx / length, dy / length];
            let tangent_ring = [-n_ring[1], n_ring[0]];
            faces.push(Face {
                owner: leaf,
                neighbor: next as i32,
                area: pitch / degree as f32,
                normal: n_ring,
                center: [
                    0.5 * (a[0] + b[0]) + 0.22 * length * tangent_ring[0],
                    0.5 * (a[1] + b[1]) + 0.22 * length * tangent_ring[1],
                ],
            });
        }
    }
    finish_mesh("high-valence", centers, volumes, faces)
}

fn fields(centers: &[[f32; 2]]) -> (Vec<Vec4>, Vec<Vec4>, Vec<Vec4>) {
    let mut q = Vec::with_capacity(centers.len());
    let mut gx = Vec::with_capacity(centers.len());
    let mut gy = Vec::with_capacity(centers.len());
    for &[x, y] in centers {
        let sx = (TAU * x).sin();
        let cx = (TAU * x).cos();
        let sy = (TAU * y).sin();
        let cy = (TAU * y).cos();
        q.push(Vec4([
            0.25 + 0.35 * sx + 0.20 * cy,
            -0.10 + 0.30 * cx - 0.25 * sy,
            1.0 + 0.08 * sx * cy,
            1.2 + 0.10 * cx + 0.07 * sy,
        ]));
        gx.push(Vec4([
            0.35 * TAU * cx,
            -0.30 * TAU * sx,
            0.08 * TAU * cx * cy,
            -0.10 * TAU * sx,
        ]));
        gy.push(Vec4([
            -0.20 * TAU * sy,
            -0.25 * TAU * cy,
            -0.08 * TAU * sx * sy,
            0.07 * TAU * cy,
        ]));
    }
    (q, gx, gy)
}

const SHADER: &str = r#"
const MU: f32 = 0.020;
// Same compact conductance as the orthogonal Rhie--Chow correction.
const PRESSURE_DIFFUSION: f32 = 0.035;
const THERMAL_DIFFUSION: f32 = 0.016;
const RHIE_CHOW_D: f32 = 0.035;
const DT: f32 = 1.0e-5;

@group(0) @binding(0) var<storage, read> owners: array<u32>;
@group(0) @binding(1) var<storage, read> neighbors: array<i32>;
@group(0) @binding(2) var<storage, read> areas: array<f32>;
@group(0) @binding(3) var<storage, read> normals: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> face_centers: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> cell_centers: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> volumes: array<f32>;
@group(0) @binding(7) var<storage, read> offsets: array<u32>;
@group(0) @binding(8) var<storage, read> incidences: array<i32>;
@group(0) @binding(9) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> grad_x: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> grad_y: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> current_flux: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> face_shared: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read_write> current_rhs: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> out_current: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> out_face_once: array<vec4<f32>>;

fn boundary_value() -> vec4<f32> {
    return vec4<f32>(0.12, -0.18, 1.03, 1.16);
}

// Returns (corrected mass flux used by bounded U/T advection, predicted mass
// flux consumed by the pressure-row DivFlux).  The shipped all-Mach flux module
// deliberately keeps these distinct: pressure diffusion supplies the orthogonal
// Rhie--Chow correction on the pressure row.
fn allmach_flux_channels(
    qo: vec4<f32>, qn: vec4<f32>,
    gxo: vec4<f32>, gyo: vec4<f32>,
    gxn: vec4<f32>, gyn: vec4<f32>,
    xo: vec2<f32>, xn: vec2<f32>, normal: vec2<f32>, area: f32,
) -> vec2<f32> {
    let u_face = 0.5 * (qo.xy + qn.xy);
    let grad_p_face = 0.5 * (vec2<f32>(gxo.z, gyo.z) + vec2<f32>(gxn.z, gyn.z));
    let projected_distance = max(abs(dot(xn - xo, normal)), 1.0e-12);
    let advective = area * dot(u_face, normal);
    let nonorthogonal = area * RHIE_CHOW_D * dot(grad_p_face, normal);
    let orthogonal = area * RHIE_CHOW_D * (qn.z - qo.z) / projected_distance;
    let predicted = advective + nonorthogonal;
    return vec2<f32>(predicted - orthogonal, predicted);
}

fn conservative_face_terms(
    qo: vec4<f32>, qn: vec4<f32>,
    gxo: vec4<f32>, gyo: vec4<f32>,
    gxn: vec4<f32>, gyn: vec4<f32>,
    xo: vec2<f32>, xn: vec2<f32>, xf: vec2<f32>,
    normal: vec2<f32>, area: f32, phis: vec4<f32>, boundary: bool,
) -> vec4<f32> {
    let projected_distance = max(abs(dot(xn - xo, normal)), 1.0e-12);
    let diffusion = area / projected_distance;
    var q_hat: vec4<f32>;
    if (boundary) {
        q_hat = select(qn, qo, phis.x >= 0.0);
    } else if (phis.x >= 0.0) {
        let dx = xf - xo;
        q_hat = qo + grad_x_component(gxo, gyo, dx);
    } else {
        let dx = xf - xn;
        q_hat = qn + grad_x_component(gxn, gyn, dx);
    }

    let gx_face = select(0.5 * (gxo + gxn), gxo, boundary);
    let gy_face = select(0.5 * (gyo + gyn), gyo, boundary);
    let div_u = gx_face.x + gy_face.y;
    let dev2_x = MU * area * (
        (gx_face.x - (2.0 / 3.0) * div_u) * normal.x + gx_face.y * normal.y
    );
    let dev2_y = MU * area * (
        gy_face.x * normal.x + (gy_face.y - (2.0 / 3.0) * div_u) * normal.y
    );
    let p_face = select(0.5 * (qo.z + qn.z), qn.z, boundary);
    return vec4<f32>(
        MU * diffusion * (qn.x - qo.x) + dev2_x - phis.x * q_hat.x - area * normal.x * p_face,
        MU * diffusion * (qn.y - qo.y) + dev2_y - phis.y * q_hat.y - area * normal.y * p_face,
        PRESSURE_DIFFUSION * diffusion * (qn.z - qo.z) - phis.z,
        THERMAL_DIFFUSION * diffusion * (qn.w - qo.w) - phis.w * q_hat.w,
    );
}

fn grad_x_component(gx: vec4<f32>, gy: vec4<f32>, dx: vec2<f32>) -> vec4<f32> {
    return gx * dx.x + gy * dx.y;
}

fn local_source(q: vec4<f32>, gx: vec4<f32>, gy: vec4<f32>) -> vec4<f32> {
    let u_dot_grad_p = q.x * gx.z + q.y * gy.z;
    let div_u = gx.x + gy.y;
    let phi_grad = 2.0 * (
        gx.x * gx.x + gy.y * gy.y +
        0.5 * (gy.x + gx.y) * (gy.x + gx.y) -
        (1.0 / 3.0) * div_u * div_u
    );
    return vec4<f32>(0.0, 0.0, 0.0, 0.008 * u_dot_grad_p + 0.004 * phi_grad);
}

@compute @workgroup_size(256)
fn current_face_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if (f >= arrayLength(&owners)) { return; }
    let o = owners[f];
    let ni = neighbors[f];
    let qo = state[o];
    let gxo = grad_x[o];
    let gyo = grad_y[o];
    let xo = cell_centers[o];
    var qn = boundary_value();
    var gxn = gxo;
    var gyn = gyo;
    var xn = face_centers[f];
    if (ni >= 0) {
        let n = u32(ni);
        qn = state[n];
        gxn = grad_x[n];
        gyn = grad_y[n];
        xn = cell_centers[n];
    }
    let channels = allmach_flux_channels(
        qo, qn, gxo, gyo, gxn, gyn, xo, xn, normals[f], areas[f]
    );
    // Shipped layout: corrected mass flux on Ux/Uy/T, predicted flux on p.
    current_flux[f] = vec4<f32>(channels.x, channels.x, channels.y, channels.x);
}

@compute @workgroup_size(256)
fn current_cell_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= arrayLength(&state)) { return; }
    let qc = state[c];
    let gxc = grad_x[c];
    let gyc = grad_y[c];
    let xc = cell_centers[c];
    let vol = volumes[c];
    // unified_assembly emits cell-local sources before its face loop; keep that
    // accumulation order identical in both paths.
    var sum = vol * local_source(qc, gxc, gyc);
    let row_begin = offsets[c];
    let row_end = offsets[c + 1u];
    for (var k = row_begin; k < row_end; k += 1u) {
        let code = incidences[k];
        var f: u32;
        var sigma: f32;
        if (code >= 0) {
            f = u32(code);
            sigma = 1.0;
        } else {
            f = u32(-code - 1);
            sigma = -1.0;
        }
        let ni = neighbors[f];
        var qother = boundary_value();
        var gxother = gxc;
        var gyother = gyc;
        var xother = face_centers[f];
        if (ni >= 0) {
            let other = select(owners[f], u32(ni), sigma > 0.0);
            qother = state[other];
            gxother = grad_x[other];
            gyother = grad_y[other];
            xother = cell_centers[other];
        }
        let outward_normal = sigma * normals[f];
        let outward_flux = sigma * current_flux[f];
        var contribution = conservative_face_terms(
            qc, qother, gxc, gyc, gxother, gyother,
            xc, xother, face_centers[f], outward_normal, areas[f], outward_flux, ni < 0,
        );
        // bounded div(phi,q) = conservative div(phi*q) - q*div(phi): this is
        // endpoint-local and cannot be folded into a single antisymmetric vector.
        contribution += outward_flux * vec4<f32>(qc.x, qc.y, 0.0, qc.w);
        sum += contribution;
    }
    current_rhs[c] = sum / vol;
}

@compute @workgroup_size(256)
fn current_rk_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= arrayLength(&state)) { return; }
    out_current[c] = state[c] + DT * current_rhs[c];
}

@compute @workgroup_size(256)
fn canonical_face_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if (f >= arrayLength(&owners)) { return; }
    let o = owners[f];
    let ni = neighbors[f];
    let qo = state[o];
    let gxo = grad_x[o];
    let gyo = grad_y[o];
    let xo = cell_centers[o];
    var qn = boundary_value();
    var gxn = gxo;
    var gyn = gyo;
    var xn = face_centers[f];
    if (ni >= 0) {
        let n = u32(ni);
        qn = state[n];
        gxn = grad_x[n];
        gyn = grad_y[n];
        xn = cell_centers[n];
    }
    let normal = normals[f];
    let area = areas[f];
    let channels = allmach_flux_channels(
        qo, qn, gxo, gyo, gxn, gyn, xo, xn, normal, area
    );
    face_shared[f] = conservative_face_terms(
        qo, qn, gxo, gyo, gxn, gyn,
        xo, xn, face_centers[f], normal, area,
        vec4<f32>(channels.x, channels.x, channels.y, channels.x), ni < 0,
    );
}

@compute @workgroup_size(256)
fn canonical_gather_rk_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= arrayLength(&state)) { return; }
    let qc = state[c];
    let gxc = grad_x[c];
    let gyc = grad_y[c];
    let vol = volumes[c];
    var sum = vol * local_source(qc, gxc, gyc);
    let row_begin = offsets[c];
    let row_end = offsets[c + 1u];
    for (var k = row_begin; k < row_end; k += 1u) {
        let code = incidences[k];
        var f: u32;
        var sigma: f32;
        if (code >= 0) {
            f = u32(code);
            sigma = 1.0;
        } else {
            f = u32(-code - 1);
            sigma = -1.0;
        }
        let face_value = face_shared[f];
        // pressure diffusion - phi_predicted = -phi_corrected, including
        // the shipped pressure boundary closure. Reuse it for bounded terms.
        let outward_flux = -sigma * face_value.z;
        var contribution = sigma * face_value;
        contribution += outward_flux * vec4<f32>(qc.x, qc.y, 0.0, qc.w);
        sum += contribution;
    }
    let rhs = sum / vol;
    out_face_once[c] = qc + DT * rhs;
}
"#;

struct Buffers {
    owners: wgpu::Buffer,
    neighbors: wgpu::Buffer,
    areas: wgpu::Buffer,
    normals: wgpu::Buffer,
    face_centers: wgpu::Buffer,
    cell_centers: wgpu::Buffer,
    volumes: wgpu::Buffer,
    offsets: wgpu::Buffer,
    incidences: wgpu::Buffer,
    state: wgpu::Buffer,
    grad_x: wgpu::Buffer,
    grad_y: wgpu::Buffer,
    current_flux: wgpu::Buffer,
    face_shared: wgpu::Buffer,
    current_rhs: wgpu::Buffer,
    out_current: wgpu::Buffer,
    out_face_once: wgpu::Buffer,
}

fn initialized<T: Pod>(device: &wgpu::Device, label: &str, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn scratch(device: &wgpu::Device, label: &str, bytes: u64, copy_src: bool) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE;
    if copy_src {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage,
        mapped_at_creation: false,
    })
}

fn make_buffers(ctx: &GpuContext, mesh: &Mesh) -> Buffers {
    let owners: Vec<u32> = mesh.faces.iter().map(|f| f.owner).collect();
    let neighbors: Vec<i32> = mesh.faces.iter().map(|f| f.neighbor).collect();
    let areas: Vec<f32> = mesh.faces.iter().map(|f| f.area).collect();
    let normals: Vec<Vec2> = mesh.faces.iter().map(|f| Vec2(f.normal)).collect();
    let face_centers: Vec<Vec2> = mesh.faces.iter().map(|f| Vec2(f.center)).collect();
    let cell_centers: Vec<Vec2> = mesh.cell_centers.iter().copied().map(Vec2).collect();
    let (state, grad_x, grad_y) = fields(&mesh.cell_centers);
    let face_bytes = mesh.faces.len() as u64;
    let cell_bytes = mesh.cell_centers.len() as u64;
    Buffers {
        owners: initialized(&ctx.device, "face-once owners", &owners),
        neighbors: initialized(&ctx.device, "face-once neighbors", &neighbors),
        areas: initialized(&ctx.device, "face-once areas", &areas),
        normals: initialized(&ctx.device, "face-once normals", &normals),
        face_centers: initialized(&ctx.device, "face-once centers", &face_centers),
        cell_centers: initialized(&ctx.device, "face-once cell centers", &cell_centers),
        volumes: initialized(&ctx.device, "face-once volumes", &mesh.volumes),
        offsets: initialized(&ctx.device, "face-once offsets", &mesh.offsets),
        incidences: initialized(&ctx.device, "face-once signed incidences", &mesh.incidences),
        state: initialized(&ctx.device, "face-once state", &state),
        grad_x: initialized(&ctx.device, "face-once grad x", &grad_x),
        grad_y: initialized(&ctx.device, "face-once grad y", &grad_y),
        current_flux: scratch(&ctx.device, "current repeated flux", 16 * face_bytes, false),
        face_shared: scratch(
            &ctx.device,
            "canonical face residual",
            16 * face_bytes,
            false,
        ),
        current_rhs: scratch(&ctx.device, "current cell rhs", 16 * cell_bytes, false),
        out_current: scratch(&ctx.device, "current RK output", 16 * cell_bytes, true),
        out_face_once: scratch(&ctx.device, "face-once RK output", 16 * cell_bytes, true),
    }
}

struct Pipelines {
    current_face: wgpu::ComputePipeline,
    current_cell: wgpu::ComputePipeline,
    current_rk: wgpu::ComputePipeline,
    canonical_face: wgpu::ComputePipeline,
    canonical_gather_rk: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
}

fn make_pipelines(ctx: &GpuContext, buffers: &Buffers) -> Pipelines {
    let mut entries = Vec::new();
    for binding in 0..17u32 {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: binding < 12,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    let bind_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("face-once bind layout"),
            entries: &entries,
        });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("face-once pipeline layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("face-once adversarial shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
    let pipeline = |entry: &str| {
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    };
    let resources = [
        &buffers.owners,
        &buffers.neighbors,
        &buffers.areas,
        &buffers.normals,
        &buffers.face_centers,
        &buffers.cell_centers,
        &buffers.volumes,
        &buffers.offsets,
        &buffers.incidences,
        &buffers.state,
        &buffers.grad_x,
        &buffers.grad_y,
        &buffers.current_flux,
        &buffers.face_shared,
        &buffers.current_rhs,
        &buffers.out_current,
        &buffers.out_face_once,
    ];
    let bind_entries: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("face-once bind group"),
        layout: &bind_layout,
        entries: &bind_entries,
    });
    Pipelines {
        current_face: pipeline("current_face_pass"),
        current_cell: pipeline("current_cell_pass"),
        current_rk: pipeline("current_rk_pass"),
        canonical_face: pipeline("canonical_face_pass"),
        canonical_gather_rk: pipeline("canonical_gather_rk_pass"),
        bind,
    }
}

fn wait(ctx: &GpuContext, submission: wgpu::SubmissionIndex) {
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("GPU wait");
}

#[derive(Clone, Copy)]
enum Variant {
    Incidence,
    FaceOnce,
}

fn encode_variant(
    ctx: &GpuContext,
    pipelines: &Pipelines,
    variant: Variant,
    face_groups: u32,
    cell_groups: u32,
    repeats: usize,
    timing: Option<(&wgpu::QuerySet, &wgpu::Buffer, &wgpu::Buffer)>,
) -> wgpu::CommandBuffer {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("face-once timing encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("face-once timing pass"),
            timestamp_writes: timing.map(|(query, _, _)| wgpu::ComputePassTimestampWrites {
                query_set: query,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        pass.set_bind_group(0, &pipelines.bind, &[]);
        for _ in 0..repeats {
            match variant {
                Variant::Incidence => {
                    pass.set_pipeline(&pipelines.current_face);
                    pass.dispatch_workgroups(face_groups, 1, 1);
                    pass.set_pipeline(&pipelines.current_cell);
                    pass.dispatch_workgroups(cell_groups, 1, 1);
                    pass.set_pipeline(&pipelines.current_rk);
                    pass.dispatch_workgroups(cell_groups, 1, 1);
                }
                Variant::FaceOnce => {
                    pass.set_pipeline(&pipelines.canonical_face);
                    pass.dispatch_workgroups(face_groups, 1, 1);
                    pass.set_pipeline(&pipelines.canonical_gather_rk);
                    pass.dispatch_workgroups(cell_groups, 1, 1);
                }
            }
        }
    }
    if let Some((query, resolve, readback)) = timing {
        encoder.resolve_query_set(query, 0..2, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 16);
    }
    encoder.finish()
}

fn timed(
    ctx: &GpuContext,
    pipelines: &Pipelines,
    variant: Variant,
    face_groups: u32,
    cell_groups: u32,
    repeats: usize,
) -> Duration {
    if ctx.timestamp_query {
        let query = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("face-once timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("face-once timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("face-once timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let command = encode_variant(
            ctx,
            pipelines,
            variant,
            face_groups,
            cell_groups,
            repeats,
            Some((&query, &resolve, &readback)),
        );
        let submission = ctx.queue.submit(Some(command));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).expect("timestamp callback");
        });
        wait(ctx, submission);
        rx.recv()
            .expect("timestamp callback channel")
            .expect("timestamp map");
        let words = slice.get_mapped_range();
        let timestamps: &[u64] = cast_slice(&words);
        let ticks = timestamps[1].wrapping_sub(timestamps[0]);
        drop(words);
        readback.unmap();
        // Some Metal adapters advertise timestamps but return identical begin/end
        // values for a compute pass.  Re-run under a queue fence instead of
        // turning that driver quirk into a bogus zero-duration result.
        if ticks != 0 {
            return Duration::from_nanos((ticks as f64 * ctx.timestamp_period_ns as f64) as u64);
        }
    }

    let command = encode_variant(
        ctx,
        pipelines,
        variant,
        face_groups,
        cell_groups,
        repeats,
        None,
    );
    let start = Instant::now();
    let submission = ctx.queue.submit(Some(command));
    wait(ctx, submission);
    start.elapsed()
}

fn read_vec4(ctx: &GpuContext, source: &wgpu::Buffer, count: usize) -> Vec<Vec4> {
    let bytes = (count * std::mem::size_of::<Vec4>()) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("face-once output readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("face-once output copy"),
        });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    let submission = ctx.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("output callback");
    });
    wait(ctx, submission);
    rx.recv()
        .expect("output callback channel")
        .expect("output map");
    let mapped = slice.get_mapped_range();
    let values = cast_slice::<u8, Vec4>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    values
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn compare(a: &[Vec4], b: &[Vec4]) -> (f32, f64, u32) {
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;
    let mut max_ulp = 0u32;
    for (a, b) in a.iter().zip(b) {
        for k in 0..4 {
            let x = a.0[k];
            let y = b.0[k];
            let error = (x - y).abs();
            max_abs = max_abs.max(error);
            sum_sq += f64::from(error) * f64::from(error);
            let ordered = |v: f32| {
                let bits = v.to_bits();
                if bits & 0x8000_0000 == 0 {
                    bits | 0x8000_0000
                } else {
                    !bits
                }
            };
            max_ulp = max_ulp.max(ordered(x).abs_diff(ordered(y)));
        }
    }
    (max_abs, (sum_sq / (4 * a.len()) as f64).sqrt(), max_ulp)
}

fn audit_typed_lowering() {
    println!("typed Term -> matrix-free DiscreteOp inventory:");
    let models = [
        scalar_transport_model().expect("scalar model"),
        allmach_thermal_model().expect("all-Mach thermal model"),
        compressible_model().expect("compressible model"),
    ];
    let schemes = SchemeRegistry::new(Scheme::SecondOrderUpwind);
    for model in models {
        model
            .validate_explicit_rk4()
            .unwrap_or_else(|error| panic!("{} explicit validation: {error}", model.id));
        let spatial =
            lower_system_unchecked(&model.system, &schemes).matrix_free_spatial_residual();
        assert!(spatial.equations.iter().all(|equation| equation
            .ops
            .iter()
            .all(|op| op.kind != DiscreteOpKind::TimeDerivative
                && op.linearize_pressure_flux.is_none()
                && !op.relative_to_mesh)));
        let mut face_lanes = 0u32;
        let mut local_lanes = 0u32;
        print!(
            "  {} (state P={}, coupled S={}):",
            model.id,
            model.state_layout.stride(),
            model.system.unknowns_per_cell(),
        );
        for equation in &spatial.equations {
            let lanes = equation.target.kind().component_count() as u32;
            print!(" {}=[", equation.target.name());
            for (index, op) in equation.ops.iter().enumerate() {
                if index != 0 {
                    print!(",");
                }
                let is_face = matches!(
                    op.kind,
                    DiscreteOpKind::Convection
                        | DiscreteOpKind::Gradient
                        | DiscreteOpKind::Diffusion
                );
                if is_face {
                    face_lanes += lanes;
                } else {
                    local_lanes += lanes;
                }
                print!(
                    "{}({};field={}{}{})",
                    op.term_op.as_str(),
                    if is_face { "face" } else { "cell" },
                    op.field.name(),
                    if op.bounded { ";bounded" } else { "" },
                    if op.transpose_dev2 {
                        ";dev2"
                    } else if op.viscous_dissipation {
                        ";visc-diss"
                    } else if op.explicit_reaction {
                        ";reaction"
                    } else {
                        ""
                    },
                );
            }
            print!("]");
        }
        println!(" face_op_lanes={face_lanes} cell_op_lanes={local_lanes}");
    }
}

fn run_case(ctx: &GpuContext, mesh: Mesh, repeats: usize, samples: usize) {
    let cells = mesh.cell_centers.len();
    let faces = mesh.faces.len();
    let interior = mesh.faces.iter().filter(|f| f.neighbor >= 0).count();
    let boundary = faces - interior;
    let incidences = mesh.incidences.len();
    let buffers = make_buffers(ctx, &mesh);
    let pipelines = make_pipelines(ctx, &buffers);
    let face_groups = (faces as u32).div_ceil(WG_SIZE);
    let cell_groups = (cells as u32).div_ceil(WG_SIZE);

    // Compile, populate both outputs, and warm storage caches before sampling.
    let warm_current = encode_variant(
        ctx,
        &pipelines,
        Variant::Incidence,
        face_groups,
        cell_groups,
        2,
        None,
    );
    let warm_face_once = encode_variant(
        ctx,
        &pipelines,
        Variant::FaceOnce,
        face_groups,
        cell_groups,
        2,
        None,
    );
    wait(ctx, ctx.queue.submit([warm_current, warm_face_once]));

    let current = read_vec4(ctx, &buffers.out_current, cells);
    let face_once = read_vec4(ctx, &buffers.out_face_once, cells);
    let (max_abs, rms, max_ulp) = compare(&current, &face_once);
    assert!(current.iter().all(|v| v.0.iter().all(|x| x.is_finite())));
    assert!(face_once.iter().all(|v| v.0.iter().all(|x| x.is_finite())));
    assert!(max_abs <= 2.0e-5, "face-once parity error {max_abs}");

    let mut current_times = Vec::with_capacity(samples);
    let mut face_once_times = Vec::with_capacity(samples);
    // Alternate order to avoid consistently assigning thermal/drift bias.
    for sample in 0..samples {
        if sample & 1 == 0 {
            current_times.push(timed(
                ctx,
                &pipelines,
                Variant::Incidence,
                face_groups,
                cell_groups,
                repeats,
            ));
            face_once_times.push(timed(
                ctx,
                &pipelines,
                Variant::FaceOnce,
                face_groups,
                cell_groups,
                repeats,
            ));
        } else {
            face_once_times.push(timed(
                ctx,
                &pipelines,
                Variant::FaceOnce,
                face_groups,
                cell_groups,
                repeats,
            ));
            current_times.push(timed(
                ctx,
                &pipelines,
                Variant::Incidence,
                face_groups,
                cell_groups,
                repeats,
            ));
        }
    }
    let current_ns = median(current_times).as_nanos() as f64 / repeats as f64;
    let face_once_ns = median(face_once_times).as_nanos() as f64 / repeats as f64;

    // Declarative storage traffic for these exact kernels, counting each WGSL
    // vector access at its declared width and each row bound once.  This is a
    // reproducible source-level byte model, not a claim about cache-line/DRAM
    // transactions after backend scalarisation. I is separate because boundary
    // faces synthesize q_b instead of loading a neighboring state/gradient.
    let current_bytes = 132usize * cells + 44 * incidences + 100 * faces + 176 * interior;
    let face_once_bytes = 76usize * cells + 20 * incidences + 100 * faces + 56 * interior;
    let current_scratch = 16usize * faces + 16 * cells;
    let face_once_scratch = 16usize * faces;
    println!(
        "{:<12} N={:<8} F={:<8} I={:<8} B={:<6} H={:<8} vmax={:<3} \
         incidence={:>8.3} ms face_once={:>8.3} ms speedup={:>5.2}x \
         parity(max={:.3e},rms={:.3e},ulp={})",
        mesh.label,
        cells,
        faces,
        interior,
        boundary,
        incidences,
        mesh.max_valence,
        current_ns / 1.0e6,
        face_once_ns / 1.0e6,
        current_ns / face_once_ns,
        max_abs,
        rms,
        max_ulp,
    );
    println!(
        "             logical_traffic={:.2}/{:.2} MiB ({:.2}x) transient_scratch={:.2}/{:.2} MiB (incidence/face_once)",
        current_bytes as f64 / (1024.0 * 1024.0),
        face_once_bytes as f64 / (1024.0 * 1024.0),
        current_bytes as f64 / face_once_bytes as f64,
        current_scratch as f64 / (1024.0 * 1024.0),
        face_once_scratch as f64 / (1024.0 * 1024.0),
    );
}

fn main() {
    let opts = options();
    audit_typed_lowering();
    let ctx = pollster::block_on(GpuContext::new(None, None)).expect("GPU context");
    println!(
        "face-once signed-gather probe: n={} repeats={} samples={} timestamps={} period_ns={}",
        opts.n, opts.repeats, opts.samples, ctx.timestamp_query, ctx.timestamp_period_ns,
    );
    run_case(&ctx, rect_mesh(opts.n, false), opts.repeats, opts.samples);
    run_case(&ctx, rect_mesh(opts.n, true), opts.repeats, opts.samples);
    run_case(
        &ctx,
        high_valence_mesh(opts.n * opts.n, opts.degree),
        opts.repeats,
        opts.samples,
    );
}
