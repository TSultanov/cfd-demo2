//! Direct affine-exactness gate for both pressure-gradient programs consumed by
//! implicit Rhie--Chow correction and explicit RK4 face-flux preparation.
#![cfg(feature = "cpu")]

use cfd2::solver::cpu::interpreter::{Buffers, Ctx, Frame, Interpreter, Value};
use cfd2::solver::cpu::lowering::model_kernel_programs;
use cfd2::solver::mesh::refresh::mesh_geometry_f32;
use cfd2::solver::mesh::{generate_structured_rect_mesh, BoundarySides, BoundaryType, Mesh};
use cfd2::solver::model::backend::SchemeRegistry;
use cfd2::solver::model::{allmach_pressure_model, FluxLayout, KernelId, ModelSpec};
use cfd2_ir::kernel::KernelProgram;

const GX: f64 = 1.3;
const GY: f64 = -0.7;

fn affine(x: f64, y: f64) -> f32 {
    (0.37 + GX * x + GY * y) as f32
}

fn skew_mesh() -> Mesh {
    let nx = 14usize;
    let ny = 11usize;
    let mut mesh = generate_structured_rect_mesh(
        nx,
        ny,
        1.0,
        0.8,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );
    let h = (1.0 / nx as f64).min(0.8 / ny as f64);
    for v in 0..mesh.num_vertices() {
        let (x, y) = (mesh.vx[v], mesh.vy[v]);
        if x > 1.0e-12 && x < 1.0 - 1.0e-12 && y > 1.0e-12 && y < 0.8 - 1.0e-12 {
            let phase = v as f64 * 12.9898;
            mesh.vx[v] += 0.30 * h * phase.sin();
            mesh.vy[v] += 0.30 * h * (1.7 * phase).cos();
        }
    }
    mesh.recalculate_geometry();
    assert!(mesh.cell_vol.iter().all(|&v| v > 0.0));
    mesh
}

fn initial_buffers(mesh: &Mesh, model: &ModelSpec, mixed_neumann: bool) -> Buffers {
    let stride = model.state_layout.stride() as usize;
    let p_off = model.state_layout.offset_for("p").expect("p offset") as usize;
    let gp_off = model
        .state_layout
        .offset_for("grad_p")
        .expect("grad_p offset") as usize;
    let flux = FluxLayout::from_system(&model.system);
    let p_rank = flux.offset_for("p").expect("p rank") as usize;
    let unknown_stride = flux.stride as usize;

    let mut state = vec![0.0f32; mesh.num_cells() * stride];
    for c in 0..mesh.num_cells() {
        state[c * stride + p_off] = affine(mesh.cell_cx[c], mesh.cell_cy[c]);
        state[c * stride + gp_off] = f32::NAN;
        state[c * stride + gp_off + 1] = f32::NAN;
    }

    // Per-face tables, not per-boundary-type tables. Exercise the Dirichlet
    // affine closure on every physical boundary; interior entries are ignored.
    let mut bc_kind = vec![0u32; mesh.num_faces() * unknown_stride];
    let mut bc_value = vec![0.0f32; mesh.num_faces() * unknown_stride];
    for f in 0..mesh.num_faces() {
        if mesh.face_neighbor[f].is_none() {
            let i = f * unknown_stride + p_rank;
            if mixed_neumann && mesh.face_boundary[f] == Some(BoundaryType::Wall) {
                bc_kind[i] = 2;
                bc_value[i] = (GX * mesh.face_nx[f] + GY * mesh.face_ny[f]) as f32;
            } else {
                bc_kind[i] = 1;
                bc_value[i] = affine(mesh.face_cx[f], mesh.face_cy[f]);
            }
        }
    }

    let geo = mesh_geometry_f32(mesh);
    let mut b = Buffers::new();
    b.insert_f32("state", state);
    b.insert_u32(
        "face_owner",
        mesh.face_owner.iter().map(|&v| v as u32).collect(),
    );
    b.insert_i32(
        "face_neighbor",
        mesh.face_neighbor
            .iter()
            .map(|v| v.map(|x| x as i32).unwrap_or(-1))
            .collect(),
    );
    b.insert_f32("face_areas", geo.face_areas);
    b.insert_vec2("face_normals", geo.face_normals);
    b.insert_vec2("face_centers", geo.face_centers);
    b.insert_vec2("face_wrap_shift", geo.face_wrap_shift);
    b.insert_vec2("cell_centers", geo.cell_centers);
    b.insert_f32("cell_vols", geo.cell_vols);
    b.insert_u32(
        "cell_face_offsets",
        mesh.cell_face_offsets.iter().map(|&v| v as u32).collect(),
    );
    b.insert_u32(
        "cell_faces",
        mesh.cell_faces.iter().map(|&v| v as u32).collect(),
    );
    b.insert_u32(
        "face_boundary",
        mesh.face_boundary
            .iter()
            .map(|v| v.map(|x| x.bc_table_index() as u32).unwrap_or(0))
            .collect(),
    );
    b.insert_u32("bc_kind", bc_kind);
    b.insert_f32("bc_value", bc_value);
    b
}

fn execute(program: &KernelProgram, buffers: &Buffers, cells: usize) {
    execute_with_ctx(program, buffers, cells, &Ctx::new());
}

fn execute_with_ctx(program: &KernelProgram, buffers: &Buffers, count: usize, ctx: &Ctx) {
    let mut stmts =
        Vec::with_capacity(program.indexing.len() + program.preamble.len() + program.body.len());
    stmts.extend_from_slice(&program.indexing);
    stmts.extend_from_slice(&program.preamble);
    stmts.extend_from_slice(&program.body);
    for idx in 0..count {
        let mut frame = Frame::new()
            .with_local("idx", Value::U32(idx as u32))
            .with_local("global_id", Value::Vec3([idx as f32, 0.0, 0.0]));
        Interpreter::new(buffers, ctx).run(&stmts, &mut frame);
    }
}

fn gradient_error(state: &[f32], model: &ModelSpec, cells: usize) -> f64 {
    let stride = model.state_layout.stride() as usize;
    let gp = model.state_layout.offset_for("grad_p").unwrap() as usize;
    (0..cells)
        .map(|c| {
            let ex = (state[c * stride + gp] as f64 - GX).abs();
            let ey = (state[c * stride + gp + 1] as f64 - GY).abs();
            ex.max(ey)
        })
        .fold(0.0, f64::max)
}

#[test]
fn direct_pressure_gradient_kernels_are_affine_exact_on_skew_mesh() {
    let mesh = skew_mesh();
    let model = allmach_pressure_model().expect("model");
    let (programs, _wgsl_only) =
        model_kernel_programs(&model, &SchemeRegistry::default()).expect("programs");

    let find = |id: KernelId| {
        programs
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, p)| p)
            .unwrap_or_else(|| panic!("missing {}", id.as_str()))
    };

    for mixed_neumann in [false, true] {
        let rc_buffers = initial_buffers(&mesh, &model, mixed_neumann);
        execute(
            find(KernelId::RHIE_CHOW_GRAD_P_UPDATE),
            &rc_buffers,
            mesh.num_cells(),
        );
        let rc_state = rc_buffers.f32_vec("state");
        let rc_err = gradient_error(&rc_state, &model, mesh.num_cells());
        assert!(
            rc_err < 2.0e-5,
            "Rhie-Chow affine error {rc_err:.3e}, mixed_neumann={mixed_neumann}"
        );

        let flux_buffers = initial_buffers(&mesh, &model, mixed_neumann);
        execute(
            find(KernelId::FLUX_MODULE_GRADIENTS),
            &flux_buffers,
            mesh.num_cells(),
        );
        let flux_state = flux_buffers.f32_vec("state");
        let flux_err = gradient_error(&flux_state, &model, mesh.num_cells());
        assert!(
            flux_err < 2.0e-5,
            "flux-gradient affine error {flux_err:.3e}, mixed_neumann={mixed_neumann}"
        );

        let stride = model.state_layout.stride() as usize;
        let gp = model.state_layout.offset_for("grad_p").unwrap() as usize;
        for c in 0..mesh.num_cells() {
            for k in 0..2 {
                assert_eq!(
                    rc_state[c * stride + gp + k].to_bits(),
                    flux_state[c * stride + gp + k].to_bits(),
                    "pressure reconstructors differ at cell {c}, component {k}, mixed_neumann={mixed_neumann}"
                );
            }
        }
    }
}

#[test]
fn pressure_dirichlet_inlet_flux_is_affine_exact_on_a_skew_face() {
    let mesh = generate_structured_rect_mesh(
        1,
        1,
        1.0,
        1.0,
        BoundarySides {
            left: BoundaryType::Inlet,
            right: BoundaryType::Outlet,
            bottom: BoundaryType::Wall,
            top: BoundaryType::Wall,
        },
    );
    let inlet_face = (0..mesh.num_faces())
        .find(|&f| mesh.face_boundary[f] == Some(BoundaryType::Inlet))
        .expect("inlet face");

    let model = allmach_pressure_model().expect("model");
    let (programs, _wgsl_only) =
        model_kernel_programs(&model, &SchemeRegistry::default()).expect("programs");
    let flux_program = programs
        .iter()
        .find(|(id, _)| *id == KernelId::FLUX_MODULE)
        .map(|(_, program)| program)
        .expect("flux module");
    let flux_layout = FluxLayout::from_system(&model.system);
    let unknown_stride = flux_layout.stride as usize;
    let p_rank = flux_layout.offset_for("p").expect("p rank") as usize;
    let ux_rank = flux_layout.offset_for("U_x").expect("U_x rank") as usize;

    let state_stride = model.state_layout.stride() as usize;
    let mut state = vec![0.0f32; mesh.num_cells() * state_stride];
    let p_off = model.state_layout.offset_for("p").expect("p offset") as usize;
    let dp_off = model
        .state_layout
        .offset_for("d_p")
        .expect("d_p offset") as usize;
    let gp_off = model
        .state_layout
        .offset_for("grad_p")
        .expect("grad_p offset") as usize;
    let rho_off = model
        .state_layout
        .offset_for("rho")
        .expect("rho offset") as usize;
    state[p_off] = affine(mesh.cell_cx[0], mesh.cell_cy[0]);
    state[dp_off] = 1.0;
    state[gp_off] = GX as f32;
    state[gp_off + 1] = GY as f32;
    state[rho_off] = 1.0;

    let geo = mesh_geometry_f32(&mesh);
    let mut face_centers = geo.face_centers;
    // Deliberately move this face center tangentially while retaining its
    // outward normal. A normal-only pressure predictor cannot cancel the
    // affine Dirichlet jump on this geometry.
    face_centers[2 * inlet_face + 1] += 0.23;

    let mut bc_kind = vec![0u32; mesh.num_faces() * unknown_stride];
    let mut bc_value = vec![0.0f32; mesh.num_faces() * unknown_stride];
    for f in 0..mesh.num_faces() {
        if mesh.face_neighbor[f].is_none() {
            let i = f * unknown_stride + p_rank;
            bc_kind[i] = 1;
            bc_value[i] = affine(
                face_centers[2 * f] as f64,
                face_centers[2 * f + 1] as f64,
            );
        }
    }

    let mut buffers = Buffers::new();
    buffers.insert_f32("state", state);
    buffers.insert_f32(
        "fluxes",
        vec![f32::NAN; mesh.num_faces() * unknown_stride],
    );
    buffers.insert_u32(
        "face_owner",
        mesh.face_owner.iter().map(|&v| v as u32).collect(),
    );
    buffers.insert_i32(
        "face_neighbor",
        mesh.face_neighbor
            .iter()
            .map(|v| v.map(|x| x as i32).unwrap_or(-1))
            .collect(),
    );
    buffers.insert_f32("face_areas", geo.face_areas);
    buffers.insert_vec2("face_normals", geo.face_normals);
    buffers.insert_vec2("face_centers", face_centers.clone());
    buffers.insert_vec2("face_wrap_shift", geo.face_wrap_shift);
    buffers.insert_vec2("cell_centers", geo.cell_centers);
    buffers.insert_u32(
        "face_boundary",
        mesh.face_boundary
            .iter()
            .map(|v| v.map(|x| x.bc_table_index() as u32).unwrap_or(0))
            .collect(),
    );
    buffers.insert_u32("bc_kind", bc_kind);
    buffers.insert_f32("bc_value", bc_value);

    let ctx = Ctx::new()
        .with_constant("constants", "stride_x", Value::U32(mesh.num_faces() as u32))
        .with_constant("constants", "density", Value::F32(1.0));
    execute_with_ctx(flux_program, &buffers, mesh.num_faces(), &ctx);

    let fluxes = buffers.f32_vec("fluxes");
    let corrected = fluxes[inlet_face * unknown_stride + ux_rank] as f64;
    assert!(
        corrected.abs() < 2.0e-6,
        "skew affine pressure-Dirichlet inlet left a corrected flux of {corrected:.6e}"
    );

    let rx = face_centers[2 * inlet_face] as f64 - mesh.cell_cx[0];
    let ry = face_centers[2 * inlet_face + 1] as f64 - mesh.cell_cy[0];
    let dist = (rx * mesh.face_nx[inlet_face] + ry * mesh.face_ny[inlet_face]).abs();
    let expected_predictor = (GX * rx + GY * ry) / dist * mesh.face_area[inlet_face];
    let predictor = fluxes[inlet_face * unknown_stride + p_rank] as f64;
    assert!(
        (predictor - expected_predictor).abs() < 2.0e-6,
        "pressure predictor {predictor:.6e} != center-line value {expected_predictor:.6e}"
    );
}
