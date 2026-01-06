use super::wgsl_ast::Stmt;
use super::wgsl_dsl as dsl;

#[derive(Debug, Clone)]
pub struct ScalarReconstruction {
    pub phi_upwind: String,
    pub phi_ho: String,
}

pub fn scalar_reconstruction(
    prefix: &str,
    scheme_id: &str,
    flux: &str,
    phi_own: &str,
    phi_neigh: &str,
    grad_own: &str,
    grad_neigh: &str,
    center: &str,
    other_center: &str,
    face_center: &str,
) -> (Vec<Stmt>, ScalarReconstruction) {
    let phi_upwind = format!("phi_upwind_{prefix}");
    let phi_ho = format!("phi_ho_{prefix}");
    let r_x = format!("r_{prefix}_x");
    let r_y = format!("r_{prefix}_y");
    let d_cd_x = format!("d_cd_{prefix}_x");
    let d_cd_y = format!("d_cd_{prefix}_y");
    let grad_term = format!("grad_term_{prefix}");

    let mut stmts = Vec::new();
    stmts.push(dsl::var(&phi_upwind, phi_own));
    stmts.push(dsl::if_block(
        &format!("{flux} < 0.0"),
        dsl::block(vec![dsl::assign(&phi_upwind, phi_neigh)]),
        None,
    ));
    stmts.push(dsl::var(&phi_ho, &phi_upwind));

    let sou_block = dsl::block(vec![dsl::if_block(
        &format!("{flux} > 0.0"),
        dsl::block(vec![
            dsl::let_(&r_x, &format!("{face_center}.x - {center}.x")),
            dsl::let_(&r_y, &format!("{face_center}.y - {center}.y")),
            dsl::assign(
                &phi_ho,
                &format!("{phi_own} + ({grad_own}.x * {r_x} + {grad_own}.y * {r_y})"),
            ),
        ]),
        Some(dsl::block(vec![
            dsl::let_(&r_x, &format!("{face_center}.x - {other_center}.x")),
            dsl::let_(&r_y, &format!("{face_center}.y - {other_center}.y")),
            dsl::assign(
                &phi_ho,
                &format!("{phi_neigh} + ({grad_neigh}.x * {r_x} + {grad_neigh}.y * {r_y})"),
            ),
        ])),
    )]);

    let quick_block = dsl::block(vec![dsl::if_block(
        &format!("{flux} > 0.0"),
        dsl::block(vec![
            dsl::let_(&d_cd_x, &format!("{other_center}.x - {center}.x")),
            dsl::let_(&d_cd_y, &format!("{other_center}.y - {center}.y")),
            dsl::let_(
                &grad_term,
                &format!("{grad_own}.x * {d_cd_x} + {grad_own}.y * {d_cd_y}"),
            ),
            dsl::assign(
                &phi_ho,
                &format!("0.625 * {phi_own} + 0.375 * {phi_neigh} + 0.125 * {grad_term}"),
            ),
        ]),
        Some(dsl::block(vec![
            dsl::let_(&d_cd_x, &format!("{center}.x - {other_center}.x")),
            dsl::let_(&d_cd_y, &format!("{center}.y - {other_center}.y")),
            dsl::let_(
                &grad_term,
                &format!("{grad_neigh}.x * {d_cd_x} + {grad_neigh}.y * {d_cd_y}"),
            ),
            dsl::assign(
                &phi_ho,
                &format!("0.625 * {phi_neigh} + 0.375 * {phi_own} + 0.125 * {grad_term}"),
            ),
        ])),
    )]);

    stmts.push(dsl::if_block(
        &format!("{scheme_id} == 1u"),
        sou_block,
        Some(dsl::block(vec![dsl::if_block(
            &format!("{scheme_id} == 2u"),
            quick_block,
            None,
        )])),
    ));

    (
        stmts,
        ScalarReconstruction {
            phi_upwind,
            phi_ho,
        },
    )
}

pub fn limited_linear_reconstruct_face(
    prefix: &str,
    side: &str,
    phi_cell: &str,
    phi_other: &str,
    grad_cell: &str,
    r_x: &str,
    r_y: &str,
) -> (Vec<Stmt>, String) {
    let diff = format!("diff_{prefix}_{side}");
    let min_diff = format!("min_diff_{prefix}_{side}");
    let max_diff = format!("max_diff_{prefix}_{side}");
    let delta = format!("delta_{prefix}_{side}");
    let delta_limited = format!("delta_{prefix}_{side}_limited");
    let phi_face = format!("{prefix}_{side}_face");

    let mut stmts = Vec::new();
    stmts.push(dsl::let_(&diff, &format!("{phi_other} - {phi_cell}")));
    stmts.push(dsl::let_(&min_diff, &format!("min({diff}, 0.0)")));
    stmts.push(dsl::let_(&max_diff, &format!("max({diff}, 0.0)")));
    stmts.push(dsl::let_(
        &delta,
        &format!("{grad_cell}.x * {r_x} + {grad_cell}.y * {r_y}"),
    ));
    stmts.push(dsl::let_(
        &delta_limited,
        &format!("min(max({delta}, {min_diff}), {max_diff})"),
    ));
    stmts.push(dsl::let_(&phi_face, &format!("{phi_cell} + {delta_limited}")));

    (stmts, phi_face)
}
