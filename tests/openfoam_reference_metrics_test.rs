#[path = "openfoam_reference/common.rs"]
mod common;

#[test]
fn max_cell_rel_error_scalar_reports_worst_cell() {
    let reference = vec![10.0, 0.0, -5.0];
    let sol = vec![10.0, 2.0, -4.0];
    let scale = common::rms(&reference).max(1e-12);
    let err = common::max_cell_rel_error_scalar(&sol, &reference, scale);
    assert_eq!(err.idx, 1);
    assert!((err.abs - 2.0).abs() < 1e-12);
}

#[test]
fn max_cell_rel_error_vec2_reports_worst_cell() {
    let reference = vec![(3.0, 4.0), (0.0, 0.0), (1.0, 0.0)];
    let sol = vec![(3.0, 4.0), (0.0, 0.2), (1.1, 0.0)];
    let scale = common::rms_vec2_mag(&reference).max(1e-12);
    let err = common::max_cell_rel_error_vec2(&sol, &reference, scale);
    assert_eq!(err.idx, 1);
    assert!((err.abs - 0.2).abs() < 1e-12);
}
