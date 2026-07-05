//! EOS-derived compressibility `psi = 1/c^2` (isentropic, `1/sound_speed^2`)
//! from the materials database, for every preset.

#![cfg(feature = "ui")]

use cfd2::solver::model::eos::EosSpec;
use cfd2::ui::fluid::Fluid;

#[test]
fn every_preset_compressibility_is_one_over_c_squared() {
    for f in Fluid::presets() {
        let c = f.sound_speed();
        let psi = f.compressibility();
        if c > 0.0 {
            let expected = 1.0 / (c * c);
            assert!(
                (psi - expected).abs() / expected < 1e-9,
                "{}: psi {psi:.4e} must equal 1/c^2 {expected:.4e} (c={c:.1})",
                f.name
            );
        } else {
            assert_eq!(psi, 0.0, "{}: incompressible EOS must give psi 0", f.name);
        }
        eprintln!("[eos-psi] {:<8} c={c:8.1} m/s  psi=1/c^2={psi:.4e}", f.name);
    }
}

#[test]
fn preset_compressibilities_match_real_thermodynamics() {
    // Real sound speeds: Air sqrt(gamma*R*T); liquids sqrt(K/rho).
    let by_name = |name: &str| Fluid::presets().into_iter().find(|f| f.name == name).unwrap();

    // Air: 1/(1.4*287*300) ~ 8.30e-6
    let air = by_name("Air").compressibility();
    assert!((8.2e-6..8.4e-6).contains(&air), "Air psi {air:.3e}");
    // Water: rho/K = 1000/2.2e9 ~ 4.55e-7
    let water = by_name("Water").compressibility();
    assert!((4.4e-7..4.7e-7).contains(&water), "Water psi {water:.3e}");
    // Mercury: 13546/2.85e10 ~ 4.75e-7
    let merc = by_name("Mercury").compressibility();
    assert!((4.6e-7..4.9e-7).contains(&merc), "Mercury psi {merc:.3e}");

    // Air is far more compressible than the liquids (lower sound speed).
    assert!(air > 10.0 * water, "a gas must be much more compressible than water");
}

#[test]
fn constant_eos_is_incompressible() {
    assert_eq!(EosSpec::Constant.compressibility(1.0), 0.0);
    assert_eq!(EosSpec::Constant.compressibility(1000.0), 0.0);
}
