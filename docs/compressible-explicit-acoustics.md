# Compressible explicit RK4 at acoustic amplitudes: diagnosis and precision plan

Status, 2026-07-12. Investigation of the "noisy solution + unexpected startup
wave" report against the GUI compressible + RK4 + GPU obstacle case.

## What the user sees, and why

### 1. The "unexpected wave" from the obstacle is physical

The compressible GUI case starts from the uniform-freestream IC (the
load-bearing stabilizer for the implicit low-Mach path, `SolverDriver::build`,
`src/sim/driver.rs`). At `t=0` the flow violates the no-slip condition on the
cylinder, so the wall radiates the classic impulsive-start acoustic dipole:
overpressure upstream, suction downstream, amplitude `~rho*c*U`, expanding at
the sound speed and decaying by 2D cylindrical spreading. The implicit default
(BDF2 + pseudo-transient `dtau=1e-3`) damps this transient away, which is why
it was never seen before; time-accurate explicit RK4 resolves it faithfully.
Headless reproduction (`tests/probe_compressible_rk4_acoustic_noise.rs`)
matches the GUI screenshot quantitatively at `t=1.6e-3 s` (range ~0.24 Pa
around the absolute `p0 = rho*R*T = 105472.5 Pa`).

### 2. The speckle is the f32 representation floor, not a solver defect

Every pressure value read back sits on the f32 quantization grid of the
absolute pressure: one ULP at 105472.5 Pa is `2^-7 = 0.0078 Pa`. The startup
wave decays to ~0.24 Pa, i.e. ~31 representable levels; the auto-normalized
colormap stretches those 31 levels (plus ~1-quantum arithmetic noise, and
~9-quantum impulse-excited content near cut cells) across the full color
range — that is the speckle. Measurements:

- GPU f32 trajectory and CPU f64 trajectory agree: neighbor-mean roughness
  rms 4.2e-3 Pa vs 4.2e-3 Pa, max 6.5e-2 vs 7.0e-2 Pa at t=1.6e-3 (both viewed
  through the f32 readback). The GPU solver adds nothing above the
  representation floor.
- At a representable amplitude the field is clean, see below.

### 3. At representable amplitudes the explicit solver is correct

Gates in `tests/compressible_explicit_acoustics_test.rs` release an isentropic
200 Pa Gaussian pulse (2.6e4 quanta) on both topologies and assert clean
cylindrical acoustics:

| case | two-monitor transit speed | final roughness rms / max (of amplitude) |
|---|---|---|
| unstructured GPU | +2.3% of c | 1.0% / 6.7% |
| unstructured CPU | +2.3% of c | 1.0% / 6.7% |
| structured GPU | +2.8% of c | 1.8% / 8.6% |
| structured CPU | +2.8% of c | 1.8% / 8.6% |

The residual % is resolution-limited dispersion/curvature: 2x structured
refinement takes roughness 1.7% -> 0.46% rms and the speed deviation shrinks.
Amplitude decay matches 2D spreading. CPU/GPU peaks agree to 0.02 Pa.

Note the KT flux's low-Mach preconditioning and pressure-coupling terms are
gated on `dtau > 0` (`src/solver/model/flux_schemes.rs`), and explicit
stepping rejects `dtau != 0` (`GpuUnifiedSolver::set_named_param`), so RK4
runs the full acoustic dissipation — this gating is correct and load-bearing:
a preconditioned c_eff (~0.03 m/s at the GUI inlet) in the time-accurate flux
would collapse the KT dissipation by 4 orders of magnitude.

## What landed

- Explicit (RK4) compressible runs now START FROM REST like the pressure-based
  models, on both topologies: the unstructured driver IC
  (`SolverDriver::build_inner`, `src/sim/driver.rs`) and the structured GUI
  seeding (`seed_structured_freestream`, `src/ui/app.rs`) zero the initial
  velocity/momentum when `time_scheme == RK4`, while the inlet BC keeps
  driving with the configured inlet velocity. The flow develops naturally
  behind the inlet's piston compression wave; the impulsive-start dipole
  slammed off the obstacle is gone. The uniform-freestream IC remains the
  implicit pseudo-transient path's stabilizer (its from-rest low-Mach inlet
  instability is an implicit-path pathology; time-accurate RK4 runs the full
  acoustic KT dissipation and soaked 1500 steps from rest on the GUI GPU
  obstacle case without incident, front speed = c, max neighbor-mean
  roughness 3-10% of the wave range vs 27% before).
- `tests/compressible_explicit_acoustics_test.rs` — the four acoustic pulse
  gates above (GPU gates skip without an adapter; `CFD2_PROBE_OUT` dumps PPM
  snapshots, `CFD2_PROBE_REFINE` refines the structured case).
- `tests/probe_compressible_rk4_acoustic_noise.rs` — diagnostic reproduction
  of the GUI case with roughness metrics and field snapshots
  (`CFD2_PROBE_BACKEND` = gpu | cpu-f32 | cpu-f64).
- Representable-precision display floor: the visualization no longer
  normalizes the colormap to a span narrower than 64 f32 ULPs of the field's
  own magnitude (`PRECISION_FLOOR_ULPS`, `src/ui/cfd_renderer.rs`;
  `src/ui/cfd_range_reduce.wgsl` for the Direct route; plot-cache floor in
  `src/ui/app.rs`). Sub-representable signals now fade toward a flat field
  instead of rendering quantization noise as full-scale speckle. Fields
  stored near zero (gauge pressure of the all-Mach models, velocities) are
  unaffected — the floor binds only when a large absolute offset hides a tiny
  signal.

## The precision fix: constant reference (gauge) conserved state

To actually *recover* the lost precision (rather than stop displaying its
absence), the density-based compressible family should store conserved
perturbations around a constant reference state — "floating reference
pressure" generalized to the full conserved state. Design validated against
the measurements above:

- State semantics: store `rho' = rho - rho_ref0`, `rho_e' = rho_e - e_ref0`,
  gauge `p' = p - p_ref0` (derived field); `rho_u` and `T` stay absolute
  (their quanta are harmless: `rho_u` is tiny at low Mach, `T` feeds only
  relative-precision consumers). References are constants of the run:
  `rho_ref0 = rho0`, `p_ref0 = EOS(rho0)`, `e_ref0 = p_ref0/(gamma-1)`.
  A *constant* reference is deliberate — it needs no atomic state rewrites,
  and it degrades gracefully back to today's behavior only if the mean state
  drifts by O(p0).
- Storage quantum for the acoustic signal becomes ~1e-7 *relative to the
  perturbation* instead of 0.0078 Pa absolute: the 0.24 Pa startup wave gets
  ~2e6 levels instead of 31.
- The KT flux must be regrouped symbolically so the reference terms cancel
  analytically (`derive_central_upwind`, `src/solver/model/flux_schemes.rs`):
  - continuity: `phi = aphiv_pos*rho'_pos + aphiv_neg*rho'_neg
    + rho_ref0*(a_pos*phiv_pos + a_neg*phiv_neg)` — the `a_sf` dissipation on
    the constant cancels exactly;
  - momentum: use gauge `p'` in the pressure term; `p_ref0 * sum(n*A)` over a
    closed cell is analytically zero, so drop it symbolically (this also
    removes today's f32 closed-surface cancellation noise at `ULP(p0*A)`);
  - energy: `aphiv*(rho_e+p) = aphiv*(rho_e'+p')
    + (e_ref0+p_ref0)*(a_pos*phiv_pos + a_neg*phiv_neg)`;
  - EOS closure: `p' = gm1*(rho_e' - |rho_u|^2/(2*(rho'+rho_ref0)))` (the
    constant part vanishes by construction of `e_ref0`).
- Landing strategy: parameterize with reference constants that DEFAULT TO
  ZERO. At zero refs the regrouped algebra degenerates to the current
  absolute form (0-valued grouped terms), so every existing MMS/GUI/parity
  anchor stays valid; then the GUI/driver compressible path opts in with real
  references. The linear-EOS fluids (Water etc.) already carry
  `eos_rho_ref/eos_p_ref` and their pressure closure is already a function of
  `rho - rho_ref` — for them the same change fixes today's even-worse 134 Pa
  quantum (`dp_drho * ULP(1000 kg/m^3)`).
- Consumer checklist for the opt-in path (each reads/writes raw conserved
  state and must add/subtract the references): driver IC + inlet BC helpers
  (`set_uniform_state`, `set_compressible_inlet_isothermal_x`,
  `seed_structured_freestream`, `setup_structured_bcs`), host dt sampling
  (`sample_compressible_explicit_state`), the autonomous health/CFL audit
  kernels (`compressible_explicit_control.rs`, incl. the
  `ideal_gas_constants_supported` gate that currently requires zero refs),
  GUI plotting/legend (display gauge `p'`, matching the all-Mach models),
  positivity checks in the GUI smoke harness, snapshot/restore, and the MMS
  harnesses (unaffected at zero refs).
- Acceptance: re-run the acoustic pulse gates at 0.5 Pa amplitude (65 quanta
  today — currently unrepresentable) and require the same cleanliness as the
  200 Pa gates; byte-parity of generated kernels at zero refs; the full
  MMS/GUI matrices.

This is a focused arc of its own (touches the shared flux lowering + shader
re-bless); it was deliberately not rushed into this change set.
