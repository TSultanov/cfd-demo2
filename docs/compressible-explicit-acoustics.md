# Compressible explicit RK4 at acoustic amplitudes: diagnosis and gauge storage

Status, 2026-07-13: the gauge-storage reformulation described at the bottom is
SHIPPED (see "Gauge storage: shipped design" for the as-built details and
acceptance results). The diagnosis below is retained as the motivating record.

Investigation of the "noisy solution + unexpected startup wave" report against
the GUI compressible + RK4 + GPU obstacle case (2026-07-12).

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

## Gauge storage: shipped design (2026-07-13)

The reformulation above landed for the whole density-based compressible
family (`compressible` / `compressible_structured`), BOTH stepping modes,
all backends.

As-built decisions:

- Four runtime constants, zero by default (absolute storage is bit-preserved):
  `eos_gauge_rho_ref`, `eos_gauge_p_ref`, `eos_gauge_e_ref`,
  `eos_gauge_p_bias` (GpuConstants words 21-24; the layout-pin test bumped to
  28 words). They are deliberately SEPARATE from the linear-EOS references
  `eos_p_ref`/`eos_rho_ref`, which every autonomous-route certification gate
  requires to stay zero for ideal gas.
- `EosSpec::runtime_params_gauged(rho0)` computes the references and the
  f64-cancelled pressure-closure bias
  (`gauge_p_bias = gm1*e_ref + p_ref - p_ref_gauge`, exactly `p_ref` when the
  gauge is off and exactly zero when on). The driver activates the gauge for
  every driver-built compressible run; the structured GUI path activates it in
  `apply_structured_compressible_runtime`.
- STATE-form pressure closure everywhere (decl, primitives, implicit p row):
  `p' = gm1*(rho_e' - |rho_u|^2/(2*(rho'+G))) + dp_drho*(rho' + (G - rho_ref))
  + gauge_p_bias`, with the reference constants grouped FIRST — exact
  constant-constant arithmetic, so a gauge-stored liquid density never
  round-trips through its large absolute value. The WGSL printer now preserves
  right-hand grouping of +,-,*,/ (it used to flatten `a + (b - c)` into
  left-associated re-evaluation; the CPU Rust emitter was already fully
  parenthesized, so this also removes a latent CPU/GPU grouping divergence).
- KT flux regrouping in `derive_central_upwind`: the analytic weight sum
  `aphiv_sum = a_pos*phiv_pos + a_neg*phiv_neg` (the +/- a_sf dissipation
  cancelled symbolically) advects the constant references
  (`G*aphiv_sum` in continuity, `(e_ref+p_ref)*aphiv_sum` in energy); the
  momentum pressure term consumes the gauge face pressure, dropping
  `p_ref*Sf` over each closed cell analytically. Dissipation terms see only
  state differences and are gauge-invariant as written.
- Algebraic recovery rows: `(rho'+G)*u = rho_u` and `(rho'+G)*R*T = p' + P`
  distribute into two target-linear products; the lowering now MERGES multiple
  target products into one implicit source term with a summed coefficient
  (new `Coefficient::Sum` variant). T and u stay absolute fields.
- Consumers converted (all keep ABSOLUTE-value host APIs and convert through
  the solver's cached references): driver IC/BC helpers
  (`set_uniform_state`, `set_state_fields`,
  `set_compressible_inlet_isothermal_x`, `evaluate_inlet_declarations`), the
  host CFL sampler, both autonomous health/audit kernels (unstructured
  `compressible_explicit_control` words 21/23; structured autonomous `Params`
  + audit WGSL), the implicit positivity/retry gate, the structured GUI
  seeds/BCs, and the GUI matrix positivity checks. The BC expression
  declarations reconstruct the absolute state inside the model
  (`bc(rho)+G`, outlet density floor applied to the absolute value).
- TRAP (cost one debugging round): the structured GPU solver has its OWN
  `set_eos` mirror; missing the gauge fields there left the audit kernel
  reconstructing `rho' + 0`, rejecting every cell — the GUI symptom is
  "GPU explicit health check halted after 0 accepted batch steps (N invalid
  cells)". Pinned by
  `gpu_structured_obstacle_direct_autonomous_accepts_all_steps`.

Semantics changes visible to users/tests:

- Readbacks and the GUI plot/legend now show GAUGE pressure (zero at the
  quiescent reference, like the all-Mach models) and gauge rho/rho_e.
  Absolute-bound assertions shift by the references
  (`gui_default_convergence_test`).

Acceptance (all green):

- `compressible_explicit_acoustics_test`: the 200 Pa gates reproduce the
  pre-gauge metrics bit-for-bit-close on all four topology/backend
  combinations, and the new 0.5 Pa gates (a signal spanning ~64 f32 quanta
  under absolute storage) are exactly as clean in relative terms
  (rms roughness ~0.4-1.8% of amplitude, transit speed +1.6/+2.3% of c) —
  the precision recovery this arc exists for.
- GUI matrix (CPU x3 backends + GPU Direct/Plot autonomous parity), implicit
  GUI defaults (backstep + obstacle), compressible MMS orders, OpenFOAM
  acoustic reference, full lib suite, WGSL snapshot re-blessed.
- The GUI obstacle soak (1500 steps, from rest): smooth reverberant acoustic
  field, no speckle; roughness floor is now discretization content, not
  representation quanta.
