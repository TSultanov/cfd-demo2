# AGENTS.md instructions for /Volumes/sources/cfd2

## Mandatory MMS gate (primary correctness oracle)

For any change touching discretization, schemes, boundary conditions, codegen, or time
integration, run the MMS convergence-order suite and require it green:

```bash
cargo test --features dev-tests --test mms_diffusion_order_test -- --test-threads 1
cargo test --features dev-tests --test mms_scalar_transport_order_test -- --test-threads 1
cargo test --features dev-tests --test mms_incompressible_order_test -- --test-threads 1
cargo test --features dev-tests --test mms_buoyant_order_test -- --test-threads 1
cargo test --features dev-tests --test mms_compressible_order_test -- --test-threads 1
```

(Glob any additional `tests/mms_*` suites as they are added.) These tests verify the
discrete operators converge at design order against manufactured exact solutions; an order
drop or a blown error cap is a hard regression. The
incompressible (saddle-point) suite is the slowest (~3 min, dominated by the SOU n=64
level); the compressible suite (~3 min) marches each level a fixed minimum t=6 to pass
the slow thermal transient. Keep new MMS cases lean — steady cases settle in ~13 steps
for the incompressible models, so cost is per-step outer iterations, not step count.
NOTE: the compressible suite asserts the physical-NS viscous operator (EXTRA_SHEAR = 0);
its `#[ignore]` probe (doubled-shear sources) must SATURATE — order-2 convergence there
means the pre-June-2026 double-counted laplacian came back (see the header of
`tests/mms_compressible_order_test.rs`).

## Mandatory ALE gate (moving-mesh path)

For any change touching the ALE codegen (`Term::relative_to_mesh`, moving-volume ddt,
`mesh_fluxes`/volume-history plumbing, `begin_ale_step`, `src/solver/mesh/ale.rs`) or the
mesh-refresh seam, run the ALE suite and require it green — same declaration style as the
MMS gate above:

```bash
cargo test --test static_wgsl_snapshot_test                    # static models byte-identical (hash-pinned)
cargo test --features meshgen,cpu --test ale_zero_flux_equivalence_test -- --test-threads 1
cargo test --features meshgen,cpu --test ale_gcl_test -- --test-threads 1
cargo test --features cpu --test ale_conservation_test -- --test-threads 1
cargo test --features dev-tests --test mms_ale_order_test -- --test-threads 1
```

The snapshot test pins every generated WGSL file's content hash (bless ritual in its
header); zero-flux equivalence pins ALE-with-zero-fluxes == static bitwise on CPU;
the GCL gate pins free-stream preservation on a deforming mesh (Euler AND BDF2, both
backends); the conservation audit pins the f64 mesh-side mass/area identities and
zero spurious ALE-injected velocity; the MMS-ALE suite pins temporal BDF2 order ~2
(measured 2.154) and spatial order ≥ 1.35 — the honest measured band (1.387,
skew-limited: the prescribed motion keeps cells persistently non-orthogonal, a
pre-existing spatial property, NOT an ALE defect) — plus the decisive in-test
equivalence assert: the moving-mesh error must match a static solve on the same
deformed geometry to ±5% (measured equal to 3 significant digits). Also run
`cargo test --features meshgen --test ale_metal_fastmath_evidence` (<5 s) — it backs
the GPU/BDF2 zero-flux tolerance gate's ~1-ulp-per-assembly justification.

Runtime honesty (measured July 2026, M2 Max): zero-flux ~2 min, GCL ~5 min,
conservation ~3 min, MMS-ALE ~8 min (the n=64 spatial level and the 80-step temporal
level dominate) — ~18 min total; the GPU legs (pipeline compilation for the ALE
models + multi-hundred-step runs) dominate. The CPU-only legs skip cleanly without
an adapter if you need a headless quick pass.
