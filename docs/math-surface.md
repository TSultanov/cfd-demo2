# The math surface: what models declare vs. what the engine owns

This document is the normative boundary between the **math layer** (what a
model author writes) and the **engine layer** (what the solver
infrastructure provides). Everything in the math layer is a declaration that
codegen lowers to GPU numerics at build time; nothing in the engine layer is
expressible — or meant to be expressible — as model math.

The buoyant Boussinesq model
([buoyant_incompressible.rs](../src/solver/model/definitions/buoyant_incompressible.rs))
is the reference example: a complete new coupled physics (momentum +
pressure + temperature with buoyancy feedback) added as one definitions file
plus a registry entry, with every kernel derived.

## Math layer (model-declared)

| Surface | Declared with | Lowered by |
|---|---|---|
| Conservation/transport equations | `typed_fvm` / `typed_fvc` term builders (`ddt`, `div`, `div_flux`, `laplacian`, `grad`, `source_*`), summed and `.eqn(target)` | `unified_assembly` (one generated assembly kernel per model) |
| Per-term schemes | `.scheme(Scheme::…)` on a term — bakes the scheme as a literal, independent of the runtime knob | scheme expansion + assembly |
| Bounded convection | `.bounded()` on an implicit div term — subtracts the continuity defect `(div φ)·φ_P` from the diagonal (OpenFOAM `bounded Gauss`) | assembly |
| Algebraic relations (EOS, primitive recovery) | `typed_alg::equation(target, lhs, rhs)` over `TypedAlgExpr` (fields, params, `mag_sqr`, `+ − ×`; clear denominators — no `/` in coupled rows) | `lower_algebraic_equation` → implicit/explicit source rows scaled by `inv_dt` |
| Uniform model parameters | `TypedParamRef<D>` consts matching a module port manifest (e.g. `eos_*`) | name-resolved to `constants.<param>` |
| Flux definitions | `FluxExprSpec::AdvectingVelocity`, `derive_rhie_chow(system, layout)`, `FluxSchemeSpec::CentralUpwind(CentralUpwindDecl)` | flux-module codegen; Rhie–Chow aux kernels auto-attached by the deriver |
| Wave speeds / EOS face relations | `AlgExpr` declarations on `CentralUpwindDecl` (`pressure`, `wave_speed_sq`, `generalized_wave_speed_sq`) | `lower_alg_to_face` into reconstructed-state face expressions |
| Boundary conditions | `BoundaryCondition` (Dirichlet/Neumann/zero-gradient) with `BcValue::Const` or `BcValue::Expr(BoundaryExpr)` over `interior(f)` / `bc(f)` / `param(p)` | static GPU tables + one generic `bc_expr_update` kernel (snapshot semantics, refreshed every outer iteration) |
| Directional body forces | `typed_fvc::source_directional(coeff, [dx, dy], target)` — scalar coefficient tree × constant direction per component | assembly explicit-source path |
| Per-component vector sources (MMS) | `typed_fvc::source_vector(field, target)` — a Vector2 state field read component-wise | assembly explicit-source path |
| MMS sources | one more declared equation term + a state field uploaded host-side | same machinery as any source |

### Model contracts worth knowing

- **Sign conventions**: terms sum to zero. `laplacian(κ, φ)` assembles as
  `−∇·(κ∇φ)` (so `ddt + laplacian = 0` is the heat equation); explicit
  sources enter the residual negatively (declaring `source_coeff(S)` makes
  the solved equation `… − S = 0`).
- **Coupled-row ordering**: `add_algebraic_equation` treats only
  already-added equation targets as unknowns — add conservation equations
  before the algebraic relations that couple to them.
- **Outlet pressure**: the derived Rhie–Chow flux special-cases outlet-type
  boundary ghosts; a model used on an outlet-bearing mesh must pin the
  pressure gauge with an outlet Dirichlet, or the pressure system is
  singular.
- **Units close at compile time** (type-level dimensions with `cast_to`
  escape hatches) and are re-validated at runtime (`validate_units`).
  Boundary-expression literals are unit wildcards (epsilon floors).

## Engine layer (never DSL-ified)

Linear solvers (FGMRES/AMG/Schur internals), outer-loop control flow,
batching and adaptive convergence breaks, dual-time retry/rollback and
positivity guards, kernel fusion, bind-group/port plumbing, and the
build.rs WGSL pipeline. Models parameterize these through policy structs
(`CoupledCapabilities`, `RelaxationDefaults`, `ModelLinearSolverSpec`) —
never through equations.

Known engine limits (tracked in the plan):

- The generic Schur preconditioner assumes the non-pressure block is exactly
  the velocity pair; models with extra scalar unknowns (e.g. temperature)
  currently use the default preconditioner.
- At pressure-Dirichlet (outlet) boundary faces, the derived Rhie–Chow
  flux's correction bracket compares the cell-centered `grad_p` against a
  one-sided compact difference centered half a cell away — an O(h) mismatch
  on faces where the normal second derivative of p is nonzero. Measured
  cost: pressure converges at ~1.36 with an outlet vs ~1.64 all-Neumann in
  the buoyant MMS; velocity and advected scalars are unaffected (order 2).

A contract worth knowing when extending codegen: the packed `grad_state`
buffer is keyed by STATE OFFSET (stride = state stride), matching the
assembly's reconstruction reads; boundary tables are keyed by unknown RANK.
The two coincide only while solved unknowns are a prefix of the state
layout — `mms_buoyant_order_test` guards the non-prefix case (its
temperature sits behind aux fields).

## Validation contract

Every model change runs the MMS gate (primary oracle: convergence orders
against manufactured solutions) and — when numerics can change — the
OpenFOAM reference suite under the no-growth policy. See `AGENTS.md` for
the exact commands and the byte-identical-WGSL skip rule.
