# Compact explicit RHS / sparse mass audit (round 4)

> Historical attack report. Every defect below is now fixed and pinned by
> `probe_sparse_mass_closure_alias_round4`,
> `probe_explicit_mass_fix_reaudit_round4`,
> `probe_closure_lowering_adversary_round4`, and
> `probe_final_mass_adversary`. The accepted implementation uses conservative
> constant-exact versus model-owned runtime mass contracts, per-connected-block
> row/column equilibration with capped matching-quality/representability guards,
> per-component build-time constant rcond checks, quiet-NaN failure, and exact
> full coupled flux stride validation. See round 6
> of `explicit-bandwidth-approach-registry.md` for the closure verdict.

No production files were changed by this audit.

## Verdict

The structural support propagation used by the sparse generated elimination is
sound for finite inputs and preserves the operation order of a dense guarded
elimination on every potentially nonzero edge. Compact RHS projection also
preserves full coupled indexing inside assembly.

The overall capability gate is nevertheless unsound: its determinant proof
treats recoverable algebraic state fields as independent symbolic atoms, even
though the model's explicit primitive closure constrains or aliases them before
every residual/stage. Two exact, always-singular models pass validation.

A separate accepted constant mass block demonstrates that “pivot only below
1e-20” is not numerically stable partial pivoting: a well-conditioned 2x2
system produces the wrong f32 solution.

## Exact closure counterexamples

### Constant closure

Use interleaved equation order `q, a, r`. `q` and `r` are differential and `a`
is a recoverable algebraic row:

```text
q row: a * ddt(q) + ddt(r)
a row: local algebraic closure
r row: ddt(q) + ddt(r)
explicit closure: a := 1.0
```

The compact differential block is

```text
M = [[a, 1],
     [1, 1]].
```

`validate_explicit_mass_determinant` sees the unconstrained polynomial
`det(M)=a-1`, calls it nonzero, and accepts the model. The authoritative closure
writes `a=1.0f` before the first residual and after every RK stage, so the
actual block is exactly

```text
[[1, 1],
 [1, 1]]
```

at every solve. Generated elimination produces `mass_1_1 -= 1*1 = 0` and then
divides the second transformed rate by the `1e-20` floor.

### Closure aliases

Use interleaved order `q, a, r, b`:

```text
q row: a * ddt(q) + ddt(r)
a row: local closure
r row: b * ddt(q) + ddt(r)
b row: local closure
explicit closures: a := q; b := q
```

The validator sees `det(M)=a-b`, with different state-field atoms, and accepts.
Both closures write the same f32 value, so the runtime rows are bitwise equal
and the determinant is always zero.

Both are reproduced by
`tests/probe_sparse_mass_closure_alias_round4.rs`. A passing probe means the
current validator incorrectly accepted the model:

```text
cargo test --test probe_sparse_mass_closure_alias_round4 -- --nocapture
2 passed
```

## Conditional pivot counterexample

The constant block

```text
M = [[1e-19, 1],
     [1,     1]],   rhs = [1, 2]
```

has 2-norm condition number about 2.62 and exact/f32-full-partial-pivot solution
`[1,1]`. It passes the constant validator: the first pivot is finite and is
larger than `1e-20`, and elimination does not overflow.

The generated policy swaps only when `abs(pivot)<1e-20`. It therefore forms a
factor near `1e19`; f32 cancellation leaves the ignored lower entry at
`5.9604645e-8`, and back substitution returns `[0,1]`.

`examples/probe_compact_mass_audit.rs` constructs the accepted model and
compares the generated policy to full partial pivoting.

This instability overlaps the old dense algorithm: sparse pruning did not
create it. Guarded zero-pivot row swaps improve the old behavior but do not
constitute ordinary partial pivoting.

## Structural support proof

Let `S[i,j]` mean generated storage `mass_i_j` may be nonzero. `S` is an
over-approximation; repeated terms, dynamic zeros, and cancellations can make
the runtime value zero while leaving `S=true`.

1. **Initialization.** Every emitted nonzero coefficient term sets its matrix
   entry's support. A constant or constant-product which emits `0f` can be
   skipped. Therefore every potentially nonzero initial entry is supported.

2. **Conditional row swap.** A candidate is considered whenever
   `S[candidate,pivot]` is true. For every column where either row has support,
   the generated branch swaps both storage values. After emitting the branch,
   both abstract rows receive the union of their old supports. On the taken and
   untaken runtime paths, actual support is therefore a subset of the union.
   Sequential candidate swaps preserve this invariant.

3. **Elimination.** If `A[row,pivot]` can be nonzero, its support is true and
   the factor/rate update is emitted. A product can affect `(row,column)` only
   when the pivot row supports that column, in which case the matrix update is
   emitted and the destination support is set. Existing destination support is
   retained. The eliminated lower entry is deliberately removed from the
   upper-triangular support used by back substitution.

4. **Back substitution.** Every possibly nonzero upper entry remains supported,
   so every dense subtraction which can affect a finite result is emitted.

Thus no finite nonzero contribution is pruned. With finite operands, skipped
operations are structurally `0*x`; sparse and dense guarded algorithms have the
same numerical values apart from possible signed-zero bits.

Concrete search evidence:

- all 1,953,125 3x3 matrices over `{-2,-1,0,1,2}`: zero sparse/dense mismatches;
- 250,000 random 4x4 matrices with support supersets, cancellations, and
  sub-floor values: zero mismatches;
- an independent audit enumerated 319,254 nonsingular ternary cases through
  dimension 3 and 523,763 random finite support-superset cases through
  dimension 8: zero mismatches.

## Repetition, cancellation, zero and NaN

- Repeated/canceling constant terms are accumulated in emitted f32 order by the
  validator. Structural support remains conservatively true.
- Dynamic terms using the same canonical atom cancel in the determinant
  polynomial. Different algebraic field names with aliased closures do not;
  this is the demonstrated validator defect.
- `+0` and `-0` constants are pruned consistently by codegen and determinant
  construction. Dynamic zero retains support and executes the dense edge.
- Nonfinite constants are rejected. Dynamic coefficient/state NaNs are not:
  `abs(NaN)<eps` is false, no row swap occurs, and `safe_pivot(NaN)` returns
  NaN. The sparse algorithm can avoid some old dense `0*NaN` contamination on
  structurally disconnected edges, so nonfinite behavior is not byte-equivalent
  to the old dense path. A finite-runtime-coefficient invariant is required.

## Compact RHS and full-rank indexing

The projection is correctly placed only at final RHS writeback:

1. `main_assembly_fn` retains all coupled equations and declares full coupled
   accumulators.
2. Equation offsets, `BcTable` stride/offsets, and face-flux offsets remain full
   coupled ranks.
3. `write_rhs_projection` maps differential destination rank to its original
   coupled accumulator.
4. The RK stage reads the same compact differential order.

For interleaved `q, algebraic, r`, the generated probe confirms:

```text
rhs[cell*2 + 0] = rhs_0
rhs[cell*2 + 1] = rhs_2
fluxes[face*3 + 2]
bc_kind[face*3 + 2]
```

There is one API hole: `validate_unified_assembly_inputs` requires only
`flux_stride > 0`. Passing compact stride 2 to the same full-rank example is
accepted and emits `fluxes[face*2 + 2]`, which aliases the next face or runs out
of bounds. Production callers currently pass full
`model.system.unknowns_per_cell()` and are correct. The codegen API should
require `flux_stride == coupled_stride` whenever fluxes are consumed.

## Recommended corrections

1. Pass the ordered explicit primitive closure into the determinant proof and
   symbolically substitute recoverable algebraic coefficient fields.
   Constants and identifier aliases must lower to the same polynomial atoms;
   identical fully-expanded unsupported expressions must at least share one
   opaque atom. Conservatively reject a determinant proof that depends on an
   unsupported closure relation.
2. Use actual runtime partial pivoting—select/swap the largest absolute entry in
   the remaining pivot column—or reject constant blocks with unacceptable
   pivot growth. Mirror the exact generated policy in constant validation.
3. State and validate a finite dynamic mass-coefficient contract. NaN cannot be
   repaired by structural support or the current pivot floor.
4. Assert full coupled flux stride at the matrix-free assembly seam.
5. Retain the sparse support algorithm; no under-propagation counterexample was
   found, and its invariant is independent of the closure-proof bug.

## Novelty versus inherited behavior

- **New and sound:** compact differential RHS/history layout, projection-only
  writeback, structural fill/support propagation, conditional-support union.
- **New but unsound:** symbolic determinant proof without explicit-closure
  substitution.
- **New partial improvement:** conditional swaps for near-zero declaration
  pivots.
- **Inherited limitation:** safe-floor dense elimination and lack of full
  partial pivoting; sparse pruning preserves that numerical policy.
- **Preserved status quo:** BC and flux tables remain coupled-rank layouts;
  only the terminal RHS storage is compact.
