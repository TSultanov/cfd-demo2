# RK4 whole-step staged-uniform supergraph

This note is an isolated design/measurement artifact. It does not change the
production solver or generated shaders.

## Decision

Keep classical RK4 and every spatial dispatch, but remove the host boundary
between stages. Pack the four stage-time constant records into aligned slices
of one uniform arena, bind slice `s` to every kernel in stage `s`, and encode
the complete RK step into one compute pass and one queue submission. The
history copy can precede the pass in the same command buffer.

This is safer than fusing the cell residual and update bodies. Both actual
generated residuals read neighbor state:

- CSR: `explicit_residual_compressible.wgsl` selects `other_idx` and reads
  `state[other_idx * stride + ...]`.
- structured: `explicit_residual_compressible_structured.wgsl` computes a
  neighbor index and reads that neighbor's state.

If an invocation updates its own `state` in the same dispatch, a peer can see
the next-stage value while still assembling the current-stage residual. WGSL
has no grid-wide barrier. Separate dispatches are therefore a correctness
requirement, not accidental overhead.

## Exact mathematics

For differential state `U`, local mass block `M`, integrated spatial residual
`R`, cell volume `V`, and ordered algebraic recovery `Z = g(U)`, define

```text
F(Y, t) = solve(M(Y, t), P R(Y, g(Y), t) / V)
c = [0, 1/2, 1/2, 1]

k1 = F(U^n,               t_n)
k2 = F(U^n + h k1/2,      t_n + h/2)
k3 = F(U^n + h k2/2,      t_n + h/2)
k4 = F(U^n + h k3,        t_n + h)
U^(n+1) = U^n + h (k1 + 2 k2 + 2 k3 + k4) / 6
```

`P` selects the compact differential rows. The existing generated recurrence
is retained exactly:

```text
stage 1: B = U; A = k1/6;        U = B + h k1/2; recover Z
stage 2:        A = A + k2/3;    U = B + h k2/2; recover Z
stage 3:        A = A + k3/3;    U = B + h k3;   recover Z
stage 4:                         U = B + h(A+k4/6); recover Z
```

Only command-buffer grouping and the source of `constants.time` change. Four
immutable records carry `time = t_n + c_s h`; all other fields are identical
for a fixed step.

## Execution schedule

Let `Q_s` be the dependency-ordered preparation, gradient, face-flux, and cell
residual dispatches for stage `s`, and `K_s` its generated RK update dispatch.

```text
copy/rotate history
begin one compute pass
  bind C0; Q0 dispatches; K0 dispatch
  bind C1; Q1 dispatches; K1 dispatch
  bind C2; Q2 dispatches; K2 dispatch
  bind C3; Q3 dispatches; K3 dispatch
end pass
submit once
```

Each dispatch remains a WebGPU usage scope. Storage writes from one dispatch
are visible to subsequent dispatches in the pass; the repository already
relies on and documents this property in `modules/graph.rs` and the structured
batched router.

Use static uniform offsets, not dynamic offsets:

```text
a = device.limits().min_uniform_buffer_offset_alignment
q = align_up(sizeof(Constants), a)
C_s = BufferBinding { buffer: stage_constants, offset: s*q,
                      size: sizeof(Constants) }
```

On the measured adapter `a = 256`. Static slices preserve the generated bind
group layouts (`has_dynamic_offset = false`) and consume no additional binding.
Structured kernels with different EOS-tail layouts can occupy separate
`(kernel, stage)` slices in one arena; pack each slice with its existing
`pack_kernel_constants` function.

## Liveness and alias obligations

| Resource | Stage liveness | Required relation |
|---|---|---|
| `state` | residual reads; update reads/writes; live across all stages | no alias with RHS/history/workspaces |
| `state_old`, `state_old_old` | read-only after history rotation | copies complete before first dispatch |
| `rk_base` (`B`) | written K1, read K2--K4 | not overwritten between stages |
| `rk_accum` (`A`) | written K1, RMW K2--K3, read K4 | dead only after K4 |
| `rhs` | written by each residual, read by matching update | may be overwritten only after K_s |
| gradients/face flux/BC values | written then read within each `Q_s` | dispatch order must be preserved |
| `C_0..C_3` | read-only for the whole command | aligned, non-overlapping, immutable until completion |

Every kernel in `Q_s`, including expression-valued BC preparation, must bind
`C_s`; binding only the residual/update would give time-dependent BCs and
sources the wrong abscissa. The ping-pong phase is advanced once before graph
encoding and must remain fixed through K4. If APIs outside the explicit graph
expect the canonical constants buffer, copy `C_3` into it after the compute
pass in the same command buffer (or issue an ordered queue write); this does
not add a submission.

## Counts and traffic

Observed production scheduling:

| Path | current submissions/step | current passes/step | candidate |
|---|---:|---:|---:|
| structured | 1 history + 4 stages = 5 | 4 | 1 submission, 1 pass |
| generic/CSR | 1 pre-copy + 4 residual + 4 update = 9 | 8 | 1 submission, 1 pass |

The generic path also writes its constants seven times per step (explicit
`dtau`, prepare, four stage setters, finalize). Structured calls
`write_kernel_constants` for every compiled kernel four times in the loop and
once at final time. A single packed arena write replaces these stage-time
writes.

No dispatch is removed, and bulk storage traffic is unchanged. With `d`
differential scalars per cell, the current compact classical recurrence has a
logical lower bound of `76d` bytes/step in its update dispatches (including
RHS reads), and residual materialization adds `16d` bytes of RHS writes, for
`92d` bytes/cell/step before primitive recovery and spatial reads. The proposed
schedule changes none of those bytes. It targets command encoding, pass, queue,
and tiny-uniform overhead.

The current generated explicit residuals use at most 20 storage bindings; the
device is requested with a limit of 31. The proposed schedule preserves the
per-pipeline count exactly, preserves four bind groups, and needs one uniform
binding as before. It only caches four variants of the bind group containing
`constants`.

## Prototype evidence

`examples/probe_rk4_single_submit.rs` implements a neighbor-reading periodic
five-point stencil and the same operator through CSR adjacency. It compares:

- `Split4`: one residual+update submission per stage (structured baseline),
- `Split8`: separate residual and update submissions (generic baseline),
- `Single`: four static uniform slices, eight ordered dispatches in one pass.

All schedules use the compact classical recurrence above. On an M2 Max with
wgpu 27, the single-pass result was bitwise equal after 17 steps for both
topologies (`bit_mismatches=0`, `max_abs=0`). Median queue-to-fence results:

| cells | structured split4 -> single | speedup | CSR split8 -> single | speedup |
|---:|---:|---:|---:|---:|
| 4,096 | 162.7 -> 36.5 us/step | 4.45x | 352.2 -> 36.9 us/step | 9.54x |
| 16,384 | 152.5 -> 40.3 us/step | 3.78x | 342.8 -> 36.4 us/step | 9.41x |
| 65,536 | 149.6 -> 56.8 us/step | 2.63x | 309.9 -> 58.0 us/step | 5.35x |
| 262,144 | 200.9 -> 170.2 us/step | 1.18x | 258.3 -> 179.8 us/step | 1.44x |
| 1,048,576 | 606.6 -> 587.7 us/step | 1.03x | 683.5 -> 646.7 us/step | 1.06x |

This is a mechanism benchmark, not a claim that a full compressible graph will
obtain the small-grid ratios: real stages contain more spatial dispatches and
work. It decisively demonstrates portability, bitwise equivalence, and that
the gain converges toward one rather than imposing a large-grid penalty.

## Rejected alternatives

### In-place residual/update kernel fusion

For two coupled cells with `F_i(U)=U_j-U_i`, `U=(0,1)`, correct stage one is
`k=(1,-1)`. If cell 0 updates first, cell 1 can read `U_0=h/2`, producing
`k_1=h/2-1` and an erroneous `h^2/4` term in its next state. A read-only
`stage_in`/separate `stage_out` ping-pong can make fusion correct, but adds a
full-state buffer or a strong alias/mirroring proof, adds bindings, and changes
every upstream stage binding. It is a separate, higher-risk optimization.

### Five-stage 2N low-storage RK4

The Carpenter--Kennedy-style recurrence

```text
D_0=0; U_0=U^n
D_i=a_i D_(i-1) + h F(U_(i-1), t_n+c_i h)
U_i=U_(i-1) + b_i D_i, i=1..5
```

can reduce capacity, but adds a fifth complete spatial evaluation. Even if the
residual tail updates `D` in-place in the RHS buffer, its local logical traffic
is about `96d` bytes versus classical's `92d`, before the 25% spatial-work
increase. Without measured CFL growth greater than 25%, it is not a speed
replacement for the existing four-stage method.

### Persistent kernel / temporal blocking

WGSL has no grid-wide workgroup barrier and no portable forward-progress
guarantee for an atomic spin barrier, so a persistent all-stage kernel can
deadlock or mix stages. Exact structured temporal blocking needs a growing
four-stage halo. The actual requested device limit is 16,352 bytes of
workgroup storage. Even with effective stencil radius one, an 8x8 interior plus
the four-cell RK halo needs `16*16*23*4 = 23,552` bytes for compressible state
alone; a 4x4 interior already uses 13,248 bytes and redundantly loads 9 halo
cells per output before gradients and fluxes. Radius two would require the
52,992-byte 24x24 tile. Arbitrary CSR meshes would additionally need graph
partitions and redundant halo evaluation.

## Production gates

An implementation changes time integration scheduling even though generated
WGSL should remain byte-identical. Require:

1. a split-vs-supergraph bitwise GPU parity test for structured and CSR models,
2. stage-time source/BC parity at all four abscissae,
3. `mms_explicit_rk4_order_test` and every mandatory `mms_*` suite,
4. generated WGSL snapshot parity,
5. release benchmarks across launch-bound and throughput-bound mesh sizes.
