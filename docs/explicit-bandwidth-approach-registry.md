# Explicit solver bandwidth approach registry

This is the live registry for the structured and unstructured explicit-solver
performance program.  Families are grouped by incompatible architectural
mechanism.  A route marked **blocked** is not assigned again unless the new
proposal supplies the stated reopening mechanism.  All production candidates
must remain model/IR-derived; handwritten physics-specific kernels are out of
scope.

## Acceptance contract

- Preserve generated-kernel mathematical equivalence and all MMS/ALE/GUI gates.
- Beat the current router on end-to-end accepted RK steps after warm-up on both
  structured and unstructured meshes.
- Report GPU timestamp time, logical bytes per accepted step, an explicit byte
  lower bound, and the resulting effective bandwidth.  CPU wall time alone is
  not evidence of bandwidth saturation.
- Demonstrate scaling on meshes large enough to escape cache and launch-bound
  regimes.  A candidate that wins only on one topology or one small mesh does
  not replace the common router.
- Survive an independent novelty and regression audit against the existing
  fusion schedule, explicit residual graph, and four low-storage RK kernels.

## Families

| ID | Architectural family | Independent mechanism | Round | Status | Owner | Reopen condition if blocked |
|---|---|---|---:|---|---|---|
| F1a | Structured spatial tiling | Radius-two workgroup tile with generated live-set placement | 3 | viable isolated mechanism; production proof pending | `structured_tiling_round1` | — |
| F1b | Structured scalar-stream canonical faces | Reuse one workgroup scalar plane across live flux components instead of storing a face tuple | 3 | viable isolated mechanism; production proof pending | `structured_tiling_round1` | — |
| F1c | Swept/temporal RK tiling for flow | Retain several RK stages in one halo-expanded tile | 2 | **blocked** | `structured_tiling_round1` | A live-set construction that fits at least a 16x16 core and keeps residual-work amplification below 1.5x |
| F1d | Persistent cross-grid workgroups | Software grid barrier or persistent tile queue | 2 | **blocked** | `structured_tiling_round1` | A portable WebGPU progress guarantee or a measured mechanism with no grid-wide barrier |
| F2a | Unstructured canonical face gather | Typed canonical face channels, signed cell-CSR gather, reducer fused with RK | 3 | viable K=4 prototype; production route pending | `unstructured_face_once_round1` | — |
| F2b | Atomic/color/scatter accumulation | Endpoint atomics, coloring, or endpoint materialization | 1 | **blocked** | `unstructured_face_once_round1` | A race-free deterministic mechanism below the `4S(F+H)+4H+8N` canonical-gather byte floor |
| F2c | Block-canonical operator streaming | Reordered graph blocks load radius-two closures, stream canonical faces in shared memory, and fuse gather/mass/RK | 4 | viable isolated mechanism; mesh-block quality gate pending | `operator_streaming_round4` | — |
| F2d | Incidence-owned flux rematerialization | Recompute canonical face flux in each cell pull and remove the global flux stream | 5 | **blocked** | `independent_bandwidth_route` | A generated no-contraction materialization primitive, or an explicit bounded-ULP numerical contract, followed by a two-topology production A/B |
| F3a | Submission/bind routing | Cache immutable bind groups and batch a generated stage schedule | 1 | accepted infrastructure; **blocked as bandwidth route** | root | Traffic reduction or cross-kernel scalar replacement, not another submission-only change |
| F3b | Differential/algebraic liveness | Compact RK workspaces and generated rows to differential ranks | 2 | **accepted candidate** | `explicit_liveness_compaction_r2` | — |
| F3c | Generated stage-dataflow fusion | Privatize RHS and ping-pong generated stage state | 2 | **blocked pending live-complement proof** | root + adversary | A mechanically checked construction that preserves every next-stage-live packed slot without erasing the RHS traffic win |
| F3d | Logical-state physicalization | Generate the ordered graph over differential, algebraic, producer, and immutable state; ping-pong only the written partition | 3 | new viable mechanism; implementation pending | root + fusion adversary | — |
| F3e | Differential residual + sparse equilibrated mass graph | Project full local residuals to differential storage; prune impossible elimination/fill edges; solve coupled blocks in row/column-equilibrated coordinates with capped matching-quality and representability gates | 6 | **accepted; adversarial audit passed** | root | — |
| F3f | Semantic face-channel liveness | Preserve the full coupled face layout for equations/BCs while routing producers and every consumer through a generated compact storage-rank projection | 7 | implemented candidate; integration and timing audit pending | `rk4_bandwidth_adversary` + root | — |
| F4 | Roofline/adversarial measurement | Establish whether bandwidth is limiting and reject false saturation claims | 3 | active; prior timestamp ceiling retracted | root + measurement adversary | New hardware-counter access or a falsifiable bandwidth proxy |
| F5a | Structured state physical layout | Generated AoSoA-32 hot-state layout selected from semantic D/A/P/I liveness | 3 | viable isolated mechanism | `explicit_liveness_compaction_r2` | — |
| F5b | Unstructured state physical layout | Compact/reordered AoS hot state selected from semantic D/A/P/I liveness | 3 | viable isolated mechanism | `explicit_liveness_compaction_r2` | — |
| F5c | Universal SoA/AoSoA state | Apply the structured coalescing layout to indirect unstructured gathers | 3 | **blocked** | `explicit_liveness_compaction_r2` | A locality/reordering mechanism that beats compact AoS under the actual indirect incidence distribution |
| F6 | Staged-uniform whole-step supergraph | Prepack four RK-time constant records and encode every existing dispatch barrier in one pass/submission | 4 | viable mechanism; structured partly subsumed, generic production pending | `low_storage_fusion_round4` | — |

## Round log

### Round 1

- Hardware under test: Apple M3 Max, 40-core GPU, Metal 4, 48 GiB unified memory.
- F1 and F2 were assigned without sharing a favored route.
- F3 and F4 are being developed independently by the root agent from the
  current router, generated shaders, timestamp infrastructure, and byte counts.

#### Baseline and byte evidence

- The initial 328--346 GB/s timestamp result is retracted: zero-initialized
  unified pages and unstable Metal timestamp domains made it an invalid
  ceiling. The corrected harness materializes pseudo-random input and uses
  queue-to-completion wall time; its current 256 MiB, 24-pass result is
  **318.8 GB/s**. GPU timestamps are diagnostic-only and reported `n/a` on
  this device.
- A 1024x1024 regular mesh represented by the unstructured topology has
  `N=1,048,576`, `F=2,099,200`, and `H=4,194,304`. The incumbent residual's
  generated mesh/index prologue alone is
  `20N + 44H + 16I = 227.969 MiB/stage`, or 911.875 MiB/RK step.
- Current all-Mach thermal schedules contain seven dispatches/stage on both
  topologies. Structured flow additionally materializes four directed face
  slots per cell; unstructured flux is already produced once per physical
  face, but its cell residual walks every incidence.

### Round 2

#### F3a router ablation

The exact legacy ablation recreates per-dispatch bind-group construction,
separate kernel submissions, and two copy submissions. The cached-separate
ablation isolates bind caching from stage batching. Median queue-to-completion
times on the M3 Max were:

| Grid/model | Cached + batched | Cached + separate | Legacy uncached | Conclusion |
|---|---:|---:|---:|---|
| 256^2 scalar | 0.223 ms | 0.286 ms | 0.327 ms | launch-bound win |
| 256^2 all-Mach | 0.932 ms | 1.039 ms | 1.124 ms | useful small-grid infrastructure |
| 512^2 scalar | 0.242 ms | 0.334 ms | 0.361 ms | launch-bound win |
| 512^2 all-Mach | 4.647 ms | 4.792 ms | 4.789 ms | about 3% |
| 1024^2 all-Mach | 21.215 ms | n/a | prior router 21.189 ms | no large-grid bandwidth win |

F3a stays in production consideration because it is strictly generated-schedule
infrastructure, but it is blocked as the requested new bandwidth algorithm.

#### F1 adversarial correction and mechanism probe

- Radius two is correct for one composed all-Mach/central-upwind spatial stage.
  The original 16x16 `vec4`-face tile was rejected: all-Mach high-order live
  gradients require at least 16,768 bytes before face storage; compressible
  high-order requires still more.
- The surviving F1b construction stores a 20x20 `vec4` state halo (6,400 B),
  six persistent gradient scalars on 18x18 (7,776 B), and one reusable scalar
  for 544 canonical tile faces (2,176 B): exactly **16,352 B**, the requested
  downlevel workgroup-storage limit. Flux components stream through that one
  scalar plane with a barrier between components. Temperature/high-order
  derivatives are recomputed from the state halo.
- A production stage must ping-pong its compact stage state. In-place writes are
  racy because one workgroup can overwrite a neighbor halo before it is loaded.
- Isolated 2048^2, 100-iteration probe: three global passes 1.483 ms; basic
  tile 0.529 ms (2.81x); exact-limit gradient+face-phase tile 0.730 ms (2.03x).
  The basic tile's 41 B/cell logical lower bound implies 325 GB/s. Because this
  slightly exceeds the corrected 318.8 GB/s wall-time stream rate, cache reuse
  and/or logical-byte undercounting is present; it proves a strong isolated
  mechanism, not literal DRAM saturation or an all-Mach solver result.

#### F2 face-once lower bound

The strongest race-free unstructured construction is canonical face
materialization followed by a signed incidence gather, with the reducer fused
into the RK update. Its synchronized byte lower bound is

`4S(F+H) + 4H + 8N bytes/stage`,

and fusing the reducer/RK removes another `8SN bytes/stage` of RHS traffic.
Atomics, coloring, and endpoint-pair materialization were blocked because they
cost at least `8SH` bytes, add contention/serialization, or lose deterministic
f32 accumulation. Full face-once gradient materialization remains a separate
cost decision: for `J=2G+5W` gradient floats/face it costs `4J(F+H)` and can be
worse than the incumbent at all-Mach/compressible live widths.

The isolated K=4 prototype now carries the complete four-scalar
all-Mach/compressible channel set. Rhie--Chow does not require a fifth channel:
`G=(rho d_p) A/d (p_n-p_o)`, `phi_corr=phi_pred-G`, and
`Q_p=G-phi_pred=-phi_corr`, so each endpoint gathers the signed corrected mass
flux directly. On a 1024 grid with 32 synthetic incidences/cell, face-once plus
fused gather is 1.40x faster on regular/skew degree-four graphs and 1.72x at
degree 32, with maximum f32 differences below `1.31e-7`. Declared traffic falls
from 859.85 to 468.09 MiB. The broad face-once method is prior art; the
repo-specific candidate is a generated `FaceChannelPlan`, the Rhie--Chow
channel alias, and fusion of its signed reducer into compact mass/RK updates.

#### F3b liveness candidate

The typed IR now distinguishes coupled rank from differential rank for explicit
RK workspaces. On compressible, the safe compact set is four conserved rows,
not all eight coupled/algebraic rows; algebraic closure remains in full packed
state. Initial release A/B medians at 1024^2:

| Scheme/topology | Before | Candidate | Change |
|---|---:|---:|---:|
| QUICK VanLeer structured | 24.23 ms | 18.95 ms | -21.8% |
| QUICK VanLeer unstructured | 25.86 ms | 19.69 ms | -23.9% |
| Upwind structured | 20.62 ms | 19.64 ms | -4.8% |
| Upwind unstructured | 20.74 ms | 19.70 ms | -5.0% |

RK stage logical traffic falls from 576 to 320 B/cell/step and RK workspace
capacity from 64 to 32 B/cell. The larger high-order win also removes a packed
gradient dispatch proven unused by the expanded `DivFlux` IR. This route passed
generated-WGSL freshness, the static snapshot, all five mandatory MMS suites,
all 11 explicit-RK MMS cases, structured and unstructured GPU RK4 temporal-order
tests (orders 4.15--4.50), CPU freestream controls, cross-ddt retention, and
adversarial primitive-closure validation. It is accepted as a model-ID-free
improvement; remaining allocation overprovisioning is capacity rather than
stage traffic.

#### F3c adversarial block

Typed residual/RK scalar replacement successfully synthesizes all four stages
for shipped structured diffusion, compressible, and all-Mach programs and
removes the global RHS binding. It also enforces distinct input/output buffers,
rejects nested/compound RHS writes, and routes local primitive-closure reads by
dominance. It is not yet routable: a stage writes only differential and derived
slots, so a fresh output buffer does not automatically contain the packed
coefficient complement. Concrete counterexamples are
`generic_diffusion_demo_structured_ibm` (slot 1 `ibm_penalty`) and all-Mach
thermal static EOS/reference slots. The route remains blocked until generated
schedule liveness proves that every next-stage read is either stage-written,
definitely regenerated before its first read, or mirrored as immutable data in
both scratch buffers. Blind per-stage full-state copies would cost more bytes
than the eliminated RHS stream and therefore do not satisfy the reopen
condition.

#### F3d logical-state physicalization

The mechanism that reopens fusion is not a copy of the untouched packed-state
complement. Codegen instead classifies logical state into differential `D`,
stage-written algebraic `A`, prerequisite-produced `P`, and immutable `I`.
`W=D union A` occupies two compact ping-pong regions, while `P union I` is kept
once in an auxiliary region. Every prerequisite, residual, mass solve, stage
update, and closure is rewritten against that physical map; a packed view is
gathered only at the public API seam. Audited shipped partitions are:

- structured IBM diffusion: `Q=2`, `W={0}`, `I={1}`;
- compressible: `Q=23`, eight `W`, nineteen `P/I` slots;
- all-Mach thermal: `Q=22`, ten `W`, two `P`, eight live `I`, and two
  public-dead slots.

Including RK base/accumulator storage, projected physical footprints fall from
24 to 20 B/cell (diffusion), 216 to 156 (compressible), and 208 to 160
(all-Mach). The construction remains pending until the ordered graph proves
producer dominance, `P` read/write race freedom, `I` immutability, `W`
must-write coverage, affine state access, and the WebGPU eight-binding limit.

#### F5 topology-specific state layouts

An isolated 2,097,152-cell state probe rejects a universal layout. For eight
affine hot components, AoS-23 takes 0.504 ms, compact AoS-8 0.190 ms, and
AoSoA-32 0.190 ms; the same indirect workload takes 0.633, 0.381, and 1.075 ms.
At twenty hot components, structured compact/AoSoA tie near 0.474 ms while the
indirect AoSoA route degrades to 2.670 ms versus 0.791 ms for compact AoS.
Therefore structured AoSoA-32 and unstructured compact/reordered AoS remain
separate live routes, while universal SoA/AoSoA is blocked.

### Round 4

#### F3e compact residual and structural mass graph

The full coupled residual algebra remains local and unchanged, including its
original BC and face-flux ranks, but its global writeback is now the ordered
projection onto differential rows. RK stages consume that same compact rank.
For compressible this removes four dead residual stores per cell/stage and
changes the RHS stride from eight to four. The mass solver also propagates a
Boolean support graph through guarded row swaps and Gaussian fill-in, emitting
only potentially nonzero elimination rows and upper back-substitution edges.
Diagonal compressible mass blocks emit no factors or row-swap scans; thermal
all-Mach emits only its live pressure/temperature 2x2 edge.

Against a preserved pre-change release binary at 1024^2 QUICK VanLeer, three
run medians moved from 18.978 to 18.604 ms/step structured (-2.0%) and 19.211
to 18.378 unstructured (-4.3%). This is incremental to F3b's 21.8--23.9%
improvement. Upwind all-Mach is neutral within measurement noise because its
spatial residual dominates and it has no algebraic residual rows. The support
lemma and f32 exceptional-state behavior subsequently passed the multi-round
audit recorded in round 5.

#### F6 staged-uniform supergraph

Direct residual/update fusion is rejected: both stencil and CSR residuals read
neighboring state, so an in-dispatch in-place update races without a portable
grid barrier. The independent alternative preserves every dispatch boundary
but prebuilds four 256-byte-aligned constant records with
`time=t_n+[0,1/2,1/2,1]dt`, then encodes history, prerequisites, residuals, and
updates in one compute pass/submission. A structured/CSR prototype stayed
bitwise equal after 17 steps. At 1,048,576 cells it improved isolated schedules
1.032x structured and 1.057x CSR; at 65,536 cells it improved 2.63x and 5.35x.
Structured production already batches within each stage, so the remaining
route crosses stage-time submissions; generic production still has the full
9-to-1 submission opportunity. Bulk storage traffic is unchanged, so F6 is
kept as routing infrastructure rather than the bandwidth algorithm.

#### F2c block-canonical operator streaming

The new incompatible face route partitions both Cartesian and reordered graph
meshes into blocks. Each workgroup loads a radius-two closure, computes unique
canonical internal faces into shared memory, gathers signed residuals locally,
then mass-solves and writes ping-pong state; only cut faces are duplicated.
The isolated vec4 probe is bitwise exact for one stage and, at 2048^2, improves
materialized-face time 3.99x structured and 6.37x on a permuted graph, reducing
requested traffic from 448 to 52 and 496 to 61 B/cell-stage. It is conditional
on partition quality: Hilbert-blocked Voronoi samples measured radius-two
amplification 2.167 and cut fraction 0.185, while random contiguous blocks
explode to 14.409 and 0.975. Production routing therefore requires a generated
block-quality gate and fallback to F2a/current CSR.

#### F4 measurement limitation

Apple Metal exposes a timestamp counter through wgpu, but timestamp marks are
not trustworthy enough here for the ceiling: cross-command marks show zero
deltas/clock-domain jumps, and even same-pass results became zero after forcing
real pseudo-random page materialization. The harness therefore uses wall time
for the corrected stream rate and solver queue-to-completion time, with
timestamps diagnostic-only and exact generated logical-byte models alongside
it. Literal DRAM-counter saturation is not claimed without a supported Metal
counter set.

A fixed-8-GiB sweep locates the honest contiguous plateau at 512--1024 MiB per
buffer: 315.4 and 315.7 GB/s. The prior 256 MiB default measures 334.0 GB/s and
is still cache-inflated. The earlier 1024^2 scalar repeats at 0.584 ms/step
structured and 1.411 unstructured are retracted: the harness used `dt=1e-5`
with unit diffusivity, while the 2-D RK4 limit on that mesh is about `3.32e-7`.
A full packed-state scan exposed the timed trajectory as non-finite. Stable
mesh-scaled reruns use `dt=0.1 min(dx,dy)^2`; no scalar solver-bandwidth claim
is retained until those reruns complete. The contiguous plateau remains valid
but is not, by itself, proof of solver saturation.
The acceptance contract now requires either external-memory counters at >=80%
of the 512/1024 reference, or a controlled +32/+64/+128 B/cell-stage traffic
perturbation whose time slope matches the reference within 10%.

### Round 5

#### F3e adversarial closure and conditioning chain

The first sparse-support audit found no missing structural edge after more than
one million support/swap states, but four independent numeric attacks defeated
successive weaker contracts:

1. ordered primitive aliases made a real-polynomial determinant appear nonzero
   although emitted rows were identical;
2. semantically equal opaque closures (`select(q,q,...)`, idempotent `min`/`abs`)
   bypassed syntactic canonicalization;
3. f32 absorption (`q + 2^-26*q == q`) defeated exact-real polynomial algebra,
   both through primitive closures and direct DDT accumulation; and
4. absolute and then row-relative pivot floors admitted finite local rates with
   60x and 4x normwise forward error.

Those attacks forced two additional rounds; the final mechanism is recorded in
round 6 below. The historical lesson is retained here: determinant nonzero,
absolute pivots, row-relative pivots, and products of normalized pivots are all
insufficient f32 conditioning contracts. `RuntimePivoted` remains a truthful
model-owned domain certificate, not a universal theorem for arbitrary
third-party nonlinear matrices.

#### F2d incidence-owned rematerialization block

This incompatible route evaluates the same canonical face expression inside
each cell-incidence pull and immediately accumulates
`R_i = S_i + sum sigma_if Phi_f`. It would remove 192 B/cell-stage on the
structured compressible path and `32F + 16H` bytes/stage on regular CSR, at the
cost of evaluating each CSR interior flux twice. The isolated prototype is
`examples/probe_incidence_flux_remat.rs`.

It is blocked before large-grid timing. Inlining let Metal contract/reassociate
the expression: 16,356 of 16,384 lanes initially differed from the stored-flux
reference. Explicit bitcast barriers after flux return and every accumulation
reduced that to 7 lanes, but did not establish exact f32 parity. Reopen only
with a code-generated no-contraction/materialization primitive or an approved
bounded-ULP contract, then require a repeated production win on both topologies.

### Round 6

#### F3e final equilibrated local-mass algorithm

The accepted algorithm keeps the model/IR as the source of truth and combines
three mechanically derived reductions:

1. RK base/accumulator and global RHS storage contain differential rows only;
   algebraic closure remains in full public state.
2. A Boolean mass-support graph is propagated through every possible guarded
   row swap and fill edge, so generated elimination contains only structurally
   possible work.
3. Every connected coupled mass block is solved as
   `E y = D_r^-1 b`, where `E = D_r^-1 M D_c^-1`, followed by
   `x = D_c^-1 y`. Isolated scalar blocks retain their one-division path.

Runtime-dependent connected blocks are restricted to rank two. Their raw
perfect-matching determinant quality is checked against `1e-5`, then capped at
one only for inverse-amplification accounting. The emitted solve rejects a
nonzero coefficient that becomes subnormal during row normalization and
   requires `min(D_c) * q >= 2^-19` by default; a model-owned analytic proof
   may declare a different power-of-two floor without a model-ID branch in
   codegen. A separate `2^-22` physical-scale check catches precision lost in
   the earlier RHS/volume division. Fully constant
components are audited independently, even when another disconnected component
is dynamic, with row/column-equilibrated `rcond_inf >= 1e-5`. Any failed guard
injects quiet NaN into the existing non-finite step path instead of returning a
plausible finite wrong rate.

The final adversarial suite has 13 executable constructions: f32 absorption,
opaque closure cancellation, 60x 2x2 and 4x 3x3 errors, independent connected
blocks, extreme row-factor underflow, volume-division loss, an unrelated
dynamic scalar disabling constant audit, subnormal coefficient loss, and
opposite-sign matching products with raw quality above one. An exact-emulation
3,000,000-case sweep measured worst accepted normwise error 10.628%, p99
0.680%, and p99.9 2.606% at the generic `2^-19` floor; no catastrophic finite
survivor remained. The analytically constrained all-Mach block uses a
model-owned `2^-22` floor: its EOS proof gives `q >= 1/gamma = 5/7` for positive
density and capped `q=1` for opposite-sign matching products. This avoids
weakening the generic contract while admitting the evolved 2,000-step nozzle
trajectory. Initial margins over the all-Mach floor are 28.44x nozzle, 77.4x
Mercury, 104.9x for the live high-speed update, and over 850,000x for default
Air.

This is deliberately a finite-f32 representability contract, not arbitrary
unit invariance. It conservatively rejects otherwise well-conditioned blocks
whose row-normalized coefficients are subnormal or whose column scaling falls
below their model-owned floor. The one-quantum argument assumes retained subnormals; WebGPU
backends that flush denormals lie outside that extreme-scale certificate. All
shipped GUI states are far from the denormal boundary.

#### Final bandwidth evidence and honesty boundary

Nine interleaved 1024² pairs with complete post-timing state validation give:

| Workload | Preserved baseline | F3b+F3e | Change |
|---|---:|---:|---:|
| QUICK structured | 24.071 ms | 18.706 ms | -22.29% |
| QUICK unstructured | 24.972 ms | 18.336 ms | -26.63% |
| Upwind structured | 20.558 ms | 19.444 ms | -5.53% |
| Upwind unstructured | 19.976 ms | 18.407 ms | -7.99% |
| Stable scalar structured/unstructured | 0.541 / 1.378 ms | 0.540 / 1.378 ms | neutral |
| All-Mach structured/unstructured | 20.694 / 21.492 ms | 20.740 / 21.479 ms | neutral |

After the final equilibrated all-Mach guards, a fresh nine-pair re-audit measured
20.608 -> 20.704 ms structured (+0.42% cost) and 21.474 -> 21.516 ms CSR
(+0.18%); the worst pair was below 1.6%, full packed state stayed finite, and
storage traffic/dispatch counts were unchanged. A five-pair QUICK confirmation
retained 21.93% structured and 26.48% CSR speedups.

QUICK dispatches fall from 28 to 24 per step. Requested-word traffic falls
22.7% structured and 27.5% on regular CSR; the removed packed-gradient pass is
provably dead because its consumer contains zero `grad_state` reads. The CSR
requested-traffic model is about 308 GB/s versus the measured 315--316 GB/s
large-buffer stream plateau, but this is not literal DRAM-counter proof: cache
reuse makes the same logical model overstate scalar bandwidth, and Metal exposes
no supported external-memory counter here. Therefore the accepted claim is a
measured 22--27% high-order solver speedup with near-stream requested traffic,
not “100% physical DRAM saturation.”

### Round 7

#### F3f semantic face-channel liveness

An independent audit found that the density-compressible systems retain eight
semantic coupled ranks (twelve for the biharmonic MMS variant), while only the
four conserved equations are ever produced into or consumed from the face
buffer. The new router derives an exact coupled-rank-to-storage-rank map from
the model graph. Boundary expressions and equations continue to see the full
semantic ordering; face allocation, producer stores, and all structured/CSR,
explicit/implicit, RHS-only, gradient-state, and fused consumers use the same
compact projection. There is no model-ID branch.

The shipped compressible base, MMS, and structured layouts compact from stride
eight to four; the biharmonic MMS layout compacts from twelve to four. Every
other model, including all ALE layouts, retains an identity projection. The
mechanical artifact audit finds exactly four generated stores (ranks zero
through three), no residual stride-eight/twelve access, and matching compact
allocation in every stepping recipe.

The exact removed traffic is sixteen bytes per face per stage, or sixty-four
bytes per face per RK4 step. At 1024² this removes about 256 MiB of structured
face stores and 64 MiB of face-buffer capacity per step; a regular CSR mesh
removes about 128 MiB of stores and 32 MiB of capacity. These are logical-byte
results, not a physical-DRAM saturation claim. F3f remains a candidate until
the full MMS/ALE/generated-shader gates and a stable, completion-fenced
end-to-end timing comparison are green.
