#!/bin/sh
# The moving-mesh/ALE validation battery, PARALLELIZED.
#
# - Test BINARIES run as concurrent processes (xargs -P): the CPU-backend
#   tests serialize on a process-local ENV_LOCK (CFD2_BACKEND is
#   process-global), so cross-binary parallelism is the safe axis — each
#   process owns its own env.
# - The SOLVER inside each test runs multi-threaded via CFD2_CPU_THREADS
#   (value-deterministic: kernels write disjoint cells/faces, reductions are
#   deterministic — thread count changes speed, not results).
# - moving_mesh_gui_test runs ALONE at the end: its worker smokes are
#   wall-clock-budgeted (expect N mesh refreshes within a deadline) and go
#   flaky under CPU contention.
#
# Sizing: BINS concurrent binaries x THREADS solver threads. CAPPED AT 6
# CORES TOTAL by default (user preference: more spins the machine fan up).
# Override via env: BINS=4 THREADS=4 scripts/moving_battery.sh
set -u
cd "$(dirname "$0")/.."
BINS="${BINS:-3}"
THREADS="${THREADS:-2}"
OUT="target/parbat"
mkdir -p "$OUT"
rm -f "$OUT"/*.log

cargo build --release --features ui --tests || exit 1

printf '%s\n' \
    moving_mesh_adapt_test allmach_ale_test gpu_moving_gcl_test \
    moving_mesh_gcl_test moving_mesh_flip_gcl_test gpu_moving_mesh_test \
    moving_boundary_test moving_mesh_flow_test moving_mesh_loop_test \
    ale_gcl_test moving_mesh_stress_test moving_mesh_adaptive_dt_test \
    | xargs -P "$BINS" -I{} sh -c \
        "CFD2_CPU_THREADS=$THREADS cargo test --release --features ui --test {} > $OUT/{}.log 2>&1; echo \"{} exit: \$?\""

CFD2_CPU_THREADS="$THREADS" cargo test --release --features ui \
    --test moving_mesh_gui_test > "$OUT/moving_mesh_gui_test.log" 2>&1
echo "moving_mesh_gui_test exit: $?"

echo "---"
grep -H "test result:" "$OUT"/*.log
! grep -lE "FAILED|panicked" "$OUT"/*.log
