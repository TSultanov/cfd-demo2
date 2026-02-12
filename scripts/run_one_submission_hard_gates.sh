#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <baseline_openfoam_metrics_file>" >&2
  exit 2
fi

BASELINE_METRICS="$1"
if [[ ! -f "${BASELINE_METRICS}" ]]; then
  echo "baseline metrics file not found: ${BASELINE_METRICS}" >&2
  exit 2
fi

LOG_DIR="${ROOT_DIR}/target/openfoam_reference_logs"
mkdir -p "${LOG_DIR}"

PARITY_LOG="${LOG_DIR}/hard_gate_parity.log"
AFTER_LOG="${LOG_DIR}/hard_gate_after.log"
AFTER_METRICS="${LOG_DIR}/hard_gate_after.metrics"

echo "==> Hard gate: numerical parity + dispatch/submission counters"
cargo test -p cfd2 --test rhie_chow_fusion_parity_test -- --nocapture \
  2>&1 | tee "${PARITY_LOG}"

ONE_SUBMISSION_LOG="${LOG_DIR}/hard_gate_one_submission.log"
FGMRES_PARITY_LOG="${LOG_DIR}/hard_gate_fgmres_parity.log"

echo "==> Hard gate: one-submission path parity + submission floor"
CFD2_ENABLE_FULL_ONE_SUBMISSION_OUTER=1 \
  cargo test -p cfd2 --test rhie_chow_fusion_parity_test \
  one_submission -- --nocapture \
  2>&1 | tee "${ONE_SUBMISSION_LOG}"

echo "==> Hard gate: encoded vs host FGMRES element-wise parity"
cargo test -p cfd2 --test fgmres_encoded_vs_host_parity_test \
  -- --nocapture \
  2>&1 | tee "${FGMRES_PARITY_LOG}"

echo "==> Hard gate: OpenFOAM diagnostics (post-change snapshot)"
CFD2_OPENFOAM_DIAG=1 bash scripts/run_openfoam_reference_tests.sh \
  2>&1 | tee "${AFTER_LOG}" || true
rg '^\[openfoam\]' "${AFTER_LOG}" > "${AFTER_METRICS}"

echo "==> Hard gate: OpenFOAM drift comparison"
if diff -u "${BASELINE_METRICS}" "${AFTER_METRICS}"; then
  echo "OpenFOAM drift gate: PASS (no regression in extracted diagnostics)"
else
  echo "OpenFOAM drift gate: FAIL (diagnostics changed vs baseline)" >&2
  exit 1
fi
