#!/usr/bin/env bash
# Guardrail: codegen must not import model backend types directly.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

CODEGEN_DIR="$ROOT_DIR/src/solver/codegen"

if [ ! -d "$CODEGEN_DIR" ]; then
  echo "ERROR: missing codegen dir: $CODEGEN_DIR" >&2
  exit 1
fi

# Enforced invariants for the incremental IR boundary step:
# - codegen may depend on the IR facade (`crate::solver::ir::*`)
# - (most) codegen must not depend on `crate::solver::model::*`
#   (legacy EI emission is temporarily exempt)
# - codegen must not reach into `crate::solver::model::backend::*` directly

if grep -R --line-number --fixed-string "crate::solver::model::backend" "$CODEGEN_DIR" --include='*.rs' >/dev/null; then
  echo "ERROR: IR boundary violation: codegen imports model backend directly." >&2
  echo "Replace with the IR facade (e.g. crate::solver::ir::{...})." >&2
  echo "" >&2
  grep -R --line-number --fixed-string "crate::solver::model::backend" "$CODEGEN_DIR" --include='*.rs' >&2
  exit 1
fi

echo "✓ IR boundary check passed (no model backend imports in codegen)"

# Stronger guard: forbid `crate::solver::model::*` in codegen outside the temporary EI bridge.
#
# This is intentionally coarse-grained (a simple grep) to avoid pulling in tooling.
MODEL_IMPORTS=$(grep -R --line-number --fixed-string "crate::solver::model::" "$CODEGEN_DIR" --include='*.rs' || true)
if [ -n "$MODEL_IMPORTS" ]; then
  # Allowlist the transitional files that still depend on model data.
  FILTERED=$(echo "$MODEL_IMPORTS" | grep -v "/codegen/ei/" | grep -v "/codegen/method_ei.rs" || true)
  if [ -n "$FILTERED" ]; then
    echo "ERROR: IR boundary violation: codegen imports model types outside the EI bridge." >&2
    echo "Move model-dependent orchestration out of codegen, or plumb data via solver::ir." >&2
    echo "" >&2
    echo "$FILTERED" >&2
    exit 1
  fi
fi

echo "✓ IR boundary check passed (no model imports outside EI bridge)"
