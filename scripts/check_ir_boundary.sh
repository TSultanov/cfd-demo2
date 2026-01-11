#!/usr/bin/env bash
# Guardrail: codegen must not import model backend types directly.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

CODEGEN_DIR="$ROOT_DIR/src/solver/codegen"

if [ ! -d "$CODEGEN_DIR" ]; then
  echo "ERROR: missing codegen dir: $CODEGEN_DIR" >&2
  exit 1
fi

# Enforced invariant for the incremental IR boundary step:
# - codegen may depend on the IR facade (`crate::solver::ir::*`)
# - codegen must not reach into `crate::solver::model::backend::*` directly

if grep -R --line-number --fixed-string "crate::solver::model::backend" "$CODEGEN_DIR" --include='*.rs' >/dev/null; then
  echo "ERROR: IR boundary violation: codegen imports model backend directly." >&2
  echo "Replace with the IR facade (e.g. crate::solver::ir::{...})." >&2
  echo "" >&2
  grep -R --line-number --fixed-string "crate::solver::model::backend" "$CODEGEN_DIR" --include='*.rs' >&2
  exit 1
fi

echo "✓ IR boundary check passed (no model backend imports in codegen)"

