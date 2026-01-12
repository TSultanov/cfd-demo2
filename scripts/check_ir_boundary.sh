#!/usr/bin/env bash
# Guardrail: codegen must not import model backend types directly.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

CODEGEN_DIR="$ROOT_DIR/crates/cfd2_codegen/src/solver/codegen"

if [ ! -d "$CODEGEN_DIR" ]; then
  echo "ERROR: missing codegen dir: $CODEGEN_DIR" >&2
  exit 1
fi

# Enforced invariants:
# - codegen may depend on the IR facade (`crate::solver::ir::*`)
# - codegen must not depend on `crate::solver::model::*` (KT migration completed)
# - codegen must not reach into `crate::solver::model::backend::*` directly

if grep -R --line-number --fixed-string "crate::solver::model::backend" "$CODEGEN_DIR" --include='*.rs' >/dev/null; then
  echo "ERROR: IR boundary violation: codegen imports model backend directly." >&2
  echo "Replace with the IR facade (e.g. crate::solver::ir::{...})." >&2
  echo "" >&2
  grep -R --line-number --fixed-string "crate::solver::model::backend" "$CODEGEN_DIR" --include='*.rs' >&2
  exit 1
fi

echo "✓ IR boundary check passed (no model backend imports in codegen)"

# Stronger guard: forbid model references in codegen entirely.
#
# NOTE: This is intentionally coarse-grained (a simple grep) to keep it portable.
MODEL_REFS=$(grep -R --line-number -E "(crate::solver::model\b|crate::solver::\{[[:space:]]*model\b|super::model\b|super::\{[[:space:]]*model\b|::model(::|[[:space:]]*(as|[;},)])))" "$CODEGEN_DIR" --include='*.rs' || true)
if [ -n "$MODEL_REFS" ]; then
  echo "ERROR: IR boundary violation: codegen references model types." >&2
  echo "Use the IR facade (crate::solver::ir::{...}) or move orchestration to compiler." >&2
  echo "" >&2
  echo "$MODEL_REFS" >&2
  exit 1
fi

echo "✓ IR boundary check passed (no model references in codegen)"
