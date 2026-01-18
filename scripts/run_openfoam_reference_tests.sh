#!/usr/bin/env bash
set -euo pipefail

# OpenFOAM reference tests are our main solver behavior gate.
#
# Why this wrapper exists:
# - `cargo test openfoam_` / `cargo test --tests openfoam_` can look like a false-green because
#   Cargo still runs many unrelated test binaries with "0 tests" due to filtering.
# - This script runs *only* the OpenFOAM integration test binaries and fails loudly if they are missing.

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${ROOT_DIR}"

test_targets=()
while IFS= read -r t; do
  test_targets+=("${t}")
done < <(
  ls -1 tests/openfoam_*_reference_test.rs 2>/dev/null \
    | xargs -n 1 basename \
    | sed 's/\.rs$//' \
    | sort
)

if [[ "${#test_targets[@]}" -eq 0 ]]; then
  echo "ERROR: No OpenFOAM reference test targets found under tests/openfoam_*_reference_test.rs" >&2
  exit 1
fi

cmd=(cargo test -p cfd2)
for t in "${test_targets[@]}"; do
  cmd+=(--test "${t}")
done

# Sanity-check we have at least one OpenFOAM test name to execute.
list_out="$("${cmd[@]}" -- --list)"
num_openfoam_tests="$(printf '%s\n' "${list_out}" | rg -c '^openfoam_.*: test$' || true)"
if [[ "${num_openfoam_tests}" -eq 0 ]]; then
  echo "ERROR: OpenFOAM reference test gate did not find any tests to run." >&2
  echo "" >&2
  echo "Debug helpers:" >&2
  echo "  cargo test -p cfd2 --test ${test_targets[0]} -- --list" >&2
  exit 1
fi

echo "==> OpenFOAM reference test binaries: ${#test_targets[@]}" 
echo "==> OpenFOAM reference tests discovered: ${num_openfoam_tests}" 
echo "==> Running: ${cmd[*]} -- --nocapture $*"

"${cmd[@]}" -- --nocapture "$@"
