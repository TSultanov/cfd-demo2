#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${ROOT_DIR}/target/openfoam_reference_runs"
OUT_DIR="${ROOT_DIR}/tests/openfoam_reference/data"

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}" "${OUT_DIR}"

run_openfoam() {
  # Use OpenFOAM's environment wrapper so tools are available.
  local cmd="$1"
  openfoam -c "${cmd}"
}

echo "==> OpenFOAM version: $(openfoam -show-api)"

echo "==> Incompressible channel (simpleFoam)"
cp -R "${ROOT_DIR}/reference/openfoam/incompressible_channel" "${WORK_DIR}/"
run_openfoam "cd '${WORK_DIR}/incompressible_channel' && blockMesh > log.blockMesh && simpleFoam > log.simpleFoam"
python3 "${ROOT_DIR}/scripts/openfoam/extract_probes_to_csv.py" \
  --case "${WORK_DIR}/incompressible_channel" \
  --probes-name probes \
  --mode incompressible_channel \
  --out "${OUT_DIR}/incompressible_channel_centerline.csv"
python3 "${ROOT_DIR}/scripts/openfoam/extract_probes_to_csv.py" \
  --case "${WORK_DIR}/incompressible_channel" \
  --probes-name probesAll \
  --mode incompressible_channel \
  --out "${OUT_DIR}/incompressible_channel_full_field.csv"

echo "==> Compressible acoustic box (rhoCentralFoam + setExprFields)"
cp -R "${ROOT_DIR}/reference/openfoam/compressible_acoustic_box" "${WORK_DIR}/"
run_openfoam "cd '${WORK_DIR}/compressible_acoustic_box' && blockMesh > log.blockMesh && setExprFields -dict system/setExprFieldsDict > log.setExprFields && rhoCentralFoam > log.rhoCentralFoam"
python3 "${ROOT_DIR}/scripts/openfoam/extract_probes_to_csv.py" \
  --case "${WORK_DIR}/compressible_acoustic_box" \
  --probes-name probes \
  --mode compressible_acoustic \
  --out "${OUT_DIR}/compressible_acoustic_centerline.csv"
python3 "${ROOT_DIR}/scripts/openfoam/extract_probes_to_csv.py" \
  --case "${WORK_DIR}/compressible_acoustic_box" \
  --probes-name probesAll \
  --mode compressible_acoustic \
  --out "${OUT_DIR}/compressible_acoustic_full_field.csv"

echo "==> Wrote:"
echo "    ${OUT_DIR}/incompressible_channel_centerline.csv"
echo "    ${OUT_DIR}/incompressible_channel_full_field.csv"
echo "    ${OUT_DIR}/compressible_acoustic_centerline.csv"
echo "    ${OUT_DIR}/compressible_acoustic_full_field.csv"
