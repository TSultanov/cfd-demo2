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

gen_probes_all() {
  local out="$1"
  shift
  python3 "${ROOT_DIR}/scripts/openfoam/generate_probes_all_cfg.py" --out "${out}" "$@"
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

echo "==> Incompressible backwards step (simpleFoam)"
cp -R "${ROOT_DIR}/reference/openfoam/incompressible_backwards_step" "${WORK_DIR}/"
gen_probes_all "${WORK_DIR}/incompressible_backwards_step/system/probes_all.cfg" \
  --write-interval 1 \
  --fields U,p \
  backwards-step --nx 30 --ny 10 --length 3 --height-outlet 1 --height-inlet 0.5 --step-x 1
run_openfoam "cd '${WORK_DIR}/incompressible_backwards_step' && blockMesh > log.blockMesh && simpleFoam > log.simpleFoam"
python3 "${ROOT_DIR}/scripts/openfoam/extract_probes_to_csv.py" \
  --case "${WORK_DIR}/incompressible_backwards_step" \
  --probes-name probesAll \
  --mode incompressible_channel \
  --out "${OUT_DIR}/incompressible_backwards_step_full_field.csv"

echo "==> Incompressible lid-driven cavity (simpleFoam)"
cp -R "${ROOT_DIR}/reference/openfoam/incompressible_lid_driven_cavity" "${WORK_DIR}/"
gen_probes_all "${WORK_DIR}/incompressible_lid_driven_cavity/system/probes_all.cfg" \
  --write-interval 1 \
  --fields U,p \
  rect --nx 20 --ny 20 --length 1 --height 1
run_openfoam "cd '${WORK_DIR}/incompressible_lid_driven_cavity' && blockMesh > log.blockMesh && simpleFoam > log.simpleFoam"
python3 "${ROOT_DIR}/scripts/openfoam/extract_probes_to_csv.py" \
  --case "${WORK_DIR}/incompressible_lid_driven_cavity" \
  --probes-name probesAll \
  --mode incompressible_channel \
  --out "${OUT_DIR}/incompressible_lid_driven_cavity_full_field.csv"

echo "==> Compressible backwards step (rhoCentralFoam)"
cp -R "${ROOT_DIR}/reference/openfoam/compressible_backwards_step" "${WORK_DIR}/"
gen_probes_all "${WORK_DIR}/compressible_backwards_step/system/probes_all.cfg" \
  --write-interval 200 \
  --fields U,p \
  backwards-step --nx 30 --ny 10 --length 3 --height-outlet 1 --height-inlet 0.5 --step-x 1
run_openfoam "cd '${WORK_DIR}/compressible_backwards_step' && blockMesh > log.blockMesh && rhoCentralFoam > log.rhoCentralFoam"
python3 "${ROOT_DIR}/scripts/openfoam/extract_probes_to_csv.py" \
  --case "${WORK_DIR}/compressible_backwards_step" \
  --probes-name probesAll \
  --mode compressible_basic \
  --out "${OUT_DIR}/compressible_backwards_step_full_field.csv"

echo "==> Compressible lid-driven cavity (rhoCentralFoam)"
cp -R "${ROOT_DIR}/reference/openfoam/compressible_lid_driven_cavity" "${WORK_DIR}/"
gen_probes_all "${WORK_DIR}/compressible_lid_driven_cavity/system/probes_all.cfg" \
  --write-interval 100 \
  --fields U,p \
  rect --nx 20 --ny 20 --length 1 --height 1
run_openfoam "cd '${WORK_DIR}/compressible_lid_driven_cavity' && blockMesh > log.blockMesh && rhoCentralFoam > log.rhoCentralFoam"
python3 "${ROOT_DIR}/scripts/openfoam/extract_probes_to_csv.py" \
  --case "${WORK_DIR}/compressible_lid_driven_cavity" \
  --probes-name probesAll \
  --mode compressible_basic \
  --out "${OUT_DIR}/compressible_lid_driven_cavity_full_field.csv"

echo "==> Compressible supersonic wedge (rhoCentralFoam)"
cp -R "${ROOT_DIR}/reference/openfoam/compressible_supersonic_wedge" "${WORK_DIR}/"
gen_probes_all "${WORK_DIR}/compressible_supersonic_wedge/system/probes_all.cfg" \
  --write-interval 200 \
  --fields U,p \
  trapezoid --nx 30 --ny 15 --length 1 --height 0.6 --ramp-height 0.2
run_openfoam "cd '${WORK_DIR}/compressible_supersonic_wedge' && blockMesh > log.blockMesh && rhoCentralFoam > log.rhoCentralFoam"
python3 "${ROOT_DIR}/scripts/openfoam/extract_probes_to_csv.py" \
  --case "${WORK_DIR}/compressible_supersonic_wedge" \
  --probes-name probesAll \
  --mode compressible_basic \
  --out "${OUT_DIR}/compressible_supersonic_wedge_full_field.csv"

echo "==> Wrote:"
echo "    ${OUT_DIR}/incompressible_channel_centerline.csv"
echo "    ${OUT_DIR}/incompressible_channel_full_field.csv"
echo "    ${OUT_DIR}/compressible_acoustic_centerline.csv"
echo "    ${OUT_DIR}/compressible_acoustic_full_field.csv"
echo "    ${OUT_DIR}/incompressible_backwards_step_full_field.csv"
echo "    ${OUT_DIR}/incompressible_lid_driven_cavity_full_field.csv"
echo "    ${OUT_DIR}/compressible_backwards_step_full_field.csv"
echo "    ${OUT_DIR}/compressible_lid_driven_cavity_full_field.csv"
echo "    ${OUT_DIR}/compressible_supersonic_wedge_full_field.csv"
