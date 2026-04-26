#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
INPUT_DIR="${ROOT_DIR}/data/input"
OUTPUT_DIR="${ROOT_DIR}/data/output"
PROOF_DIR="${ROOT_DIR}/proof/latest"

mkdir -p "${PROOF_DIR}"
rm -rf "${INPUT_DIR}" "${OUTPUT_DIR}" "${PROOF_DIR}/outputs"
mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}" "${PROOF_DIR}/outputs"

python3 "${ROOT_DIR}/scripts/generate_dataset.py" \
  --output "${INPUT_DIR}" \
  --count 256 \
  --width 256 \
  --height 256 \
  --seed 7 | tee "${PROOF_DIR}/dataset.log"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j

"${BUILD_DIR}/cuda_batch_edges" \
  --input "${INPUT_DIR}" \
  --output "${OUTPUT_DIR}" \
  --threshold 88 \
  --warmup 1 | tee "${PROOF_DIR}/run.log"

cp "${OUTPUT_DIR}/summary.csv" "${PROOF_DIR}/summary.csv"
find "${OUTPUT_DIR}" -maxdepth 1 -name 'synthetic_*.pgm' | sort | head -n 12 | while read -r image; do
  cp "${image}" "${PROOF_DIR}/outputs/"
done
python3 "${ROOT_DIR}/scripts/make_contact_sheet.py" \
  --input "${PROOF_DIR}/outputs" \
  --output "${PROOF_DIR}/contact_sheet.png" \
  --limit 12

echo "Proof artifacts written to ${PROOF_DIR}"
