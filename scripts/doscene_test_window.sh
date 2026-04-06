#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OMNIDRIVE_DIR="${REPO_ROOT}/omnidrive"

# ===== Defaults (can be overridden by env) =====
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
CONDA_ENV="${CONDA_ENV:-omnidrive}"

CONFIG="${CONFIG:-projects/configs/OmniDrive/mask_eva_lane_det_vlm_doscenes_only.py}"
CKPT="${CKPT:-work_dirs/baseline/latest.pth}"
OBS_LEN="${OBS_LEN:-4}"
FUT_LEN="${FUT_LEN:-12}"
WINDOW_STRIDE="${WINDOW_STRIDE:-1}"
NO_INSTRUCTION="${NO_INSTRUCTION:-1}"   # 1 => add --no-instruction

EXP_ROOT="${EXP_ROOT:-challenge}"
EXP_SUFFIX="${EXP_SUFFIX:-no_lang}"     # output folder will be expXX_${EXP_SUFFIX}

# Optional:
# - If EXP_ID is set, use it directly (e.g., EXP_ID=7 -> exp07_no_lang)
# - If EXP_ID is unset, auto pick the next id by scanning EXP_ROOT
EXP_ID="${EXP_ID:-}"

mkdir -p "${OMNIDRIVE_DIR}/${EXP_ROOT}"

if [[ -z "${EXP_ID}" ]]; then
  max_id=0
  while IFS= read -r d; do
    name="$(basename "$d")"
    if [[ "${name}" =~ ^exp([0-9]+)_${EXP_SUFFIX}$ ]]; then
      n="${BASH_REMATCH[1]}"
      n=$((10#${n}))
      if (( n > max_id )); then
        max_id=$n
      fi
    fi
  done < <(find "${OMNIDRIVE_DIR}/${EXP_ROOT}" -mindepth 1 -maxdepth 1 -type d)
  EXP_ID=$((max_id + 1))
fi

EXP_NAME="$(printf "exp%02d_%s" "${EXP_ID}" "${EXP_SUFFIX}")"
SAVE_DIR="${EXP_ROOT}/${EXP_NAME}/preds"
INST_JSON_DIR="${EXP_ROOT}/${EXP_NAME}/instruction_jsons"
METRICS_JSON="${EXP_ROOT}/${EXP_NAME}/metrics.json"

mkdir -p "${OMNIDRIVE_DIR}/${SAVE_DIR}" "${OMNIDRIVE_DIR}/${INST_JSON_DIR}"

# Infer nproc from CUDA_VISIBLE_DEVICES unless overridden.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
fi

echo "EXP_NAME=${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CONFIG=${CONFIG}"
echo "CKPT=${CKPT}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "INST_JSON_DIR=${INST_JSON_DIR}"
echo "METRICS_JSON=${METRICS_JSON}"

cd "${OMNIDRIVE_DIR}"

cmd=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  python -u -m torch.distributed.run
  --nproc_per_node="${NPROC_PER_NODE}"
  tools/test_doscenes_sliding_window_eval.py
  "${CONFIG}"
  "${CKPT}"
  --launcher pytorch
  --obs-len "${OBS_LEN}"
  --fut-len "${FUT_LEN}"
  --window-stride "${WINDOW_STRIDE}"
  --save-dir "${SAVE_DIR}"
  --per-instruction-json-dir "${INST_JSON_DIR}"
  --metrics-json "${METRICS_JSON}"
)

if [[ "${NO_INSTRUCTION}" == "1" ]]; then
  cmd+=(--no-instruction)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" "${cmd[@]}"


# CUDA_VISIBLE_DEVICES=4,5,6,7 EXP_ID=8 bash test/doscene_test_window.sh
