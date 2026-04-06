#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OMNIDRIVE_DIR="${REPO_ROOT}/omnidrive"

# ===== Defaults (can be overridden by env) =====
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
CONDA_ENV="${CONDA_ENV:-omnidrive}"

CONFIG="${CONFIG:-projects/configs/OmniDrive/mask_eva_lane_det_vlm.py}"
CKPT="${CKPT:-work_dirs/baseline/latest.pth}"
DOSCENES_CSV="${DOSCENES_CSV:-/home/fzj/data_2/challenge/doScenes-VLM-Planning/data/doScenes/annotated_doscenes.csv}"

FRAME_STRIDE="${FRAME_STRIDE:-1}"
HISTORY_SECONDS="${HISTORY_SECONDS:-2.0}"
HISTORY_FRAMES="${HISTORY_FRAMES:-0}"  # >0 will override HISTORY_SECONDS
DROP_TAIL_FRAMES="${DROP_TAIL_FRAMES:-12}"  # drop last N frames (set 0 to disable)
NO_INSTRUCTION="${NO_INSTRUCTION:-0}"  # 1 => add --no-instruction
RECORD_WARMUP_OUTPUTS="${RECORD_WARMUP_OUTPUTS:-0}"  # 1 => add --record-warmup-outputs

EXP_ROOT="${EXP_ROOT:-work_dirs/full_scene_eval}"
EXP_SUFFIX="${EXP_SUFFIX:-full_scene}"

# Optional:
# - If EXP_ID is set, use it directly (e.g., EXP_ID=7 -> exp07_full_scene)
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
INDEX_JSON="${EXP_ROOT}/${EXP_NAME}/index.json"
mkdir -p "${OMNIDRIVE_DIR}/${SAVE_DIR}"

# Infer nproc from CUDA_VISIBLE_DEVICES unless overridden.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
fi

echo "EXP_NAME=${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CONFIG=${CONFIG}"
echo "CKPT=${CKPT}"
echo "DOSCENES_CSV=${DOSCENES_CSV}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "INDEX_JSON=${INDEX_JSON}"
echo "FRAME_STRIDE=${FRAME_STRIDE}"
echo "HISTORY_SECONDS=${HISTORY_SECONDS}"
echo "HISTORY_FRAMES=${HISTORY_FRAMES}"
echo "DROP_TAIL_FRAMES=${DROP_TAIL_FRAMES}"

cd "${OMNIDRIVE_DIR}"

cmd=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  python -u -m torch.distributed.run
  --nproc_per_node="${NPROC_PER_NODE}"
  tools/test_doscenes_full_scene_eval.py
  "${CONFIG}"
  "${CKPT}"
  --launcher pytorch
  --doscenes-csv "${DOSCENES_CSV}"
  --frame-stride "${FRAME_STRIDE}"
  --history-seconds "${HISTORY_SECONDS}"
  --drop-tail-frames "${DROP_TAIL_FRAMES}"
  --save-dir "${SAVE_DIR}"
  --index-json "${INDEX_JSON}"
)

if [[ "${HISTORY_FRAMES}" != "0" ]]; then
  cmd+=(--history-frames "${HISTORY_FRAMES}")
fi

if [[ "${NO_INSTRUCTION}" == "1" ]]; then
  cmd+=(--no-instruction)
fi

if [[ "${RECORD_WARMUP_OUTPUTS}" == "1" ]]; then
  cmd+=(--record-warmup-outputs)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" "${cmd[@]}"

# DROP_TAIL_FRAMES=12 CUDA_VISIBLE_DEVICES=4,5,6,7 bash test/doscene_test_full_sc
# ene.sh
