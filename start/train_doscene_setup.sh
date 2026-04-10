#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -z "${GPUS:-}" ]]; then
  GPUS="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
fi
NUM_GPUS="${NUM_GPUS:-${GPUS}}"

CONFIG="${CONFIG:-projects/configs/OmniDrive/mask_eva_lane_det_vlm_train_doscenes_only.py}"
EXP_ROOT="${EXP_ROOT:-log_train}"
EXP_NAME="${EXP_NAME:-exp01}"
WORK_DIR="${WORK_DIR:-${EXP_ROOT}/${EXP_NAME}}"
ENABLE_DOSCENES="${ENABLE_DOSCENES:-1}"
RANDOM_DOSCENES="${RANDOM_DOSCENES:-1}"
ONLY_DOSCENES_SAMPLES="${ONLY_DOSCENES_SAMPLES:-0}"
DCSV="${DCSV:-data/annotated_doscenes.csv}"

echo "CONFIG=${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "GPUS=${GPUS}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "EXP_ROOT=${EXP_ROOT}"
echo "EXP_NAME=${EXP_NAME}"
echo "WORK_DIR=${WORK_DIR}"
echo "ENABLE_DOSCENES=${ENABLE_DOSCENES}"
echo "RANDOM_DOSCENES=${RANDOM_DOSCENES}"
echo "ONLY_DOSCENES_SAMPLES=${ONLY_DOSCENES_SAMPLES}"
echo "DCSV=${DCSV}"

CONFIG="${CONFIG}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
GPUS="${GPUS}" \
NUM_GPUS="${NUM_GPUS}" \
EXP_ROOT="${EXP_ROOT}" \
EXP_NAME="${EXP_NAME}" \
WORK_DIR="${WORK_DIR}" \
ENABLE_DOSCENES="${ENABLE_DOSCENES}" \
RANDOM_DOSCENES="${RANDOM_DOSCENES}" \
ONLY_DOSCENES_SAMPLES="${ONLY_DOSCENES_SAMPLES}" \
DCSV="${DCSV}" \
bash tools/doscene_train.sh


# GPUS=4 EXP_NAME=exp01 bash start/train_doscene_setup.sh