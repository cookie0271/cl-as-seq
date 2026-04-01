#!/usr/bin/env bash
set -euo pipefail

MODEL_CONFIG=${MODEL_CONFIG:-cfg/model/omniglot_tf_pure_baseline.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/omniglot.yaml}
RUN_ROOT=${RUN_ROOT:-runs/omniglot_tf_pure_baseline}
SEEDS=${SEEDS:-"0 1 2"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

usage() {
  cat <<USAGE
Usage:
  $0 smoke [seed]
  $0 train [seed]
  $0 resume [seed] [target_steps]
  $0 train-multiseed
  $0 eval <log_dir>

Examples:
  $0 smoke 0
  $0 train 0
  $0 resume 0 50000
  $0 train-multiseed
  $0 eval runs/omniglot_tf_pure_baseline/seed0
USAGE
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

MODE=$1
SEED=${2:-0}

common_overrides() {
  local seed="$1"
  echo "tasks=20|train_shots=5|test_shots=1|seed=${seed}|enable_correction=False|enable_highway_gate=False|freeze_backbone=False|train_correction=False|train_gate=False"
}

case "${MODE}" in
  smoke)
    LOG_DIR="${RUN_ROOT}/smoke_seed${SEED}"
    python train.py \
      -mc "${MODEL_CONFIG}" \
      -dc "${DATA_CONFIG}" \
      -l "${LOG_DIR}" \
      -o "$(common_overrides "${SEED}")|max_train_steps=500|eval_interval=250|summary_interval=100|ckpt_interval=500|eval_iters=32|batch_size=256|eval_batch_size=32|num_workers=8"
    ;;

  train)
    LOG_DIR="${RUN_ROOT}/seed${SEED}"
    python train.py \
      -mc "${MODEL_CONFIG}" \
      -dc "${DATA_CONFIG}" \
      -l "${LOG_DIR}" \
      -o "$(common_overrides "${SEED}")|max_train_steps=50000|eval_interval=1000|summary_interval=250|ckpt_interval=2500|eval_iters=32|batch_size=256|eval_batch_size=64"
    ;;

  resume)
    TARGET_STEPS=${3:-50000}
    LOG_DIR="${RUN_ROOT}/seed${SEED}"
    python train.py \
      --resume \
      -mc "${MODEL_CONFIG}" \
      -dc "${DATA_CONFIG}" \
      -l "${LOG_DIR}" \
      -o "$(common_overrides "${SEED}")|max_train_steps=${TARGET_STEPS}|eval_interval=1000|summary_interval=250|ckpt_interval=2500|eval_iters=64|batch_size=256|eval_batch_size=32"
    ;;

  train-multiseed)
    for seed in ${SEEDS}; do
      "${0}" train "${seed}"
    done
    ;;

  eval)
    if [ $# -lt 2 ]; then
      echo "eval mode requires log_dir as second argument"
      exit 1
    fi
    LOG_DIR=$2
    python measure_forgetting.py \
      -mc "${MODEL_CONFIG}" \
      -dc "${DATA_CONFIG}" \
      -l "${LOG_DIR}" \
      -o "tasks=20|train_shots=5|test_shots=1|max_train_steps=50000|eval_iters=32|eval_batch_size=64|batch_size=256|num_workers=8|enable_correction=False|enable_highway_gate=False|freeze_backbone=False|train_correction=False|train_gate=False"
    ;;

  *)
    usage
    exit 1
    ;;
esac