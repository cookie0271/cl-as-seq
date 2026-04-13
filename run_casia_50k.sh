#!/usr/bin/env bash
set -euo pipefail

# CASIA comparison for 6 partial-freeze algorithms (50k by default):
# 1) head_only
# 2) head_last1
# 3) head_last2
# 4) head_last2_correction
# 5) head_last2_gate
# 6) head_last2_gate_correction

TF_MODEL_CONFIG=${TF_MODEL_CONFIG:-cfg/model/tf.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/casia.yaml}
RUN_ROOT=${RUN_ROOT:-runs/casia_partial6_50k}
RESULTS_CSV=${RESULTS_CSV:-results/casia_partial6_50k.csv}
SEEDS=${SEEDS:-"0 1 2"}
TARGET_STEPS=${TARGET_STEPS:-50000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES
CUDA_DEVICE_FIRST3=${CUDA_DEVICE_FIRST3:-0}
CUDA_DEVICE_LAST3=${CUDA_DEVICE_LAST3:-1}

# Performance defaults (override via env if needed)
TASKS=${TASKS:-20}
TRAIN_SHOTS=${TRAIN_SHOTS:-5}
TEST_SHOTS=${TEST_SHOTS:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
EVAL_ITERS=${EVAL_ITERS:-16}
NUM_WORKERS=${NUM_WORKERS:-8}
SUMMARY_INTERVAL=${SUMMARY_INTERVAL:-500}
EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
CKPT_INTERVAL=${CKPT_INTERVAL:-2500}
LR=${LR:-0.0001}
GATE_BIAS_INIT=${GATE_BIAS_INIT:--1.0}
GATE_DROPOUT=${GATE_DROPOUT:-0.1}
CORR_DROPOUT=${CORR_DROPOUT:-0.1}

# Data pipeline + transfer knobs
DATALOADER_PIN_MEMORY=${DATALOADER_PIN_MEMORY:-True}
DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS:-True}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-4}
TRANSFER_NON_BLOCKING=${TRANSFER_NON_BLOCKING:-True}
CUDNN_BENCHMARK=${CUDNN_BENCHMARK:-True}

# Runtime overhead knobs
NO_BACKUP=${NO_BACKUP:-1}  # 1 => pass --no-backup to train.py
RUN_SCORE_AFTER_TRAIN=${RUN_SCORE_AFTER_TRAIN:-0}  # 1 => run meta_train_score right after each training run

train_extra_flags() {
  if [ "${NO_BACKUP}" = "1" ]; then
    echo "--no-backup"
  fi
}

common_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${target_steps}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|num_workers=${NUM_WORKERS}|summary_interval=${SUMMARY_INTERVAL}|eval_interval=${EVAL_INTERVAL}|ckpt_interval=${CKPT_INTERVAL}|optim_args.lr=${LR}|gate_bias_init=${GATE_BIAS_INIT}|gate_dropout=${GATE_DROPOUT}|corr_dropout=${CORR_DROPOUT}|dataloader_pin_memory=${DATALOADER_PIN_MEMORY}|dataloader_persistent_workers=${DATALOADER_PERSISTENT_WORKERS}|dataloader_prefetch_factor=${DATALOADER_PREFETCH_FACTOR}|transfer_non_blocking=${TRANSFER_NON_BLOCKING}|cudnn_benchmark=${CUDNN_BENCHMARK}"
}

score_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${target_steps}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|num_workers=${NUM_WORKERS}|optim_args.lr=${LR}|gate_bias_init=${GATE_BIAS_INIT}|gate_dropout=${GATE_DROPOUT}|corr_dropout=${CORR_DROPOUT}|dataloader_pin_memory=${DATALOADER_PIN_MEMORY}|dataloader_persistent_workers=${DATALOADER_PERSISTENT_WORKERS}|dataloader_prefetch_factor=${DATALOADER_PREFETCH_FACTOR}|transfer_non_blocking=${TRANSFER_NON_BLOCKING}|cudnn_benchmark=${CUDNN_BENCHMARK}"
}

usage() {
  cat <<USAGE
Usage:
  $0 run-one <head_only|head_last1|head_last2|head_last2_correction|head_last2_gate|head_last2_gate_correction> [seed] [target_steps]
  $0 run-seed [seed] [target_steps]
  $0 run-seed-split [seed] [target_steps]
  $0 run-multiseed [target_steps]
  $0 eval-one <method> <seed> [target_steps]
  $0 collect
USAGE
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

method_settings() {
  local method="$1"
  local seed="$2"
  local exp_name="${method}_seed${seed}"
  local extra=""

  case "${method}" in
    head_only)
      extra="freeze_backbone=True|partial_freeze_mode=none|train_last_tf_layers=0|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|train_head=True"
      ;;
    head_last1)
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=1|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|train_head=True"
      ;;
    head_last2)
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|train_head=True"
      ;;
    head_last2_correction)
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=True|enable_highway_gate=False|train_correction=True|train_gate=False|train_head=True"
      ;;
    head_last2_gate)
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=False|enable_highway_gate=True|train_correction=False|train_gate=True|train_head=True"
      ;;
    head_last2_gate_correction)
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=True|enable_highway_gate=True|train_correction=True|train_gate=True|train_head=True"
      ;;
    *)
      echo "Unknown method: ${method}"
      exit 1
      ;;
  esac

  printf '%s\n%s\n' "${exp_name}" "${extra}"
}

run_one() {
  local method="$1"
  local seed="$2"
  local target_steps="${3:-${TARGET_STEPS}}"
  local settings
  settings="$(method_settings "${method}" "${seed}")"
  local exp_name extra
  exp_name="$(echo "${settings}" | sed -n '1p')"
  extra="$(echo "${settings}" | sed -n '2p')"

  local log_dir="${RUN_ROOT}/${exp_name}"
  local train_override
  local score_override_str
  train_override="$(common_overrides "${target_steps}")|seed=${seed}|${extra}"
  score_override_str="$(score_overrides "${target_steps}")|seed=${seed}|${extra}"

  echo "=== method=${method} seed=${seed} target_steps=${target_steps} log_dir=${log_dir} ==="
  python train.py $(train_extra_flags) -mc "${TF_MODEL_CONFIG}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "${train_override}"

  if [ "${RUN_SCORE_AFTER_TRAIN}" = "1" ]; then
    python meta_train_score.py -mc "${TF_MODEL_CONFIG}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "${score_override_str}"
  fi
}

run_seed_split() {
  local seed="$1"
  local target_steps="${2:-${TARGET_STEPS}}"

  (
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_FIRST3}"
    run_one head_only "${seed}" "${target_steps}"
    run_one head_last1 "${seed}" "${target_steps}"
    run_one head_last2 "${seed}" "${target_steps}"
  ) &
  local pid_first3=$!

  (
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LAST3}"
    run_one head_last2_correction "${seed}" "${target_steps}"
    run_one head_last2_gate "${seed}" "${target_steps}"
    run_one head_last2_gate_correction "${seed}" "${target_steps}"
  ) &
  local pid_last3=$!

  wait "${pid_first3}"
  wait "${pid_last3}"
}

case "$1" in
  run-one)
    method="${2:-}"
    seed="${3:-0}"
    target_steps="${4:-${TARGET_STEPS}}"
    if [ -z "${method}" ]; then
      usage
      exit 1
    fi
    run_one "${method}" "${seed}" "${target_steps}"
    ;;

  run-seed)
    seed="${2:-0}"
    target_steps="${3:-${TARGET_STEPS}}"
    run_one head_only "${seed}" "${target_steps}"
    run_one head_last1 "${seed}" "${target_steps}"
    run_one head_last2 "${seed}" "${target_steps}"
    run_one head_last2_correction "${seed}" "${target_steps}"
    run_one head_last2_gate "${seed}" "${target_steps}"
    run_one head_last2_gate_correction "${seed}" "${target_steps}"
    ;;

  run-seed-split)
    seed="${2:-0}"
    target_steps="${3:-${TARGET_STEPS}}"
    run_seed_split "${seed}" "${target_steps}"
    ;;

  run-multiseed)
    target_steps="${2:-${TARGET_STEPS}}"
    for seed in ${SEEDS}; do
      "$0" run-seed "${seed}" "${target_steps}"
    done
    "$0" collect
    ;;

  eval-one)
    method="${2:-}"
    seed="${3:-0}"
    target_steps="${4:-${TARGET_STEPS}}"
    if [ -z "${method}" ]; then
      usage
      exit 1
    fi
    settings="$(method_settings "${method}" "${seed}")"
    exp_name="$(echo "${settings}" | sed -n '1p')"
    extra="$(echo "${settings}" | sed -n '2p')"
    log_dir="${RUN_ROOT}/${exp_name}"
    score_override_str="$(score_overrides "${target_steps}")|seed=${seed}|${extra}"
    python meta_train_score.py -mc "${TF_MODEL_CONFIG}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "${score_override_str}"
    ;;

  collect)
    python collect_refine_ablation_results.py --run-root "${RUN_ROOT}" --output "${RESULTS_CSV}"
    ;;

  *)
    usage
    exit 1
    ;;
esac
