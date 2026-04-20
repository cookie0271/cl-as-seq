#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

MODEL_CONFIG=${MODEL_CONFIG:-cfg/model/tf.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/omniglot.yaml}
RUN_ROOT=${RUN_ROOT:-runs/refine_bitfit_standard_ablation}
RESULTS_CSV=${RESULTS_CSV:-results/refine_bitfit_standard_ablation.csv}
SEEDS=${SEEDS:-"0 1 2"}
TARGET_STEPS=${TARGET_STEPS:-50000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

TASKS=${TASKS:-20}
TRAIN_SHOTS=${TRAIN_SHOTS:-5}
TEST_SHOTS=${TEST_SHOTS:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
EVAL_ITERS=${EVAL_ITERS:-32}
NUM_WORKERS=${NUM_WORKERS:-16}

common_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${target_steps}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|summary_interval=250|eval_interval=1000|ckpt_interval=1000|num_workers=${NUM_WORKERS}|dataloader_pin_memory=True|dataloader_persistent_workers=True|dataloader_prefetch_factor=4|transfer_non_blocking=True|cudnn_benchmark=True"
}

score_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${target_steps}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|num_workers=${NUM_WORKERS}|dataloader_pin_memory=True|dataloader_persistent_workers=True|dataloader_prefetch_factor=4|transfer_non_blocking=True|cudnn_benchmark=True"
}

usage() {
  cat <<USAGE
Usage:
  $0 run-one <bitfit_standard> [seed] [target_steps]
  $0 run-seed <seed> [target_steps]
  $0 run-multiseed [target_steps]
  $0 eval-one <log_dir> [target_steps]
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
  local exp_name=""
  local extra=""

  case "${method}" in
    bitfit_standard)
      exp_name="bitfit_standard_seed${seed}"
      extra="freeze_backbone=True|partial_freeze_mode=bitfit_standard|train_last_tf_layers=0|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|enable_adapter=False|train_adapter=False|enable_lora=False|train_lora=False|enable_bitfit=True|train_bitfit=True|bitfit_scope=transformer_only|bitfit_include_layernorm_bias=True|bitfit_include_head_bias=True|bitfit_strict_match=False|train_head=True"
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

  echo "=== bitfit standard ablation method=${method} seed=${seed} target_steps=${target_steps} log_dir=${log_dir} ==="

  python train.py -mc "${MODEL_CONFIG}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "$(common_overrides "${target_steps}")|seed=${seed}|${extra}"
  python meta_train_score.py -mc "${MODEL_CONFIG}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "$(score_overrides "${target_steps}")|seed=${seed}|${extra}"
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
    run_one bitfit_standard "${seed}" "${target_steps}"
    ;;

  run-multiseed)
    target_steps="${2:-${TARGET_STEPS}}"
    for seed in ${SEEDS}; do
      bash "${SCRIPT_PATH}" run-seed "${seed}" "${target_steps}"
    done
    bash "${SCRIPT_PATH}" collect
    ;;

  eval-one)
    log_dir="${2:-}"
    target_steps="${3:-${TARGET_STEPS}}"
    if [ -z "${log_dir}" ]; then
      usage
      exit 1
    fi
    if [ ! -f "${log_dir}/config.yaml" ]; then
      echo "Missing ${log_dir}/config.yaml"
      exit 1
    fi
    python meta_train_score.py -mc "${log_dir}/config.yaml" -dc "${log_dir}/config.yaml" -l "${log_dir}" -o "max_train_steps=${target_steps}"
    ;;


  collect)
    python collect_refine_ablation_results.py --run-root "${RUN_ROOT}" --output "${RESULTS_CSV}"
    ;;

  *)
    usage
    exit 1
    ;;
esac