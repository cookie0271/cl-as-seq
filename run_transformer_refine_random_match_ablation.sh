#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

MODEL_CONFIG=${MODEL_CONFIG:-cfg/model/omniglot_tf_fulltrain_12k.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/omniglot.yaml}
RUN_ROOT=${RUN_ROOT:-runs/omniglot_refine_random_match_ablation_12k}
RESUME_RUN_ROOT=${RESUME_RUN_ROOT:-runs/omniglot_refine_random_match_ablation_50k_resume}
RESULTS_CSV=${RESULTS_CSV:-results/refine_random_match_ablation_12k.csv}
SEEDS=${SEEDS:-"0 1 2"}
TARGET_STEPS=${TARGET_STEPS:-50000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

# Dataset/task defaults (override via env for CIFAR100 or other datasets)
TASKS=${TASKS:-20}
TRAIN_SHOTS=${TRAIN_SHOTS:-5}
TEST_SHOTS=${TEST_SHOTS:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
EVAL_ITERS=${EVAL_ITERS:-32}
NUM_WORKERS=${NUM_WORKERS:-16}

common_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${target_steps}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|summary_interval=250|eval_interval=1000|ckpt_interval=1000|freeze_backbone=True|train_head=True|freeze_encoder=True|num_workers=${NUM_WORKERS}|dataloader_pin_memory=True|dataloader_persistent_workers=True|dataloader_prefetch_factor=4|transfer_non_blocking=True|cudnn_benchmark=True"
}

score_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${target_steps}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|freeze_backbone=True|train_head=True|freeze_encoder=True|num_workers=${NUM_WORKERS}|dataloader_pin_memory=True|dataloader_persistent_workers=True|dataloader_prefetch_factor=4|transfer_non_blocking=True|cudnn_benchmark=True"
}

usage() {
  cat <<USAGE
Usage:
  $0 run-one <random_match_no_corr_gate|last2_no_corr_gate|last2_gate_only|last2_corr_gate> [seed] [target_steps]
  $0 resume-one <random_match_no_corr_gate|last2_no_corr_gate|last2_gate_only|last2_corr_gate> [seed] [target_steps]
  $0 run-seed <seed>
  $0 resume-seed <seed> [target_steps]
  $0 run-multiseed
  $0 resume-multiseed [target_steps]
  $0 eval-one <log_dir>
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
    random_match_no_corr_gate)
      exp_name="random_match_no_corr_gate_seed${seed}"
      extra="partial_freeze_mode=random_match|train_last_tf_layers=0|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|random_match_seed=${seed}|random_match_scope=tf_only|random_match_unit=layer_or_block|random_match_target_last_tf_layers=2"
      ;;
    last2_no_corr_gate)
      exp_name="last2_no_corr_gate_seed${seed}"
      extra="partial_freeze_mode=last_n_tf|train_last_tf_layers=2|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False"
      ;;
    last2_gate_only)
      exp_name="last2_gate_only_seed${seed}"
      extra="partial_freeze_mode=last_n_tf|train_last_tf_layers=2|enable_correction=False|enable_highway_gate=True|train_correction=False|train_gate=True|gate_bias_init=-1.0"
      ;;
    last2_corr_gate)
      exp_name="last2_corr_gate_seed${seed}"
      extra="partial_freeze_mode=last_n_tf|train_last_tf_layers=2|enable_correction=True|enable_highway_gate=True|train_correction=True|train_gate=True|gate_bias_init=-1.0"
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
  local target_steps="${3:-12000}"
  local settings
  settings="$(method_settings "${method}" "${seed}")"
  local exp_name
  local extra
  exp_name="$(echo "${settings}" | sed -n '1p')"
  extra="$(echo "${settings}" | sed -n '2p')"
  local log_dir="${RUN_ROOT}/${exp_name}"

  echo "=== random-match ablation method=${method} seed=${seed} target_steps=${target_steps} log_dir=${log_dir} ==="

  python train.py \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "$(common_overrides "${target_steps}")|seed=${seed}|${extra}"

  python meta_train_score.py \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "$(score_overrides "${target_steps}")|seed=${seed}|${extra}"
}

latest_ckpt_in_dir() {
  local log_dir="$1"
  local ckpt_path=""
  ckpt_path="$(ls "${log_dir}"/ckpt-*.pt 2>/dev/null | sort | tail -n1 || true)"
  printf '%s\n' "${ckpt_path}"
}

resume_one() {
  local method="$1"
  local seed="$2"
  local target_steps="${3:-${TARGET_STEPS}}"
  local settings
  settings="$(method_settings "${method}" "${seed}")"
  local exp_name
  local extra
  exp_name="$(echo "${settings}" | sed -n '1p')"
  extra="$(echo "${settings}" | sed -n '2p')"

  local src_log_dir="${RUN_ROOT}/${exp_name}"
  local dst_log_dir="${RESUME_RUN_ROOT}/${exp_name}"
  local ckpt_path
  ckpt_path="$(latest_ckpt_in_dir "${src_log_dir}")"
  if [ -z "${ckpt_path}" ]; then
    echo "No checkpoint found in ${src_log_dir}"
    exit 1
  fi

  mkdir -p "${dst_log_dir}"
  if ! ls "${dst_log_dir}"/ckpt-*.pt >/dev/null 2>&1; then
    cp "${ckpt_path}" "${dst_log_dir}/"
  fi

  echo "=== resume method=${method} seed=${seed} target_steps=${target_steps} src=${src_log_dir} dst=${dst_log_dir} ==="

  python train.py \
    --resume \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${dst_log_dir}" \
    -o "$(common_overrides "${target_steps}")|seed=${seed}|${extra}"

  python meta_train_score.py \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${dst_log_dir}" \
    -o "$(score_overrides "${target_steps}")|seed=${seed}|${extra}"
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

  resume-one)
    method="${2:-}"
    seed="${3:-0}"
    target_steps="${4:-${TARGET_STEPS}}"
    if [ -z "${method}" ]; then
      usage
      exit 1
    fi
    resume_one "${method}" "${seed}" "${target_steps}"
    ;;

  run-seed)
    seed="${2:-0}"
    run_one random_match_no_corr_gate "${seed}"
    run_one last2_no_corr_gate "${seed}"
    run_one last2_gate_only "${seed}"
    run_one last2_corr_gate "${seed}"
    ;;

  resume-seed)
    seed="${2:-0}"
    target_steps="${3:-${TARGET_STEPS}}"
    resume_one random_match_no_corr_gate "${seed}" "${target_steps}"
    resume_one last2_no_corr_gate "${seed}" "${target_steps}"
    resume_one last2_gate_only "${seed}" "${target_steps}"
    resume_one last2_corr_gate "${seed}" "${target_steps}"
    ;;

  run-multiseed)
    for seed in ${SEEDS}; do
      bash "${SCRIPT_PATH}" run-seed "${seed}"
    done
    bash "${SCRIPT_PATH}" collect
    ;;

  resume-multiseed)
    target_steps="${2:-${TARGET_STEPS}}"
    for seed in ${SEEDS}; do
      bash "${SCRIPT_PATH}" resume-seed "${seed}" "${target_steps}"
    done
    bash "${SCRIPT_PATH}" collect
    ;;

  eval-one)
    log_dir="${2:-}"
    if [ -z "${log_dir}" ]; then
      usage
      exit 1
    fi
    if [ ! -f "${log_dir}/config.yaml" ]; then
      echo "Missing ${log_dir}/config.yaml"
      exit 1
    fi
    python meta_train_score.py \
      -mc "${log_dir}/config.yaml" \
      -dc "${log_dir}/config.yaml" \
      -l "${log_dir}" \
      -o "max_train_steps=${TARGET_STEPS}"
    ;;

  collect)
    python collect_refine_ablation_results.py \
      --run-root "${RUN_ROOT}" \
      --output "${RESULTS_CSV}"
    ;;

  *)
    usage
    exit 1
    ;;
esac
