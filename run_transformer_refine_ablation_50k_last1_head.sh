#!/usr/bin/env bash
set -euo pipefail

MODEL_CONFIG=${MODEL_CONFIG:-cfg/model/omniglot_tf_fulltrain_12k.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/omniglot.yaml}
RUN_ROOT=${RUN_ROOT:-runs/omniglot_refine_ablation_50k_last1_head}
RESULTS_CSV=${RESULTS_CSV:-results/refine_ablation_50k_last1_head.csv}
SEEDS=${SEEDS:-"0 1 2"}
TARGET_STEPS=${TARGET_STEPS:-50000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

common_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  echo "tasks=20|train_shots=5|test_shots=1|max_train_steps=${target_steps}|batch_size=256|eval_batch_size=64|eval_iters=32|summary_interval=250|eval_interval=1000|ckpt_interval=1000|freeze_backbone=True|train_head=True"
}

score_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  echo "tasks=20|train_shots=5|test_shots=1|max_train_steps=${target_steps}|batch_size=256|eval_batch_size=64|eval_iters=32|freeze_backbone=True|train_head=True"
}

usage() {
  cat <<USAGE
Usage:
  $0 run-one <head_only|pf_last1> [seed] [target_steps]
  $0 resume-one <head_only|pf_last1> [seed] [target_steps]
  $0 run-seed [seed] [target_steps]
  $0 resume-seed [seed] [target_steps]
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
    head_only)
      exp_name="head_only_seed${seed}"
      extra="partial_freeze_mode=none|train_last_tf_layers=0|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|train_head=True"
      ;;
    pf_last1)
      exp_name="pf_last1_seed${seed}"
      extra="partial_freeze_mode=last_n_tf|train_last_tf_layers=1|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|train_head=True"
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
  local exp_name
  local extra
  exp_name="$(echo "${settings}" | sed -n '1p')"
  extra="$(echo "${settings}" | sed -n '2p')"
  local log_dir="${RUN_ROOT}/${exp_name}"

  echo "=== method=${method} seed=${seed} target_steps=${target_steps} log_dir=${log_dir} ==="

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
  local log_dir="${RUN_ROOT}/${exp_name}"

  echo "=== resume method=${method} seed=${seed} target_steps=${target_steps} log_dir=${log_dir} ==="

  python train.py \
    --resume \
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
    target_steps="${3:-${TARGET_STEPS}}"
    run_one head_only "${seed}" "${target_steps}"
    run_one pf_last1 "${seed}" "${target_steps}"
    ;;

  resume-seed)
    seed="${2:-0}"
    target_steps="${3:-${TARGET_STEPS}}"
    resume_one head_only "${seed}" "${target_steps}"
    resume_one pf_last1 "${seed}" "${target_steps}"
    ;;

  run-multiseed)
    target_steps="${2:-${TARGET_STEPS}}"
    for seed in ${SEEDS}; do
      "$0" run-seed "${seed}" "${target_steps}"
    done
    "$0" collect
    ;;

  eval-one)
    log_dir="${2:-}"
    target_steps="${3:-${TARGET_STEPS}}"
    if [ -z "${log_dir}" ]; then
      usage
      exit 1
    fi
    python meta_train_score.py \
      -mc "${MODEL_CONFIG}" \
      -dc "${DATA_CONFIG}" \
      -l "${log_dir}" \
      -o "$(score_overrides "${target_steps}")"
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