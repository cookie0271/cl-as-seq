#!/usr/bin/env bash
set -euo pipefail

# CIFAR100 11-way algorithm comparison at 50k steps.
# Algorithms:
# 1) full_outer_loop
# 2) head_only
# 3) head_last1
# 4) head_last2
# 5) head_last2_correction
# 6) head_last2_gate
# 7) head_last2_gate_correction
# 8) pn
# 9) gemcl
# 10) oml
# 11) anml

TF_MODEL_CONFIG=${TF_MODEL_CONFIG:-cfg/model/tf.yaml}
PN_MODEL_CONFIG=${PN_MODEL_CONFIG:-cfg/model/pn.yaml}
GEMCL_MODEL_CONFIG=${GEMCL_MODEL_CONFIG:-cfg/model/gemcl.yaml}
OML_MODEL_CONFIG=${OML_MODEL_CONFIG:-cfg/model/oml.yaml}
ANML_MODEL_CONFIG=${ANML_MODEL_CONFIG:-cfg/model/anml.yaml}

DATA_CONFIG=${DATA_CONFIG:-cfg/data/cifar100.yaml}
RUN_ROOT=${RUN_ROOT:-runs/cifar100_11alg_50k}
RESULTS_CSV=${RESULTS_CSV:-results/cifar100_11alg_50k.csv}

SEEDS=${SEEDS:-"0 1 2"}
TARGET_STEPS=${TARGET_STEPS:-50000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

# Recommended CIFAR100 defaults (adjustable via env):
TASKS=${TASKS:-20}
TRAIN_SHOTS=${TRAIN_SHOTS:-5}
TEST_SHOTS=${TEST_SHOTS:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
EVAL_ITERS=${EVAL_ITERS:-32}
NUM_WORKERS=${NUM_WORKERS:-16}

# gate/correction related knobs (important for CIFAR transfer)
GATE_BIAS_INIT=${GATE_BIAS_INIT:--1.0}
GATE_DROPOUT=${GATE_DROPOUT:-0.1}
CORR_DROPOUT=${CORR_DROPOUT:-0.1}
LR=${LR:-0.0001}

common_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${target_steps}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|num_workers=${NUM_WORKERS}|summary_interval=250|eval_interval=1000|ckpt_interval=1000|optim_args.lr=${LR}|gate_bias_init=${GATE_BIAS_INIT}|gate_dropout=${GATE_DROPOUT}|corr_dropout=${CORR_DROPOUT}|dataloader_pin_memory=True|dataloader_persistent_workers=True|dataloader_prefetch_factor=4|transfer_non_blocking=True|cudnn_benchmark=True"
}

score_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${target_steps}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|num_workers=${NUM_WORKERS}|optim_args.lr=${LR}|gate_bias_init=${GATE_BIAS_INIT}|gate_dropout=${GATE_DROPOUT}|corr_dropout=${CORR_DROPOUT}|dataloader_pin_memory=True|dataloader_persistent_workers=True|dataloader_prefetch_factor=4|transfer_non_blocking=True|cudnn_benchmark=True"
}

usage() {
  cat <<USAGE
Usage:
  $0 run-one <method> [seed] [target_steps]
  $0 run-seed [seed] [target_steps]
  $0 run-multiseed [target_steps]
  $0 eval-one <method> <seed> [target_steps]
  $0 collect

Methods:
  full_outer_loop
  head_only
  head_last1
  head_last2
  head_last2_correction
  head_last2_gate
  head_last2_gate_correction
  pn
  gemcl
  oml
  anml
USAGE
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

method_settings() {
  local method="$1"
  local seed="$2"
  local model_config=""
  local extra=""
  local exp_name="${method}_seed${seed}"

  case "${method}" in
    full_outer_loop)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=False|partial_freeze_mode=none|train_last_tf_layers=0|freeze_encoder=False|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|train_head=True"
      ;;
    head_only)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=True|partial_freeze_mode=none|train_last_tf_layers=0|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|train_head=True"
      ;;
    head_last1)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=1|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|train_head=True"
      ;;
    head_last2)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|train_head=True"
      ;;
    head_last2_correction)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=True|enable_highway_gate=False|train_correction=True|train_gate=False|train_head=True"
      ;;
    head_last2_gate)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=False|enable_highway_gate=True|train_correction=False|train_gate=True|train_head=True"
      ;;
    head_last2_gate_correction)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=True|enable_highway_gate=True|train_correction=True|train_gate=True|train_head=True"
      ;;
    pn)
      model_config="${PN_MODEL_CONFIG}"
      extra=""
      ;;
    gemcl)
      model_config="${GEMCL_MODEL_CONFIG}"
      extra=""
      ;;
    oml)
      model_config="${OML_MODEL_CONFIG}"
      extra=""
      ;;
    anml)
      model_config="${ANML_MODEL_CONFIG}"
      extra=""
      ;;
    *)
      echo "Unknown method: ${method}"
      exit 1
      ;;
  esac

  printf '%s\n%s\n%s\n' "${exp_name}" "${model_config}" "${extra}"
}

run_one() {
  local method="$1"
  local seed="$2"
  local target_steps="${3:-${TARGET_STEPS}}"
  local settings
  settings="$(method_settings "${method}" "${seed}")"
  local exp_name model_config extra
  exp_name="$(echo "${settings}" | sed -n '1p')"
  model_config="$(echo "${settings}" | sed -n '2p')"
  extra="$(echo "${settings}" | sed -n '3p')"

  local log_dir="${RUN_ROOT}/${exp_name}"
  local train_override
  local score_override_str
  train_override="$(common_overrides "${target_steps}")|seed=${seed}"
  score_override_str="$(score_overrides "${target_steps}")|seed=${seed}"
  if [ -n "${extra}" ]; then
    train_override="${train_override}|${extra}"
    score_override_str="${score_override_str}|${extra}"
  fi
  echo "=== method=${method} seed=${seed} target_steps=${target_steps} log_dir=${log_dir} ==="

  python train.py -mc "${model_config}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "${train_override}"
  python meta_train_score.py -mc "${model_config}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "${score_override_str}"
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
    run_one full_outer_loop "${seed}" "${target_steps}"
    run_one head_only "${seed}" "${target_steps}"
    run_one head_last1 "${seed}" "${target_steps}"
    run_one head_last2 "${seed}" "${target_steps}"
    run_one head_last2_correction "${seed}" "${target_steps}"
    run_one head_last2_gate "${seed}" "${target_steps}"
    run_one head_last2_gate_correction "${seed}" "${target_steps}"
    run_one pn "${seed}" "${target_steps}"
    run_one gemcl "${seed}" "${target_steps}"
    run_one oml "${seed}" "${target_steps}"
    run_one anml "${seed}" "${target_steps}"
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
    model_config="$(echo "${settings}" | sed -n '2p')"
    extra="$(echo "${settings}" | sed -n '3p')"
    log_dir="${RUN_ROOT}/${exp_name}"
    score_override_str="$(score_overrides "${target_steps}")|seed=${seed}"
    if [ -n "${extra}" ]; then
      score_override_str="${score_override_str}|${extra}"
    fi

    python meta_train_score.py -mc "${model_config}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "${score_override_str}"
    ;;

  collect)
    python collect_refine_ablation_results.py --run-root "${RUN_ROOT}" --output "${RESULTS_CSV}"
    ;;

  *)
    usage
    exit 1
    ;;
esac