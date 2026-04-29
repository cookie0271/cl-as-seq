#!/usr/bin/env bash
set -euo pipefail

# Unified with run_cifar100_11alg_50k.sh style, but for regression datasets.
# Included methods are the CIFAR100 11-alg list minus PN/GeMCL (classification-only).

TF_MODEL_CONFIG=${TF_MODEL_CONFIG:-cfg/model/tf.yaml}
OML_MODEL_CONFIG=${OML_MODEL_CONFIG:-cfg/model/oml.yaml}
ANML_MODEL_CONFIG=${ANML_MODEL_CONFIG:-cfg/model/anml.yaml}

RUN_ROOT=${RUN_ROOT:-runs_regression}
RESULTS_CSV=${RESULTS_CSV:-results/regression_9alg.csv}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TARGET_STEPS=${TARGET_STEPS:-50000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

DATASETS=(
  sine_cifar_like
  casia_rot_cifar_like
  casia_comp_cifar_like
)

TASKS=${TASKS:-5}
TRAIN_SHOTS=${TRAIN_SHOTS:-5}
TEST_SHOTS=${TEST_SHOTS:-10}
BATCH_SIZE=${BATCH_SIZE:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
EVAL_ITERS=${EVAL_ITERS:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-0.0001}

common_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  local seed="$2"
  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${target_steps}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|num_workers=${NUM_WORKERS}|summary_interval=250|eval_interval=1000|ckpt_interval=1000|optim_args.lr=${LR}|seed=${seed}"
}

score_overrides() {
  local target_steps="${1:-${TARGET_STEPS}}"
  local seed="$2"
  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${target_steps}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|num_workers=${NUM_WORKERS}|optim_args.lr=${LR}|seed=${seed}"
}

usage() {
  cat <<USAGE
Usage:
  $0 run-one <dataset> <method> [seed] [target_steps]
  $0 run-seed <dataset> [seed] [target_steps]
  $0 run-dataset <dataset> [target_steps]
  $0 run-all [target_steps]
  $0 resume-one <dataset> <method> [seed] [target_steps]
  $0 resume-seed <dataset> [seed] [target_steps]
  $0 resume-dataset <dataset> [target_steps]
  $0 resume-all [target_steps]
  $0 eval-one <dataset> <method> [seed] [target_steps]
  $0 collect

Datasets:
  sine_cifar_like
  casia_rot_cifar_like
  casia_comp_cifar_like

Methods (CIFAR100-11alg minus PN/GeMCL):
  full_outer_loop_train
  head_only
  head_last1
  head_last2
  head_last2_correction
  head_last2_gate
  head_last2_gate_correction
  oml
  anml
USAGE
}

methods=(
  full_outer_loop_train
  head_only
  head_last1
  head_last2
  head_last2_correction
  head_last2_gate
  head_last2_gate_correction
  oml
  anml
)

method_settings() {
  local method="$1"
  local seed="$2"
  local model_config=""
  local extra=""
  local exp_name="${method}_seed${seed}"

  case "${method}" in
    full_outer_loop_train)
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
    oml)
      model_config="${OML_MODEL_CONFIG}"
      extra=""
      ;;
    anml)
      model_config="${ANML_MODEL_CONFIG}"
      extra=""
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 1
      ;;
  esac

  printf '%s\n%s\n%s\n' "${exp_name}" "${model_config}" "${extra}"
}

assert_dataset() {
  local dataset="$1"
  case "${dataset}" in
    sine_cifar_like|casia_rot_cifar_like|casia_comp_cifar_like) ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
}

run_one() {
  local dataset="$1"
  local method="$2"
  local seed="$3"
  local target_steps="${4:-${TARGET_STEPS}}"
  local resume_mode="${5:-0}"
  assert_dataset "${dataset}"

  local settings exp_name model_config extra
  settings="$(method_settings "${method}" "${seed}")"
  exp_name="$(echo "${settings}" | sed -n '1p')"
  model_config="$(echo "${settings}" | sed -n '2p')"
  extra="$(echo "${settings}" | sed -n '3p')"

  local log_dir="${RUN_ROOT}/${dataset}/${exp_name}"
  local train_override score_override_str
  train_override="$(common_overrides "${target_steps}" "${seed}")"
  score_override_str="$(score_overrides "${target_steps}" "${seed}")"
  if [ -n "${extra}" ]; then
    train_override="${train_override}|${extra}"
    score_override_str="${score_override_str}|${extra}"
  fi

  echo "=== dataset=${dataset} method=${method} seed=${seed} target_steps=${target_steps} log_dir=${log_dir} ==="
  if [ "${resume_mode}" = "1" ]; then
    python train.py --resume -mc "${model_config}" -dc "cfg/data/${dataset}.yaml" -l "${log_dir}" -o "${train_override}" --no-backup
  else
    python train.py -mc "${model_config}" -dc "cfg/data/${dataset}.yaml" -l "${log_dir}" -o "${train_override}" --no-backup
  fi
  python meta_train_score.py -mc "${model_config}" -dc "cfg/data/${dataset}.yaml" -l "${log_dir}" -o "${score_override_str}"
}

run_seed() {
  local dataset="$1"
  local seed="$2"
  local target_steps="${3:-${TARGET_STEPS}}"
  local resume_mode="${4:-0}"
  for method in "${methods[@]}"; do
    run_one "${dataset}" "${method}" "${seed}" "${target_steps}" "${resume_mode}"
  done
}

run_dataset() {
  local dataset="$1"
  local target_steps="${2:-${TARGET_STEPS}}"
  local resume_mode="${3:-0}"
  for seed in ${SEEDS}; do
    run_seed "${dataset}" "${seed}" "${target_steps}" "${resume_mode}"
  done
}

run_all() {
  local target_steps="${1:-${TARGET_STEPS}}"
  local resume_mode="${2:-0}"
  for dataset in "${DATASETS[@]}"; do
    run_dataset "${dataset}" "${target_steps}" "${resume_mode}"
  done
  "$0" collect
}

case "${1:-}" in
  run-one)
    dataset="${2:-}"
    method="${3:-}"
    seed="${4:-0}"
    target_steps="${5:-${TARGET_STEPS}}"
    if [ -z "${dataset}" ] || [ -z "${method}" ]; then
      usage
      exit 1
    fi
    run_one "${dataset}" "${method}" "${seed}" "${target_steps}"
    ;;
  run-seed)
    dataset="${2:-}"
    seed="${3:-0}"
    target_steps="${4:-${TARGET_STEPS}}"
    if [ -z "${dataset}" ]; then
      usage
      exit 1
    fi
    run_seed "${dataset}" "${seed}" "${target_steps}"
    ;;
  run-dataset)
    dataset="${2:-}"
    target_steps="${3:-${TARGET_STEPS}}"
    if [ -z "${dataset}" ]; then
      usage
      exit 1
    fi
    run_dataset "${dataset}" "${target_steps}"
    ;;
  run-all)
    target_steps="${2:-${TARGET_STEPS}}"
    run_all "${target_steps}" "0"
    ;;
  resume-one)
    dataset="${2:-}"
    method="${3:-}"
    seed="${4:-0}"
    target_steps="${5:-${TARGET_STEPS}}"
    if [ -z "${dataset}" ] || [ -z "${method}" ]; then
      usage
      exit 1
    fi
    run_one "${dataset}" "${method}" "${seed}" "${target_steps}" "1"
    ;;
  resume-seed)
    dataset="${2:-}"
    seed="${3:-0}"
    target_steps="${4:-${TARGET_STEPS}}"
    if [ -z "${dataset}" ]; then
      usage
      exit 1
    fi
    run_seed "${dataset}" "${seed}" "${target_steps}" "1"
    ;;
  resume-dataset)
    dataset="${2:-}"
    target_steps="${3:-${TARGET_STEPS}}"
    if [ -z "${dataset}" ]; then
      usage
      exit 1
    fi
    run_dataset "${dataset}" "${target_steps}" "1"
    ;;
  resume-all)
    target_steps="${2:-${TARGET_STEPS}}"
    run_all "${target_steps}" "1"
    ;;
  eval-one)
    dataset="${2:-}"
    method="${3:-}"
    seed="${4:-0}"
    target_steps="${5:-${TARGET_STEPS}}"
    if [ -z "${dataset}" ] || [ -z "${method}" ]; then
      usage
      exit 1
    fi
    settings="$(method_settings "${method}" "${seed}")"
    exp_name="$(echo "${settings}" | sed -n '1p')"
    model_config="$(echo "${settings}" | sed -n '2p')"
    extra="$(echo "${settings}" | sed -n '3p')"
    log_dir="${RUN_ROOT}/${dataset}/${exp_name}"
    score_override_str="$(score_overrides "${target_steps}" "${seed}")"
    if [ -n "${extra}" ]; then
      score_override_str="${score_override_str}|${extra}"
    fi
    python meta_train_score.py -mc "${model_config}" -dc "cfg/data/${dataset}.yaml" -l "${log_dir}" -o "${score_override_str}"
    ;;
  collect)
    python collect_regression_results.py --run-root "${RUN_ROOT}" --output "${RESULTS_CSV}"
    ;;
  *)
    usage
    exit 1
    ;;
esac