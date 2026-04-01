#!/usr/bin/env bash
set -euo pipefail

MODEL_CONFIG=${MODEL_CONFIG:-cfg/model/omniglot_tf_fulltrain_12k.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/omniglot.yaml}
RUN_ROOT=${RUN_ROOT:-runs/omniglot_refine_ablation_12k_warm_frozen}
RESULTS_CSV=${RESULTS_CSV:-results/refine_ablation_12k_warm_frozen.csv}
SEEDS=${SEEDS:-"0 1 2"}
TARGET_STEPS=${TARGET_STEPS:-16000}
INIT_CKPT_TEMPLATE=${INIT_CKPT_TEMPLATE:-runs/omniglot_refine_ablation_12k_fulltrain/baseline_seed{seed}/ckpt-012000.pt}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

common_overrides() {
  local target_steps="${1:-16000}"
  echo "tasks=20|train_shots=5|test_shots=1|max_train_steps=${target_steps}|batch_size=256|eval_batch_size=64|freeze_backbone=True|summary_interval=250|eval_interval=1000|ckpt_interval=1000|eval_iters=32"
}

score_overrides() {
  local target_steps="${1:-16000}"
  echo "tasks=20|train_shots=5|test_shots=1|max_train_steps=${target_steps}|batch_size=256|eval_batch_size=64|eval_iters=32|freeze_backbone=True"
}

usage() {
  cat <<USAGE
Usage:
  $0 run-one <baseline|correction_only|gate_only|corr_gate> [seed] [target_steps] [init_ckpt]
  $0 run-seed <seed> [target_steps] [init_ckpt]
  $0 run-multiseed [target_steps]
  $0 eval-one <log_dir> [target_steps]
  $0 collect

Environment:
  TARGET_STEPS (default: 16000)
  INIT_CKPT_TEMPLATE (default: home/hongchang/changzhuo/cl-as.seq-main/runs/omniglot_refine_ablation_12k_fulltrain/baseline_seed{seed}/ckpt-012000.pt)
USAGE
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

method_settings() {
  local method="$1"
  local seed="$2"
  local extra=""
  local exp_name=""

  case "${method}" in
    baseline)
      exp_name="warm_baseline_seed${seed}"
      extra="enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False"
      ;;
    correction_only)
      exp_name="warm_correction_only_seed${seed}"
      extra="enable_correction=True|enable_highway_gate=False|train_correction=True|train_gate=False"
      ;;
    gate_only)
      exp_name="warm_gate_only_seed${seed}"
      extra="enable_correction=False|enable_highway_gate=True|train_correction=False|train_gate=True|gate_bias_init=-1.0"
      ;;
    corr_gate)
      exp_name="warm_corr_gate_seed${seed}"
      extra="enable_correction=True|enable_highway_gate=True|train_correction=True|train_gate=True|gate_bias_init=-1.0"
      ;;
    *)
      echo "Unknown method: ${method}"
      exit 1
      ;;
  esac
  printf '%s\n%s\n' "${exp_name}" "${extra}"
}

resolve_init_ckpt() {
  local seed="$1"
  local explicit_ckpt="${2:-}"
  if [ -n "${explicit_ckpt}" ]; then
    echo "${explicit_ckpt}"
  else
    echo "${INIT_CKPT_TEMPLATE/\{seed\}/${seed}}"
  fi
}

bootstrap_logdir_ckpt() {
  local log_dir="$1"
  local init_ckpt="$2"

  if ls "${log_dir}"/ckpt-*.pt >/dev/null 2>&1; then
    return
  fi

  if [ ! -f "${init_ckpt}" ]; then
    echo "Init checkpoint not found: ${init_ckpt}"
    exit 1
  fi

  mkdir -p "${log_dir}"
  cp "${init_ckpt}" "${log_dir}/"
}

run_one() {
  local method="$1"
  local seed="$2"
  local target_steps="${3:-${TARGET_STEPS}}"
  local explicit_ckpt="${4:-}"
  local settings
  settings="$(method_settings "${method}" "${seed}")"
  local exp_name
  local extra
  exp_name="$(echo "${settings}" | sed -n '1p')"
  extra="$(echo "${settings}" | sed -n '2p')"

  local init_ckpt
  init_ckpt="$(resolve_init_ckpt "${seed}" "${explicit_ckpt}")"
  local log_dir="${RUN_ROOT}/${exp_name}"
  bootstrap_logdir_ckpt "${log_dir}" "${init_ckpt}"

  echo "=== warm-start method=${method} seed=${seed} target_steps=${target_steps} init_ckpt=${init_ckpt} ==="
  python train.py \
    --resume \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "$(common_overrides "${target_steps}")|seed=${seed}|init_ckpt='${init_ckpt}'|${extra}"

  python meta_train_score.py \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "$(score_overrides "${target_steps}")|seed=${seed}|init_ckpt='${init_ckpt}'|${extra}"
}

case "$1" in
  run-one)
    method="${2:-}"
    seed="${3:-0}"
    target_steps="${4:-${TARGET_STEPS}}"
    init_ckpt="${5:-}"
    if [ -z "${method}" ]; then
      usage
      exit 1
    fi
    run_one "${method}" "${seed}" "${target_steps}" "${init_ckpt}"
    ;;

  run-seed)
    seed="${2:-0}"
    target_steps="${3:-${TARGET_STEPS}}"
    init_ckpt="${4:-}"
    run_one baseline "${seed}" "${target_steps}" "${init_ckpt}"
    run_one correction_only "${seed}" "${target_steps}" "${init_ckpt}"
    run_one gate_only "${seed}" "${target_steps}" "${init_ckpt}"
    run_one corr_gate "${seed}" "${target_steps}" "${init_ckpt}"
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