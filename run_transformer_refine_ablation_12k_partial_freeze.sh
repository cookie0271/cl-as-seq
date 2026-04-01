#!/usr/bin/env bash
set -euo pipefail

MODEL_CONFIG=${MODEL_CONFIG:-cfg/model/omniglot_tf_fulltrain_12k.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/omniglot.yaml}
RUN_ROOT=${RUN_ROOT:-runs/omniglot_refine_ablation_12k_partial_freeze}
RESULTS_CSV=${RESULTS_CSV:-results/refine_ablation_12k_partial_freeze.csv}
SEEDS=${SEEDS:-"0 1 2"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

common_overrides() {
  echo "tasks=20|train_shots=5|test_shots=1|max_train_steps=12000|batch_size=256|eval_batch_size=64|eval_iters=32|summary_interval=250|eval_interval=1000|ckpt_interval=1000|freeze_backbone=True|enable_correction=True|enable_highway_gate=True|train_correction=True|train_gate=True|gate_bias_init=-1.0|train_head=True"
}

score_overrides() {
  echo "tasks=20|train_shots=5|test_shots=1|max_train_steps=12000|batch_size=256|eval_batch_size=64|eval_iters=32|freeze_backbone=True|enable_correction=True|enable_highway_gate=True|train_correction=True|train_gate=True|gate_bias_init=-1.0|train_head=True"
}

usage() {
  cat <<USAGE
Usage:
  $0 run-one <pf_last1_corr_gate|pf_last2_corr_gate|pf_tf_all_corr_gate> [seed]
  $0 run-seed <seed>
  $0 run-multiseed
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
    pf_last1_corr_gate)
      exp_name="pf_last1_corr_gate_seed${seed}"
      extra="partial_freeze_mode=last_n_tf|train_last_tf_layers=1|freeze_encoder=True"
      ;;
    pf_last2_corr_gate)
      exp_name="pf_last2_corr_gate_seed${seed}"
      extra="partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True"
      ;;
    pf_tf_all_corr_gate)
      exp_name="pf_tf_all_corr_gate_seed${seed}"
      extra="partial_freeze_mode=tf_all|train_last_tf_layers=0|freeze_encoder=True"
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
  local settings
  settings="$(method_settings "${method}" "${seed}")"
  local exp_name
  local extra
  exp_name="$(echo "${settings}" | sed -n '1p')"
  extra="$(echo "${settings}" | sed -n '2p')"
  local log_dir="${RUN_ROOT}/${exp_name}"

  echo "=== partial-freeze method=${method} seed=${seed} log_dir=${log_dir} ==="

  python train.py \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "$(common_overrides)|seed=${seed}|${extra}"

  python meta_train_score.py \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "$(score_overrides)|seed=${seed}|${extra}"
}

case "$1" in
  run-one)
    method="${2:-}"
    seed="${3:-0}"
    if [ -z "${method}" ]; then
      usage
      exit 1
    fi
    run_one "${method}" "${seed}"
    ;;

  run-seed)
    seed="${2:-0}"
    run_one pf_last1_corr_gate "${seed}"
    run_one pf_last2_corr_gate "${seed}"
    run_one pf_tf_all_corr_gate "${seed}"
    ;;

  run-multiseed)
    for seed in ${SEEDS}; do
      "$0" run-seed "${seed}"
    done
    "$0" collect
    ;;

  eval-one)
    log_dir="${2:-}"
    if [ -z "${log_dir}" ]; then
      usage
      exit 1
    fi
    python meta_train_score.py \
      -mc "${MODEL_CONFIG}" \
      -dc "${DATA_CONFIG}" \
      -l "${log_dir}" \
      -o "$(score_overrides)"
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