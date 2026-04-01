#!/usr/bin/env bash
set -euo pipefail

MODEL_CONFIG=${MODEL_CONFIG:-cfg/model/omniglot_tf_fulltrain_12k.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/omniglot.yaml}
RUN_ROOT=${RUN_ROOT:-runs/omniglot_refine_ablation_12k_fulltrain}
RESULTS_CSV=${RESULTS_CSV:-results/refine_ablation_12k.csv}
SEEDS=${SEEDS:-"0 1 2"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

# Keep aligned with layer-1 baseline except 12K/batch_size changes requested in layer-2.
COMMON="tasks=20|train_shots=5|test_shots=1|max_train_steps=12000|batch_size=256|eval_batch_size=64|freeze_backbone=False|summary_interval=250|eval_interval=1000|ckpt_interval=1000"
# Stable evaluation: 8 * 256 = 2048 episodes
SCORE_COMMON="tasks=20|train_shots=5|test_shots=1|max_train_steps=12000|batch_size=256|eval_batch_size=64|eval_iters=32|freeze_backbone=False"

usage() {
  cat <<USAGE
Usage:
  $0 run-one <baseline|correction_only|gate_only|corr_gate> [seed]
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

run_one () {
  local method="$1"
  local seed="$2"
  local extra=""
  local exp_name=""

  case "${method}" in
    baseline)
      exp_name="baseline_seed${seed}"
      extra="enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False"
      ;;
    correction_only)
      exp_name="correction_only_seed${seed}"
      extra="enable_correction=True|enable_highway_gate=False|train_correction=True|train_gate=False"
      ;;
    gate_only)
      exp_name="gate_only_seed${seed}"
      extra="enable_correction=False|enable_highway_gate=True|train_correction=False|train_gate=True|gate_bias_init=-1.0"
      ;;
    corr_gate)
      exp_name="corr_gate_seed${seed}"
      extra="enable_correction=True|enable_highway_gate=True|train_correction=True|train_gate=True|gate_bias_init=-1.0"
      ;;
    *)
      echo "Unknown method: ${method}"
      exit 1
      ;;
  esac

  local log_dir="${RUN_ROOT}/${exp_name}"
  echo "=== method=${method} seed=${seed} log_dir=${log_dir} ==="

  python train.py \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "${COMMON}|seed=${seed}|${extra}"

  python meta_train_score.py \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "${SCORE_COMMON}|seed=${seed}|${extra}"
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
    run_one baseline "${seed}"
    run_one correction_only "${seed}"
    run_one gate_only "${seed}"
    run_one corr_gate "${seed}"
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
      -o "${SCORE_COMMON}"
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