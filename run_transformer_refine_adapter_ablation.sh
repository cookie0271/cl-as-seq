#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

MODEL_CONFIG=${MODEL_CONFIG:-cfg/model/tf.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/omniglot.yaml}
RUN_ROOT=${RUN_ROOT:-runs/refine_adapter_standard_ablation}
RESULTS_CSV=${RESULTS_CSV:-results/refine_adapter_standard_ablation.csv}
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
  $0 run-one <adapter_standard|adapter_last2|last2_only|last2_corr_gate> [seed] [target_steps]
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
    adapter_standard)
      exp_name="adapter_standard_seed${seed}"
      extra="freeze_backbone=True|partial_freeze_mode=adapter_standard|train_last_tf_layers=0|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|enable_adapter=True|train_adapter=True|adapter_layers=all|adapter_location=post_layer|adapter_strict_match=False|adapter_dim_candidates=[16,32,64,128,256,512]|adapter_default_dim=64|train_head=True"
      ;;
    adapter_last2)
      exp_name="adapter_last2_seed${seed}"
      extra="freeze_backbone=True|partial_freeze_mode=adapter_last2|train_last_tf_layers=0|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|enable_adapter=True|train_adapter=True|adapter_layers=last2|adapter_location=post_layer|adapter_strict_match=False|adapter_dim_candidates=[16,32,64,128,256,512]|train_head=True"
      ;;
    last2_only)
      exp_name="last2_only_seed${seed}"
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|enable_adapter=False|train_adapter=False|train_head=True"
      ;;
    last2_corr_gate)
      exp_name="last2_corr_gate_seed${seed}"
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=True|enable_highway_gate=True|train_correction=True|train_gate=True|enable_adapter=False|train_adapter=False|gate_bias_init=-1.0|train_head=True"
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

  echo "=== adapter standard ablation method=${method} seed=${seed} target_steps=${target_steps} log_dir=${log_dir} ==="

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
    run_one adapter_standard "${seed}" "${target_steps}"
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
    eval_override="max_train_steps=${target_steps}"
    if [ -f "${log_dir}/trainable_summary.yaml" ]; then
      adapter_dim="$(python - <<PY
import yaml
from pathlib import Path
p = Path(r'''${log_dir}/trainable_summary.yaml''')
data = yaml.safe_load(p.read_text()) if p.exists() else {}
v = data.get('adapter_dim', '')
print(v if v not in ('', None) else '')
PY
)"
      adapter_location="$(python - <<PY
import yaml
from pathlib import Path
p = Path(r'''${log_dir}/trainable_summary.yaml''')
data = yaml.safe_load(p.read_text()) if p.exists() else {}
v = data.get('adapter_location', '')
print(v if v not in ('', None) else '')
PY
)"
      if [ -n "${adapter_dim}" ]; then
        eval_override="${eval_override}|adapter_dim=${adapter_dim}|enable_adapter=True"
      fi
      if [ -n "${adapter_location}" ]; then
        eval_override="${eval_override}|adapter_location=${adapter_location}"
      fi
    fi
    python meta_train_score.py -mc "${log_dir}/config.yaml" -dc "${log_dir}/config.yaml" -l "${log_dir}" -o "${eval_override}"
    ;;

  collect)
    python collect_refine_ablation_results.py --run-root "${RUN_ROOT}" --output "${RESULTS_CSV}"
    ;;

  *)
    usage
    exit 1
    ;;
esac