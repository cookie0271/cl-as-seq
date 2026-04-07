#!/usr/bin/env bash
set -euo pipefail

MODEL_CONFIG=${MODEL_CONFIG:-cfg/model/omniglot_tf_fulltrain_12k.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/omniglot.yaml}
RUN_ROOT=${RUN_ROOT:-runs/omniglot_refine_ablation_12k_partial_freeze}
RESUME_RUN_ROOT=${RESUME_RUN_ROOT:-runs/omniglot_refine_ablation_50k_partial_freeze_resume}
BASELINE_RUN_ROOT=${BASELINE_RUN_ROOT:-runs/omniglot_refine_ablation_12k_partial_freeze}
RESUME_BASELINE_RUN_ROOT=${RESUME_BASELINE_RUN_ROOT:-runs/omniglot_refine_ablation_50k_partial_freeze_baseline_resume}
RESULTS_CSV=${RESULTS_CSV:-results/refine_ablation_12k_partial_freeze.csv}
SEEDS=${SEEDS:-"0 1 2"}
TARGET_STEPS=${TARGET_STEPS:-50000}
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
  $0 run-one <baseline_pf_last2|pf_last1_corr_gate|pf_last2_corr_gate|pf_last2_cor_only|pf_last2_gate_only|pf_tf_all_corr_gate> [seed]
  $0 resume-one <baseline_pf_last2|pf_last1_corr_gate|pf_last2_corr_gate|pf_last2_cor_only|pf_last2_gate_only|pf_tf_all_corr_gate> [seed] [target_steps]
  $0 resume-last2-pair [seed] [target_steps]
  $0 resume-last2-ablation [seed] [target_steps]
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
    baseline_pf_last2)
      exp_name="baseline_pf_last2_seed${seed}"
      extra="partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False"
      ;;
    pf_last1_corr_gate)
      exp_name="pf_last1_corr_gate_seed${seed}"
      extra="partial_freeze_mode=last_n_tf|train_last_tf_layers=1|freeze_encoder=True"
      ;;
    pf_last2_corr_gate)
      exp_name="pf_last2_corr_gate_seed${seed}"
      extra="partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True"
      ;;
    pf_last2_cor_only)
      exp_name="pf_last2_cor_only_seed${seed}"
      extra="partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=True|enable_highway_gate=False|train_correction=True|train_gate=False"
      ;;
    pf_last2_gate_only)
      exp_name="pf_last2_gate_only_seed${seed}"
      extra="partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=False|enable_highway_gate=True|train_correction=False|train_gate=True|gate_bias_init=-1.0"
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

latest_ckpt_in_dir() {
  local log_dir="$1"
  local ckpt_path=""
  ckpt_path="$(ls "${log_dir}"/ckpt-*.pt 2>/dev/null | sort | tail -n1 || true)"
  printf '%s\n' "${ckpt_path}"
}

fallback_src_exp_name() {
  local method="$1"
  local seed="$2"
  case "${method}" in
    pf_last2_cor_only|pf_last2_gate_only)
      printf 'baseline_pf_last2_seed%s\n' "${seed}"
      ;;
    *)
      printf '\n'
      ;;
  esac
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
  local run_root="${RUN_ROOT}"
  if [ "${method}" = "baseline_pf_last2" ]; then
    run_root="${BASELINE_RUN_ROOT}"
  fi
  local log_dir="${run_root}/${exp_name}"

  echo "=== partial-freeze method=${method} seed=${seed} target_steps=${target_steps} log_dir=${log_dir} ==="

  python train.py \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "$(common_overrides)|seed=${seed}|max_train_steps=${target_steps}|${extra}"

  python meta_train_score.py \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "$(score_overrides)|seed=${seed}|max_train_steps=${target_steps}|${extra}"
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

  local src_root="${RUN_ROOT}"
  local dst_root="${RESUME_RUN_ROOT}"
  if [ "${method}" = "baseline_pf_last2" ]; then
    src_root="${BASELINE_RUN_ROOT}"
    dst_root="${RESUME_BASELINE_RUN_ROOT}"
  fi
  local src_log_dir="${src_root}/${exp_name}"
  local dst_log_dir="${dst_root}/${exp_name}"
  local ckpt_path
  ckpt_path="$(latest_ckpt_in_dir "${src_log_dir}")"
  if [ -z "${ckpt_path}" ]; then
    local fallback_exp_name
    fallback_exp_name="$(fallback_src_exp_name "${method}" "${seed}")"
    if [ -n "${fallback_exp_name}" ]; then
      local fallback_src_log_dir="${src_root}/${fallback_exp_name}"
      ckpt_path="$(latest_ckpt_in_dir "${fallback_src_log_dir}")"
      if [ -n "${ckpt_path}" ]; then
        echo "No checkpoint found in ${src_log_dir}, fallback to ${fallback_src_log_dir}"
      fi
    fi
  fi
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
    -o "$(common_overrides)|seed=${seed}|max_train_steps=${target_steps}|${extra}"

  python meta_train_score.py \
    -mc "${MODEL_CONFIG}" \
    -dc "${DATA_CONFIG}" \
    -l "${dst_log_dir}" \
    -o "$(score_overrides)|seed=${seed}|max_train_steps=${target_steps}|${extra}"
}

case "$1" in
  run-one)
    method="${2:-}"
    seed="${3:-0}"
    if [ -z "${method}" ]; then
      usage
      exit 1
    fi
    run_one "${method}" "${seed}" 12000
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

  resume-last2-pair)
    seed="${2:-0}"
    target_steps="${3:-${TARGET_STEPS}}"
    resume_one baseline_pf_last2 "${seed}" "${target_steps}"
    resume_one pf_last2_corr_gate "${seed}" "${target_steps}"
    ;;

  resume-last2-ablation)
    seed="${2:-0}"
    target_steps="${3:-${TARGET_STEPS}}"
    resume_one baseline_pf_last2 "${seed}" "${target_steps}"
    resume_one pf_last2_cor_only "${seed}" "${target_steps}"
    resume_one pf_last2_gate_only "${seed}" "${target_steps}"
    resume_one pf_last2_corr_gate "${seed}" "${target_steps}"
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
