#!/usr/bin/env bash
set -euo pipefail

TF_MODEL_CONFIG=${TF_MODEL_CONFIG:-cfg/model/tf.yaml}
OML_MODEL_CONFIG=${OML_MODEL_CONFIG:-cfg/model/oml.yaml}
ANML_MODEL_CONFIG=${ANML_MODEL_CONFIG:-cfg/model/anml.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/cifar100.yaml}

RUN_ROOT=${RUN_ROOT:-runs/cifar100_meta_train_scale}
RESULTS_CSV=${RESULTS_CSV:-results/cifar100_meta_train_scale.csv}

META_TRAIN_SCALES=${META_TRAIN_SCALES:-"20 40 60 80"}
TARGET_STEPS=${TARGET_STEPS:-50000}
SEEDS=${SEEDS:-"0"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

TASKS=${TASKS:-20}
TRAIN_SHOTS=${TRAIN_SHOTS:-5}
TEST_SHOTS=${TEST_SHOTS:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
EVAL_ITERS=${EVAL_ITERS:-32}
NUM_WORKERS=${NUM_WORKERS:-16}
LR=${LR:-0.0001}
NUM_BITES=${NUM_BITES:-1}

# OML / ANML use higher-order gradients in the inner loop and are usually more
# memory-sensitive than transformer baselines. We keep per-method knobs so OML
# can be pushed harder (better utilization) without forcing ANML to OOM.
# Backward-compat: if old OML_ANML_* variables are set, they are used as
# fallbacks.
OML_BATCH_SIZE=${OML_BATCH_SIZE:-${OML_ANML_BATCH_SIZE:-128}}
OML_EVAL_BATCH_SIZE=${OML_EVAL_BATCH_SIZE:-${OML_ANML_EVAL_BATCH_SIZE:-64}}
OML_NUM_BITES=${OML_NUM_BITES:-${OML_ANML_NUM_BITES:-2}}

ANML_BATCH_SIZE=${ANML_BATCH_SIZE:-${OML_ANML_BATCH_SIZE:-64}}
ANML_EVAL_BATCH_SIZE=${ANML_EVAL_BATCH_SIZE:-${OML_ANML_EVAL_BATCH_SIZE:-32}}
ANML_NUM_BITES=${ANML_NUM_BITES:-${OML_ANML_NUM_BITES:-4}}

META_TEST_TASKS=${META_TEST_TASKS:-20}
CIFAR_CLASS_SPLIT_SEED=${CIFAR_CLASS_SPLIT_SEED:-0}
META_TRAIN_NESTED_CLASS_POOL=${META_TRAIN_NESTED_CLASS_POOL:-True}

GATE_BIAS_INIT=${GATE_BIAS_INIT:--1.0}
GATE_DROPOUT=${GATE_DROPOUT:-0.1}
CORR_DROPOUT=${CORR_DROPOUT:-0.1}



usage() {
  cat <<USAGE
Usage:
  $0 run-one <method> <meta_train_num_classes> [seed] [max_train_steps]
  $0 run-scale <method> [seed] [max_train_steps]
  $0 run-nested-growth <method> [seed] [max_train_steps]
  $0 run-all-methods <meta_train_num_classes> [seed] [max_train_steps]
  $0 run-all [seed] [max_train_steps]
  $0 eval-one <log_dir>
  $0 collect

Methods:
  full_train
  last2_only
  last2_corr_gate
  adapter_standard
  lora_standard
  oml
  anml
USAGE
}

methods=(
  full_train
  last2_only
  last2_corr_gate
  adapter_standard
  lora_standard
  oml
  anml
)

compute_default_meta_test_ids() {
  python - <<PY
import random
seed = int("${CIFAR_CLASS_SPLIT_SEED}")
n_test = int("${META_TEST_TASKS}")
classes = list(range(100))
rng = random.Random(seed)
rng.shuffle(classes)
print(classes[:n_test])
PY
}

META_TEST_CLASS_IDS=${META_TEST_CLASS_IDS:-$(compute_default_meta_test_ids)}

print_nested_split_plan() {
  local meta_test_ids="$1"
  local scales="$2"
  python - <<PY
meta_test = set(${meta_test_ids})
scales = [int(x) for x in "${scales}".split()]
candidate_pool = [c for c in range(100) if c not in meta_test]
print(f"fixed_meta_test_size={len(meta_test)}")
print(f"meta_train_candidate_pool_size={len(candidate_pool)}")
print(f"meta_train_candidate_pool={candidate_pool}")
for n in scales:
    print(f"nested_train_{n}={candidate_pool[:n]}")
PY
}

common_overrides() {
  local target_steps="$1"
  local seed="$2"
  local meta_train_num_classes="$3"
  echo "dataset=cifar100|tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|num_bites=${NUM_BITES}|test_shots=${TEST_SHOTS}|meta_test_tasks=${META_TEST_TASKS}|meta_test_class_ids=${META_TEST_CLASS_IDS}|cifar100_class_split_seed=${CIFAR_CLASS_SPLIT_SEED}|meta_train_nested_class_pool=${META_TRAIN_NESTED_CLASS_POOL}|meta_train_num_classes=${meta_train_num_classes}|max_train_steps=${target_steps}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|num_workers=${NUM_WORKERS}|summary_interval=250|eval_interval=1000|ckpt_interval=1000|optim_args.lr=${LR}|gate_bias_init=${GATE_BIAS_INIT}|gate_dropout=${GATE_DROPOUT}|corr_dropout=${CORR_DROPOUT}|dataloader_pin_memory=True|dataloader_persistent_workers=True|dataloader_prefetch_factor=4|transfer_non_blocking=True|cudnn_benchmark=True|seed=${seed}"
}

method_settings() {
  local method="$1"
  local seed="$2"
  local mtc="$3"
  local model_config=""
  local extra=""
  local exp_name="${method}_mtrain${mtc}_seed${seed}"

  case "${method}" in
    full_train)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=False|partial_freeze_mode=none|train_last_tf_layers=0|freeze_encoder=False|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|enable_adapter=False|train_adapter=False|enable_lora=False|train_lora=False|train_head=True"
      ;;
    last2_only)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|enable_adapter=False|train_adapter=False|enable_lora=False|train_lora=False|train_head=True"
      ;;
    last2_corr_gate)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=True|enable_highway_gate=True|train_correction=True|train_gate=True|gate_bias_init=${GATE_BIAS_INIT}|enable_adapter=False|train_adapter=False|enable_lora=False|train_lora=False|train_head=True"
      ;;
    adapter_standard)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=True|partial_freeze_mode=adapter_standard|train_last_tf_layers=0|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|enable_adapter=True|train_adapter=True|adapter_layers=all|adapter_location=post_layer|adapter_strict_match=False|adapter_dim_candidates=[16,32,64,128,256,512]|adapter_default_dim=64|enable_lora=False|train_lora=False|train_head=True"
      ;;
    lora_standard)
      model_config="${TF_MODEL_CONFIG}"
      extra="freeze_backbone=True|partial_freeze_mode=lora_standard|train_last_tf_layers=0|freeze_encoder=True|enable_correction=False|enable_highway_gate=False|train_correction=False|train_gate=False|enable_adapter=False|train_adapter=False|enable_lora=True|train_lora=True|lora_layers=all|lora_target_modules=attn_only|lora_rank_candidates=[4,8,16,32,64]|lora_default_rank=16|lora_alpha=16|lora_dropout=0.0|lora_strict_match=False|train_head=True"
      ;;
    oml)
      model_config="${OML_MODEL_CONFIG}"
      extra="batch_size=${OML_BATCH_SIZE}|eval_batch_size=${OML_EVAL_BATCH_SIZE}|num_bites=${OML_NUM_BITES}"
      ;;
    anml)
      model_config="${ANML_MODEL_CONFIG}"
      extra="batch_size=${ANML_BATCH_SIZE}|eval_batch_size=${ANML_EVAL_BATCH_SIZE}|num_bites=${ANML_NUM_BITES}"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      exit 1
      ;;
  esac

  printf '%s\n%s\n%s\n' "${exp_name}" "${model_config}" "${extra}"
}

run_one() {
  local method="$1"
  local mtc="$2"
  local seed="$3"
  local target_steps="$4"

  local settings exp_name model_config extra
  settings="$(method_settings "${method}" "${seed}" "${mtc}")"
  exp_name="$(echo "${settings}" | sed -n '1p')"
  model_config="$(echo "${settings}" | sed -n '2p')"
  extra="$(echo "${settings}" | sed -n '3p')"

  local log_dir="${RUN_ROOT}/${exp_name}"
  local override
  override="$(common_overrides "${target_steps}" "${seed}" "${mtc}")"
  if [ -n "${extra}" ]; then
    override="${override}|${extra}"
  fi

  echo "=== controlled meta-train scale run start ==="
  echo "method=${method}"
  echo "meta_train_num_classes=${mtc}"
  echo "meta_test_class_ids=${META_TEST_CLASS_IDS}"
  echo "max_train_steps=${target_steps}"
  echo "seed=${seed}"
  echo "model_config=${model_config}"
  echo "extra=${extra}"
  echo "log_dir=${log_dir}"

  local resume_flag=""
  local latest_step=""
  latest_step="$(python - <<PY
from pathlib import Path
import re
log_dir = Path("${log_dir}")
steps = []
if log_dir.exists():
    for p in log_dir.glob("ckpt-*.pt"):
        m = re.match(r"ckpt-(\\d+)\\.pt$", p.name)
        if m:
            steps.append(int(m.group(1)))
print(max(steps) if steps else "")
PY
)"
  if [ -n "${latest_step}" ]; then
    echo "found_latest_checkpoint_step=${latest_step}"
    if [ "${latest_step}" -lt "${target_steps}" ]; then
      echo "resume_training_from=${latest_step} -> ${target_steps}"
      resume_flag="--resume"
    else
      echo "checkpoint already reached target_steps=${target_steps}, skip training."
    fi
  fi

  if [ -z "${latest_step}" ] || [ "${latest_step}" -lt "${target_steps}" ]; then
    python train.py -mc "${model_config}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "${override}" ${resume_flag}
  fi
  python meta_train_score.py -mc "${model_config}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "${override}"
}

run_scale() {
  local method="$1"
  local seed="$2"
  local target_steps="$3"
  for mtc in ${META_TRAIN_SCALES}; do
    run_one "${method}" "${mtc}" "${seed}" "${target_steps}"
  done
}

run_nested_growth() {
  local method="$1"
  local seed="$2"
  local target_steps="$3"
  local fixed_meta_test_tasks=20
  local fixed_scales="20 40 60 80"

  if [ "${META_TEST_TASKS}" != "${fixed_meta_test_tasks}" ]; then
    echo "Warning: forcing META_TEST_TASKS=${fixed_meta_test_tasks} for nested-growth mode (was ${META_TEST_TASKS})."
  fi
  META_TEST_TASKS="${fixed_meta_test_tasks}"
  META_TRAIN_SCALES="${fixed_scales}"
  META_TEST_CLASS_IDS="$(compute_default_meta_test_ids)"

  echo "=== nested-growth mode (single method ladder) ==="
  echo "method=${method}"
  echo "fixed_meta_test_tasks=${META_TEST_TASKS}"
  echo "fixed_meta_test_class_ids=${META_TEST_CLASS_IDS}"
  echo "meta_train_scales=${META_TRAIN_SCALES}"
  print_nested_split_plan "${META_TEST_CLASS_IDS}" "${META_TRAIN_SCALES}"

  run_scale "${method}" "${seed}" "${target_steps}"
}

run_all_methods() {
  local mtc="$1"
  local seed="$2"
  local target_steps="$3"
  for m in "${methods[@]}"; do
    run_one "${m}" "${mtc}" "${seed}" "${target_steps}"
  done
}

run_all() {
  local seed="$1"
  local target_steps="$2"
  for mtc in ${META_TRAIN_SCALES}; do
    run_all_methods "${mtc}" "${seed}" "${target_steps}"
  done
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

case "$1" in
  run-one)
    method="${2:-}"
    mtc="${3:-}"
    seed="${4:-0}"
    target_steps="${5:-${TARGET_STEPS}}"
    if [ -z "${method}" ] || [ -z "${mtc}" ]; then
      usage
      exit 1
    fi
    run_one "${method}" "${mtc}" "${seed}" "${target_steps}"
    ;;
  run-scale)
    method="${2:-}"
    seed="${3:-0}"
    target_steps="${4:-${TARGET_STEPS}}"
    if [ -z "${method}" ]; then
      usage
      exit 1
    fi
    run_scale "${method}" "${seed}" "${target_steps}"
    ;;
  run-nested-growth)
    method="${2:-}"
    seed="${3:-0}"
    target_steps="${4:-${TARGET_STEPS}}"
    if [ -z "${method}" ]; then
      usage
      exit 1
    fi
    run_nested_growth "${method}" "${seed}" "${target_steps}"
    ;;
  run-all-methods)
    mtc="${2:-}"
    seed="${3:-0}"
    target_steps="${4:-${TARGET_STEPS}}"
    if [ -z "${mtc}" ]; then
      usage
      exit 1
    fi
    run_all_methods "${mtc}" "${seed}" "${target_steps}"
    ;;
  run-all)
    seed="${2:-0}"
    target_steps="${3:-${TARGET_STEPS}}"
    run_all "${seed}" "${target_steps}"
    ;;
  eval-one)
    log_dir="${2:-}"
    if [ -z "${log_dir}" ]; then
      usage
      exit 1
    fi
    python meta_train_score.py -mc "${log_dir}/config.yaml" -dc "${log_dir}/config.yaml" -l "${log_dir}" -o ""
    ;;
  collect)
    python collect_refine_ablation_results.py --run-root "${RUN_ROOT}" --output "${RESULTS_CSV}" --experiment-tag "cifar100_meta_train_scale"
    ;;
  *)
    usage
    exit 1
    ;;
esac