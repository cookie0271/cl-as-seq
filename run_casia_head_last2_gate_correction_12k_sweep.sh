#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep for CASIA:
# method = head + last2 layers + gate + correction
# fixed steps = 12k (default)

MODEL_CONFIG=${MODEL_CONFIG:-cfg/model/tf.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/casia.yaml}
RUN_ROOT=${RUN_ROOT:-runs/casia_head_last2_gate_correction_12k_sweep}
SUMMARY_CSV=${SUMMARY_CSV:-results/casia_head_last2_gate_correction_12k_sweep_summary.csv}
TARGET_STEPS=${TARGET_STEPS:-12000}
SEED=${SEED:-0}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

TASKS=${TASKS:-20}
TRAIN_SHOTS=${TRAIN_SHOTS:-5}
TEST_SHOTS=${TEST_SHOTS:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
EVAL_ITERS=${EVAL_ITERS:-32}
NUM_WORKERS=${NUM_WORKERS:-8}

# Sweep grid (can be overridden by env vars).
# Use decimal literals by default to avoid string parsing issues in overrides.
LRS=${LRS:-"0.00005 0.0001 0.0002"}
GATE_BIAS_INITS=${GATE_BIAS_INITS:-"-2.5 -2.0 -1.5 -1.0 -0.5"}
GATE_DROPOUTS=${GATE_DROPOUTS:-"0 0.1"}
CORR_DROPOUTS=${CORR_DROPOUTS:-"0 0.1"}

usage() {
  cat <<USAGE
Usage:
  $0 run-one <lr> <gate_bias_init> <gate_dropout> <corr_dropout> [seed]
  $0 run-grid [seed]
  $0 summarize

Environment (optional):
  RUN_ROOT, SUMMARY_CSV, TARGET_STEPS, CUDA_VISIBLE_DEVICES
  TASKS, TRAIN_SHOTS, TEST_SHOTS, BATCH_SIZE, EVAL_BATCH_SIZE, EVAL_ITERS, NUM_WORKERS
  LRS, GATE_BIAS_INITS, GATE_DROPOUTS, CORR_DROPOUTS
USAGE
}

common_overrides() {
  local lr="$1"
  local gate_bias_init="$2"
  local gate_dropout="$3"
  local corr_dropout="$4"
  local seed="$5"
  local lr_num
  lr_num="$(python - <<PY
v = float(${lr@Q})
print(v)
PY
)"

  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${TARGET_STEPS}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|num_workers=${NUM_WORKERS}|summary_interval=250|eval_interval=1000|ckpt_interval=1000|seed=${seed}|optim_args.lr=${lr_num}|gate_bias_init=${gate_bias_init}|gate_dropout=${gate_dropout}|corr_dropout=${corr_dropout}|freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=True|enable_highway_gate=True|train_correction=True|train_gate=True|train_head=True|dataloader_pin_memory=True|dataloader_persistent_workers=True|dataloader_prefetch_factor=4|transfer_non_blocking=True|cudnn_benchmark=True"
}

score_overrides() {
  local lr="$1"
  local gate_bias_init="$2"
  local gate_dropout="$3"
  local corr_dropout="$4"
  local seed="$5"
  local lr_num
  lr_num="$(python - <<PY
v = float(${lr@Q})
print(v)
PY
)"

  echo "tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${TARGET_STEPS}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|eval_iters=${EVAL_ITERS}|num_workers=${NUM_WORKERS}|seed=${seed}|optim_args.lr=${lr_num}|gate_bias_init=${gate_bias_init}|gate_dropout=${gate_dropout}|corr_dropout=${corr_dropout}|freeze_backbone=True|partial_freeze_mode=last_n_tf|train_last_tf_layers=2|freeze_encoder=True|enable_correction=True|enable_highway_gate=True|train_correction=True|train_gate=True|train_head=True|dataloader_pin_memory=True|dataloader_persistent_workers=True|dataloader_prefetch_factor=4|transfer_non_blocking=True|cudnn_benchmark=True|dataloader_pin_memory=True|dataloader_persistent_workers=True|dataloader_prefetch_factor=4|transfer_non_blocking=True|cudnn_benchmark=True"
}

exp_name() {
  local lr="$1"
  local gate_bias_init="$2"
  local gate_dropout="$3"
  local corr_dropout="$4"
  local seed="$5"

  echo "h2gc_lr${lr}_gb${gate_bias_init}_gd${gate_dropout}_cd${corr_dropout}_s${seed}" | tr ' ' '_' | tr '.' 'p' | tr '-' 'm'
}

run_one() {
  local lr="$1"
  local gate_bias_init="$2"
  local gate_dropout="$3"
  local corr_dropout="$4"
  local seed="${5:-${SEED}}"

  local name
  name="$(exp_name "${lr}" "${gate_bias_init}" "${gate_dropout}" "${corr_dropout}" "${seed}")"
  local log_dir="${RUN_ROOT}/${name}"

  echo "=== sweep lr=${lr} gate_bias_init=${gate_bias_init} gate_dropout=${gate_dropout} corr_dropout=${corr_dropout} seed=${seed} ==="

  python train.py -mc "${MODEL_CONFIG}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "$(common_overrides "${lr}" "${gate_bias_init}" "${gate_dropout}" "${corr_dropout}" "${seed}")"
  python meta_train_score.py -mc "${MODEL_CONFIG}" -dc "${DATA_CONFIG}" -l "${log_dir}" -o "$(score_overrides "${lr}" "${gate_bias_init}" "${gate_dropout}" "${corr_dropout}" "${seed}")"
}

run_grid() {
  local seed="${1:-${SEED}}"
  for lr in ${LRS}; do
    for gb in ${GATE_BIAS_INITS}; do
      for gd in ${GATE_DROPOUTS}; do
        for cd in ${CORR_DROPOUTS}; do
          run_one "${lr}" "${gb}" "${gd}" "${cd}" "${seed}"
        done
      done
    done
  done
}

summarize() {
  mkdir -p "$(dirname "${SUMMARY_CSV}")"
  python - <<PY
import csv
from pathlib import Path
import torch

run_root = Path(${RUN_ROOT@Q})
out_csv = Path(${SUMMARY_CSV@Q})
rows = []

for d in sorted(run_root.glob('h2gc_*')):
    score_path = d / 'meta_train_scores.pt'
    if not score_path.exists():
        continue
    scores = torch.load(score_path, map_location='cpu')
    acc_train = float(scores.get('acc/train', float('nan')))
    loss_train = float(scores.get('loss/train', float('nan')))
    rows.append({
        'run': d.name,
        'acc_train': acc_train,
        'meta_train_error': 1.0 - acc_train,
        'loss_train': loss_train,
    })

rows.sort(key=lambda x: (x['meta_train_error'], x['loss_train']))
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['run', 'acc_train', 'meta_train_error', 'loss_train'])
    w.writeheader()
    w.writerows(rows)

print(f'Saved summary: {out_csv}')
if rows:
    print('Top 10 by meta_train_error:')
    for r in rows[:10]:
        print(r)
PY
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

case "$1" in
  run-one)
    if [ $# -lt 5 ]; then
      usage
      exit 1
    fi
    run_one "$2" "$3" "$4" "$5" "${6:-${SEED}}"
    ;;
  run-grid)
    run_grid "${2:-${SEED}}"
    ;;
  summarize)
    summarize
    ;;
  *)
    usage
    exit 1
    ;;
esac
