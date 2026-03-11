#!/usr/bin/env bash
set -euo pipefail

# Multi-seed ON/OFF comparison for correction+highway gate.
# Usage:
#   bash multi_seed_gate_compare.sh
#   SEEDS="0 1 2 3 4" STEPS=500 bash multi_seed_gate_compare.sh

MODEL_CONFIG=${MODEL_CONFIG:-cfg/model/linear_tf.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/omniglot.yaml}
RUN_PREFIX=${RUN_PREFIX:-runs/gate_multiseed}
SEEDS=${SEEDS:-"0 1 2 3 4"}

# Default to single-GPU for stability/reproducibility in quick ablations.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

# Fast defaults (override via env)
TASKS=${TASKS:-5}
TRAIN_SHOTS=${TRAIN_SHOTS:-1}
TEST_SHOTS=${TEST_SHOTS:-1}
STEPS=${STEPS:-200}
SUMMARY_INTERVAL=${SUMMARY_INTERVAL:-20}
EVAL_ITERS_TRAIN=${EVAL_ITERS_TRAIN:-4}
EVAL_ITERS_SCORE=${EVAL_ITERS_SCORE:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-0}
TF_LAYERS=${TF_LAYERS:-1}
HIDDEN_DIM=${HIDDEN_DIM:-128}
TF_HEADS=${TF_HEADS:-4}
TF_FF_DIM=${TF_FF_DIM:-256}

COMMON_OVERRIDE="tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${STEPS}|eval_interval=${STEPS}|summary_interval=${SUMMARY_INTERVAL}|ckpt_interval=${STEPS}|eval_iters=${EVAL_ITERS_TRAIN}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|num_workers=${NUM_WORKERS}|tf_layers=${TF_LAYERS}|hidden_dim=${HIDDEN_DIM}|tf_heads=${TF_HEADS}|tf_ff_dim=${TF_FF_DIM}|attn_loss=0.0|distributed_loss=False"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (set CUDA_VISIBLE_DEVICES to override)"

SCORE_OVERRIDE="tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${STEPS}|eval_iters=${EVAL_ITERS_SCORE}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|num_workers=${NUM_WORKERS}|tf_layers=${TF_LAYERS}|hidden_dim=${HIDDEN_DIM}|tf_heads=${TF_HEADS}|tf_ff_dim=${TF_FF_DIM}|attn_loss=0.0|distributed_loss=False"

for seed in ${SEEDS}; do
  for mode in on off; do
    if [[ "${mode}" == "on" ]]; then
      EXTRA="enable_correction=True|enable_highway_gate=True|seed=${seed}"
    else
      EXTRA="enable_correction=False|enable_highway_gate=False|seed=${seed}"
    fi

    LOG_DIR="${RUN_PREFIX}_${mode}_seed${seed}"
    echo "=== [${mode}] seed=${seed} log_dir=${LOG_DIR} ==="

    python train.py \
      -mc "${MODEL_CONFIG}" \
      -dc "${DATA_CONFIG}" \
      -l "${LOG_DIR}" \
      -o "${COMMON_OVERRIDE}|${EXTRA}"

    python meta_train_score.py \
      -mc "${MODEL_CONFIG}" \
      -dc "${DATA_CONFIG}" \
      -l "${LOG_DIR}" \
      -o "${SCORE_OVERRIDE}|${EXTRA}"
  done
done

python - <<'PY'
import os
import torch
import statistics as stats

run_prefix = os.environ.get('RUN_PREFIX', 'runs/gate_multiseed')
seeds = [int(s) for s in os.environ.get('SEEDS', '0 1 2 3 4').split()]

rows = []
for seed in seeds:
    on_path = f"{run_prefix}_on_seed{seed}/meta_train_scores.pt"
    off_path = f"{run_prefix}_off_seed{seed}/meta_train_scores.pt"
    on = torch.load(on_path)
    off = torch.load(off_path)
    on_acc = float(on['acc/train'])
    off_acc = float(off['acc/train'])
    delta = on_acc - off_acc
    rows.append((seed, on_acc, off_acc, delta))

print('\n==== Multi-seed gate comparison (acc/train) ====')
for seed, on_acc, off_acc, delta in rows:
    print(f'seed={seed:>2} | on={on_acc:.6f} | off={off_acc:.6f} | delta={delta:+.6f}')

on_vals = [r[1] for r in rows]
off_vals = [r[2] for r in rows]
deltas = [r[3] for r in rows]

print('-----------------------------------------------')
print(f'ON  mean={stats.mean(on_vals):.6f} std={stats.pstdev(on_vals):.6f}')
print(f'OFF mean={stats.mean(off_vals):.6f} std={stats.pstdev(off_vals):.6f}')
print(f'DELTA mean={stats.mean(deltas):+.6f} std={stats.pstdev(deltas):.6f}')
print(f'Positive-delta seeds: {sum(d > 0 for d in deltas)}/{len(deltas)}')
PY
