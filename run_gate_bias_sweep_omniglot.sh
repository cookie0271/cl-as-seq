#!/usr/bin/env bash
set -euo pipefail

# Omniglot small-scale experiment plan for:
# A) baseline
# B) correction only
# C) gate only
# D) correction+gate with gate_bias sweep
# + OML / ANML (if available in repo)

MODEL_LINEAR=${MODEL_LINEAR:-cfg/model/linear_tf.yaml}
MODEL_OML=${MODEL_OML:-cfg/model/oml.yaml}
MODEL_ANML=${MODEL_ANML:-cfg/model/anml.yaml}
DATA_CONFIG=${DATA_CONFIG:-cfg/data/omniglot.yaml}
RUN_ROOT=${RUN_ROOT:-runs/exp_gate_bias_omniglot_small}
RESULTS_CSV=${RESULTS_CSV:-results/gate_bias_sweep_omniglot.csv}

# Small but informative defaults
TASKS=${TASKS:-20}
TRAIN_SHOTS=${TRAIN_SHOTS:-5}
TEST_SHOTS=${TEST_SHOTS:-1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
SUMMARY_INTERVAL=${SUMMARY_INTERVAL:-100}
CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
EVAL_ITERS_TRAIN=${EVAL_ITERS_TRAIN:-5}
EVAL_ITERS_SCORE=${EVAL_ITERS_SCORE:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-0}
SEEDS=${SEEDS:-"0 1 2"}
GATE_BIASES=${GATE_BIASES:-"-3.0 -2.5 -2.0 -1.0"}

# Use single GPU by default for stability and comparability
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

COMMON="tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${MAX_TRAIN_STEPS}|eval_interval=${EVAL_INTERVAL}|summary_interval=${SUMMARY_INTERVAL}|ckpt_interval=${CKPT_INTERVAL}|eval_iters=${EVAL_ITERS_TRAIN}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|num_workers=${NUM_WORKERS}|analysis_export.enable=True|analysis_export.max_batches=5|analysis_export.output_dir=analysis_outputs"
SCORE_COMMON="tasks=${TASKS}|train_shots=${TRAIN_SHOTS}|test_shots=${TEST_SHOTS}|max_train_steps=${MAX_TRAIN_STEPS}|eval_iters=${EVAL_ITERS_SCORE}|batch_size=${BATCH_SIZE}|eval_batch_size=${EVAL_BATCH_SIZE}|num_workers=${NUM_WORKERS}|analysis_export.enable=True|analysis_export.max_batches=5|analysis_export.output_dir=analysis_outputs"

run_pair () {
  local name=$1
  local model_cfg=$2
  local extra=$3
  local seed=$4
  local log_dir="${RUN_ROOT}/${name}_seed${seed}"

  echo "=== ${name} seed=${seed} ==="
  python train.py \
    -mc "${model_cfg}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "${COMMON}|seed=${seed}|${extra}"

  python meta_train_score.py \
    -mc "${model_cfg}" \
    -dc "${DATA_CONFIG}" \
    -l "${log_dir}" \
    -o "${SCORE_COMMON}|seed=${seed}|${extra}"
}

for seed in ${SEEDS}; do
  # A) baseline (no correction, no gate)
  run_pair "linear_baseline" "${MODEL_LINEAR}" "enable_correction=False|enable_highway_gate=False" "${seed}"

  # B) correction only
  run_pair "linear_correction_only" "${MODEL_LINEAR}" "enable_correction=True|enable_highway_gate=False" "${seed}"

  # C) gate only
  run_pair "linear_gate_only" "${MODEL_LINEAR}" "enable_correction=False|enable_highway_gate=True|gate_bias_init=-3.0" "${seed}"

  # D) correction + gate sweep
  for bias in ${GATE_BIASES}; do
    bias_tag=$(echo "${bias}" | sed 's/-/m/g; s/\./p/g')
    run_pair "linear_ours_bias${bias_tag}" "${MODEL_LINEAR}" "enable_correction=True|enable_highway_gate=True|gate_bias_init=${bias}" "${seed}"
  done

  # OML / ANML baselines (repository-provided)
  run_pair "oml" "${MODEL_OML}" "" "${seed}"
  run_pair "anml" "${MODEL_ANML}" "" "${seed}"
done

python scripts/collect_gate_bias_results.py \
  --run-root "${RUN_ROOT}" \
  --output "${RESULTS_CSV}"

echo "Saved: ${RESULTS_CSV}"