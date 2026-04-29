#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import torch
import yaml
from tensorboard.backend.event_processing import event_accumulator


def load_yaml(path: Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_meta_train_scores(log_dir: Path):
    score_path = log_dir / 'meta_train_scores.pt'
    if not score_path.exists():
        return {}
    data = torch.load(score_path, map_location='cpu')
    return data if isinstance(data, dict) else {}


def load_trainable_summary(log_dir: Path):
    summary_path = log_dir / 'trainable_summary.yaml'
    if not summary_path.exists():
        return {}
    with open(summary_path, 'r') as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_last_scalar_from_events(log_dir: Path, tag: str):
    event_paths = sorted(log_dir.glob('events.out.tfevents.*'))
    if not event_paths:
        return float('nan')
    latest = event_paths[-1]
    try:
        ea = event_accumulator.EventAccumulator(
            str(latest),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()
        if tag not in ea.Tags().get('scalars', []):
            return float('nan')
        events = ea.Scalars(tag)
        if len(events) == 0:
            return float('nan')
        return float(events[-1].value)
    except Exception:
        return float('nan')


def infer_method_name(exp_name: str):
    known = [
        'full_outer_loop_train',
        'head_only',
        'head_last1',
        'head_last2',
        'head_last2_correction',
        'head_last2_gate',
        'head_last2_gate_correction',
        'oml',
        'anml',
    ]
    for m in known:
        if exp_name.startswith(m + '_') or exp_name == m:
            return m
    return 'unknown'


def collect_rows(run_root: Path):
    rows = []
    for dataset_dir in sorted(run_root.glob('*')):
        if not dataset_dir.is_dir():
            continue
        for log_dir in sorted(dataset_dir.glob('*')):
            if not log_dir.is_dir():
                continue
            cfg_path = log_dir / 'config.yaml'
            if not cfg_path.exists():
                continue

            cfg = load_yaml(cfg_path)
            scores = load_meta_train_scores(log_dir)
            trainable = load_trainable_summary(log_dir)
            loss_train = scores.get('loss/train', float('nan'))
            loss_test = load_last_scalar_from_events(log_dir, 'loss/test')
            loss_gap = loss_test - loss_train if math.isfinite(loss_train) and math.isfinite(loss_test) else float('nan')

            rows.append({
                'dataset': dataset_dir.name,
                'experiment_name': log_dir.name,
                'method_name': infer_method_name(log_dir.name),
                'seed': cfg.get('seed', ''),
                'max_train_steps': cfg.get('max_train_steps', ''),
                'loss/train': loss_train,
                'loss/test': loss_test,
                'meta_overfitting_gap(loss_test-loss_train)': loss_gap,
                'trainable_params': trainable.get('trainable_params', ''),
                'target_trainable_params': trainable.get('target_trainable_params', ''),
                'trainable_param_gap': trainable.get('trainable_param_gap', ''),
                'partial_freeze_mode': cfg.get('partial_freeze_mode', ''),
                'random_match_seed': cfg.get('random_match_seed', ''),
                'log_dir': str(log_dir),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', default='results/regression_9alg.csv')
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = collect_rows(run_root)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'dataset', 'experiment_name', 'method_name', 'seed', 'max_train_steps',
        'loss/train', 'loss/test', 'meta_overfitting_gap(loss_test-loss_train)',
        'trainable_params', 'target_trainable_params', 'trainable_param_gap',
        'partial_freeze_mode', 'random_match_seed', 'log_dir',
    ]
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Collected {len(rows)} runs -> {output}')


if __name__ == '__main__':
    main()
