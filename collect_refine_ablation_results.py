#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import yaml
import torch


def load_yaml(path: Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_meta_train_scores(log_dir: Path):
    score_path = log_dir / 'meta_train_scores.pt'
    if not score_path.exists():
        return {}
    data = torch.load(score_path, map_location='cpu')
    return data if isinstance(data, dict) else {}


def load_analysis_summary(run_name: str):
    summary_path = Path('analysis_outputs') / run_name / 'summary.csv'
    if not summary_path.exists():
        return {}

    rows = []
    with open(summary_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    def mean_of(col):
        vals = []
        for r in rows:
            v = r.get(col, '')
            if v in ('', None):
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            if math.isfinite(fv):
                vals.append(fv)
        return sum(vals) / len(vals) if vals else float('nan')

    return {
        'mean_gate': mean_of('mean_gate'),
        'mean_gate_train': mean_of('mean_gate_train'),
        'mean_gate_test': mean_of('mean_gate_test'),
        'mean_delta_norm': mean_of('mean_delta_norm'),
        'mean_h_minus_prev_norm': mean_of('mean_h_minus_prev_norm'),
    }


def infer_method_name(exp_name: str):
    if exp_name.startswith('baseline_'):
        return 'baseline'
    if exp_name.startswith('correction_only_'):
        return 'correction_only'
    if exp_name.startswith('gate_only_'):
        return 'gate_only'
    if exp_name.startswith('corr_gate_'):
        return 'correction_plus_gate'
    return 'unknown'


def collect_rows(run_root: Path):
    rows = []
    for log_dir in sorted(run_root.glob('*')):
        if not log_dir.is_dir():
            continue
        cfg_path = log_dir / 'config.yaml'
        if not cfg_path.exists():
            continue

        cfg = load_yaml(cfg_path)
        scores = load_meta_train_scores(log_dir)
        analysis = load_analysis_summary(log_dir.name)

        rows.append({
            'experiment_name': log_dir.name,
            'method_name': infer_method_name(log_dir.name),
            'seed': cfg.get('seed', ''),
            'acc/train': scores.get('acc/train', float('nan')),
            'loss/train': scores.get('loss/train', float('nan')),
            'max_train_steps': cfg.get('max_train_steps', ''),
            'freeze_backbone': cfg.get('freeze_backbone', ''),
            'use_correction': cfg.get('enable_correction', ''),
            'use_highway_gate': cfg.get('enable_highway_gate', ''),
            'gate_bias_init': cfg.get('gate_bias_init', ''),
            'mean_gate': analysis.get('mean_gate', float('nan')),
            'mean_gate_train': analysis.get('mean_gate_train', float('nan')),
            'mean_gate_test': analysis.get('mean_gate_test', float('nan')),
            'mean_delta_norm': analysis.get('mean_delta_norm', float('nan')),
            'mean_h_minus_prev_norm': analysis.get('mean_h_minus_prev_norm', float('nan')),
            'log_dir': str(log_dir),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', default='results/refine_ablation_12k.csv')
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = collect_rows(run_root)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'experiment_name', 'method_name', 'seed',
        'acc/train', 'loss/train',
        'max_train_steps', 'freeze_backbone', 'use_correction', 'use_highway_gate', 'gate_bias_init',
        'mean_gate', 'mean_gate_train', 'mean_gate_test', 'mean_delta_norm', 'mean_h_minus_prev_norm',
        'log_dir',
    ]
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Collected {len(rows)} runs -> {output}')


if __name__ == '__main__':
    main()