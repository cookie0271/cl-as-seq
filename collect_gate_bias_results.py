#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

try:
    import torch
except Exception:
    torch = None




def _parse_scalar(text):
    t = text.strip()
    if t.lower() in {'true', 'false'}:
        return t.lower() == 'true'
    try:
        if '.' in t:
            return float(t)
        return int(t)
    except ValueError:
        return t


def load_config(cfg_path: Path):
    if yaml is not None:
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)

    # lightweight fallback parser for top-level "key: value" pairs
    cfg = {}
    with open(cfg_path, 'r') as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if line.startswith(' ') or line.startswith('	'):
                continue
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            cfg[k.strip()] = _parse_scalar(v)
    return cfg

def load_meta_train_scores(log_dir: Path):
    pt = log_dir / 'meta_train_scores.pt'
    if not pt.exists() or torch is None:
        return {}
    try:
        data = torch.load(str(pt), map_location='cpu')
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def load_analysis_summary(log_dir: Path):
    run_name = log_dir.name
    summary_path = Path('analysis_outputs') / run_name / 'summary.csv'
    if not summary_path.exists():
        return {}

    rows = []
    with open(summary_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {}

    def mean_of(col):
        vals = []
        for r in rows:
            v = r.get(col, '')
            if v is None or v == '':
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
        'mean_delta_norm': mean_of('mean_delta_norm'),
        'mean_h_minus_prev_norm': mean_of('mean_h_minus_prev_norm'),
        'mean_gate_train': mean_of('mean_gate_train'),
        'mean_gate_test': mean_of('mean_gate_test'),
    }


def infer_method_name(exp_name: str):
    if exp_name.startswith('linear_baseline'):
        return 'baseline'
    if exp_name.startswith('linear_correction_only'):
        return 'correction_only'
    if exp_name.startswith('linear_gate_only'):
        return 'gate_only'
    if exp_name.startswith('linear_ours'):
        return 'ours'
    if exp_name.startswith('oml'):
        return 'OML'
    if exp_name.startswith('anml'):
        return 'ANML'
    return exp_name


def parse_bias(exp_name: str, config: dict):
    if 'gate_bias_init' in config:
        return config['gate_bias_init']
    if 'bias' in exp_name:
        token = exp_name.split('bias', 1)[-1]
        token = token.replace('m', '-').replace('p', '.')
        try:
            return float(token)
        except ValueError:
            return ''
    return ''


def collect(run_root: Path):
    rows = []
    for log_dir in sorted(run_root.glob('*')):
        if not log_dir.is_dir():
            continue
        cfg_path = log_dir / 'config.yaml'
        if not cfg_path.exists():
            continue

        cfg = load_config(cfg_path)

        exp_name = log_dir.name
        method_name = infer_method_name(exp_name)
        scores = load_meta_train_scores(log_dir)
        analysis = load_analysis_summary(log_dir)

        row = {
            'experiment_name': exp_name,
            'method_name': method_name,
            'gate_bias_init': parse_bias(exp_name, cfg),
            'use_correction': cfg.get('enable_correction', ''),
            'use_highway_gate': cfg.get('enable_highway_gate', ''),
            'train_backbone': not cfg.get('freeze_backbone', True),
            'main_metric': scores.get('acc/train', float('nan')),
            'loss_train': scores.get('loss/train', float('nan')),
            'mean_gate': analysis.get('mean_gate', float('nan')),
            'mean_delta_norm': analysis.get('mean_delta_norm', float('nan')),
            'mean_h_minus_prev_norm': analysis.get('mean_h_minus_prev_norm', float('nan')),
            'mean_gate_train': analysis.get('mean_gate_train', float('nan')),
            'mean_gate_test': analysis.get('mean_gate_test', float('nan')),
            'seed': cfg.get('seed', ''),
            'tasks': cfg.get('tasks', ''),
            'train_shots': cfg.get('train_shots', ''),
            'test_shots': cfg.get('test_shots', ''),
            'max_train_steps': cfg.get('max_train_steps', ''),
            'log_dir': str(log_dir),
        }
        rows.append(row)
    return rows


def write_csv(rows, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'experiment_name', 'method_name', 'gate_bias_init',
        'use_correction', 'use_highway_gate', 'train_backbone',
        'main_metric', 'loss_train',
        'mean_gate', 'mean_delta_norm', 'mean_h_minus_prev_norm',
        'mean_gate_train', 'mean_gate_test',
        'seed', 'tasks', 'train_shots', 'test_shots', 'max_train_steps',
        'log_dir',
    ]
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output', default='results/gate_bias_sweep_omniglot.csv')
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = collect(run_root)
    write_csv(rows, Path(args.output))
    print(f'Collected {len(rows)} runs -> {args.output}')
    if torch is None:
        print('Warning: torch is unavailable, meta_train_scores.pt metrics may be empty.')


if __name__ == '__main__':
    main()