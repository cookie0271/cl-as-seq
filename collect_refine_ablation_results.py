#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import yaml
import torch
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
    if exp_name.startswith('bitfit_standard'):
        return 'bitfit_standard'
    if exp_name.startswith('lora_standard'):
        return 'lora_standard'
    if exp_name.startswith('random_match_no_corr_gate'):
        return 'random_match_no_corr_gate'
    if exp_name.startswith('last2_no_corr_gate'):
        return 'last2_no_corr_gate'
    if exp_name.startswith('last2_gate_only'):
        return 'last2_gate_only'
    if exp_name.startswith('last2_corr_gate'):
        return 'last2_corr_gate'
    if exp_name.startswith('adapter_last2_standard'):
        return 'adapter_last2_standard'
    if exp_name.startswith('adapter_standard'):
        return 'adapter_standard'
    if exp_name.startswith('adapter_last2'):
        return 'adapter_last2'
    if exp_name.startswith('baseline_pf_last2'):
        return 'baseline_pf_last2'
    if exp_name.startswith('pf_last1_corr_gate'):
        return 'pf_last1_corr_gate'
    if exp_name.startswith('pf_last2_corr_gate'):
        return 'pf_last2_corr_gate'
    if exp_name.startswith('pf_last2_cor_only'):
        return 'pf_last2_cor_only'
    if exp_name.startswith('pf_last2_gate_only'):
        return 'pf_last2_gate_only'
    if exp_name.startswith('pf_tf_all_corr_gate'):
        return 'pf_tf_all_corr_gate'
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
        trainable = load_trainable_summary(log_dir)

        rows.append({
            'experiment_name': log_dir.name,
            'method_name': infer_method_name(log_dir.name),
            'seed': cfg.get('seed', ''),
            'acc/train': scores.get('acc/train', float('nan')),
            'acc/test': load_last_scalar_from_events(log_dir, 'acc/test'),
            'loss/train': scores.get('loss/train', float('nan')),
            'loss/test': load_last_scalar_from_events(log_dir, 'loss/test'),
            'trainable_params': trainable.get('trainable_params', ''),
            'target_trainable_params': trainable.get('target_trainable_params', ''),
            'trainable_param_gap': trainable.get('trainable_param_gap', ''),
            'selected_modules': trainable.get('selected_modules', ''),
            'max_train_steps': cfg.get('max_train_steps', ''),
            'freeze_backbone': cfg.get('freeze_backbone', ''),
            'use_correction': cfg.get('enable_correction', ''),
            'use_highway_gate': cfg.get('enable_highway_gate', ''),
            'partial_freeze_mode': cfg.get('partial_freeze_mode', ''),
            'random_match_seed': cfg.get('random_match_seed', ''),
            'random_match_scope': cfg.get('random_match_scope', ''),
            'random_match_unit': cfg.get('random_match_unit', ''),
            'enable_adapter': cfg.get('enable_adapter', ''),
            'adapter_dim': trainable.get('adapter_dim', cfg.get('adapter_dim', '')),
            'adapter_location': trainable.get('adapter_location', cfg.get('adapter_location', '')),
            'adapter_layers': trainable.get('adapter_layers', cfg.get('adapter_layers', '')),
            'adapter_strict_match': trainable.get('adapter_strict_match', cfg.get('adapter_strict_match', '')),
            'enable_lora': trainable.get('enable_lora', cfg.get('enable_lora', '')),
            'lora_rank': trainable.get('lora_rank', cfg.get('lora_rank', '')),
            'lora_alpha': trainable.get('lora_alpha', cfg.get('lora_alpha', '')),
            'lora_dropout': trainable.get('lora_dropout', cfg.get('lora_dropout', '')),
            'lora_layers': trainable.get('lora_layers', cfg.get('lora_layers', '')),
            'lora_target_modules': trainable.get('lora_target_modules', cfg.get('lora_target_modules', '')),
            'lora_strict_match': trainable.get('lora_strict_match', cfg.get('lora_strict_match', cfg.get('strict_match', ''))),
            'enable_bitfit': trainable.get('enable_bitfit', cfg.get('enable_bitfit', '')),
            'train_bitfit': trainable.get('train_bitfit', cfg.get('train_bitfit', '')),
            'bitfit_scope': trainable.get('bitfit_scope', cfg.get('bitfit_scope', '')),
            'bitfit_include_layernorm_bias': trainable.get(
            'bitfit_include_layernorm_bias', cfg.get('bitfit_include_layernorm_bias', '')),
            'bitfit_include_head_bias': trainable.get(
            'bitfit_include_head_bias', cfg.get('bitfit_include_head_bias', '')),
            'bitfit_strict_match': trainable.get(
            'bitfit_strict_match', cfg.get('bitfit_strict_match', cfg.get('strict_match', ''))),
            'trainable_bias_params': trainable.get('trainable_bias_params', ''),
            'trainable_head_params': trainable.get('trainable_head_params', ''),
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
        'acc/train', 'acc/test', 'loss/train', 'loss/test',
        'trainable_params', 'target_trainable_params', 'trainable_param_gap', 'selected_modules',
        'max_train_steps', 'freeze_backbone', 'use_correction', 'use_highway_gate',
        'partial_freeze_mode', 'random_match_seed', 'random_match_scope', 'random_match_unit',
        'partial_freeze_mode', 'random_match_seed', 'random_match_scope', 'random_match_unit',
        'enable_adapter', 'adapter_dim', 'adapter_location', 'adapter_layers', 'adapter_strict_match',
        'gate_bias_init',
        'enable_lora', 'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_layers', 'lora_target_modules',
        'lora_strict_match',
        'enable_bitfit', 'train_bitfit', 'bitfit_scope', 'bitfit_include_layernorm_bias',
        'bitfit_include_head_bias', 'bitfit_strict_match', 'trainable_bias_params', 'trainable_head_params',
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