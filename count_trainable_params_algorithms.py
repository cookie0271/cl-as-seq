#!/usr/bin/env python3
import argparse
import csv
from copy import deepcopy
from pathlib import Path

import yaml

from models import MODEL
from models.trainable_utils import set_trainable_modules


def get_config(config_path: str):
    with open(config_path, 'r') as f:
        new_config = yaml.full_load(f)
    config = {}
    if isinstance(new_config, dict) and 'include' in new_config:
        include_config = get_config(new_config['include'])
        config.update(include_config)
        del new_config['include']
    if new_config:
        config.update(new_config)
    return config


def apply_overrides(config: dict, overrides: dict):
    out = deepcopy(config)
    out.update(overrides)
    if out.get('y_vocab', None) is None and 'tasks' in out:
        out['y_vocab'] = out['tasks']
    return out


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_cases(args):
    # Shared defaults to make model initialization independent from missing cfg/data files.
    common = {
        'tasks': args.tasks,
        'train_shots': args.train_shots,
        'test_shots': args.test_shots,
        'x_c': args.x_c,
        'x_h': args.x_h,
        'x_w': args.x_w,
        'input_type': 'image',
        'output_type': 'class',
    }

    tf_cfg = args.tf_model_config
    return [
        {
            'name': 'Full outer-loop train',
            'model_config': tf_cfg,
            'overrides': {
                **common,
                'freeze_backbone': False,
                'partial_freeze_mode': 'none',
                'train_last_tf_layers': 0,
                'freeze_encoder': False,
                'enable_correction': False,
                'enable_highway_gate': False,
                'train_correction': False,
                'train_gate': False,
                'train_head': True,
            },
        },
        {
            'name': '只训head',
            'model_config': tf_cfg,
            'overrides': {
                **common,
                'freeze_backbone': True,
                'partial_freeze_mode': 'none',
                'train_last_tf_layers': 0,
                'freeze_encoder': True,
                'enable_correction': False,
                'enable_highway_gate': False,
                'train_correction': False,
                'train_gate': False,
                'train_head': True,
            },
        },
        {
            'name': '训head + last 1 layer',
            'model_config': tf_cfg,
            'overrides': {
                **common,
                'freeze_backbone': True,
                'partial_freeze_mode': 'last_n_tf',
                'train_last_tf_layers': 1,
                'freeze_encoder': True,
                'enable_correction': False,
                'enable_highway_gate': False,
                'train_correction': False,
                'train_gate': False,
                'train_head': True,
            },
        },
        {
            'name': '训head + last 2 layer',
            'model_config': tf_cfg,
            'overrides': {
                **common,
                'freeze_backbone': True,
                'partial_freeze_mode': 'last_n_tf',
                'train_last_tf_layers': 2,
                'freeze_encoder': True,
                'enable_correction': False,
                'enable_highway_gate': False,
                'train_correction': False,
                'train_gate': False,
                'train_head': True,
            },
        },
        {
            'name': '训head + last 2 layers + correction',
            'model_config': tf_cfg,
            'overrides': {
                **common,
                'freeze_backbone': True,
                'partial_freeze_mode': 'last_n_tf',
                'train_last_tf_layers': 2,
                'freeze_encoder': True,
                'enable_correction': True,
                'enable_highway_gate': False,
                'train_correction': True,
                'train_gate': False,
                'train_head': True,
            },
        },
        {
            'name': '训head + last 2 layers + gate',
            'model_config': tf_cfg,
            'overrides': {
                **common,
                'freeze_backbone': True,
                'partial_freeze_mode': 'last_n_tf',
                'train_last_tf_layers': 2,
                'freeze_encoder': True,
                'enable_correction': False,
                'enable_highway_gate': True,
                'train_correction': False,
                'train_gate': True,
                'train_head': True,
            },
        },
        {
            'name': '训head + last 2 layer + gate + correction',
            'model_config': tf_cfg,
            'overrides': {
                **common,
                'freeze_backbone': True,
                'partial_freeze_mode': 'last_n_tf',
                'train_last_tf_layers': 2,
                'freeze_encoder': True,
                'enable_correction': True,
                'enable_highway_gate': True,
                'train_correction': True,
                'train_gate': True,
                'train_head': True,
            },
        },
        {'name': 'PN', 'model_config': args.pn_model_config, 'overrides': common},
        {'name': 'GeMCL', 'model_config': args.gemcl_model_config, 'overrides': common},
        {'name': 'OML', 'model_config': args.oml_model_config, 'overrides': common},
        {'name': 'ANML', 'model_config': args.anml_model_config, 'overrides': common},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf-model-config', default='cfg/model/omniglot_tf_fulltrain_12k.yaml')
    parser.add_argument('--pn-model-config', default='cfg/model/pn.yaml')
    parser.add_argument('--gemcl-model-config', default='cfg/model/gemcl.yaml')
    parser.add_argument('--oml-model-config', default='cfg/model/oml.yaml')
    parser.add_argument('--anml-model-config', default='cfg/model/anml.yaml')
    parser.add_argument('--tasks', type=int, default=20)
    parser.add_argument('--train-shots', type=int, default=5)
    parser.add_argument('--test-shots', type=int, default=1)
    parser.add_argument('--x-c', type=int, default=1)
    parser.add_argument('--x-h', type=int, default=32)
    parser.add_argument('--x-w', type=int, default=32)
    parser.add_argument('--csv', default='results/trainable_params_algorithms.csv')
    args = parser.parse_args()

    cases = build_cases(args)
    rows = []

    for case in cases:
        cfg = get_config(case['model_config'])
        cfg = apply_overrides(cfg, case['overrides'])
        model = MODEL[cfg['model']](cfg)
        set_trainable_modules(
            model,
            train_backbone=cfg.get('freeze_backbone', True) is False,
            train_corr=cfg.get('train_correction', True),
            train_gate=cfg.get('train_gate', True),
            train_head=cfg.get('train_head', True),
            partial_freeze_mode=cfg.get('partial_freeze_mode', 'none'),
            train_last_tf_layers=cfg.get('train_last_tf_layers', 0),
            freeze_encoder=cfg.get('freeze_encoder', False),
        )
        total, trainable = count_params(model)
        rows.append({
            'algorithm': case['name'],
            'model': cfg['model'],
            'total_params': total,
            'trainable_params': trainable,
            'trainable_ratio': f"{trainable / total:.6f}",
        })

    print('| algorithm | model | total_params | trainable_params | trainable_ratio |')
    print('|---|---:|---:|---:|---:|')
    for r in rows:
        print(f"| {r['algorithm']} | {r['model']} | {r['total_params']} | {r['trainable_params']} | {r['trainable_ratio']} |")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to: {csv_path}")


if __name__ == '__main__':
    main()