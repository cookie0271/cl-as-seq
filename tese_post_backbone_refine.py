import torch

from models.continual_transformer import ContinualTransformer
from models.post_backbone_refine import PostBackboneRefineAndGate


def _base_config():
    return {
        'hidden_dim': 32,
        'input_type': 'vector',
        'output_type': 'class',
        'x_dim': 8,
        'y_vocab': 5,
        'y_len': 1,
        'tasks': 2,
        'train_shots': 2,
        'test_shots': 2,
        'tf_attn': 'elu',
        'tf_layers': 1,
        'tf_heads': 2,
        'tf_ff_dim': 64,
        'tf_dropout': 0.0,
        'tf_ln': False,
        'qk_dim': 16,
        'v_dim': 16,
        'favor_dim': 16,
        'favor_stabilize': True,
        'attn_loss': 0.0,
        'attn_loss_heads': 1,
        'distributed_loss': False,
        'enable_correction': True,
        'enable_highway_gate': True,
        'corr_hidden_ratio': 0.25,
        'corr_dropout': 0.0,
        'corr_zero_init': True,
        'use_prev_state_in_corr': False,
        'gate_type': 'mlp2',
        'gate_hidden_ratio': 0.25,
        'gate_dropout': 0.0,
        'gate_bias_init': -3.0,
        'gate_input_type': 'prev_z',
        'lambda_rate': 0.1,
        'r_target': 0.1,
        'freeze_backbone': True,
        'train_correction': True,
        'train_gate': True,
        'train_head': True,
    }


def test_random_forward_and_shapes():
    cfg = _base_config()
    model = ContinualTransformer(cfg)
    b = 3
    train_num = cfg['tasks'] * cfg['train_shots']
    test_num = cfg['tasks'] * cfg['test_shots']
    train_x = torch.randn(b, train_num, cfg['x_dim'])
    test_x = torch.randn(b, test_num, cfg['x_dim'])
    train_y = torch.randint(0, cfg['y_vocab'], (b, train_num))
    test_y = torch.randint(0, cfg['y_vocab'], (b, test_num))

    out = model(train_x, train_y, test_x, test_y)
    assert out['loss'].shape[0] == b
    assert out['logit'].shape[:2] == (b, test_num)


def test_disable_modules_identity_path():
    cfg = _base_config()
    cfg['enable_correction'] = False
    cfg['enable_highway_gate'] = False
    module = PostBackboneRefineAndGate(cfg)
    u = torch.randn(2, 7, cfg['hidden_dim'])
    h, _ = module(u, u, input_layout='BLD')
    assert torch.equal(h, u)


def test_gate_bias_init_low_mean_gate():
    cfg = _base_config()
    module = PostBackboneRefineAndGate(cfg)
    u = torch.randn(4, 12, cfg['hidden_dim'])
    h, aux = module(u, u, input_layout='BLD')
    assert h.shape == u.shape
    assert aux['gates'].mean().item() < 0.5


def test_freeze_backbone_trainables():
    cfg = _base_config()
    model = ContinualTransformer(cfg)
    model.set_trainable_modules(train_backbone=False, train_corr=True, train_gate=True, train_head=True)

    tf_trainable = any(p.requires_grad for p in model.tf_layers.parameters())
    corr_trainable = any(p.requires_grad for p in model.post_backbone_refine.correction.parameters())
    gate_trainable = any(p.requires_grad for p in model.post_backbone_refine.gate.parameters())
    head_trainable = any(p.requires_grad for p in model.output.parameters())

    assert not tf_trainable
    assert corr_trainable
    assert gate_trainable
    assert head_trainable


if __name__ == '__main__':
    test_random_forward_and_shapes()
    test_disable_modules_identity_path()
    test_gate_bias_init_low_mean_gate()
    test_freeze_backbone_trainables()
    print('ok')