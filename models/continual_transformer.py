import math
import random
from functools import lru_cache

import torch
import torch.nn as nn
from einops import rearrange, repeat, pack, unpack, reduce
from fast_transformers.feature_maps import Favor, ActivationFunctionFeatureMap
from torch import nn as nn, Tensor

from models import Model
from models.encoders.classification import ClassEncoder
from utils import cross_entropy, angle_loss
from models.encoders import MlpEncoder, X_ENCODER
from models.post_backbone_refine import PostBackboneRefineAndGate, gate_aux_stats



class ContinualAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if config['tf_attn'] == 'vanilla':
            self.feature_map = None
        elif config['tf_attn'] == 'elu':
            # Linear transformer: elu(x) + 1 feature map with much better precision
            self.feature_map = ActivationFunctionFeatureMap.factory(
                lambda x: torch.where(x > 0, x + 1, torch.exp(torch.minimum(x, torch.ones([], device=x.device))))
            )(None)
        elif config['tf_attn'] == 'favor':
            # Performer
            self.feature_map = Favor(
                query_dimensions=config['qk_dim'], n_dims=config['favor_dim'],
                stabilize=config['favor_stabilize'], redraw=1)  # always redraw
            # Manually control redraws to avoid resampling random features when loading from checkpoint
            self.register_buffer('feature_map_calls', torch.zeros([], dtype=torch.long))
        else:
            raise ValueError(f'Unknown feature map {config["feature_map"]}')

    def forward(self, queries, keys, values, attach_test_after, train_len=0, past_state=None, return_state=False):
        """Compute the attention.

        Args:
            queries: [batch, q_len, heads, qk_dim]
            keys: [batch, kv_len, heads, qk_dim]
            values: [batch, kv_len, heads, v_dim]
            attach_test_after: where to attach the test examples in the train sequence [batch, test_num]
            train_len: the length of the train sequence
            past_state: [batch_size, 1, heads, f_dim, v_dim+1]
            return_state: whether to return the last state after train sequence
        """
        batch, q_len, heads, qk_dim = queries.shape
        batch, test_num = attach_test_after.shape
        test_chunk_size = (q_len - train_len) // test_num
        test_len = test_num * test_chunk_size
        assert test_len + train_len == q_len

        if self.feature_map is not None:
            if 'favor_redraw' in self.config:
                # Performer
                if self.feature_map_calls.item() % self.config['favor_redraw'] == 0:
                    self.feature_map.new_feature_map(queries.device)
                self.feature_map_calls += 1
            queries = self.feature_map.forward_queries(queries)
            keys = self.feature_map.forward_keys(keys)

        q = rearrange(queries, 'b l h d -> b h l d')
        k = rearrange(keys, 'b l h d -> b h l d')
        v = rearrange(values, 'b l h d -> b h l d')

        if past_state is not None:
            past_k, past_v = past_state
            k, _ = pack([past_k, k], 'b h * d')
            v, _ = pack([past_v, v], 'b h * d')

            past_len = past_k.shape[-2]
            attach_test_after = attach_test_after + past_len
            train_len += past_len

        train_len = k.shape[-2] - test_num * test_chunk_size
        aux_output = {
            'state': (k[:, :, :train_len], v[:, :, :train_len]) if return_state else None,
        }

        attn_logit = torch.einsum('bhmd,bhnd->bhmn', q, k)
        if self.feature_map is None:
            attn_logit = attn_logit / (qk_dim ** 0.5)

        # Build attention mask
        mask = get_continual_mask(*attn_logit.shape[-2:], test_num, test_chunk_size, device=q.device)
        mask = repeat(mask, 'q k -> b h q k', b=batch, h=1).contiguous()

        # Prevent some of the attention from test queries to train keys according to attach_test_after
        indices = rearrange(torch.arange(train_len, device=q.device), 'train_len -> () () () train_len')
        attach_test_after = rearrange(attach_test_after, 'b n -> b () n ()')
        test_q_train_k_mask = repeat(
            indices <= attach_test_after,
            'b h test_num train_len -> b h (test_num c) train_len', c=test_chunk_size
        ).float()
        mask[:, :, q_len - test_len:, :train_len] = test_q_train_k_mask

        if self.feature_map is None:
            # Vanilla attention
            mask = torch.zeros_like(mask).masked_fill(~mask.bool(), torch.finfo(attn_logit.dtype).min)
            attn_logit = attn_logit + mask
            attn = torch.softmax(attn_logit, dim=-1)
            aux_output['attn_logit'] = attn_logit
        else:
            # Efficient transformers
            attn_logit = attn_logit + 1e-9  # for stability
            attn_logit = attn_logit * mask
            attn = attn_logit / attn_logit.sum(dim=-1, keepdim=True)
            aux_output['attn'] = attn

        output = torch.einsum('bhmn,bhnd->bhmd', attn, v)
        output = rearrange(output, 'b h l d -> b l (h d)')
        return output, aux_output


@lru_cache(maxsize=8)
def get_continual_mask(query_num: int, key_num: int, test_num: int, test_chunk_size: int, device='cuda'):
    """Build mask that simulates each test example being added to the end of train sequence one by one.

    Example:
        query_num = 8
        key_num = 12
        test_num = 3
        test_chunk_size = 2
        mask = tensor([[1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1.]])
    """
    assert query_num <= key_num

    # Basic causal mask
    mask = torch.tril(torch.ones((query_num, key_num), device=device), diagonal=key_num - query_num)

    # Block-diagonal mask for test chunks
    test_chunks = [torch.tril(torch.ones([test_chunk_size, test_chunk_size], device=device), diagonal=0)] * test_num
    test_allowed = torch.block_diag(*test_chunks)

    # Paste the block-diagonal mask to the causal mask
    q_train = query_num - test_num * test_chunk_size
    k_train = key_num - test_num * test_chunk_size
    mask[q_train:, k_train:] = test_allowed
    return mask


class ContinualAttentionLayer(nn.Module):
    """Attention layer optimized for continual learning."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dropout = nn.Dropout(config['tf_dropout'])

        self.qkv_linear = nn.Linear(config['hidden_dim'], config['tf_heads'] * (2 * config['qk_dim'] + config['v_dim']))
        self.attn = ContinualAttention(config, layer_idx)
        self.proj_linear = nn.Linear(config['tf_heads'] * config['v_dim'], config['hidden_dim'])

    def forward(self, x, attach_test_after, train_len=0, past_state=None, return_state=False):
        # Linear projection for Q, K, V
        qkv = self.qkv_linear(x)
        q, k, v = unpack(qkv, [
            [self.config['tf_heads'] * self.config['qk_dim']],
            [self.config['tf_heads'] * self.config['qk_dim']],
            [self.config['tf_heads'] * self.config['v_dim']]
        ], 'b l *')
        q = rearrange(q, 'b l (h d) -> b l h d', h=self.config['tf_heads'])
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.config['tf_heads'])
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.config['tf_heads'])

        # Compute attention
        attn_output, state = self.attn(
            q, k, v,
            attach_test_after=attach_test_after, train_len=train_len,
            past_state=past_state, return_state=return_state)

        x = x + self.dropout(self.proj_linear(attn_output))
        return x, state


class ContinualTransformerLayer(nn.Module):
    """Transformer layer optimized for continual learning."""

    def __init__(self, config, layer_idx, use_adapter=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_layer_norm = nn.LayerNorm(config['hidden_dim']) if config['tf_ln'] else nn.Identity()
        self.attn_layer = ContinualAttentionLayer(config, layer_idx)
        self.mlp_layer_norm = nn.LayerNorm(config['hidden_dim']) if config['tf_ln'] else nn.Identity()
        self.mlp_layer = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['tf_ff_dim']),
            nn.GELU(),
            nn.Linear(config['tf_ff_dim'], config['hidden_dim']),
            nn.Dropout(config['tf_dropout']),
        )
        self.adapter = BottleneckAdapter(config) if use_adapter else None

    def forward(self, xy_enc, attach_test_after, train_len=0, past_state=None, return_state=False):
        x, state = self.attn_layer(
            self.attn_layer_norm(xy_enc), attach_test_after=attach_test_after, train_len=train_len,
            past_state=past_state, return_state=return_state)
        x = x + self.mlp_layer(self.mlp_layer_norm(x))
        if self.adapter is not None:
            x = x + self.adapter(x)
        return x, state

class BottleneckAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config['hidden_dim'])
        adapter_dim = int(config.get('adapter_dim', 16))
        self.adapter_dim = adapter_dim
        self.location = config.get('adapter_location', 'post_layer')
        self.down_proj = nn.Linear(hidden_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)
        act_name = str(config.get('adapter_activation', 'gelu')).lower()
        if act_name == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()

        # Near-identity init at step 0.
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        delta = self.up_proj(self.act(self.down_proj(x)))
        return x + delta

class LoRALinear(nn.Module):
    """Standard LoRA wrapper for a frozen linear layer."""

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f'LoRA rank must be > 0, got {rank}')
        self.base_linear = base_linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False)

        # LoRA init: start near identity mapping (delta ~= 0).
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for param in self.base_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        base = self.base_linear(x)
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + delta



def sample_test_attachment(train_y, test_y):
    """Sample where to attach each test example in the train sequence

    Each test example can be evaluated after the task it belongs to or after any task after that.
    E.g., a test example of task3 can be tested after training task3, task4, ..., or taskN

    This function requires minimal assumptions about the task structure.
    Each sequence in the batch can have a different number of tasks and a different number of examples per task.
    The only assumption is that each sequence in train_y should start from 0 and monotonically increase by 0 or 1.
    E.g., [0, 0, 1, 1, 1, 2] is valid, but [0, 0, 1, 2, 1, 2] is not.

    Args:
        train_y: [batch, train_num]
        test_y: [batch, test_num]

    Returns:
        test_attachment: train example index [batch, test_num]
    """
    # For each test example, randomly sample which task to test after (sample in range [test_y, tasks))
    tasks = train_y.max(dim=1, keepdim=True).values + 1  # [batch, 1]
    num_options = tasks - test_y  # [batch, test_num]
    test_after = tasks - 1 - (torch.rand(test_y.shape, device=test_y.device) * num_options).long()

    # Find the indices of the ends of tasks in train sequence
    # E.g., if train_y = [[0, 0, 1, 2, 2, 2], [0, 1, 1, 2, 2, 3]], then task_end_indices = [1, 2, 5, 0, 2, 4, 5]
    # The first three comes from the first row, and the last four comes from the second row
    is_task_end = nn.functional.pad(train_y[:, 1:] != train_y[:, :-1], [0, 1], value=True)  # [batch, train_num]
    idx_arange = repeat(torch.arange(train_y.shape[1], device=train_y.device), 'l -> b l', b=train_y.shape[0])
    task_end_indices = torch.masked_select(idx_arange, is_task_end)  # [total_number_of_tasks]

    # Convert batch-wise task indices in test_after to global task indices that can be used to index task_end_indices
    num_boundaries = is_task_end.sum(dim=1)  # [batch]
    global_task_idx_offset = torch.cumsum(num_boundaries, dim=0, dtype=torch.long)
    global_task_idx_offset = nn.functional.pad(global_task_idx_offset[:-1], [1, 0], value=0)
    global_task_idx_offset = rearrange(global_task_idx_offset, 'b -> b 1')
    test_after_global = test_after + global_task_idx_offset  # [batch, test_num]

    return task_end_indices[test_after_global]


class ContinualTransformer(Model):
    """Decoder-only transformer optimized for continual learning."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_type = config['input_type']
        self.output_type = config['output_type']

        max_len = config['tasks'] * (
                config['train_shots'] * (1 + config['y_len']) +
                config['test_shots'] * config['y_len']
        )
        self.pos_enc = PositionalEncoding(config['hidden_dim'], max_len=max_len)
        if self.input_type == 'image':
            self.x_encoder = X_ENCODER[config['x_encoder']](config)
            self.x_proj = nn.Linear(256 * (config['x_h'] // 16) * (config['x_w'] // 16), config['hidden_dim'])
        elif self.input_type == 'vector':
            self.x_encoder = MlpEncoder(config, input_dim=config['x_dim'])
        else:
            raise NotImplementedError

        if self.output_type == 'class':
            self.y_encoder = ClassEncoder(config)
            self.output = nn.Linear(config['hidden_dim'], config['y_vocab'])
        elif self.output_type == 'vector':
            self.y_encoder = MlpEncoder(config, input_dim=config['y_dim'])
            self.output = nn.Linear(config['hidden_dim'], config['y_dim'])
        else:
            raise NotImplementedError

        adapter_layer_indices = self._resolve_adapter_layer_indices(
            config.get('adapter_layers', 'last2'),
            config['tf_layers'],
        ) if config.get('enable_adapter', False) else set()
        self.tf_layers = nn.ModuleList([
            ContinualTransformerLayer(
                config,
                layer_idx,
                use_adapter=(layer_idx in adapter_layer_indices),
            ) for layer_idx in range(config['tf_layers'])
        ])
        self.enable_correction = config.get('enable_correction', True)
        self.enable_highway_gate = config.get('enable_highway_gate', True)
        self.post_backbone_refine = PostBackboneRefineAndGate(config)
        self._lora_attached = False

    @staticmethod
    def _resolve_lora_layer_indices(lora_layers, num_layers):
        return ContinualTransformer._resolve_adapter_layer_indices(lora_layers, num_layers)

    @staticmethod
    def _normalize_lora_targets(lora_target_modules):
        text = str(lora_target_modules).strip().lower()
        if text in {'attn_only', 'attn'}:
            return 'attn_only'
        if text in {'attn_and_mlp', 'all', 'full'}:
            return 'attn_and_mlp'
        raise ValueError(f'Unsupported lora_target_modules={lora_target_modules}')

    def _iter_lora_target_linears(self, lora_layers, lora_target_modules):
        layer_indices = sorted(self._resolve_lora_layer_indices(lora_layers, len(self.tf_layers)))
        if len(layer_indices) == 0:
            raise ValueError('LoRA requires non-empty lora_layers.')
        target_mode = self._normalize_lora_targets(lora_target_modules)

        targets = []
        for layer_idx in layer_indices:
            layer = self.tf_layers[layer_idx]
            targets.append((f'tf_layers.{layer_idx}.attn_layer.qkv_linear', layer.attn_layer.qkv_linear))
            targets.append((f'tf_layers.{layer_idx}.attn_layer.proj_linear', layer.attn_layer.proj_linear))
            if target_mode == 'attn_and_mlp':
                targets.append((f'tf_layers.{layer_idx}.mlp_layer.0', layer.mlp_layer[0]))
                targets.append((f'tf_layers.{layer_idx}.mlp_layer.2', layer.mlp_layer[2]))
        return layer_indices, target_mode, targets

    def _attach_lora_modules(self, lora_rank, lora_alpha, lora_dropout, lora_layers, lora_target_modules):
        layer_indices, target_mode, targets = self._iter_lora_target_linears(lora_layers, lora_target_modules)
        for module_name, linear in targets:
            if isinstance(linear, LoRALinear):
                continue
            ref_param = next(linear.parameters(), None)
            if ref_param is not None:
                device, dtype = ref_param.device, ref_param.dtype
            else:
                model_param = next(self.parameters())
                device, dtype = model_param.device, model_param.dtype
            lora_linear = LoRALinear(linear, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout).to(
                device=device, dtype=dtype
            )
            parent_name, attr_name = module_name.rsplit('.', 1)
            parent = self.get_submodule(parent_name)
            setattr(parent, attr_name, lora_linear)
        self._lora_attached = True
        return layer_indices, target_mode

    @staticmethod
    def _resolve_adapter_layer_indices(adapter_layers, num_layers):
        if isinstance(adapter_layers, int):
            n = max(0, min(int(adapter_layers), num_layers))
            return set(range(num_layers - n, num_layers))
        if isinstance(adapter_layers, (list, tuple)):
            out = set()
            for idx in adapter_layers:
                i = int(idx)
                if 0 <= i < num_layers:
                    out.add(i)
            return out

        text = str(adapter_layers).strip().lower()
        if text == 'all':
            return set(range(num_layers))
        if text.startswith('last'):
            n = int(text[4:]) if len(text) > 4 else 0
            n = max(0, min(n, num_layers))
            return set(range(num_layers - n, num_layers))
        if text == 'none' or text == '':
            return set()

        out = set()
        for token in text.split(','):
            token = token.strip()
            if token == '':
                continue
            i = int(token)
            if 0 <= i < num_layers:
                out.add(i)
        return out

    def set_trainable_modules(
            self,
            train_backbone=False,
            train_corr=True,
            train_gate=True,
            train_bitfit=False,
            train_adapter=False,
            train_lora = False,
            train_head=True,
            partial_freeze_mode='none',
            train_last_tf_layers=0,
            freeze_encoder=False,
            random_match_target_trainable_params=None,
            random_match_seed=0,
            random_match_scope='tf_only',
            random_match_unit='layer_or_block',
            random_match_target_last_tf_layers=2,
            adapter_dim=None,
            adapter_layers='last2',
            adapter_location='post_layer',
            lora_rank=None,
            lora_alpha=16,
            lora_dropout=0.0,
            lora_layers='all',
            lora_target_modules='attn_only',
            lora_strict_match=False,
            lora_rank_candidates=None,
            bitfit_scope='transformer_only',
            bitfit_include_layernorm_bias=True,
            bitfit_include_head_bias=True,
            bitfit_strict_match=False,
            target_trainable_params=None):
        def _module_param_count(module):
            return sum(p.numel() for p in module.parameters())

        def _params_count(params):
            return sum(p.numel() for p in params)

        def _estimate_random_match_target():
            explicit_target = random_match_target_trainable_params
            if explicit_target is None:
                explicit_target = target_trainable_params
            if explicit_target is not None:
                return int(explicit_target)

            n_last = int(random_match_target_last_tf_layers)
            n_last = max(0, min(n_last, len(self.tf_layers)))
            target = _module_param_count(self.output) if train_head else 0
            if n_last > 0:
                target += sum(_module_param_count(layer) for layer in self.tf_layers[-n_last:])

            refine_cfg = dict(self.config)
            refine_cfg['enable_correction'] = True
            refine_cfg['enable_highway_gate'] = True
            refine_target = PostBackboneRefineAndGate(refine_cfg)
            if refine_target.correction is not None:
                target += _module_param_count(refine_target.correction)
            if refine_target.gate is not None:
                target += _module_param_count(refine_target.gate)
            return int(target)

        def _adapter_param_count_for_dim(dim, num_adapter_layers):
            h = int(self.config['hidden_dim'])
            d = int(dim)
            # down(h->d) + up(d->h), both with bias
            return num_adapter_layers * ((h * d + d) + (d * h + h))

        def _build_adapter_dim_candidates(adapter_budget, num_adapter_layers):
            hidden_dim = int(self.config['hidden_dim'])
            explicit = self.config.get('adapter_dim_candidates', None)
            if explicit is not None:
                candidates = {max(1, int(x)) for x in explicit}
            else:
                # Standard PEFT-friendly bottleneck candidates.
                candidates = {16, 32, 64, 128, 256, 512}
            if adapter_dim is not None:
                candidates.add(max(1, int(adapter_dim)))
            # Keep adapter bottleneck semantically valid; do not exceed hidden size.
            bounded = sorted({c for c in candidates if c <= hidden_dim})
            if len(bounded) == 0:
                bounded = [hidden_dim]
            return bounded

        def _replace_adapters(selected_layer_indices, selected_adapter_dim):
            for layer_idx in selected_layer_indices:
                old_adapter = self.tf_layers[layer_idx].adapter
                if old_adapter is not None:
                    ref_param = next(old_adapter.parameters(), None)
                    device = ref_param.device if ref_param is not None else next(self.parameters()).device
                    dtype = ref_param.dtype if ref_param is not None else next(self.parameters()).dtype
                else:
                    ref_param = next(self.tf_layers[layer_idx].parameters(), None)
                    device = ref_param.device if ref_param is not None else next(self.parameters()).device
                    dtype = ref_param.dtype if ref_param is not None else next(self.parameters()).dtype

                adapter_cfg = dict(self.config)
                adapter_cfg['adapter_dim'] = int(selected_adapter_dim)
                adapter_cfg['adapter_location'] = adapter_location
                new_adapter = BottleneckAdapter(adapter_cfg).to(device=device, dtype=dtype)
                self.tf_layers[layer_idx].adapter = new_adapter

        def _lora_param_count_for_rank(rank, targets):
            total = 0
            for _, linear in targets:
                base = linear.base_linear if isinstance(linear, LoRALinear) else linear
                out_dim, in_dim = base.weight.shape
                total += rank * (in_dim + out_dim)
            return int(total)

        def _is_layernorm_bias(name):
            return name.endswith('attn_layer_norm.bias') or name.endswith('mlp_layer_norm.bias')

        def _is_bitfit_bias_name(name):
            return name.endswith('.bias') or name == 'bias'

        def _extract_tf_layer_index(name):
            prefix = 'tf_layers.'
            if not name.startswith(prefix):
                return None
            remain = name[len(prefix):]
            token = remain.split('.', 1)[0]
            try:
                return int(token)
            except ValueError:
                return None

        def _bitfit_scope_match(param_name):
            if bitfit_scope == 'transformer_only':
                return param_name.startswith('tf_layers.')
            if bitfit_scope == 'last2_only':
                idx = _extract_tf_layer_index(param_name)
                return idx is not None and idx >= max(0, len(self.tf_layers) - 2)
            if bitfit_scope == 'transformer_and_encoder':
                return (
                        param_name.startswith('tf_layers.')
                        or param_name.startswith('x_encoder.')
                        or param_name.startswith('x_proj.')
                )
            if bitfit_scope == 'all_backbone':
                return (
                        param_name.startswith('tf_layers.')
                        or param_name.startswith('x_encoder.')
                        or param_name.startswith('x_proj.')
                        or param_name.startswith('y_encoder.')
                        or param_name.startswith('pos_enc.')
                )
            raise ValueError(f'Unsupported bitfit_scope={bitfit_scope}')

        backbone_modules = [self.x_encoder, self.y_encoder, self.pos_enc, self.tf_layers]
        if hasattr(self, 'x_proj'):
            backbone_modules.append(self.x_proj)
        head_modules = [self.output]

        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad = train_backbone

        if self.post_backbone_refine.correction is not None:
            for param in self.post_backbone_refine.correction.parameters():
                param.requires_grad = train_corr
        if self.post_backbone_refine.gate is not None:
            for param in self.post_backbone_refine.gate.parameters():
                param.requires_grad = train_gate
        for layer in self.tf_layers:
            if getattr(layer, 'adapter', None) is not None:
                for param in layer.adapter.parameters():
                    param.requires_grad = train_adapter
            for module in layer.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.weight.requires_grad = bool(train_lora)
                    module.lora_B.weight.requires_grad = bool(train_lora)

        for module in head_modules:
            for param in module.parameters():
                param.requires_grad = train_head

        selected_groups = []
        target_params = None

        # Optional finer-grained partial-freeze controls.
        # Existing train_backbone behavior remains default when partial_freeze_mode is 'none'.
        if partial_freeze_mode != 'none':
            # Encoder freeze control (CNN/MLP encoder + projection).
            if freeze_encoder:
                for param in self.x_encoder.parameters():
                    param.requires_grad = False
                if hasattr(self, 'x_proj'):
                    for param in self.x_proj.parameters():
                        param.requires_grad = False

            if partial_freeze_mode == 'last_n_tf':
                # Freeze all TF layers first, then unfreeze only last N.
                for layer in self.tf_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

                n = int(train_last_tf_layers) if train_last_tf_layers is not None else 0
                n = max(0, min(n, len(self.tf_layers)))
                if n > 0:
                    for layer in self.tf_layers[-n:]:
                        for param in layer.parameters():
                            param.requires_grad = True
            elif partial_freeze_mode == 'tf_all':
                for layer in self.tf_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
            elif partial_freeze_mode in {'adapter_last2', 'adapter_standard'}:
                adapter_layer_indices = sorted(self._resolve_adapter_layer_indices(adapter_layers, len(self.tf_layers)))
                if len(adapter_layer_indices) == 0:
                    raise ValueError(f'{partial_freeze_mode} mode requires non-empty adapter_layers.')

                for layer in self.tf_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

                target_params = _estimate_random_match_target()
                head_params = _module_param_count(self.output) if train_head else 0
                adapter_budget = max(1, target_params - head_params)
                adapter_candidates = _build_adapter_dim_candidates(adapter_budget, len(adapter_layer_indices))
                strict_match = bool(self.config.get('adapter_strict_match', False))

                if partial_freeze_mode == 'adapter_standard':
                    if strict_match:
                        print('Warning: adapter_strict_match=True ignored for adapter_standard baseline.')
                    strict_match = False
                    if adapter_dim is not None:
                        best_dim = min(int(adapter_dim), int(self.config['hidden_dim']))
                    else:
                        preferred = int(self.config.get('adapter_default_dim', 64))
                        if preferred in adapter_candidates:
                            best_dim = preferred
                        else:
                            best_dim = min(adapter_candidates, key=lambda d: abs(d - preferred))
                    best_total = head_params + _adapter_param_count_for_dim(best_dim, len(adapter_layer_indices))
                else:
                    if strict_match:
                        print(
                            'Warning: adapter_strict_match=True is deprecated; using bounded bottleneck candidates only.')
                    best_dim = None
                    best_gap = None
                    best_total = None
                    for candidate_dim in adapter_candidates:
                        adapter_count = _adapter_param_count_for_dim(candidate_dim, len(adapter_layer_indices))
                        total_candidate = head_params + adapter_count
                        gap = abs(total_candidate - target_params)
                        if best_gap is None or gap < best_gap:
                            best_gap = gap
                            best_dim = int(candidate_dim)
                            best_total = int(total_candidate)

                print('Adapter candidate trainable params:')
                for candidate_dim in adapter_candidates:
                    candidate_total = head_params + _adapter_param_count_for_dim(candidate_dim,
                                                                                 len(adapter_layer_indices))
                    print(
                        f'  - dim={candidate_dim}: trainable={candidate_total:,} (gap={candidate_total - target_params:+,})')

                _replace_adapters(adapter_layer_indices, best_dim)
                self.config['adapter_dim'] = int(best_dim)
                self.config['adapter_location'] = adapter_location
                self.config['adapter_layers'] = adapter_layers

                selected_count = 0
                for layer_idx in adapter_layer_indices:
                    layer_adapter = self.tf_layers[layer_idx].adapter
                    if layer_adapter is None:
                        continue
                    adapter_params = list(layer_adapter.parameters())
                    module_count = _params_count(adapter_params)
                    selected_groups.append((f'tf_layers.{layer_idx}.adapter', module_count))
                    selected_count += module_count
                    for param in adapter_params:
                        param.requires_grad = bool(train_adapter)

                print(f'Adapter selection details ({partial_freeze_mode}):')
                print(
                    f'  - adapter_layers={adapter_layers} ({adapter_layer_indices}) | '
                    f'adapter_location={adapter_location} | adapter_dim={best_dim}')
                print(
                    f'  - adapter_params={selected_count:,} | head_params={head_params:,} | '
                    f'target={target_params:,} | total_selected={best_total:,} | gap={best_total - target_params:+,}')
                print(
                    f'  - hidden_dim={self.config["hidden_dim"]} | strict_match={strict_match} | '
                    f'candidate_dims={adapter_candidates}')
            elif partial_freeze_mode == 'lora_standard':
                for layer in self.tf_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

                if bool(lora_strict_match):
                    print('Warning: lora_strict_match=True ignored for standard LoRA baseline.')
                lora_layer_indices, normalized_targets, target_linears = self._iter_lora_target_linears(
                    lora_layers=lora_layers,
                    lora_target_modules=lora_target_modules,
                )
                target_params = _estimate_random_match_target()
                head_params = _module_param_count(self.output) if train_head else 0

                candidates = lora_rank_candidates if lora_rank_candidates is not None else self.config.get(
                    'lora_rank_candidates', [4, 8, 16, 32, 64])
                rank_candidates = sorted({max(1, int(r)) for r in candidates})
                if len(rank_candidates) == 0:
                    rank_candidates = [8]
                if lora_rank is not None:
                    selected_rank = max(1, int(lora_rank))
                    if selected_rank not in rank_candidates:
                        rank_candidates.append(selected_rank)
                        rank_candidates = sorted(rank_candidates)
                else:
                    default_rank = int(self.config.get('lora_default_rank', 16))
                    selected_rank = default_rank if default_rank in rank_candidates else min(
                        rank_candidates, key=lambda r: abs(r - default_rank))

                print('LoRA candidate trainable params:')
                for cand_rank in rank_candidates:
                    lora_count = _lora_param_count_for_rank(cand_rank, target_linears)
                    total_candidate = head_params + lora_count
                    print(
                        f'  - rank={cand_rank}: trainable={total_candidate:,} (gap={total_candidate - target_params:+,})')

                lora_layer_indices, normalized_targets = self._attach_lora_modules(
                    lora_rank=selected_rank,
                    lora_alpha=float(lora_alpha),
                    lora_dropout=float(lora_dropout),
                    lora_layers=lora_layers,
                    lora_target_modules=normalized_targets,
                )

                lora_selected_count = 0
                for layer_idx in lora_layer_indices:
                    layer = self.tf_layers[layer_idx]
                    for module_name, module in layer.named_modules():
                        if isinstance(module, LoRALinear):
                            a_count = module.lora_A.weight.numel()
                            b_count = module.lora_B.weight.numel()
                            lora_selected_count += a_count + b_count
                            selected_groups.append((f'tf_layers.{layer_idx}.{module_name}.lora', a_count + b_count))
                            module.lora_A.weight.requires_grad = bool(train_lora)
                            module.lora_B.weight.requires_grad = bool(train_lora)
                            for param in module.base_linear.parameters():
                                param.requires_grad = False

                selected_total = head_params + lora_selected_count
                self.config['enable_lora'] = True
                self.config['lora_rank'] = int(selected_rank)
                self.config['lora_alpha'] = float(lora_alpha)
                self.config['lora_dropout'] = float(lora_dropout)
                self.config['lora_layers'] = lora_layers
                self.config['lora_target_modules'] = normalized_targets
                print('LoRA selection details (lora_standard):')
                print(
                    f'  - baseline=standard_lora | hidden_dim={self.config["hidden_dim"]} | '
                    f'lora_rank={selected_rank} | lora_alpha={float(lora_alpha):g} | lora_dropout={float(lora_dropout):g}')
                print(
                    f'  - lora_target_modules={normalized_targets} | lora_layers={lora_layers} ({lora_layer_indices})')
                print(
                    f'  - lora_params={lora_selected_count:,} | head_params={head_params:,} | '
                    f'target={target_params:,} | total_selected={selected_total:,} | gap={selected_total - target_params:+,}')
                print(
                    f'  - strict_match=False | train_modules=lora+head | rank_candidates={rank_candidates}')
            elif partial_freeze_mode == 'bitfit_standard':
                if bool(bitfit_strict_match):
                    print('Warning: bitfit_strict_match=True ignored for standard BitFit baseline.')
                # Freeze all backbone params first; only selected bias params will be re-enabled.
                for module in backbone_modules:
                    for param in module.parameters():
                        param.requires_grad = False

                selected_bias = []
                skipped_layernorm = []
                for name, param in self.named_parameters():
                    if not _is_bitfit_bias_name(name):
                        continue
                    if not _bitfit_scope_match(name):
                        continue
                    if (not bool(bitfit_include_layernorm_bias)) and _is_layernorm_bias(name):
                        skipped_layernorm.append(name)
                        continue
                    param.requires_grad = bool(train_bitfit)
                    if bool(train_bitfit):
                        selected_bias.append((name, param.numel()))
                        selected_groups.append((name, param.numel()))

                # Standard BitFit baseline keeps full output head trainable by design.
                for param in self.output.parameters():
                    param.requires_grad = bool(train_head)
                if not bool(bitfit_include_head_bias) and self.output.bias is not None:
                    self.output.bias.requires_grad = False

                trainable_bias_params = sum(
                    p.numel() for name, p in self.named_parameters()
                    if p.requires_grad and _is_bitfit_bias_name(name)
                )
                trainable_head_params = sum(p.numel() for p in self.output.parameters() if p.requires_grad)
                selected_bias_params = sum(count for _, count in selected_bias)
                strict_match = False
                self.config['enable_bitfit'] = True
                self.config['train_bitfit'] = bool(train_bitfit)
                self.config['bitfit_scope'] = bitfit_scope
                self.config['bitfit_include_layernorm_bias'] = bool(bitfit_include_layernorm_bias)
                self.config['bitfit_include_head_bias'] = bool(bitfit_include_head_bias)
                self.config['bitfit_strict_match'] = strict_match
                print('BitFit selection details (bitfit_standard):')
                print(
                    f'  - baseline=standard_bitfit_bias_only | bitfit_scope={bitfit_scope} | '
                    f'include_layernorm_bias={bool(bitfit_include_layernorm_bias)} | '
                    f'include_head_bias={bool(bitfit_include_head_bias)}')
                print(
                    f'  - trainable_backbone_bias_params={selected_bias_params:,} | '
                    f'trainable_head_params={trainable_head_params:,} | '
                    f'trainable_bias_params(total)={trainable_bias_params:,}')
                print(
                    f'  - strict_match={strict_match} | train_modules=bitfit_bias+head | '
                    f'skipped_layernorm_bias={len(skipped_layernorm)}')
            elif partial_freeze_mode == 'random_match':
                if random_match_scope != 'tf_only':
                    raise ValueError(f'Unsupported random_match_scope: {random_match_scope}')

                for layer in self.tf_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

                rng = random.Random(int(random_match_seed))
                target_params = _estimate_random_match_target()
                selected_layer_idx = set()

                layer_candidates = []
                for layer_idx, layer in enumerate(self.tf_layers):
                    layer_params = list(layer.parameters())
                    layer_candidates.append({
                        'name': f'tf_layers.{layer_idx}',
                        'params': layer_params,
                        'count': _params_count(layer_params),
                        'layer_idx': layer_idx,
                    })
                rng.shuffle(layer_candidates)

                selected_count = 0
                for candidate in layer_candidates:
                    if selected_count + candidate['count'] <= target_params:
                        for param in candidate['params']:
                            param.requires_grad = True
                        selected_groups.append((candidate['name'], candidate['count']))
                        selected_count += candidate['count']
                        selected_layer_idx.add(candidate['layer_idx'])

                if random_match_unit == 'layer_or_block' and selected_count < target_params:
                    block_candidates = []
                    for layer_idx, layer in enumerate(self.tf_layers):
                        if layer_idx in selected_layer_idx:
                            continue
                        block_specs = [
                            ('attn_layer_norm', layer.attn_layer_norm),
                            ('attn_layer', layer.attn_layer),
                            ('mlp_layer_norm', layer.mlp_layer_norm),
                            ('mlp_layer', layer.mlp_layer),
                        ]
                        for block_name, block_module in block_specs:
                            block_params = list(block_module.parameters())
                            if len(block_params) == 0:
                                continue
                            block_candidates.append({
                                'name': f'tf_layers.{layer_idx}.{block_name}',
                                'params': block_params,
                                'count': _params_count(block_params),
                            })
                    rng.shuffle(block_candidates)

                    for candidate in block_candidates:
                        with_candidate = selected_count + candidate['count']
                        if abs(target_params - with_candidate) <= abs(target_params - selected_count):
                            for param in candidate['params']:
                                param.requires_grad = True
                            selected_groups.append((candidate['name'], candidate['count']))
                            selected_count = with_candidate
                elif random_match_unit != 'layer':
                    raise ValueError(f'Unsupported random_match_unit: {random_match_unit}')

                print('Random-match selection details:')
                for name, count in selected_groups:
                    print(f'  - {name}: {count:,} params')
                print(
                    f'Random-match totals: selected={selected_count:,} | target={target_params:,} | '
                    f'gap={selected_count - target_params:+,} | seed={random_match_seed}')
            else:
                raise ValueError(f'Unknown partial_freeze_mode: {partial_freeze_mode}')

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        selected_modules_text = ';'.join([f'{name}:{count}' for name, count in selected_groups])
        self._trainable_summary = {
            'trainable_params': int(trainable_params),
            'target_trainable_params': int(target_params) if target_params is not None else '',
            'trainable_param_gap': int(trainable_params - target_params) if target_params is not None else '',
            'selected_modules': selected_modules_text,
            'partial_freeze_mode': partial_freeze_mode,
            'random_match_seed': int(random_match_seed) if partial_freeze_mode == 'random_match' else '',
            'random_match_scope': random_match_scope if partial_freeze_mode == 'random_match' else '',
            'random_match_unit': random_match_unit if partial_freeze_mode == 'random_match' else '',
            'adapter_dim': int(self.config.get('adapter_dim', 0)) if partial_freeze_mode in {'adapter_last2',
                                                                                             'adapter_standard'} else '',
            'adapter_location': adapter_location if partial_freeze_mode in {'adapter_last2',
                                                                            'adapter_standard'} else '',
            'enable_adapter': bool(self.config.get('enable_adapter', False)) if partial_freeze_mode in {'adapter_last2',
                                                                                                        'adapter_standard'} else '',
            'adapter_strict_match': bool(self.config.get('adapter_strict_match', False)) if partial_freeze_mode in {
                'adapter_last2', 'adapter_standard'} else '',
            'adapter_layers': str(self.config.get('adapter_layers', '')) if partial_freeze_mode in {'adapter_last2',
                                                                                                    'adapter_standard'} else '',
            'adapter_baseline': partial_freeze_mode if partial_freeze_mode in {'adapter_last2',
                                                                               'adapter_standard'} else '',
            'enable_lora': bool(
                self.config.get('enable_lora', False)) if partial_freeze_mode == 'lora_standard' else '',
            'lora_rank': int(self.config.get('lora_rank', 0)) if partial_freeze_mode == 'lora_standard' else '',
            'lora_alpha': float(self.config.get('lora_alpha', 0.0)) if partial_freeze_mode == 'lora_standard' else '',
            'lora_dropout': float(
                self.config.get('lora_dropout', 0.0)) if partial_freeze_mode == 'lora_standard' else '',
            'lora_layers': str(self.config.get('lora_layers', '')) if partial_freeze_mode == 'lora_standard' else '',
            'lora_target_modules': str(
                self.config.get('lora_target_modules', '')) if partial_freeze_mode == 'lora_standard' else '',
            'lora_strict_match': bool(lora_strict_match) if partial_freeze_mode == 'lora_standard' else '',
            'lora_baseline': partial_freeze_mode if partial_freeze_mode == 'lora_standard' else '',
            'enable_bitfit': bool(
                self.config.get('enable_bitfit', False)) if partial_freeze_mode == 'bitfit_standard' else '',
            'train_bitfit': bool(
                self.config.get('train_bitfit', False)) if partial_freeze_mode == 'bitfit_standard' else '',
            'bitfit_scope': str(
                self.config.get('bitfit_scope', '')) if partial_freeze_mode == 'bitfit_standard' else '',
            'bitfit_include_layernorm_bias': bool(self.config.get('bitfit_include_layernorm_bias',
                                                                  True)) if partial_freeze_mode == 'bitfit_standard' else '',
            'bitfit_include_head_bias': bool(
                self.config.get('bitfit_include_head_bias', True)) if partial_freeze_mode == 'bitfit_standard' else '',
            'bitfit_strict_match': bool(
                self.config.get('bitfit_strict_match', False)) if partial_freeze_mode == 'bitfit_standard' else '',
            'bitfit_baseline': partial_freeze_mode if partial_freeze_mode == 'bitfit_standard' else '',
            'trainable_bias_params': int(sum(
                p.numel() for n, p in self.named_parameters()
                if p.requires_grad and (n.endswith('.bias') or n == 'bias')
            )),
            'trainable_head_params': int(sum(p.numel() for p in self.output.parameters() if p.requires_grad)),
        }
        module_status = {
            'backbone': train_backbone,
            'correction': train_corr and self.post_backbone_refine.correction is not None,
            'gate': train_gate and self.post_backbone_refine.gate is not None,
            'adapter': train_adapter,
            'lora': train_lora,
            'bitfit': train_bitfit,
            'head': train_head,
            'partial_freeze_mode': partial_freeze_mode,
            'train_last_tf_layers': train_last_tf_layers,
            'freeze_encoder': freeze_encoder,
        }
        print(
            'Trainable modules: '
            f'{module_status} | total params={total_params:,} | trainable params={trainable_params:,}')

    def get_trainable_summary(self):
        return getattr(self, '_trainable_summary', None)

    def forward(self, train_x, train_y, test_x, test_y, evaluate=False):
        if self.input_type == 'image':
            batch, train_num, c, h, w = train_x.shape
            batch, test_num, c, h, w = test_x.shape

            # Encode images
            x, _ = pack([
                rearrange(train_x, 'b l c h w -> (b l) c h w'),
                rearrange(test_x, 'b l c h w -> (b l) c h w'),
            ], '* c h w')
            x_enc = self.x_encoder(x)
            x_enc = rearrange(x_enc, 'bl c h w -> bl (c h w)')
            x_enc = self.x_proj(x_enc)
            train_x_enc = rearrange(x_enc[:batch * train_num], '(b l) h -> b l h', b=batch, l=train_num)
            test_x_enc = rearrange(x_enc[batch * train_num:], '(b l) h -> b l h', b=batch, l=test_num)
        elif self.input_type == 'vector':
            batch, train_num, x_dim = train_x.shape
            batch, test_num, x_dim = test_x.shape

            # Encode x vectors
            x, x_ps = pack([
                rearrange(train_x, 'b l d -> (b l) d'),
                rearrange(test_x, 'b l d -> (b l) d'),
            ], '* d')
            x_enc = self.x_encoder(x)
            train_x_enc, test_x_enc = unpack(x_enc, x_ps, '* h')
            train_x_enc = rearrange(train_x_enc, '(b l) h -> b l h', b=batch, l=train_num)
            test_x_enc = rearrange(test_x_enc, '(b l) h -> b l h', b=batch, l=test_num)
        else:
            raise NotImplementedError

        if self.output_type == 'class':
            batch, train_num = train_y.shape
            batch, test_num = test_y.shape

            # Encode labels
            y_codebook = self.y_encoder.sample_codebook(batch, device=train_y.device)
            batch, num_classes, y_len = y_codebook.shape
            train_y_code = self.y_encoder.y2code(train_y, y_codebook)  # [batch, train_num, y_len]
            test_y_code = self.y_encoder.y2code(test_y, y_codebook)  # [batch, test_num, y_len]
            train_y_enc = self.y_encoder.encode(train_y_code)  # [batch, train_num, y_len, hidden]
            test_y_enc = self.y_encoder.encode(test_y_code)  # [batch, test_num, y_len, hidden]
        elif self.output_type == 'vector':
            batch, train_num, y_dim = train_y.shape
            batch, test_num, y_dim = test_y.shape

            if self.config['output_activation'] == 'tanh':
                assert train_y.dtype == torch.uint8
                train_y = train_y.float() * 2 / 255 - 1
                test_y = test_y.float() * 2 / 255 - 1

            # Encode y vectors
            y, y_ps = pack([
                rearrange(train_y, 'b l d -> (b l) d'),
                rearrange(test_y, 'b l d -> (b l) d'),
            ], '* d')
            y_enc = self.y_encoder(y)
            train_y_enc, test_y_enc = unpack(y_enc, y_ps, '* d')
            y_len = 1
            train_y_enc = rearrange(train_y_enc, '(b l) h -> b l 1 h', b=batch, l=train_num)
            test_y_enc = rearrange(test_y_enc, '(b l) h -> b l 1 h', b=batch, l=test_num)
        else:
            raise NotImplementedError

        # Interleave train_x_enc and train_y_enc to build train sequence
        train_xy_enc, _ = pack([train_x_enc, train_y_enc], 'b l * h')
        train_xy_enc = rearrange(train_xy_enc, 'b l chunk h -> b (l chunk) h', chunk=1 + y_len)

        # Add positional encoding to train sequence
        train_xy_enc = self.pos_enc(train_xy_enc)

        loss_weight = None
        if not evaluate and self.config['distributed_loss']:
            # Sample where to attach each test example
            task_idx = torch.arange(self.config['tasks'], device=train_y.device)
            train_task = repeat(task_idx, 't -> b (t s)', b=batch, s=self.config['train_shots'])
            test_task = repeat(task_idx, 't -> b (t s)', b=batch, s=self.config['test_shots'])
            test_attachment = sample_test_attachment(train_task, test_task)  # [batch, test_num]
            if 'distributed_loss_weighted' in self.config and self.config['distributed_loss_weighted']:
                loss_weight_mean = (self.config['tasks'] + 1) / 2
                loss_weight = (self.config['tasks'] - test_task) / loss_weight_mean  # [batch, test_num]
        else:
            # Attach all test examples after the last train example
            test_attachment = repeat(
                torch.tensor(train_num - 1, dtype=torch.long, device=train_y.device),
                ' -> b n', b=batch, n=test_num)

        # Since test_attachment is the indices of train examples, convert it to token indices
        attach_test_after = test_attachment * (1 + y_len) + y_len

        test_xy_enc = self.build_test_xy(test_x_enc, test_y_enc, attach_test_after)

        xy_enc, _ = pack([train_xy_enc, test_xy_enc], 'b * h')
        train_len = train_num * (1 + y_len)
        hidden, aux_outputs = self.forward_tf(xy_enc, attach_test_after, train_len=train_len)
        analysis_cfg = self.config.get('analysis_export', {})
        collect_analysis = evaluate and analysis_cfg.get('enable', False)
        refined_hidden, refine_aux = self.post_backbone_refine(
            hidden,
            xy_enc,
            input_layout='BLD',
            collect_analysis=collect_analysis,
            train_len=train_len,
        )
        hidden_for_output = refined_hidden

        test_hidden = hidden_for_output[:, train_len:]
        logit = self.output(test_hidden)
        if self.output_type == 'class':
            logit = rearrange(logit, 'b (n y) v -> b n y v', n=test_num, y=y_len)
            loss = cross_entropy(logit, test_y_code)
        elif self.output_type == 'vector':
            if self.config['output_activation'] == 'angle':
                loss = angle_loss(logit, test_y)
            elif self.config['output_activation'] == 'tanh':
                logit = torch.tanh(logit)
                loss = reduce(((logit - test_y) ** 2), 'b n h -> b n', 'mean')
            else:
                loss = reduce(((logit - test_y) ** 2), 'b n h -> b n', 'mean')
        else:
            raise NotImplementedError

        if loss_weight is not None:
            if len(loss.shape) == 3:
                loss_weight = repeat(loss_weight, 'b n -> b n y', y=loss.shape[-1])
            loss = loss_weight * loss

        output = {
            'loss': loss,
            'logit': logit.detach(),
        }

        if refine_aux is not None:
            gate_stats = gate_aux_stats(refine_aux)
            if gate_stats:
                output.update(gate_stats)
            gates = refine_aux.get('gates')
            if gates is not None:
                mean_gate = gates.mean()
                output['gate_rate_loss'] = (mean_gate - self.config.get('r_target', 0.1)) ** 2
            if collect_analysis and 'analysis' in refine_aux:
                output['analysis'] = refine_aux['analysis']

        # Compute attention loss
        if self.config['attn_loss'] > 0:
            attn_losses = []
            for aux_output in aux_outputs:
                if 'attn_logit' in aux_output:
                    # Get attention logit of test queries and train keys
                    attn_logit = aux_output['attn_logit'][:, :self.config['attn_loss_heads'], train_len:, :train_len]

                    # Compute log-sum-exp for train keys in each task
                    attn_logit = rearrange(
                        attn_logit, 'b h q (t l) -> b h q t l',
                        t=self.config['tasks'], l=(1 + y_len) * self.config['train_shots'])
                    task_logit = torch.logsumexp(attn_logit, dim=-1)
                elif 'attn' in aux_output:
                    # Get attention of test queries and train keys
                    attn = aux_output['attn'][:, :self.config['attn_loss_heads'], train_len:, :train_len]

                    # Compute sum for train keys in each task
                    task_attn = reduce(
                        attn, 'b h q (t l) -> b h q t', reduction='sum',
                        t=self.config['tasks'], l=(1 + y_len) * self.config['train_shots'])
                    task_logit = (task_attn + 1e-9).log()
                else:
                    raise RuntimeError('No attention logit or attention found in aux_output')

                task_gt = repeat(
                    torch.arange(self.config['tasks'], device=task_logit.device),
                    't -> b h (t s c)',
                    b=batch, h=self.config['attn_loss_heads'],
                    s=self.config['test_shots'], c=y_len)
                attn_loss = cross_entropy(task_logit, task_gt).mean()
                attn_losses.append(attn_loss)
            output['attn_losses'] = attn_losses

        if not evaluate:
            return output

        ############
        # Evaluate #
        ############

        if self.output_type == 'vector':
            # Simply use the loss as evaluation
            if self.config['output_activation'] == 'tanh':
                output['logit'] = logit
            return output

        if self.config['y_len'] == 1:
            # Simple evaluation
            y_code_pred = logit.argmax(dim=-1)
            evaluation = rearrange(y_code_pred == test_y_code, 'b n 1 -> b n')
            output['evaluation'] = evaluation
            return output

        # Evaluate by comparing the likelihoods of all possible y_codes
        with torch.no_grad():
            # When evaluating, every test example is attached after the last train example
            attach_test_after = repeat(
                torch.tensor(train_len - 1, dtype=torch.long, device=train_y.device),
                ' -> b n', b=batch, n=test_num)

            c_losses = []
            for c in range(num_classes):
                c_y_code = y_codebook[:, c]
                c_y_code = repeat(c_y_code, f'b y -> b {test_num} y')
                c_y_enc = self.y_encoder.encode(c_y_code)  # [batch, test_num, y_len, hidden]
                c_xy_enc = self.build_test_xy(test_x_enc, c_y_enc, attach_test_after)
                # c_hidden, _ = self.forward_tf(
                #     c_xy_enc, attach_test_after, train_len=0, past_states=train_states, return_states=False)
                hidden, _ = self.forward_tf(
                    pack([train_xy_enc, c_xy_enc], 'b * h')[0], attach_test_after, train_len=train_len)

                c_hidden = hidden[:, train_len:]
                c_logit = self.output(c_hidden)
                c_logit = rearrange(c_logit, 'b (t y) v -> b t y v', t=test_num, y=y_len)
                c_loss = cross_entropy(c_logit, c_y_code)
                c_loss = reduce(c_loss, 'b t y -> b t', 'sum')
                c_losses.append(c_loss)
            c_losses, _ = pack(c_losses, 'b t *')
            pred = c_losses.argmin(dim=-1)
            evaluation = pred == test_y
            output['evaluation'] = evaluation

        return output

    def build_test_xy(self, test_x_enc, test_y_enc, attach_test_after):
        batch, test_num, y_len, hidden = test_y_enc.shape
        test_xy_enc, _ = pack([test_x_enc, test_y_enc[:, :, :-1]], 'b l * h')  # [batch, test_num, y_len, hidden]

        # Add positional encoding to test examples
        pos_idx = repeat(attach_test_after, f'b n -> b n {y_len}')
        pos_idx = pos_idx + rearrange(torch.arange(1, 1 + y_len, device=test_x_enc.device), 'l -> 1 1 l')
        test_pe = self.pos_enc.pe[pos_idx]  # [batch, test_num, y_len, hidden]
        test_xy_enc = test_xy_enc + test_pe
        test_xy_enc = rearrange(test_xy_enc, 'b n y h -> b (n y) h')

        return test_xy_enc

    def forward_tf(self, xy_enc, attach_test_after, train_len=0, past_states=None, return_states=False):
        if past_states is None:
            past_states = [None] * len(self.tf_layers)

        aux_outputs = []
        hidden = xy_enc
        for tf_layer, past_state in zip(self.tf_layers, past_states):
            hidden, aux_output = tf_layer(
                hidden, attach_test_after=attach_test_after, train_len=train_len,
                past_state=past_state, return_state=return_states)
            aux_outputs.append(aux_output)

        return hidden, aux_outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len, requires_grad=False):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.pe = None
        self.requires_grad = requires_grad
        self.build_pe(max_len)

    def build_pe(self, max_len):
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(max_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe, requires_grad=self.requires_grad)

    def forward(self, x: Tensor, offset=0) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch, seq_len, hidden]
            offset: int, offset of the first position
        """
        x_len = x.size(-2)
        return x + self.pe[offset:offset + x_len]
