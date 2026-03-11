import torch
import torch.nn as nn

from models.correction_head import CorrectionHead
from models.highway_gate import HighwayScalarGate


class PostBackboneRefineAndGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config['hidden_dim']
        self.enable_correction = config.get('enable_correction', True)
        self.enable_highway_gate = config.get('enable_highway_gate', True)
        self.use_learnable_h0 = config.get('use_learnable_h0', False)

        self.correction = CorrectionHead(
            hidden_dim=self.hidden_dim,
            corr_hidden_ratio=config.get('corr_hidden_ratio', 0.25),
            corr_dropout=config.get('corr_dropout', 0.1),
            corr_zero_init=config.get('corr_zero_init', True),
            use_prev_state_in_corr=config.get('use_prev_state_in_corr', False),
        ) if self.enable_correction else None

        self.gate = HighwayScalarGate(
            hidden_dim=self.hidden_dim,
            gate_type=config.get('gate_type', 'mlp2'),
            gate_hidden_ratio=config.get('gate_hidden_ratio', 0.25),
            gate_dropout=config.get('gate_dropout', 0.1),
            gate_bias_init=config.get('gate_bias_init', -3.0),
            gate_input_type=config.get('gate_input_type', 'prev_z'),
        ) if self.enable_highway_gate else None

        if self.use_learnable_h0:
            self.h0 = nn.Parameter(torch.zeros(self.hidden_dim))
        else:
            self.register_parameter('h0', None)

    @staticmethod
    def _to_bld(x, input_layout='BLD'):
        if x.dim() != 3:
            raise ValueError(f'Expected 3D tensor, got shape {x.shape}')
        if input_layout == 'BLD':
            return x, 'BLD'
        if input_layout == 'LBD':
            return x.transpose(0, 1), 'LBD'
        raise ValueError(f'Unknown input_layout: {input_layout}')

    @staticmethod
    def _from_bld(x, layout):
        return x if layout == 'BLD' else x.transpose(0, 1)

    def forward(self, u_hidden, z_hidden, valid_mask=None, input_layout='BLD'):
        u_bld, layout = self._to_bld(u_hidden, input_layout=input_layout)
        z_bld, _ = self._to_bld(z_hidden, input_layout=input_layout)

        if (not self.enable_correction) and (not self.enable_highway_gate):
            aux = {
                'gates': None,
                'deltas': None,
                'candidates': u_bld,
                'base_hidden': u_bld,
                'refined_hidden': u_bld,
            }
            return u_hidden, aux

        batch, seq_len, hidden_dim = u_bld.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f'Hidden dim mismatch: {hidden_dim} vs {self.hidden_dim}')

        if self.h0 is None:
            prev = torch.zeros(batch, hidden_dim, device=u_bld.device, dtype=u_bld.dtype)
        else:
            prev = self.h0.unsqueeze(0).expand(batch, -1)

        deltas, cands, gates, refined = [], [], [], []

        if self.enable_correction and not self.enable_highway_gate:
            prev_corr = prev
            for t in range(seq_len):
                delta_t = self.correction(u_bld[:, t], z_bld[:, t], prev=prev_corr)
                cand_t = u_bld[:, t] + delta_t
                deltas.append(delta_t)
                cands.append(cand_t)
                refined.append(cand_t)
                prev_corr = cand_t
            h_bld = torch.stack(refined, dim=1)
            delta_bld = torch.stack(deltas, dim=1)
            cand_bld = torch.stack(cands, dim=1)
            aux = {
                'gates': None,
                'deltas': delta_bld,
                'candidates': cand_bld,
                'base_hidden': u_bld,
                'refined_hidden': h_bld,
            }
            return self._from_bld(h_bld, layout), aux

        for t in range(seq_len):
            u_t = u_bld[:, t]
            z_t = z_bld[:, t]

            if self.enable_correction:
                delta_t = self.correction(u_t, z_t, prev=prev)
                cand_t = u_t + delta_t
            else:
                delta_t = torch.zeros_like(u_t)
                cand_t = u_t

            g_t = self.gate(prev, z_t, cand_t=cand_t)
            h_t = (1 - g_t) * prev + g_t * cand_t

            if valid_mask is not None:
                valid_t = valid_mask[:, t].unsqueeze(-1).to(dtype=h_t.dtype)
                h_t = valid_t * h_t + (1 - valid_t) * prev
                g_t = g_t * valid_t

            deltas.append(delta_t)
            cands.append(cand_t)
            gates.append(g_t)
            refined.append(h_t)
            prev = h_t

        h_bld = torch.stack(refined, dim=1)
        aux = {
            'gates': torch.stack(gates, dim=1),
            'deltas': torch.stack(deltas, dim=1),
            'candidates': torch.stack(cands, dim=1),
            'base_hidden': u_bld,
            'refined_hidden': h_bld,
        }
        return self._from_bld(h_bld, layout), aux


def gate_aux_stats(aux, valid_mask=None):
    stats = {}
    gates = aux.get('gates')
    if gates is not None:
        gate_values = gates.squeeze(-1)
        if valid_mask is not None:
            valid = valid_mask.bool()
            gate_values = gate_values[valid]
        else:
            gate_values = gate_values.reshape(-1)
        if gate_values.numel() > 0:
            stats.update({
                'mean_gate': gate_values.mean(),
                'std_gate': gate_values.std(unbiased=False),
                'min_gate': gate_values.min(),
                'max_gate': gate_values.max(),
            })

    if aux.get('deltas') is not None:
        stats['mean_delta_norm'] = aux['deltas'].norm(dim=-1).mean()
    if aux.get('candidates') is not None:
        stats['candidate_norm'] = aux['candidates'].norm(dim=-1).mean()
    if aux.get('refined_hidden') is not None:
        stats['refined_hidden_norm'] = aux['refined_hidden'].norm(dim=-1).mean()
    return stats