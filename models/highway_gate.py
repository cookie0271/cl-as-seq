import torch
import torch.nn as nn


class HighwayScalarGate(nn.Module):
    def __init__(
            self,
            hidden_dim,
            gate_type='mlp2',
            gate_hidden_ratio=0.25,
            gate_dropout=0.1,
            gate_bias_init=-3.0,
            gate_input_type='prev_z'):
        super().__init__()
        self.gate_type = gate_type
        self.gate_input_type = gate_input_type

        if gate_input_type == 'prev_z':
            in_mult = 2
        elif gate_input_type == 'prev_cand':
            in_mult = 2
        elif gate_input_type == 'prev_z_cand':
            in_mult = 3
        else:
            raise ValueError(f'Unknown gate_input_type: {gate_input_type}')

        in_dim = hidden_dim * in_mult
        self.prev_ln = nn.LayerNorm(hidden_dim)
        self.z_ln = nn.LayerNorm(hidden_dim) if gate_input_type in {'prev_z', 'prev_z_cand'} else nn.Identity()
        self.cand_ln = nn.LayerNorm(hidden_dim) if gate_input_type in {'prev_cand', 'prev_z_cand'} else nn.Identity()

        if gate_type == 'mlp2':
            gate_hidden_dim = max(int(hidden_dim * gate_hidden_ratio), 16)
            self.net = nn.Sequential(
                nn.Linear(in_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(gate_dropout),
                nn.Linear(gate_hidden_dim, 1),
            )
            nn.init.constant_(self.net[-1].bias, gate_bias_init)
        elif gate_type == 'linear':
            self.net = nn.Linear(in_dim, 1)
            nn.init.constant_(self.net.bias, gate_bias_init)
        else:
            raise ValueError(f'Unknown gate_type: {gate_type}')

    def forward(self, prev, z_t, cand_t=None, return_logits=False):
        prev_ln = self.prev_ln(prev)
        z_ln = self.z_ln(z_t)

        if self.gate_input_type == 'prev_z':
            gate_inp = torch.cat([prev_ln, z_ln], dim=-1)
        elif self.gate_input_type == 'prev_cand':
            if cand_t is None:
                raise ValueError('cand_t is required for gate_input_type=prev_cand')
            gate_inp = torch.cat([prev_ln, self.cand_ln(cand_t)], dim=-1)
        elif self.gate_input_type == 'prev_z_cand':
            if cand_t is None:
                raise ValueError('cand_t is required for gate_input_type=prev_z_cand')
            gate_inp = torch.cat([prev_ln, z_ln, self.cand_ln(cand_t)], dim=-1)
        else:
            raise ValueError(f'Unknown gate_input_type: {self.gate_input_type}')

        gate_logits = self.net(gate_inp)
        gate = torch.sigmoid(gate_logits)
        if return_logits:
            return gate, gate_logits
        return gate