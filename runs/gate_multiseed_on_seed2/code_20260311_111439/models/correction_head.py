import torch
import torch.nn as nn


class CorrectionHead(nn.Module):
    def __init__(
            self,
            hidden_dim,
            corr_hidden_ratio=0.25,
            corr_dropout=0.1,
            corr_zero_init=True,
            use_prev_state_in_corr=False):
        super().__init__()
        self.use_prev_state_in_corr = use_prev_state_in_corr
        in_dim = hidden_dim * (3 if use_prev_state_in_corr else 2)
        corr_hidden_dim = max(int(hidden_dim * corr_hidden_ratio), 16)

        self.prev_ln = nn.LayerNorm(hidden_dim)
        self.u_ln = nn.LayerNorm(hidden_dim)
        self.z_ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(in_dim, corr_hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(corr_dropout)
        self.fc2 = nn.Linear(corr_hidden_dim, hidden_dim)

        if corr_zero_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, u_t, z_t, prev=None):
        parts = []
        if self.use_prev_state_in_corr:
            if prev is None:
                prev = torch.zeros_like(u_t)
            parts.append(self.prev_ln(prev))
        parts.extend([self.u_ln(u_t), self.z_ln(z_t)])
        inp = torch.cat(parts, dim=-1)
        hidden = self.dropout(self.act(self.fc1(inp)))
        return self.fc2(hidden)