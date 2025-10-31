import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryAdapterMLP(nn.Module):

    def __init__(
        self,
        d: int,
        d_k: int,
        *,
        hidden: int | None = None,
        dropout: float = 0.0,
        ln_eps: float = 1e-5,
        l2_normalize: bool = False,  # set True if you plan cosine scoring
    ):
        super().__init__()
        hidden = hidden or (d // 2)

        self.ln = nn.LayerNorm(d, eps=ln_eps)
        self.fc1 = nn.Linear(d, hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_k, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.l2_normalize = l2_normalize


    def forward(self, h_txt: torch.Tensor) -> torch.Tensor:
        # Accept [T, d] or [B, T, d]
        is_batched = (h_txt.dim() == 3)
        if not is_batched:
            h_txt = h_txt.unsqueeze(0)  # -> [1, T, d]

        x = self.ln(h_txt)             # [B, T, d]
        x = self.fc1(x)                # [B, T, hidden]
        x = self.act(x)
        x = self.dropout(x)
        q = self.fc2(x)                # [B, T, d_k]

        if self.l2_normalize:
            q = F.normalize(q, p=2, dim=-1)

        if not is_batched:
            q = q.squeeze(0)           # -> [T, d_k]
        return q