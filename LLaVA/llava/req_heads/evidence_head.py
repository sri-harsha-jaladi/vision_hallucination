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


class ValueAdapterMLP(nn.Module):

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
    
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SingleHeadQueryAwareScorer(nn.Module):
    """
    Single-head, query-aware per-token classifier with a shared set of image tokens across the batch.

    Shapes:
      img_tokens:  [N, D]      (shared across the batch)
      text_tokens: [B, D]      (one target text token per item)
      labels:      [B, N]      (optional, 0/1)

    Returns:
      logits:      [B, N]
      loss:        scalar or None
    """
    def __init__(self, d: int, d_k: int = 512, mlp_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln_q  = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)

        self.W_q = nn.Linear(d, d_k, bias=False)
        self.W_k = nn.Linear(d, d_k, bias=False)

        self.scale = d_k ** 0.5
        self.dropout = nn.Dropout(dropout)

        # Feature per token: [k_i, q, q⊙k_i, |q-k_i|, sim] -> size = 4*d_k + 1
        feat_dim = 4 * d_k + 1
        self.head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(
        self,
        img_tokens: torch.Tensor,          # [N, D]
        text_tokens: torch.Tensor,          # [B, D]
        labels: Optional[torch.Tensor] = None,  # [B, N]
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if img_tokens.dim() != 2 or text_tokens.dim() != 2:
            raise ValueError(f"img_tokens [N,D], text_tokens [B,D], got {img_tokens.shape}, {text_tokens.shape}")

        N, D = img_tokens.shape
        B, Dt = text_tokens.shape
        if Dt != D:
            raise ValueError(f"Dim mismatch: D_img={D} vs D_txt={Dt}")

        # project once (shared K over batch)
        # K: [N, d_k], Q: [B, d_k]
        K = self.W_k(self.ln_kv(img_tokens))      # [N, d_k]
        Q = self.W_q(self.ln_q(text_tokens))      # [B, d_k]

        # similarity per (b, i): [B, N]
        # sim[b, i] = (Q[b] · K[i]) / sqrt(d_k)
        sim = (Q @ K.T) / self.scale              # [B, N]
        sim = self.dropout(sim)
        sim_feat = sim.unsqueeze(-1)              # [B, N, 1]

        # Expand for tokenwise features without copying unnecessarily
        # q_exp: [B, N, d_k], k_exp: [B, N, d_k]
        q_exp = Q.unsqueeze(1).expand(B, N, -1)
        k_exp = K.unsqueeze(0).expand(B, N, -1)

        prod  = q_exp * k_exp                     # [B, N, d_k]
        diff  = (q_exp - k_exp).abs()             # [B, N, d_k]

        feats = torch.cat([k_exp, q_exp, prod, diff, sim_feat], dim=-1)  # [B, N, 4*d_k+1]
        logits = self.head(feats).squeeze(-1)     # [B, N]

        loss = None
        if labels is not None:
            if labels.shape != (B, N):
                raise ValueError(f"labels must be [B,N]={B,N}, got {labels.shape}")
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=pos_weight, reduction=reduction
            )
        return logits, loss

    @torch.no_grad()
    def predict_proba(self, img_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(img_tokens, text_tokens, labels=None)
        return torch.sigmoid(logits)  # [B, N]


# M: [B, N] probabilities (or logits -> pass through sigmoid first)
# optional word reliabilities r: [B] in [0,1] (e.g., 1 - token entropy, or |p-0.5|)
# soft-OR with reliability weights



# --- helpers (reuse your own if you already defined them) ---
def aggregate_importance(M, r=None, eps=1e-6, temp=0.8):
    # M: [B, N] in [0,1]
    if r is None: r = M.new_ones(M.size(0))
    r = r / (r.sum() + eps)
    log_comp = (r.unsqueeze(1) * torch.log1p(-M.clamp(0,1) + eps)).sum(dim=0)
    w = 1.0 - torch.exp(log_comp)                         # [N]
    # optional temperature calibration
    w = torch.sigmoid((torch.logit(w.clamp(1e-6,1-1e-6)))/temp)
    return w

def smooth_and_sparsify(w, H=24, W=24, sigma=1.0, keep_ratio=0.25, floor=0.05):
    # gaussian smooth on 24x24 then soft top-k style sparsify
    k = int(2*round(2.5*sigma)+1)
    g = torch.arange(k, device=w.device) - k//2
    gauss1d = torch.exp(-(g**2)/(2*sigma**2)); gauss1d /= gauss1d.sum()
    G = (gauss1d[:,None] @ gauss1d[None,:]).view(1,1,k,k)

    Wmap = w.view(1,1,H,W)
    pad = (k//2,)*4
    Wmap = torch.nn.functional.conv2d(torch.nn.functional.pad(Wmap, pad, mode='replicate'), G)
    w_sm = Wmap.view(-1)

    k_keep = max(1, int(keep_ratio * w_sm.numel()))
    tau = 0.25
    scores = (w_sm - w_sm.topk(k_keep).values.min()).clamp_min(0)
    w_sp = torch.softmax(scores/tau, dim=0)

    w_final = floor + (1 - floor) * (w_sp / w_sp.max().clamp_min(1e-6))
    return w_final.clamp(0,1)

def weights_to_bias(w, beta=2.0, clip=6.0):
    return (beta * torch.logit(w.clamp(1e-5, 1-1e-5))).clamp(-clip, clip)


def agg_max(M):
    # Hard union: per-token max over words
    return M.max(dim=0).values  # [N]

# --- toggleable builder ---
def build_importance(M, do_smoothing: bool = False,
                     H=24, W=24, sigma=1.0, keep_ratio=0.25,
                     temp=0.8, beta=2.0, clip=6.0):
    """M: [B, N] probs in [0,1] -> (w, b)"""
    # w = aggregate_importance(M, temp=temp)                # [N]
    w = agg_max(M)
    if do_smoothing:
        w = smooth_and_sparsify(w, H=H, W=W, sigma=sigma, keep_ratio=keep_ratio)
    else:
        # simple safety: clamp + mild normalization so scales are comparable
        w = w.clamp(0,1)
        # optional: renormalize to similar mean as smoothed run
        # w = (w - w.mean()) / (w.std()+1e-6); w = torch.sigmoid(w)
    b = weights_to_bias(w, beta=beta, clip=clip)
    return w, b
