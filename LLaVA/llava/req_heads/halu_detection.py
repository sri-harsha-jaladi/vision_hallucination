import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple



class HaluDetectionHead30(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim1=1024, hidden_dim2=512, num_classes=3, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class HaluDetectionHead24(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim1: int = 1024,
        hidden_dim2: int = 512,
        num_classes: int = 2,          # <-- use 1 for binary (recommended). Use 2 for softmax(2).
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,                          # [B, D]
        labels: Optional[torch.Tensor] = None,    # binary: [B] or [B,1] in {0,1}; multi-class: [B] in {0..C-1}
        *,
        pos_weight: Optional[torch.Tensor] = None,       # used only when num_classes == 1
        class_weights: Optional[torch.Tensor] = None,    # used when num_classes >= 2
        label_smoothing: float = 0.0,                    # used when num_classes >= 2
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        logits = self.net(x)

        # Binary single-logit normalization
        if self.num_classes == 1:
            logits = logits.squeeze(-1)  # [B]

        loss = None
        if labels is not None:
            if self.num_classes == 1:
                # BCE-with-logits expects float targets in {0,1}
                if labels.dim() == 2 and labels.size(1) == 1:
                    labels = labels.squeeze(1)
                labels = labels.float()
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight, reduction=reduction
                )
            elif self.num_classes >= 2:
                # Cross-entropy expects class indices in {0..C-1}
                if labels.dim() != 1:
                    labels = labels.view(-1)
                labels = labels.long()
                # class_weights: Tensor[C] or None
                loss = F.cross_entropy(
                    logits, labels,
                    weight=class_weights,
                    reduction=reduction,
                    label_smoothing=label_smoothing
                )
            else:
                raise ValueError(
                    f"num_classes must be 1 (binary single-logit) or >=2 (softmax), got {self.num_classes}"
                )

        return logits, loss

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probabilities in [0,1]."""
        logits, _ = self.forward(x, labels=None)
        if self.num_classes == 1:
            return torch.sigmoid(logits)               # [B]
        else:  # num_classes == 2
            return F.softmax(logits, dim=-1)[:, 1]     # P(class=1), [B]



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SingleHeadDetectionClassifier(nn.Module):
    """
    Attention -> Single MLP head for per-target-token hallucination classification.

    Inputs:
      img_tokens:  [N, D]        (image patch tokens, shared across the batch)
      text_tokens: [B, D]        (one target text token per item)
      labels:     [B] or [B,1]   (optional, 0/1; 1 = hallucinated)

    Returns:
      logits:     [B]            (per-token hallucination logit; >0 => hallucinated)
      loss:       scalar or None
    """
    def __init__(
        self,
        d: int,
        d_k: int = 1024,
        mlp_hidden: int = 1024,
        dropout: float = 0.1,
        attn_temp: float = 1.0,   # temperature on attention logits (1.0 = none)
    ):
        super().__init__()
        # Pre-norms
        self.ln_q  = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)

        # Projections
        self.W_q = nn.Linear(d, d_k, bias=False)
        self.W_k = nn.Linear(d, d_k, bias=False)
        self.W_v = nn.Linear(d, d_k, bias=False)

        self.scale   = d_k ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.attn_temp = attn_temp

        # Single classification head over compact features
        # Feature vector: [q, ctx, q⊙ctx, |q-ctx|, sim_mean, sim_max, sim_lse]
        # Size = 4*d_k + 3
        feat_dim = 4 * d_k + 3
        self.head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(
        self,
        img_tokens: torch.Tensor,            # [N, D]
        text_tokens: torch.Tensor,           # [B, D]
        labels: Optional[torch.Tensor] = None,  # [B] or [B,1]
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if img_tokens.dim() != 2 or text_tokens.dim() != 2:
            raise ValueError(f"img_tokens [N,D], text_tokens [B,D], got {img_tokens.shape}, {text_tokens.shape}")

        N, D = img_tokens.shape
        B, Dt = text_tokens.shape
        if Dt != D:
            raise ValueError(f"Dim mismatch: D_img={D} vs D_txt={Dt}")

        # Normalize + project
        K = self.W_k(self.ln_kv(img_tokens))      # [N, d_k]
        V = self.W_v(self.ln_kv(img_tokens))      # [N, d_k]
        Q = self.W_q(self.ln_q(text_tokens))      # [B, d_k]

        # Attention over ALL image patches (no top-k)
        # sim[b, i] = (Q[b] · K[i]) / sqrt(d_k)
        sim = (Q @ K.T) / self.scale              # [B, N]
        if self.attn_temp is not None and self.attn_temp > 0:
            sim = sim / self.attn_temp
        sim = self.dropout(sim)

        attn = F.softmax(sim, dim=1)              # [B, N]
        # Context per target token
        # ctx[b] = Σ_i attn[b,i] * V[i]
        ctx = attn @ V                            # [B, d_k]

        # Compose minimal, information-dense features
        prod = Q * ctx                             # [B, d_k]
        diff = (Q - ctx).abs()                     # [B, d_k]
        sim_mean = sim.mean(dim=1, keepdim=True)   # [B,1]
        sim_max  = sim.amax(dim=1, keepdim=True)   # [B,1]
        sim_lse  = torch.logsumexp(sim, dim=1, keepdim=True)  # [B,1]

        feats = torch.cat([Q, ctx, prod, diff, sim_mean, sim_max, sim_lse], dim=-1)  # [B, 4*d_k+3]

        logits = self.head(feats).squeeze(-1)     # [B]
        
        # labels = labels.float()
        # n_pos = labels.sum()
        # n_neg = labels.numel() - n_pos

        # if n_pos == 0 or n_neg == 0:
        #     # fallback: uniform weights
        #     weight = torch.ones_like(labels, dtype=torch.float32)
        # else:
        #     w_pos = n_neg / (n_pos + n_neg)
        #     w_neg = n_pos / (n_pos + n_neg)
        #     weight = torch.where(labels == 1, w_pos, w_neg)
        

        loss = None
        if labels is not None:
            labels = labels.view(-1).float()      # [B]
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, reduction=reduction, pos_weight=pos_weight#, weight=weight 
            )
        return logits, loss

    @torch.no_grad()
    def predict_proba(self, img_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(img_tokens, text_tokens, labels=None)
        return torch.sigmoid(logits)  # [B]

class EvidenceConditionedHallucinationDetector(nn.Module):
    """
    Query-aware hallucination detector that conditions on frozen evidence maps.

    Inputs:
      img_tokens:       [N, D]  (shared across batch; layer-24 image tokens)
      text_tokens:      [B, D]  (one target text token per item; layer-24)
      evidence_logits:  [B, N]  (from frozen evidence head; logits or probs)
      labels (optional):[B]     (0/1; token hallucinated?)

    Returns:
      logits: [B]  (hallucination logit per token)
      loss:   scalar or None
    """
    def __init__(self, d: int, d_k: int = 1024, mlp_hidden: int = 1024,
                 dropout: float = 0.1,):
        super().__init__()

        # normalize + project
        self.ln_q  = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.W_q   = nn.Linear(d, d_k, bias=False)
        self.W_k   = nn.Linear(d, d_k, bias=False)

        self.scale = d_k ** 0.5
        self.dropout = nn.Dropout(dropout)

        # trainable evidence temperature (calibration inside detector only)
        self.log_gamma = nn.Parameter(torch.zeros(1))  # gamma = exp(log_gamma) ∈ (0, +inf)

        # final head over concatenated features
        # features: [q, ctx_e, ctx_sim, q⊙ctx_e, |q-ctx_e|, stats(e)] -> 4*d_k + d_k + S
        # where S = 6 stats (mean, max, entropy, topk_mean, mass>0.5, align score)
        stats_dim = 6
        feat_dim  = 5 * d_k + stats_dim # q, ctx_e, ctx_sim, q⊙ctx_e, |q-ctx_e|
        self.head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def _stats_from_evidence(self, e: torch.Tensor, sim: torch.Tensor, thresh: float = 0.4) -> torch.Tensor:
        B, N = e.shape
        eps = 1e-8

        # mean / max
        e_mean = e.mean(dim=1, keepdim=True)
        e_max  = e.max(dim=1, keepdim=True).values

        # entropy (binary-prob entropy, averaged across patches)
        ent   = -(e.clamp(eps,1-eps)*torch.log(e.clamp(eps,1-eps)) +
                (1-e).clamp(eps,1-eps)*torch.log((1-e).clamp(eps,1-eps)))
        e_ent = ent.mean(dim=1, keepdim=True)

        # thresholded mean over patches where e > thresh
        mask = (e > thresh).float()                     # [B, N]
        count = mask.sum(dim=1, keepdim=True)           # [B, 1]
        # If no patch passes the threshold, fall back to the global mean to avoid NaNs.
        thresh_sum  = (e * mask).sum(dim=1, keepdim=True)            # [B, 1]
        thresh_mean = torch.where(count > 0, thresh_sum / (count + eps), e_mean)

        # mass above threshold (fraction of patches passing)
        mass_gt = count / float(N)                      # [B, 1]

        # alignment with similarity attention
        sim_w = F.softmax(sim, dim=1)                   # [B, N]
        align = (e * sim_w).sum(dim=1, keepdim=True)    # [B, 1]

        return torch.cat([e_mean, e_max, e_ent, thresh_mean, mass_gt, align], dim=1)


    def forward(
        self,
        img_tokens: torch.Tensor,          # [N, D]
        text_tokens: torch.Tensor,         # [B, D]
        evidence_logits: torch.Tensor,     # [B, N] (from frozen scorer)
        labels: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if img_tokens.dim() != 2 or text_tokens.dim() != 2:
            raise ValueError(f"img_tokens [N,D], text_tokens [B,D] expected; got {img_tokens.shape}, {text_tokens.shape}")

        N, D = img_tokens.shape
        B, Dt = text_tokens.shape
        if Dt != D:
            raise ValueError(f"D mismatch: D_img={D} vs D_txt={Dt}")
        if evidence_logits.shape != (B, N):
            raise ValueError(f"evidence_logits must be [B,N]={B,N}, got {evidence_logits.shape}")

        # Project once
        K = self.W_k(self.ln_kv(img_tokens))      # [N, d_k]
        Q = self.W_q(self.ln_q(text_tokens))      # [B, d_k]

        # Query–image similarity
        sim = (Q @ K.T) / self.scale              # [B, N]
        sim = self.dropout(sim)

        # Evidence probs (detach to keep evidence head frozen)
        with torch.no_grad():
            e_in = evidence_logits.detach()
        # Learnable temperature on evidence within detector
        gamma = torch.exp(self.log_gamma).clamp_min(1e-3)
        e = torch.sigmoid(e_in / gamma)           # [B, N] calibrated evidence

        # Evidence-weighted context from image keys
        e_norm = e / (e.sum(dim=1, keepdim=True) + 1e-8)
        ctx_e  = e_norm @ K                        # [B, d_k]

        # Similarity-weighted context (second pooling stream)
        w_sim  = F.softmax(sim, dim=1)            # [B, N]
        ctx_s  = w_sim @ K                        # [B, d_k]

        # Feature assembly
        prod   = Q * ctx_e                         # [B, d_k]
        diff   = (Q - ctx_e).abs()                 # [B, d_k]
        stats  = self._stats_from_evidence(e, sim) # [B, 6]

        feats = torch.cat([Q, ctx_e, ctx_s, prod, diff, stats], dim=1)  # [B, 4*d_k + 6]
        logits = self.head(feats).squeeze(-1)                           # [B]

        loss = None
        if labels is not None:
            if labels.shape != (B,):
                raise ValueError(f"labels must be [B], got {labels.shape}")
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=pos_weight, reduction=reduction
            )
        return logits, loss

    @torch.no_grad()
    def predict_proba(self, img_tokens, text_tokens, evidence_logits):
        logits, _ = self.forward(img_tokens, text_tokens, evidence_logits, labels=None)
        return torch.sigmoid(logits)  # [B


