# model.py
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import DROPOUT, N_EMBD, N_HEAD, N_LAYER, USE_GRAD_CKPT

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int = N_EMBD, n_head: int = N_HEAD, dropout: float = DROPOUT):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head

        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))

class FeedForward(nn.Module):
    def __init__(self, n_embd: int = N_EMBD, dropout: float = DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        self._act_out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        self._act_out = h.detach()
        return self.drop(self.fc2(h))

class Block(nn.Module):
    def __init__(
        self,
        attn_res: bool = True,
        ffn_res: bool = True,
        n_embd: int = N_EMBD,
        n_head: int = N_HEAD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)
        self.attn_res = attn_res
        self.ffn_res = ffn_res
        self._h_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sa = self.attn(self.ln1(x))
        x = x + sa if self.attn_res else sa

        ff = self.ffwd(self.ln2(x))
        x = x + ff if self.ffn_res else ff

        self._h_norm = x.detach().norm(2, dim=-1).mean().item()
        return x

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = N_LAYER,
        n_embd: int = N_EMBD,
        n_head: int = N_HEAD,
        dropout: float = DROPOUT,
        attn_res: bool = True,
        ffn_res: bool = True,
        use_grad_ckpt: bool = USE_GRAD_CKPT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(attn_res=attn_res, ffn_res=ffn_res, n_embd=n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.tok_emb.weight = self.lm_head.weight
        self.use_grad_ckpt = use_grad_ckpt
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * len(self.blocks)))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.block_size}")

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            if self.use_grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_ffn_layers(self):
        return [b.ffwd for b in self.blocks]

    def get_blocks(self):
        return list(self.blocks)

    def n_params(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e6