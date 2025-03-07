from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class TransformerMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, use_geglu: bool):
        super().__init__()
        self.use_geglu = use_geglu
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        if self.use_geglu:
            self.w_g = nn.Linear(d_model, d_ff)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_norm = self.norm(x)
        x_transform = self.w_up(x_norm)
        if self.use_geglu:
            x_transform = x_transform * self.w_g(x_norm)
        x_transform = F.gelu(x_transform)
        x_transform = self.w_down(x_transform)
        return x + x_transform


class TransformerAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_heads: Optional[int] = None):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.kv_heads = n_heads if kv_heads is None else kv_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.d_head * self.kv_heads, bias=False)
        self.w_v = nn.Linear(d_model, self.d_head * self.kv_heads, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)
        self.register_buffer(
            "scale", torch.tensor(np.sqrt(d_model), dtype=torch.float32)
        )

        theta = torch.pow(10000, -2 * (torch.arange(0, self.d_head) // 2) / self.d_head)
        max_seq_len = 8096
        cos = torch.outer(torch.arange(max_seq_len), theta)
        sin = torch.outer(torch.arange(max_seq_len), theta)
        self.register_buffer("cos", torch.cos(cos))
        self.register_buffer("sin", torch.sin(sin))
        self.sin[1::2] = -self.sin[1::2]

    def forward(
        self, x: torch.tensor, attention_mask: Optional[torch.tensor] = None
    ) -> torch.tensor:
        b, n, _ = x.shape
        x_norm = self.norm(x)  # (B, N, D)
        q = self.w_q(x_norm)  # (B, N, H * A)
        k, v = self.w_k(x_norm), self.w_v(x_norm)  # (B, N, K * A)

        # reshaping
        q = rearrange(q, "b n (h a) -> b h n a", h=self.n_heads)
        k = rearrange(k, "b n (k a) -> b k n a", k=self.kv_heads)
        v = rearrange(v, "b n (k a) -> b k n a", k=self.kv_heads)

        # rotary embedding
        cos = self.cos[:n]
        sin = self.sin[:n]
        q = q * cos + q * sin
        k = k * cos + k * sin
        # matmulling
        reps = self.n_heads // self.kv_heads
        k = k.repeat(1, reps, 1, 1)
        v = v.repeat(1, reps, 1, 1)
        qkt = q @ k.transpose(2, 3) / self.scale

        if attention_mask is not None:
            # Expand mask for all heads
            expanded_mask = attention_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            qkt = qkt.masked_fill(~expanded_mask, float("-inf"))

        s = torch.softmax(qkt, dim=-1) @ v

        s = rearrange(s, "b h n a -> b n (h a)")
        o = self.w_o(s)

        return s + o


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        use_geglu: bool,
        n_heads: int,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.attention = TransformerAttention(d_model, n_heads, kv_heads)
        self.mlp = TransformerMLP(d_model, d_ff, use_geglu)

    def forward(
        self, x: torch.tensor, attention_mask: Optional[torch.tensor] = None
    ) -> torch.tensor:
        x = self.attention(x, attention_mask)
        x = self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        use_geglu: bool = True,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_geglu = use_geglu
        self.kv_heads = n_heads if kv_heads is None else kv_heads
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, d_ff, use_geglu, n_heads, kv_heads)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.RMSNorm(d_model)
        self.w_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, x: torch.tensor, attention_mask: Optional[torch.tensor] = None
    ) -> torch.tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.norm(x)
        x = self.w_proj(x)
        return x
