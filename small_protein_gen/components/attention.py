from functools import partial
from small_protein_gen.components.pos_embedding import (
    PositionalEmbedding,
    RelativePositionalEmbedding,
)
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SelfAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        n_heads: int,
        pos_embed: PositionalEmbedding = "absolute",
        max_rel_pos_dist: int = 127,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.n_heads = n_heads
        hidden_dim = head_dim * n_heads

        scale = torch.sqrt(torch.FloatTensor([head_dim]))
        self.register_buffer("scale", scale)

        self.to_qkv = nn.Linear(input_dim, 3 * hidden_dim)

        self.rel_pos_embed_key = None
        if pos_embed == "relative_key":
            self.rel_pos_embed_key = RelativePositionalEmbedding(
                max_rel_pos_dist, head_dim
            )

        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None, bias: Optional[Tensor] = None
    ) -> Tensor:
        # x: (B x S x D), mask: (B x S)
        qkv_shape = (*x.shape[:-1], self.head_dim, self.n_heads)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda u: u.view(qkv_shape), qkv)

        attn_score = self.scale * torch.einsum("bqdh,bkdh->bqkh", q, k)

        if self.rel_pos_embed_key is not None:
            seq_len = x.shape[1]
            rel_pos = self.rel_pos_embed_key(seq_len)
            # Q(K + A)^T = QK^T + QA^T
            attn_score = attn_score + torch.einsum("bqdh,qkd->bqkh", q, rel_pos)

        if bias is not None:
            attn_score = attn_score + bias

        if mask is not None:
            attn_score[mask == 0] = -1e8

        attn_logits = F.softmax(attn_score, dim=2)
        attn = torch.einsum("bqkh,bkdh->bqdh", attn_logits, v)
        out = self.out_proj(attn.flatten(-2))

        return out


class EncoderLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ff_dim: int,
        attn_head_dim: int,
        n_attn_heads: int,
        dropout: float = 0.1,
        pos_embed: PositionalEmbedding = "absolute",
    ) -> None:
        super().__init__()
        self.norm = partial(F.layer_norm, normalized_shape=(input_dim,))
        self.attn = SelfAttention(input_dim, attn_head_dim, n_attn_heads, pos_embed)

        self.proj1 = nn.Linear(input_dim, ff_dim)
        self.proj2 = nn.Linear(ff_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        attn = self.attn(self.norm(x), mask)
        x = x + attn
        x = x + self.proj2(self.dropout(F.gelu(self.proj1(self.norm(x)))))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ff_dim: int,
        attn_head_dim: int,
        n_attn_heads: int,
        n_encoder_blocks: int,
        dropout: float = 0.1,
        pos_embed: PositionalEmbedding = "absolute",
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    input_dim, ff_dim, attn_head_dim, n_attn_heads, dropout, pos_embed
                )
                for _ in range(n_encoder_blocks)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
