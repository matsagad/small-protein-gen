import torch
from torch import Tensor
import torch.nn as nn
from typing import Literal

PositionalEmbedding = Literal[
    "absolute",
    "relative_key",  # Act on keys (Shaw et al, 2018) https://arxiv.org/abs/1803.02155
]


class LearnedPositionalEmbedding(nn.Embedding):
    def forward(self, mask: Tensor) -> Tensor:
        pos = torch.cumsum(mask, dim=0) * mask
        return super()(pos)


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, max_dist: int, proj_dim: int) -> None:
        super().__init__()
        self.max_dist = max_dist
        self.embed_rel_pos = nn.Embedding(2 * max_dist - 1, proj_dim)

    def forward(self, seq_len: int) -> Tensor:
        _range = torch.arange(seq_len)
        dists = _range[None, :] - _range[:, None]

        bound = self.max_dist - 1
        rel_pos = bound + torch.clamp(dists, -bound, bound)
        out = self.embed_rel_pos(rel_pos)

        return out
