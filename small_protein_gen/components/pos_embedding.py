from functools import lru_cache
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
        return super().forward(pos)


class RelativePositionalEmbedding(nn.Embedding):
    def __init__(self, max_dist: int, proj_dim: int) -> None:
        super().__init__(2 * max_dist - 1, proj_dim)
        self.max_dist = max_dist
        self.param_for_device = nn.Parameter(torch.empty(0))

    @lru_cache(maxsize=None)
    def get_rel_pos(self, seq_len: int) -> Tensor:
        _range = torch.arange(seq_len, device=self.param_for_device.device)
        dists = _range[None, :] - _range[:, None]

        bound = self.max_dist - 1
        rel_pos = bound + torch.clamp(dists, -bound, bound)
        return rel_pos

    def forward(self, seq_len: int) -> Tensor:
        rel_pos = self.get_rel_pos(seq_len)
        out = super().forward(rel_pos)
        return out
