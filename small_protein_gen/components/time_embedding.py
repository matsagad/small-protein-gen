from functools import partial
from small_protein_gen.utils.registry import register
import torch
from torch import Tensor
import torch.nn as nn
from typing import Literal

TimeEmbedding = Literal["random_fourier"]

TIME_EMBEDDING_REGISTRY = {}
register_time_embedding = partial(register, registry=TIME_EMBEDDING_REGISTRY)
get_time_embedding = TIME_EMBEDDING_REGISTRY.get


@register_time_embedding("random_fourier")
class RandomSinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        w = torch.randn((1, embed_dim // 2), requires_grad=False)
        self.register_buffer("w", w)

    def forward(self, t: Tensor) -> Tensor:
        t_proj = t[:, None] * self.w * 2 * torch.pi
        t_embed = torch.cat([torch.cos(t_proj), torch.sin(t_proj)], dim=-1)
        return t_embed
