from small_protein_gen.components.attention import Encoder
from small_protein_gen.components.pos_embedding import (
    LearnedPositionalEmbedding,
    PositionalEmbedding,
)
from small_protein_gen.components.time_embedding import (
    get_time_embedding,
    TimeEmbedding,
)
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class AngleNet(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 6) -> None:
        super().__init__()
        self.proj1 = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj2 = nn.Linear(input_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj2(self.norm(F.gelu(self.proj1(x))))


class BERTDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        ff_dim: int,
        attn_head_dim: int,
        n_attn_heads: int,
        n_encoder_blocks: int,
        dropout: float = 0.1,
        pos_embed: PositionalEmbedding = "relative_key",
        time_embed: TimeEmbedding = "random_fourier",
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()

        hidden_dim = attn_head_dim * n_attn_heads
        self.proj = nn.Linear(input_dim, hidden_dim)

        self.embed_pos = None
        if pos_embed == "absolute":
            self.embed_pos = LearnedPositionalEmbedding(max_seq_len, hidden_dim)
        self.embed_time = get_time_embedding(time_embed)(hidden_dim)

        self.encoder = Encoder(
            hidden_dim,
            ff_dim,
            attn_head_dim,
            n_attn_heads,
            n_encoder_blocks,
            dropout,
            pos_embed,
        )
        self.decoder = AngleNet(hidden_dim, out_dim)

    def forward(self, x: Tensor, t: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
          x: (B x N x 6)
        """
        x = self.proj(x)
        if self.embed_pos is not None:
            x = x + self.embed_pos(mask)
        x = x + self.embed_time(t).unsqueeze(1)

        embeddings = self.encoder(x, mask)
        decoded = self.decoder(embeddings)

        return decoded
