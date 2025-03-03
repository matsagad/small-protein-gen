import abc
from functools import partial
from small_protein_gen.utils.registry import register
import torch
from torch import Tensor
import torch.nn as nn
from typing import Literal

NoiseSchedule = Literal["cosine", "literal"]

NOISE_SCHEDULER_REGISTRY = {}
register_scheduler = partial(register, registry=NOISE_SCHEDULER_REGISTRY)
get_scheduler = NOISE_SCHEDULER_REGISTRY.get


class NoiseScheduler(abc.ABC, nn.Module):
    def __init__(self, T: int) -> None:
        super().__init__()

        self.T = T

        self.build_beta_table()
        if not hasattr(self, "beta") or self.beta is None:
            raise Exception("Variance values (betas) have yet to be populated.")

        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_cum_prod", torch.cumprod(self.alpha, dim=-1))
        self.register_buffer("one_minus_alpha_cum_prod", 1 - self.alpha_cum_prod)

        self.register_buffer("sqrt_alpha", torch.sqrt(self.alpha))
        self.register_buffer("sqrt_beta", torch.sqrt(self.beta))
        self.register_buffer("sqrt_alpha_cum_prod", torch.sqrt(self.alpha_cum_prod))
        self.register_buffer(
            "sqrt_one_minus_alpha_cum_prod", torch.sqrt(self.one_minus_alpha_cum_prod)
        )

    def sample_uniform_time(self, mask: Tensor, device: str = "cpu") -> Tensor:
        return torch.randint(1, self.T, (mask.shape[0],), device=device)

    @abc.abstractmethod
    def build_beta_table(self) -> None:
        raise NotImplementedError()


@register_scheduler("cosine")
class CosineNoiseScheduler(NoiseScheduler):
    def __init__(self, T: int, offset: float = 8e-3, exponent: float = 2) -> None:
        self.T = T
        self.offset = offset
        self.exponent = exponent

        super().__init__(T)

    def _f(self, t: Tensor) -> Tensor:
        s = self.offset
        return torch.cos((((t / self.T) + s) / (1 + s)) * torch.pi / 2) ** self.exponent

    def build_beta_table(self) -> None:
        t = torch.arange(self.T + 1)
        alpha_cumprod = self._f(t) / self._f(t[:1])
        beta = torch.zeros((self.T + 1,))
        beta[1:] = torch.clamp(1 - alpha_cumprod[1:] / (alpha_cumprod[:-1]), max=0.999)
        self.register_buffer("beta", beta)


@register_scheduler("linear")
class LinearNoiseScheduler(NoiseScheduler):
    def __init__(self, T: int, start: float, end: float) -> None:
        self.T = T
        self.start = start
        self.end = end

        super().__init__(T)

    def build_beta_table(self) -> None:
        self.register_buffer("beta", torch.linspace(self.start, self.end, self.T + 1))
