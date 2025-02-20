from small_protein_gen.components.noise_scheduler import NoiseSchedule
from small_protein_gen.models.base_model import BaseDenoiser, DefaultLightningModule
import torch
from torch import Tensor
from typing import Tuple


class FoldingDiff(BaseDenoiser, DefaultLightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        n_time_steps: int = 250,
        noise_schedule: NoiseSchedule = None,
        l2_thresh: float = 0.1 * torch.pi,
    ):
        super().__init__(
            net, optimizer, scheduler, compile, n_time_steps, noise_schedule
        )

        self.register_buffer("_beta", torch.Tensor([l2_thresh]))
        self.register_buffer("_pi", torch.Tensor([torch.pi]))

    def _wrap(self, x: Tensor) -> Tensor:
        return torch.fmod(x + self._pi, 2 * self._pi) - self._pi

    def forward(self, x: Tensor, t: Tensor, mask: Tensor) -> torch.Tensor:
        return self.net(x, t, mask)

    def model_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        x, mask = batch
        t = self.noise_scheduler.sample_uniform_time(mask, device=self.device)
        _t = t.view(-1, 1, 1)
        epsilon = torch.rand(x.shape, device=self.device)

        x_noised = self._wrap(
            self.noise_scheduler.sqrt_alpha_cum_prod[_t] * x
            + self.noise_scheduler.sqrt_one_minus_alpha_cum_prod[_t] * epsilon
        )
        epsilon_hat = self.forward(x_noised, t, mask)

        d = torch.abs(self._wrap(epsilon - epsilon_hat))
        L = torch.where(d < self._beta, 0.5 * (d**2) / self._beta, d - 0.5 * self._beta)
        loss = torch.sum(L * mask.unsqueeze(-1)) / torch.sum(mask)

        return loss

    def sample_protein(self, mask: Tensor) -> Tensor:
        # TODO: implement inference loop
        return super().sample_protein(mask)
