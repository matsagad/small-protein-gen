import random
from small_protein_gen.components.noise_scheduler import NoiseSchedule
from small_protein_gen.protein.structure import ProteinStructure
from small_protein_gen.models.base_model import (
    BaseDenoiser,
    BaseGenerator,
    DefaultLightningModule,
)
from small_protein_gen.utils.data import angles_to_backbone
import torch
from torch import Tensor
from typing import List, Tuple


class FoldingDiff(BaseDenoiser, BaseGenerator, DefaultLightningModule):
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

        self.register_buffer("mu", torch.Tensor([0] * 6))
        self.register_buffer("_beta", torch.Tensor([l2_thresh]))
        self.register_buffer("_pi", torch.Tensor([torch.pi]))

    def on_train_start(self) -> None:
        super().on_train_start()
        if not hasattr(self.trainer.datamodule, "mu"):
            raise Exception(
                "Datamodule does not have dataset mean stored in a parameter `mu`."
            )
        self.mu = self.trainer.datamodule.mu

    def _wrap(self, x: Tensor) -> Tensor:
        return torch.remainder(x + self._pi, 2 * self._pi) - self._pi

    def forward(self, x: Tensor, t: Tensor, mask: Tensor) -> torch.Tensor:
        return self.net(x, t, mask)

    def model_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        x, mask = batch
        F = x.shape[-1]
        ns = self.noise_scheduler

        t = ns.sample_uniform_time(mask, device=self.device)
        _t = t.view(-1, 1, 1)
        epsilon = torch.randn(x.shape, device=self.device)

        x_noised = self._wrap(
            ns.sqrt_alpha_cum_prod[_t] * x
            + ns.sqrt_one_minus_alpha_cum_prod[_t] * epsilon
        )
        epsilon_hat = self.forward(x_noised, t, mask)

        d = torch.abs(self._wrap(epsilon - epsilon_hat))
        L = torch.where(d < self._beta, 0.5 * (d**2) / self._beta, d - 0.5 * self._beta)
        loss = torch.sum(L * mask.unsqueeze(-1)) / torch.sum(F * mask)

        return loss

    def sample_protein(self, mask: Tensor) -> List[ProteinStructure]:
        B = mask.shape[0]
        x = self.sample_loop(mask)

        bb_coords = angles_to_backbone(x, mask.cpu())
        structs = [
            ProteinStructure.from_backbone(
                bb_coords[i, :, : int(mask[i].sum().item()) + 1]
            )
            for i in range(B)
        ]
        return structs

    def sample_trajectory(
        self, mask: Tensor, center_at_origin: bool = True
    ) -> List[List[ProteinStructure]]:
        x = self.sample_loop(mask, get_trajectory=True)
        B, T, N, F = x.shape
        flat_mask = mask.repeat(T, 1)

        bb_coords = angles_to_backbone(x.view((B * T, N, F)), flat_mask.cpu())
        trajs = []
        for i in range(B):
            traj = []
            for t in range(T):
                j = i * T + t
                struct = ProteinStructure.from_backbone(
                    bb_coords[j, :, : int(flat_mask[j].sum().item()) + 1]
                )
                if center_at_origin:
                    struct.center_at_origin()
                traj.append(struct)
            trajs.append(traj)
        return trajs

    def sample_loop(self, mask: Tensor, get_trajectory: bool = False) -> Tensor:
        self.net.eval()
        ns = self.noise_scheduler

        B, N, F = *mask.shape, 6
        T = ns.T
        x_T = self._wrap(torch.randn((B, N, F), device=self.device))
        x_T[mask == 0] = 0
        x_t = x_T

        if get_trajectory:
            trajectory = [self._wrap(x_T + self.mu).detach().cpu()]

        for i in reversed(range(1, T + 1)):
            t = torch.tensor([i] * B, device=self.device).long()
            with torch.no_grad():
                epsilon_hat = self.forward(x_t, t, mask)
            _t = t.view(-1, 1, 1)

            sqrt_alpha_t = ns.sqrt_alpha[_t]
            beta_t = ns.beta[_t]
            sqrt_beta_t = ns.sqrt_beta[_t]
            sqrt_one_minus_alpha_bar_t = ns.sqrt_one_minus_alpha_cum_prod[_t]
            sqrt_one_minus_alpha_bar_t_minus_one = (
                ns.sqrt_one_minus_alpha_cum_prod[_t - 1] if i > 1 else 0
            )
            sigma_t = (
                sqrt_one_minus_alpha_bar_t_minus_one
                * sqrt_beta_t
                / sqrt_one_minus_alpha_bar_t
            )

            z = (
                torch.randn((B, N, F), device=self.device) * mask.unsqueeze(-1)
                if i > 1
                else 0
            )
            x_t = self._wrap(
                (x_t - beta_t * epsilon_hat / sqrt_one_minus_alpha_bar_t) / sqrt_alpha_t
                + sigma_t * z
            )
            if get_trajectory:
                trajectory.append(self._wrap(x_t + self.mu).detach().cpu())

        if get_trajectory:
            return torch.stack(trajectory, dim=1)  # B x T x N x 6

        x_zero = self._wrap(x_t + self.mu).detach().cpu()
        return x_zero
