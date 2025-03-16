from small_protein_gen.protein.structure import ProteinStructure
from small_protein_gen.models.base_model import (
    BaseFlowModel,
    BaseGenerator,
    DefaultLightningModule,
)
from small_protein_gen.utils.data import angles_to_backbone
import torch
from torch import Tensor
from typing import List, Tuple


class FoldingFlow(BaseFlowModel, BaseGenerator):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        sig_min: float = 0.001,
    ):
        super().__init__(net, optimizer, scheduler, compile)

        self.register_buffer("mu", torch.Tensor([0] * 6))
        self.register_buffer("_pi", torch.Tensor([torch.pi]))
        self.register_buffer("_sig_min", torch.Tensor([sig_min]))
        self.register_buffer("_eps", torch.Tensor([1e-5]))

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
        x_1, mask = batch
        B, _, F = x_1.shape

        t = torch.rand(1, device=self.device) + torch.arange(B, device=self.device) / B
        t = t % (1 - self._eps)
        _t = t.view(-1, 1, 1)

        # Gaussian probability path with independent coupling
        ## Based on torus flow matching from https://arxiv.org/pdf/2405.06642
        x_0 = self._wrap(torch.randn(x_1.shape, device=self.device))
        x_1 = x_0 + self.logmap(x_0, x_1)
        mu_t = _t * x_1 + (1 - _t * (1 - self._sig_min)) * x_0

        sigma_t = 1 - _t * (1 - self._sig_min)
        epsilon = torch.randn(x_1.shape, device=self.device)

        x_t = self._wrap(sigma_t * epsilon + mu_t)

        u_t_hat = self.forward(x_t, t, mask)
        u_t = self.logmap(x_t, x_1) / (1 - _t * (1 - self._sig_min))

        L = (u_t_hat - u_t) ** 2
        loss = torch.sum(L * mask.unsqueeze(-1)) / torch.sum(F * mask)

        return loss

    # Exponential and logarithmic maps on a torus
    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return (x + u) % (2 * torch.pi)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.atan2(torch.sin(y - x), torch.cos(y - x))

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

    def sample_loop(
        self, mask: Tensor, n_steps: int = 250, get_trajectory: bool = False
    ) -> Tensor:
        self.net.eval()

        B, N, F = *mask.shape, 6
        x_0 = self._wrap(torch.randn((B, N, F), device=self.device))
        x_t = x_0

        if get_trajectory:
            trajectory = []

        for i in range(n_steps):
            t = torch.tensor([i / n_steps] * B, device=self.device)
            with torch.no_grad():
                u_t = self.net(x_t, t, mask)

            # Euler method
            x_t = self._wrap(x_t + (1 / n_steps) * u_t)

            if get_trajectory:
                trajectory.append(self._wrap(x_t + self.mu).detach().cpu())

        if get_trajectory:
            return torch.stack(trajectory, dim=1)  # B x T x N x 6

        x_zero = self._wrap(x_t + self.mu).detach().cpu()
        return x_zero
