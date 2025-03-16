import abc
import lightning as L
from small_protein_gen.protein.structure import ProteinStructure
from small_protein_gen.components.noise_scheduler import NoiseSchedule, get_scheduler
import torch
from torch import Tensor
from torchmetrics import MeanMetric, MinMetric
from typing import Any, Dict, List, Optional, Tuple


class BaseGenerator(abc.ABC):

    @abc.abstractmethod
    def sample_protein(self, mask: Tensor) -> List[ProteinStructure]:
        """Sample protein backbones given mask.

        Args:
          mask: (N x L) binary mask indicating AA positions and total length.
        Returns:
          list of protein structure objects
        """
        raise NotImplementedError()


class DefaultLightningModule(abc.ABC, L.LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss_best = MinMetric()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def model_step(self, *args, **kwargs):
        raise NotImplementedError()

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.val_loss_best.reset()

    def training_step(self, batch: Tuple[Tensor, Tensor], _: int) -> Tensor:
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], _: int) -> Tensor:
        loss = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_loss = self.val_loss.compute()
        self.val_loss_best(val_loss)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: Tuple[Tensor, Tensor], _: int) -> Tensor:
        loss = self.model_step(batch)
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def setup(self, stage: Optional[str] = None) -> None:
        if self.hparams.compile and (stage == "fit" or stage is None):
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        keys = list(state_dict.keys())
        for key in keys:
            # Handle checkpoint saving when model is compiled
            if key.startswith("net._orig_mod"):
                new_key = "".join(key.split("._orig_mod"))
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        return super().on_save_checkpoint(checkpoint)


class BaseDenoiser(DefaultLightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        n_time_steps: int,
        noise_schedule: NoiseSchedule,
    ) -> None:
        super().__init__(net, optimizer, scheduler, compile)
        self.register_module(
            "noise_scheduler", get_scheduler(noise_schedule)(n_time_steps)
        )


BaseFlowModel = DefaultLightningModule
