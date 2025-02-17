import lightning as L
from small_protein_gen.utils.data import get_backbone_angles_from_directory
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from typing import Any, Optional, Tuple


class AngleDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_val_test_split: Tuple[int, int, int],
        batch_size: int,
        seed: int = 0,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage: Optional[str] = None) -> None:
        angles, masks = map(
            torch.stack,
            zip(*get_backbone_angles_from_directory(self.hparams.data_path)),
        )
        data_full = TensorDataset(angles, masks)
        self.data_train, self.data_val, self.data_test = random_split(
            data_full,
            self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train, batch_size=self.hparams.batch_size, shuffle=False
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val, batch_size=self.hparams.batch_size, shuffle=False
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test, batch_size=self.hparams.batch_size, shuffle=False
        )
