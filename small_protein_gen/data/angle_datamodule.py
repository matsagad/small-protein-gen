import lightning as L
from small_protein_gen.utils.data import get_backbone_angles_from_directory
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Optional, Tuple, Union


class AngleDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_val_test_split: Union[Tuple[int, int, int], Tuple[float, float, float]],
        batch_size: int,
        seed: int = 0,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.mu = None
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage: Optional[str] = None) -> None:
        angles, angle_masks, masks = map(
            torch.stack,
            zip(*get_backbone_angles_from_directory(self.hparams.data_path)),
        )

        # Randomly partition data according to split config
        n_data = len(angles)
        perm = torch.randperm(
            n_data, generator=torch.Generator().manual_seed(self.hparams.seed)
        )
        n_train, n_val, n_test = self.hparams.train_val_test_split
        if n_train + n_val + n_test == 1:
            n_train = int(n_train * n_data)
            n_val = int(n_val * n_data)
            n_test = n_data - n_train - n_val
        is_train, is_val, is_test = torch.split(perm, [n_train, n_val, n_test])

        # Subtract train dataset mean from entire dataset
        self.mu = angles[is_train].sum((0, 1)) / angle_masks[is_train].sum((0, 1))
        angles -= self.mu
        angles[angle_masks == 0] = 0

        self.data_train = TensorDataset(
            angles[is_train], angle_masks[is_train], masks[is_train]
        )
        self.data_val = TensorDataset(
            angles[is_val], angle_masks[is_val], masks[is_val]
        )
        self.data_test = TensorDataset(
            angles[is_test], angle_masks[is_test], masks[is_test]
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
