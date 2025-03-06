import hydra
import lightning as L
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    data: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    fit_kwargs = {}
    if cfg.ckpt_path is not None:
        fit_kwargs["ckpt_path"] = cfg.ckpt_path
    trainer.fit(model, data, **fit_kwargs)


if __name__ == "__main__":
    main()
