import hydra
import lightning as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    dm = instantiate(cfg.data)
    model = instantiate(cfg.model)

    logger = TensorBoardLogger(save_dir="tb_logs", name="video_clip")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="videoclip-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
