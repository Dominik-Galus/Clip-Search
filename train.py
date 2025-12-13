import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from clip_search.datamodule import VideoDataModule
from clip_search.module import VideoSearchLightningModule


def main() -> None:
    data_dir = "data"
    model_name = "openai/clip-vit-base-patch32"
    batch_size = 4
    num_workers = 0
    max_epochs = 5
    learning_rate = 1e-5
    weight_decay = 0.01

    dm = VideoDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = VideoSearchLightningModule(
        model_name=model_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    logger = TensorBoardLogger(save_dir="tb_logs", name="video_clip")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="videoclip-{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="16-mixed",
        log_every_n_steps=1,
        accumulate_grad_batches=8,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
