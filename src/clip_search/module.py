import lightning as pl
import torch
from torch import nn

from clip_search.model import VideoSearchEngine


class VideoSearchLightningModule(pl.LightningModule):
    def __init__(self, model_name: str, learning_rate: float, weight_decay: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = VideoSearchEngine(model_name)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(pixel_values, input_ids, attention_masks)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pixel_values, input_ids, attention_mask = (
            batch["pixel_values"],
            batch["input_ids"],
            batch["attention_mask"],
        )

        image_features, text_features = self(
            pixel_values,
            input_ids,
            attention_mask,
        )

        cos_sim = torch.matmul(image_features, text_features.T)
        cos_sim = cos_sim * self.model.clip_model.logit_scale.exp()

        label = torch.arange(pixel_values.shape[0], device=self.device)

        loss_v2t = nn.functional.cross_entropy(cos_sim, label)
        loss_t2v = nn.functional.cross_entropy(cos_sim.T, label)
        total_loss = (loss_v2t + loss_t2v) / 2

        self.log("train_loss", total_loss, prog_bar=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pixel_values, input_ids, attention_mask = (
            batch["pixel_values"],
            batch["input_ids"],
            batch["attention_mask"],
        )

        image_features, text_features = self(
            pixel_values,
            input_ids,
            attention_mask,
        )

        cos_sim = torch.matmul(image_features, text_features.T)
        cos_sim = cos_sim * self.model.clip_model.logit_scale.exp()

        label = torch.arange(pixel_values.shape[0], device=self.device)

        loss_v2t = nn.functional.cross_entropy(cos_sim, label)
        loss_t2v = nn.functional.cross_entropy(cos_sim.T, label)
        total_loss = (loss_v2t + loss_t2v) / 2

        self.log("val_loss", total_loss, prog_bar=True, on_epoch=True)

        return total_loss
