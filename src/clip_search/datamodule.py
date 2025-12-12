from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from clip_search.dataset import VideoDataset


class VideoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 4,
        num_workers: int = 4,
        segments: int = 8,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.segments = segments
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

    def _collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        videos = [item["video"] for item in batch]
        texts = [item["text"] for item in batch]

        pixel_values = torch.stack(videos)

        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = VideoDataset(
                root_dir=self.data_dir,
                mode="train",
                segments=self.segments
            )

            self.val_dataset = VideoDataset(
                root_dir=self.data_dir,
                mode="test",
                segments=self.segments
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=False if torch.backends.mps.is_available() else True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=False if torch.backends.mps.is_available() else True,
            collate_fn=self._collate_fn
        )
