from pathlib import Path  # noqa: I001
import json
import torch
from einops import rearrange
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from clip_search.module import VideoSearchLightningModule
from clip_search.dataset import VideoDataset

import faiss  # segmentation fault with pytorch somehow when correctly sorted before torch :(


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    videos = [item["video"] for item in batch]
    paths = [item["path"] for item in batch]

    return {
        "pixel_values": torch.stack(videos),
        "paths": paths,
    }


def create_vector_index(
    root_dir: str,
    checkpoint_path: str,
    device: str = "cpu",
    batch_size: int = 1
) -> None:
    module = VideoSearchLightningModule.load_from_checkpoint(checkpoint_path)
    module.eval()
    module.to(device)

    clip_model = module.model

    dataset = VideoDataset(
        root_dir=Path(root_dir),
        mode="test",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    all_embeddings = []
    all_paths = []

    for batch in tqdm(dataloader):
        video = batch["pixel_values"].to(device)
        paths = batch["paths"]

        with torch.no_grad():
            b, _, t, _, _ = video.shape
            video_reshaped = rearrange(video, "b c t h w -> (b t) c h w")

            video_features = clip_model.clip_model.get_image_features(video_reshaped)

            video_features = F.normalize(video_features, dim=1)

            video_features = rearrange(video_features, "(b t) d -> b t d", b=b, t=t)
            video_features = video_features.mean(dim=1)

            video_features = F.normalize(video_features, dim=1)

        emb = video_features.cpu().numpy().astype("float32")

        all_embeddings.append(emb)
        all_paths.extend(paths)

    final_embeddings = np.concatenate(all_embeddings, axis=0)

    d = final_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(final_embeddings)

    faiss.write_index(index, "vector.index")

    index_mapping = {
        int(i): path
        for i, path in enumerate(all_paths)
    }

    with open("paths.json", "w", encoding="utf-8") as f:
        json.dump(index_mapping, f, indent=4)


if __name__ == "__main__":
    create_vector_index(
        root_dir="data",
        checkpoint_path="best.ckpt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
