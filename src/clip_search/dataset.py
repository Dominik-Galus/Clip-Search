from pathlib import Path
from typing import Literal, TypedDict

import albumentations as A
import av
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetResult(TypedDict):
    video: torch.Tensor
    label: int
    text: str
    path: str


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        transform: A.ReplayCompose | None = None,
        mode: Literal["train", "val", "test"] = "train",
        segments: int = 8,
    ) -> None:
        self.mode = mode
        self.classes = sorted(path.stem for path in (root_dir / mode).iterdir())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.segments = segments

        self.images_path = sorted((root_dir / mode).rglob("*.avi"))
        if transform is None:
            if mode == "train":
                self.transform = A.ReplayCompose([
                    A.Resize(256, 256),
                    A.RandomCrop(224, 224),
                    A.HorizontalFlip(p=0.5),
                ])
            else:
                self.transform = A.ReplayCompose([
                    A.Resize(224, 224),
                ])
        else:
            self.transform = transform

    def __getitem__(self, index: int) -> DatasetResult:
        item = self.images_path[index]
        class_name = item.parent.name
        label = self.class_to_idx[class_name]

        transformed_frames: list[np.ndarray] = []
        replay_data: dict | None = None

        with av.open(item) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            if stream.duration is None or stream.frames == 0:
                msg: str = f"Video {item} is broken."
                raise ValueError(msg)

            segment_duration = stream.duration // self.segments

            timestamps = []
            for i in range(self.segments):
                start = i * segment_duration
                end = start + segment_duration

                if self.mode == "train":
                    timestamp = np.random.randint(start, end)
                else:
                    timestamp = start + (segment_duration // 2)
                timestamps.append(timestamp)

            frames = []
            for pts in timestamps:
                container.seek(pts, stream=stream)

                for frame in container.decode(stream):
                    frames.append(frame.to_ndarray(format="rgb24"))
                    break  # Taking only one, sparse sampling implementation

            while len(frames) < self.segments:
                frames.append(frames[-1])  # Padding

            for img in frames:
                if replay_data is None:
                    res = self.transform(image=img)
                    replay_data = res["replay"]
                    transformed_frames.append(res["image"])
                else:
                    res = self.transform.replay(replay_data, image=img)
                    transformed_frames.append(res["image"])

        video_tensor = torch.stack([torch.tensor(img) for img in transformed_frames])
        video_tensor = video_tensor.permute(3, 0, 1, 2)

        return {
            "video": video_tensor,  # Shape: [3, 8, 224, 224]
            "label": label,
            "text": f"A video of {class_name}",
            "path": str(item),
        }

    def __len__(self) -> int:
        return len(self.images_path)


if __name__ == "__main__":
    dataset = VideoDataset(root_dir=Path("data"))
    sample, label, text = dataset[0]["video"], dataset[0]["label"], dataset[0]["text"]
    print(sample.shape, label, text)
