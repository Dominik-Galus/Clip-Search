import albumentations
import av
import numpy as np
import torch
from transformers import CLIPTokenizer

from clip_search.module import VideoSearchLightningModule


def load_video(path: str, transform: albumentations.Compose, segments: int = 8) -> torch.Tensor:
    transformed_frames: list[np.ndarray] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        if stream.duration is None or stream.frames == 0:
            msg: str = f"Video {path} is broken."
            raise ValueError(msg)

        segment_duration = stream.duration // segments

        timestamps = []
        for i in range(segments):
            start = i * segment_duration

            timestamp = start + (segment_duration // 2)
            timestamps.append(timestamp)

        frames = []
        for pts in timestamps:
            container.seek(pts, stream=stream)

            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
                break

        while len(frames) < segments:
            frames.append(frames[-1])  # Padding

        for img in frames:
            res = transform(image=img)
            transformed_frames.append(res["image"])

    video_tensor = torch.stack([torch.tensor(img) for img in transformed_frames])
    video_tensor = video_tensor.permute(3, 0, 1, 2)

    return video_tensor


def main() -> None:
    checkpoint_path = "../best.ckpt"
    video_path = "../data/test1/TennisSwing/v_TennisSwing_g01_c06.avi"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VideoSearchLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize()
    ])

    video_tensor = load_video(video_path, transform=transform).unsqueeze(0).to(device)

    texts = [
        "A video of a person playing tennis",
        "A video of a person applying makeup",
        "A video of a fast car racing",
        "A video of a dog running",
        "A video of slicing vegetables"
    ]

    tokenizer = CLIPTokenizer.from_pretrained(model.model_name)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)  # pyrefly: ignore

    with torch.no_grad():
        image_features, text_features = model(
            video_tensor,
            inputs["input_ids"],
            inputs["attention_mask"]
        )

        logits = torch.matmul(image_features, text_features.T)
        logits = logits * model.model.clip_model.logit_scale.exp()

        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    print("Results:")
    for text, prob in zip(texts, probs, strict=False):
        print(f"{text}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
