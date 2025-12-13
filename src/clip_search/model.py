import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer

CONFIG = CLIPConfig()


class VideoSearchEngine(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", freeze_backbone: bool = True) -> None:
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)

        if freeze_backbone:
            self._freeze_params()

    def _freeze_params(self) -> None:
        for param in self.clip_model.parameters():
            param.requires_grad = False

        for name, param in self.clip_model.named_parameters():
            if "visual_projection" in name or "text_projection" in name:
                param.requires_grad = True

            if "layer_norm" in name:
                param.requires_grad = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, t, _, _ = pixel_values.shape
        image_values = rearrange(pixel_values, "b c t h w -> (b t) c h w")  # (B * T, C, H, W)

        image_features = self.clip_model.get_image_features(image_values)
        image_features = F.normalize(image_features, dim=1)
        image_features = rearrange(image_features, "(b t) d -> b t d", b=b, t=t)
        pooled_features = image_features.mean(dim=1)  # (B, D)

        text_features = self.clip_model.get_text_features(input_ids, attention_mask)
        text_features = F.normalize(text_features, dim=1)

        return pooled_features, text_features


if __name__ == "__main__":
    B, T = 2, 2
    video = torch.randn(B, 3, T, 224, 224)

    seq_len = 8
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    input_ids = torch.randint(0, tokenizer.vocab_size, (B, seq_len))
    attention_mask = torch.ones(B, seq_len, dtype=torch.long)

    model = VideoSearchEngine()
    output = model(video, input_ids, attention_mask)
    print(output[0].shape, output[1].shape)
