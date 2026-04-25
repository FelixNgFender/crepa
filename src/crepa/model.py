import torch
import transformers
from PIL import Image
from torch import nn
from transformers import modeling_outputs


class HFImageClassifier(nn.Module):
    def __init__(
        self, processor: transformers.AutoImageProcessor, net: transformers.AutoModelForImageClassification
    ) -> None:
        super().__init__()
        self.processor = processor
        self.net = net

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        model_meta: dict | None = None,
    ) -> "HFImageClassifier":
        model_meta = model_meta or {}
        processor = transformers.AutoImageProcessor.from_pretrained(model_name, backend="torchvision")
        net = transformers.AutoModelForImageClassification.from_pretrained(model_name, **model_meta)
        return cls(processor, net)

    def collate_fn(self, batch: list[tuple[Image.Image, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        images = self.processor(images=[ex[0] for ex in batch], return_tensors="pt").pixel_values  # ty:ignore[call-non-callable]
        labels = torch.tensor([ex[1] for ex in batch], dtype=torch.long)
        return images, labels

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        return_logits: bool = True,
        meta: dict | None = None,
    ) -> torch.Tensor | modeling_outputs.ImageClassifierOutput:
        meta = meta or {}
        # pixel_values: (B, C, H, W) float tensor, already preprocessed by collate_fn
        outputs = self.net(pixel_values=pixel_values, labels=labels, **meta)  # ty:ignore[call-non-callable]
        return outputs.logits if return_logits else outputs
