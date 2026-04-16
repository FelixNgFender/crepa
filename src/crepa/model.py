import torch
import transformers
from PIL import Image
from torch import nn
from transformers import modeling_outputs


class IJepaImageClassifier(nn.Module):
    def __init__(
        self, processor: transformers.AutoImageProcessor, net: transformers.IJepaForImageClassification
    ) -> None:
        super().__init__()
        self.processor = processor
        self.net = net

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.LongTensor | None = None,
        *,
        return_logits: bool = True,
        interpolate_pos_encoding: bool | None = True,
    ) -> torch.Tensor | modeling_outputs.ImageClassifierOutput:
        # pixel_values: (B, C, H, W) float tensor, already preprocessed by transform()
        outputs = self.net(pixel_values=pixel_values, labels=labels, interpolate_pos_encoding=interpolate_pos_encoding)
        return outputs.logits if return_logits else outputs

    def transform(self, image: Image.Image) -> torch.Tensor:
        """
        torchvision-compatible single-image transform.
        ImageFolder calls this on individual PIL images before collation.
        Returns a (C, H, W) float tensor.
        """
        result = self.processor(images=image, return_tensors="pt")  # ty:ignore[call-non-callable]
        # processor returns shape (1, C, H, W), squeeze the batch dim
        return result.pixel_values.squeeze(0)

    @classmethod
    def from_pretrained(cls, model_name: str, num_labels: int, token: str | None = None) -> "IJepaImageClassifier":
        processor = transformers.AutoImageProcessor.from_pretrained(model_name, token=token)
        net = transformers.IJepaForImageClassification.from_pretrained(model_name, token=token, num_labels=num_labels)
        return cls(processor, net)

    def freeze_backbone(self) -> None:
        """Freeze backbone and only finetune the classifier head"""
        self.net.requires_grad_(requires_grad=False)
        self.net.classifier.requires_grad_(requires_grad=True)
