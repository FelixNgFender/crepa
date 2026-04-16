import torch
import transformers
from torch import nn


class IJepaImageClassifier(nn.Module):
    def __init__(
        self, processor: transformers.AutoImageProcessor, net: transformers.IJepaForImageClassification
    ) -> None:
        super().__init__()
        self.processor = processor
        self.net = net

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(images, return_tensors="pt", device=images.device.type)  # ty:ignore[call-non-callable]
        outputs = self.net(**inputs)
        return outputs.logits

    @classmethod
    def from_pretrained(cls, model_name: str, num_labels: int, token: str | None = None) -> "IJepaImageClassifier":
        processor = transformers.AutoImageProcessor.from_pretrained(model_name, token=token)
        net = transformers.IJepaForImageClassification.from_pretrained(model_name, token=token, num_labels=num_labels)
        return cls(processor, net)

    def freeze_backbone(self) -> None:
        """Freeze backbone and only finetune the classifier head"""
        self.net.ijepa.requires_grad_(requires_grad=False)
        self.net.classifier.requires_grad_()
