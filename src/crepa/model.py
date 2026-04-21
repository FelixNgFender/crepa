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

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_labels: int,
    ) -> "IJepaImageClassifier":
        processor = transformers.AutoImageProcessor.from_pretrained(model_name, backend="torchvision")
        net = transformers.IJepaForImageClassification.from_pretrained(model_name, num_labels=num_labels)
        return cls(processor, net)

    def freeze_backbone(self) -> None:
        """Freeze backbone and only finetune the classifier head"""
        self.net.requires_grad_(requires_grad=False)
        self.net.classifier.requires_grad_(requires_grad=True)

    def load_classifier_head(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a state dict containing only the classifier head weights, e.g. from a checkpoint"""
        self.net.classifier.load_state_dict(state_dict)

    def collate_fn(self, batch: list[tuple[Image.Image, int]]) -> tuple[torch.Tensor, torch.LongTensor]:
        images = self.processor(images=[ex[0] for ex in batch], return_tensors="pt").pixel_values  # ty:ignore[call-non-callable]
        labels = torch.tensor([ex[1] for ex in batch], dtype=torch.long)
        return images, labels  # ty:ignore[invalid-return-type]

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        return_logits: bool = True,
        interpolate_pos_encoding: bool | None = True,
    ) -> torch.Tensor | modeling_outputs.ImageClassifierOutput:
        # pixel_values: (B, C, H, W) float tensor, already preprocessed by collate_fn
        outputs = self.net(pixel_values=pixel_values, labels=labels, interpolate_pos_encoding=interpolate_pos_encoding)
        return outputs.logits if return_logits else outputs
