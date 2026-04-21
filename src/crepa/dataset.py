import enum
import functools
import pathlib
from collections.abc import Callable
from typing import Literal

import torch
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import v2 as transforms


class ImageNet1kClean:
    def __init__(
        self,
        root: str | pathlib.Path,
        *,
        split: Literal["train", "val"],
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.root = pathlib.Path(root) / split
        self.split = split
        self.transform = transform
        self.ds = datasets.ImageFolder(
            self.root,
            transform=self.transform,
        )

    def create_loader(
        self, *, ddp: bool, batch_size: int, workers: int, collate_fn: Callable | None = None
    ) -> data.DataLoader:
        if collate_fn is None:
            collate_fn = data.dataloader.default_collate

        common_loader = functools.partial(
            data.DataLoader,
            dataset=self.ds,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        if ddp:
            return common_loader(
                shuffle=False,
                sampler=data.distributed.DistributedSampler(
                    self.ds, shuffle=self.split == "train", drop_last=self.split == "val"
                ),
            )
        return common_loader(shuffle=self.split == "train")


class ImageNetCDistortion(enum.StrEnum):
    GAUSSIAN_NOISE = "gaussian_noise"
    SHOT_NOISE = "shot_noise"
    IMPULSE_NOISE = "impulse_noise"

    DEFOCUS_BLUR = "defocus_blur"
    GLASS_BLUR = "glass_blur"
    MOTION_BLUR = "motion_blur"
    ZOOM_BLUR = "zoom_blur"

    SNOW = "snow"
    FROST = "frost"
    FOG = "fog"
    BRIGHTNESS = "brightness"

    CONTRAST = "contrast"
    ELASTIC_TRANSFORM = "elastic_transform"
    PIXELATE = "pixelate"
    JPEG_COMPRESSION = "jpeg_compression"

    SPECKLE_NOISE = "speckle_noise"
    GAUSSIAN_BLUR = "gaussian_blur"
    SPATTER = "spatter"
    SATURATE = "saturate"


def imagenet_c_transform() -> transforms.Compose:
    """Standard ImageNet-C transform: https://github.com/hendrycks/robustness/blob/master/ImageNet-C/test.py#L244"""
    return transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ImageNetC:
    def __init__(
        self,
        root: str | pathlib.Path,
        distortion: ImageNetCDistortion,
        severity: Literal[1, 2, 3, 4, 5],
        transform: Callable | None = imagenet_c_transform,
    ) -> None:
        super().__init__()
        self.root = pathlib.Path(root) / distortion / str(severity)
        self.transform = transform
        self.ds = datasets.ImageFolder(
            self.root,
            transform=transform,
        )

    def create_loader(
        self, *, ddp: bool, batch_size: int, workers: int, collate_fn: Callable | None = None
    ) -> data.DataLoader:
        if collate_fn is None:
            collate_fn = data.dataloader.default_collate

        common_loader = functools.partial(
            data.DataLoader,
            dataset=self.ds,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn,
        )
        if ddp:
            return common_loader(
                sampler=data.distributed.DistributedSampler(self.ds, shuffle=False),
            )
        return common_loader()
