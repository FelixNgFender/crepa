import functools
import pathlib
from collections.abc import Callable
from typing import Literal

import torch
from torch.utils import data
from torchvision import datasets


class ImageNet1kClean(data.Dataset):
    def __init__(
        self,
        root: str | pathlib.Path,
        split: Literal["train", "val"],
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        root = pathlib.Path(root) / split
        self.root = root
        self.split = split
        self.transform = transform
        self.ds = datasets.ImageFolder(
            self.root,
            transform=self.transform,
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # ty:ignore[invalid-method-override]
        return self.ds.__getitem__(idx)

    def create_loader(self, *, batch_size: int, workers: int, ddp: bool) -> data.DataLoader:
        common_loader = functools.partial(
            data.DataLoader, dataset=self.ds, batch_size=batch_size, num_workers=workers, pin_memory=True
        )
        if ddp:
            return common_loader(
                shuffle=False,
                sampler=data.distributed.DistributedSampler(
                    self.ds, shuffle=self.split == "train", drop_last=self.split == "val"
                ),
            )
        return common_loader(shuffle=self.split == "train")
