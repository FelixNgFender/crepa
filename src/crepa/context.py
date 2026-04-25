import dataclasses
import logging
import time

import torch
from torch.utils import data

from crepa import metrics, model, track

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Eval:
    """Run-specific context for evaluating a model on a validation set."""

    # compute
    device: torch.device
    use_mixed_precision: bool

    # ddp
    ddp: bool
    world_size: int
    is_master_process: bool

    # dataloading
    workers: int
    batch_size: int
    val_loader: data.DataLoader

    # log
    log_freq: int

    # model
    arch: str
    """Model architecture. Used for logging and checkpointing."""
    forward_meta: dict | None
    """Extra keyword args to be passed to the model's forward function."""
    _ddp_model: torch.nn.parallel.DistributedDataParallel | None
    """DDP wrapper for `model` if DDP is used."""
    _raw_model: model.HFImageClassifier
    """Raw model weights. Use `model` property instead."""

    # experiment tracking
    tracker: track.Tracker

    @property
    def model(self) -> model.HFImageClassifier | torch.nn.parallel.DistributedDataParallel:
        """The model to be used. Wrapped with DDP if DDP is enabled."""
        if self.ddp:
            assert self._ddp_model is not None, "ddp_model should not be None when is_ddp is True"  # noqa: S101
            return self._ddp_model
        return self._raw_model

    def evaluate(
        self,
    ) -> float:
        """Evaluate on the validation set. Returns the top-1 accuracy."""

        def run_eval(loader: data.DataLoader, base_progress: int = 0) -> None:
            if self.forward_meta is None:
                self.forward_meta = {}
            with (
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_mixed_precision),
            ):
                end = time.perf_counter()
                for i, (images, target) in enumerate(loader):
                    global_i = base_progress + i
                    images_d = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                    target_d = target.to(self.device, non_blocking=True)

                    logits = self.model(pixel_values=images_d, return_logits=True, meta=self.forward_meta)
                    acc1, acc5 = metrics.accuracy(logits, target_d, topk=(1, 5))

                    top1.update(acc1, images.size(0))
                    top5.update(acc5, images.size(0))
                    err1.update(100.0 - acc1, images.size(0))

                    # measure elapsed time
                    batch_time.update(time.perf_counter() - end)
                    end = time.perf_counter()
                    if self.is_master_process:
                        self.tracker.log(
                            {
                                "batch_time": batch_time.val,
                                "acc@1/batch": top1.val,
                                "acc@1/running_avg": top1.avg,
                                "acc@5/batch": top5.val,
                                "acc@5/running_avg": top5.avg,
                                "error@1/batch": err1.val,
                                "error@1/running_avg": err1.avg,
                            }
                        )
                    if global_i % self.log_freq == 0:
                        progress.display(global_i + 1)

        self.model.eval()

        batch_time = metrics.AverageMeter("time", self.device, ":6.3f", metrics.Summary.NONE)
        top1 = metrics.AverageMeter("acc@1", self.device, ":6.2f", metrics.Summary.AVERAGE)
        top5 = metrics.AverageMeter("acc@5", self.device, ":6.2f", metrics.Summary.AVERAGE)
        err1 = metrics.AverageMeter("err@1", self.device, ":6.2f", metrics.Summary.AVERAGE)
        progress = metrics.ProgressMeter(
            len(self.val_loader)
            + (self.ddp and (len(self.val_loader.sampler) * self.world_size < len(self.val_loader.dataset))),  # ty:ignore[invalid-argument-type]
            [batch_time, top1, top5, err1],
            prefix="test: ",
            is_master_process=self.is_master_process,
        )

        if self.is_master_process:
            self.tracker.init()

        run_eval(self.val_loader)
        if self.ddp:
            top1.all_reduce()
            top5.all_reduce()
            err1.all_reduce()

        if self.ddp and (len(self.val_loader.sampler) * self.world_size < len(self.val_loader.dataset)):  # ty:ignore[invalid-argument-type]
            aux_val_dataset = data.Subset(
                self.val_loader.dataset,
                range(len(self.val_loader.sampler) * self.world_size, len(self.val_loader.dataset)),  # ty:ignore[invalid-argument-type]
            )
            aux_val_loader = data.DataLoader(
                aux_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers, pin_memory=True
            )
            run_eval(aux_val_loader, len(self.val_loader))

        progress.display_summary()
        if self.is_master_process:
            self.tracker.log(
                {
                    "acc@1/final": top1.avg,
                    "acc@5/final": top5.avg,
                    "error@1/final": err1.avg,
                }
            )
        return top1.avg
