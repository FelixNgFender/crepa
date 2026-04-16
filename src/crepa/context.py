import dataclasses
import logging
import pathlib
import time

import torch
import trackio
from torch import nn
from torch.utils import data

from crepa import constants, metric, utils

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
    _ddp_model: torch.nn.parallel.DistributedDataParallel | None
    """DDP wrapper for `model` if DDP is used."""
    _raw_model: nn.Module
    """Raw model weights. Use `model` property instead."""

    @property
    def model(self) -> nn.Module | torch.nn.parallel.DistributedDataParallel:
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
            with (
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_mixed_precision),
            ):
                end = time.perf_counter()
                for i, (images, target) in enumerate(loader):
                    global_i = base_progress + i
                    images_d = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                    target_d = target.to(self.device, non_blocking=True)

                    logits = self.model(images_d)
                    acc1, acc5 = metric.accuracy(logits, target_d, topk=(1, 5))

                    top1.update(acc1, images.size(0))
                    top5.update(acc5, images.size(0))
                    err1.update(100.0 - acc1, images.size(0))

                    # measure elapsed time
                    batch_time.update(time.perf_counter() - end)
                    end = time.perf_counter()
                    if self.is_master_process:
                        trackio.log(
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

        batch_time = metric.AverageMeter("time", self.device, ":6.3f", metric.Summary.NONE)
        top1 = metric.AverageMeter("acc@1", self.device, ":6.2f", metric.Summary.AVERAGE)
        top5 = metric.AverageMeter("acc@5", self.device, ":6.2f", metric.Summary.AVERAGE)
        err1 = metric.AverageMeter("err@1", self.device, ":6.2f", metric.Summary.AVERAGE)
        progress = metric.ProgressMeter(
            len(self.val_loader)
            + (self.ddp and (len(self.val_loader.sampler) * self.world_size < len(self.val_loader.dataset))),  # ty:ignore[invalid-argument-type]
            [batch_time, top1, top5, err1],
            prefix="test: ",
            is_master_process=self.is_master_process,
        )

        if self.is_master_process:
            trackio.init(
                project="crepa",
                name=f"imagenet-clean-eval-{self.arch}-{utils.current_dt_human()}",
                config={"arch": self.arch, "batch_size": self.batch_size},
                # TODO: group, name?
            )
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
            trackio.log(
                {
                    "acc@1/final": top1.avg,
                    "acc@5/final": top5.avg,
                    "error@1/final": err1.avg,
                }
            )
        return top1.avg


@dataclasses.dataclass
class Train(Eval):
    """Run-specific context for training and evaluating a model on a dataset."""

    # dataloading
    train_loader: data.DataLoader

    # hyperparameters
    epochs: int
    lr: float

    # trainers
    criterion: nn.Module
    optimizer: torch.optim.Optimizer

    # checkpointing
    ckpt_dir: pathlib.Path

    def train(
        self,
    ) -> None:
        """Entrypoint for training loop. Handles checkpointing, validation, and sampling."""

        def train_one_epoch(
            epoch: int,
        ) -> None:
            batch_time = metric.AverageMeter("time", self.device, ":6.3f", metric.Summary.NONE)
            data_time = metric.AverageMeter("data", self.device, ":6.3f", metric.Summary.NONE)
            losses = metric.AverageMeter("loss", self.device, ":.4e", metric.Summary.NONE)
            top1 = metric.AverageMeter("acc@1", self.device, ":6.2f", metric.Summary.NONE)
            top5 = metric.AverageMeter("acc@5", self.device, ":6.2f", metric.Summary.NONE)
            progress = metric.ProgressMeter(
                len(self.train_loader),
                [batch_time, data_time, losses, top1, top5],
                prefix=f"epoch: [{epoch}]",
                is_master_process=self.is_master_process,
            )

            self.model.train()

            end = time.perf_counter()
            for i, (images, target) in enumerate(self.train_loader):
                # measure data loading time
                data_time.update(time.perf_counter() - end)

                # move data to the same device as model
                images_d = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                target_d = target.to(self.device, non_blocking=True)

                # compute output
                output = self.model(images_d)
                loss = self.criterion(output, target_d)

                # measure accuracy and record loss
                acc1, acc5 = metric.accuracy(output, target_d, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1, images.size(0))
                top5.update(acc5, images.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()

                if self.is_master_process:
                    trackio.log(
                        {
                            "epoch": epoch,
                            "batch_time": batch_time.val,
                            "data_time": data_time.val,
                            "loss/batch": losses.val,
                            "loss/running_avg": losses.avg,
                            "acc@1/batch": top1.val,
                            "acc@1/running_avg": top1.avg,
                            "acc@5/batch": top5.val,
                            "acc@5/running_avg": top5.avg,
                        }
                    )

                if i % self.log_freq == 0:
                    progress.display(i + 1)

            # epoch-level summary
            if self.is_master_process:
                trackio.log(
                    {
                        "epoch": epoch,
                        "loss/epoch_avg": losses.avg,
                        "acc@1/epoch_avg": top1.avg,
                        "acc@5/epoch_avg": top5.avg,
                    }
                )

        if self.is_master_process:
            trackio.init(
                project="crepa",
                name=f"imagenet-clean-eval-{self.arch}-{utils.current_dt_human()}",
                config={"arch": self.arch, "batch_size": self.batch_size, "epochs": self.epochs, "lr": self.lr},
            )
        best_acc1 = 0.0
        for epoch in range(self.epochs):
            if self.ddp:
                self.train_loader.sampler.set_epoch(epoch)  # ty:ignore[unresolved-attribute]

            train_one_epoch(epoch)
            acc1 = self.evaluate()

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if self.is_master_process:
                trackio.log(
                    {
                        "epoch": epoch,
                        "val/acc@1": acc1,
                        "val/best_acc@1": best_acc1,
                        "val/is_best": int(is_best),
                    }
                )
                self.checkpoint(constants.BEST_CKPT if is_best else constants.EPOCH_CKPT_TMPL.format(epoch=epoch))

    def checkpoint(
        self,
        filename: str | pathlib.Path,
    ) -> None:
        """Checkpoint the model. Only has effect if called by the master process."""
        if not self.is_master_process:
            return

        ckpt = self.ckpt_dir / filename
        torch.save(self._raw_model, ckpt)
        logger.info("checkpoint saved: %s", ckpt)
