import atexit
import dataclasses
import logging
import pathlib
import time

import torch
from torch import nn
from torch.utils import data

from crepa import constants, context, dataset, metric, model, settings, track, utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class IJepaFinetuneCtx(context.Eval):
    """Run-specific context for finetuning I-JEPA on clean ImageNet-1K."""

    # dataloading
    train_loader: data.DataLoader

    # hyperparameters
    epochs: int
    lr: float

    # trainers
    optimizer: torch.optim.Optimizer

    # checkpointing
    ckpt_dir: pathlib.Path

    # model-specific
    classifier: nn.Linear | nn.Identity

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
                outputs = self.model(
                    pixel_values=images_d, labels=target_d, return_logits=False, interpolate_pos_encoding=True
                )
                loss = outputs.loss

                # measure accuracy and record loss
                acc1, acc5 = metric.accuracy(outputs.logits, target_d, topk=(1, 5))
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
                    self.tracker.log(
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
                self.tracker.log(
                    {
                        "epoch": epoch,
                        "loss/epoch_avg": losses.avg,
                        "acc@1/epoch_avg": top1.avg,
                        "acc@5/epoch_avg": top5.avg,
                    }
                )

        if self.is_master_process:
            self.tracker.init()

            def checkpoint_final() -> None:
                self.checkpoint(constants.FINAL_CKPT)

            atexit.register(checkpoint_final)

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
                self.tracker.log(
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
        torch.save(self.classifier.state_dict(), ckpt)
        logger.info("checkpoint saved: %s", ckpt)


def finetune(args: settings.Finetune) -> None:
    device = utils.compute_init(
        use_accelerator=args.use_accelerator,
        seed=args.seed,
        fp32_matmul_precision=args.fp32_matmul_precision,
        ddp=args.ddp,
        local_rank=args.local_rank,
        rank=args.rank,
        world_size=args.world_size,
    )

    _model = model.IJepaImageClassifier.from_pretrained(f"facebook/{args.arch}", num_labels=1000)
    _model.freeze_backbone()
    collate_fn = _model.collate_fn

    _model.to(device, memory_format=torch.channels_last)  # ty:ignore[no-matching-overload]
    optimizer = torch.optim.AdamW(_model.net.classifier.parameters(), lr=args.lr)

    if not args.ddp:
        ddp_model = None
        _model.compile()
    else:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            _model,
            device_ids=[args.local_rank],
        )
        ddp_model.compile()

    train_ds = dataset.ImageNet1kClean(args.imagenet_dir, split="train")
    val_ds = dataset.ImageNet1kClean(args.imagenet_dir, split="val")
    rank_batch_size = args.batch_size // args.world_size

    train_loader = train_ds.create_loader(
        batch_size=rank_batch_size, workers=args.workers, ddp=args.ddp, collate_fn=collate_fn
    )
    val_loader = val_ds.create_loader(
        batch_size=rank_batch_size, workers=args.workers, ddp=args.ddp, collate_fn=collate_fn
    )

    current_dt = utils.current_dt()
    run_checkpoint_dir = args.ckpt_dir / args.arch / current_dt
    if args.is_master_process:
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tracker = track.Tracker(
        project="crepa",
        name=f"imagenet-clean-finetune-{current_dt}",
        config={"arch": args.arch, "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr},
        enabled=args.tracker,
    )
    ctx = IJepaFinetuneCtx(
        device=device,
        use_mixed_precision=args.use_mixed_precision,
        ddp=args.ddp,
        world_size=args.world_size,
        is_master_process=args.is_master_process,
        workers=args.workers,
        batch_size=args.batch_size,
        val_loader=val_loader,
        log_freq=args.log_freq,
        arch=args.arch,
        _raw_model=_model,
        _ddp_model=ddp_model,
        tracker=tracker,
        # finetune-specific
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        optimizer=optimizer,
        ckpt_dir=run_checkpoint_dir,
        classifier=_model.net.classifier,
    )
    ctx.train()
