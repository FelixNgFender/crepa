import dataclasses
import logging
import pathlib
from typing import override

import torch
from torch import nn

from crepa import context, dataset, model, settings, utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class IJepaFinetuneCtx(context.Train):
    classifier: nn.Linear | nn.Identity

    @override
    def checkpoint(
        self,
        filename: str | pathlib.Path,
    ) -> None:
        """Override to checkpoint only the classifier head. Only has effect if called by the master process."""
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

    _model = model.IJepaImageClassifier.from_pretrained(f"facebook/{args.arch}", num_labels=1000, token=args.hf_token)
    _model.freeze_backbone()
    _model.eval().to(device, memory_format=torch.channels_last)  # ty:ignore[no-matching-overload]
    criterion = nn.CrossEntropyLoss()
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
    train_loader = train_ds.create_loader(batch_size=rank_batch_size, workers=args.workers, ddp=args.ddp)
    val_loader = val_ds.create_loader(batch_size=rank_batch_size, workers=args.workers, ddp=args.ddp)

    run_checkpoint_dir = args.ckpt_dir / args.arch / utils.current_dt()
    if args.is_master_process:
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        # train-specific
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        criterion=criterion,
        optimizer=optimizer,
        ckpt_dir=run_checkpoint_dir,
        # ijepa finetune-specific
        classifier=_model.net.classifier,
    )
    ctx.train()
