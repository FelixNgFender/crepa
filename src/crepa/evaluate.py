import logging

import torch
from torchvision import models

from crepa import context, dataset, model, settings, track, utils

logger = logging.getLogger(__name__)


def evaluate(args: settings.Eval) -> None:
    device = utils.compute_init(
        use_accelerator=args.use_accelerator,
        seed=args.seed,
        fp32_matmul_precision=args.fp32_matmul_precision,
        ddp=args.ddp,
        local_rank=args.local_rank,
        rank=args.rank,
        world_size=args.world_size,
    )

    transform = None
    collate_fn = None
    match args.arch:
        case "alexnet":
            if args.ckpt is None:
                _model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            else:
                _model = models.alexnet(weights=None)
                _model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
            transform = models.AlexNet_Weights.DEFAULT.transforms()
        case "resnet18":
            if args.ckpt is None:
                _model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                _model = models.resnet18(weights=None)
                _model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
            transform = models.ResNet18_Weights.DEFAULT.transforms()
        case "resnet50":
            if args.ckpt is None:
                _model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                _model = models.resnet50(weights=None)
                _model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
            transform = models.ResNet50_Weights.DEFAULT.transforms()
        case "ijepa_vith14_1k" | "ijepa_vith16_1k" | "ijepa_vith14_22k" | "ijepa_vitg16_22k":
            _model = model.IJepaImageClassifier.from_pretrained(f"facebook/{args.arch}", num_labels=1000)
            if args.ckpt is not None:
                _model.load_classifier_head(torch.load(args.ckpt, map_location=device, weights_only=True))
                logger.info("loaded classifier head from checkpoint %s for architecture %s", args.ckpt, args.arch)
            else:
                logger.warning("no checkpoint provided, using randomly initialized classifier head of %s", args.arch)
            collate_fn = _model.collate_fn

        case (
            "dinov2-with-registers-small-imagenet1k-1-layer"
            | "dinov2-with-registers-base-imagenet1k-1-layer"
            | "dinov2-with-registers-large-imagenet1k-1-layer"
            | "dinov2-with-registers-giant-imagenet1k-1-layer"
        ):
            _model = model.HFImageClassifier.from_pretrained(f"facebook/{args.arch}")
            collate_fn = _model.collate_fn
        case _:
            msg = f"unsupported architecture {args.arch}"
            raise RuntimeError(msg)

    # switch from the default NCHW layout to channels-last NHWC memory format, improving data locality and unlocking
    # optimized convolution kernels
    _model.to(device, memory_format=torch.channels_last)  # ty:ignore[no-matching-overload]
    ddp_model = None
    # clean val is small enough that compilation overhead is not worth it
    should_compile = args.corrupted
    if not args.ddp:
        if should_compile:
            _model.compile()
    else:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            _model,
            device_ids=[args.local_rank],
        )
        if should_compile:
            ddp_model.compile()

    # each rank only processes a subset of the effective batch
    rank_bs = args.batch_size // args.world_size

    if not args.corrupted:
        tracker = track.Tracker(
            project="crepa",
            name=f"imagenet-clean-eval-{utils.current_dt()}",
            config={
                "arch": args.arch,
                "batch_size": args.batch_size,
            },
            enabled=args.tracker,
        )
        ds = dataset.ImageNet1kClean(args.imagenet_dir, split="val", transform=transform)
        loader = ds.create_loader(ddp=args.ddp, batch_size=rank_bs, workers=args.workers, collate_fn=collate_fn)
        _eval_with_loader(
            device=device,
            args=args,
            tracker=tracker,
            loader=loader,
            model=_model,
            ddp_model=ddp_model,
        )
        return

    error_rates = []
    for distortion in dataset.ImageNetCDistortion:
        errs = []
        for severity in range(1, 6):
            current_dt = utils.current_dt()
            tracker = track.Tracker(
                project="crepa",
                name=f"imagenet-c-eval-{current_dt}",
                config={
                    "arch": args.arch,
                    "batch_size": args.batch_size,
                    "distortion": distortion,
                    "severity": severity,
                },
                enabled=args.tracker,
            )

            ds = dataset.ImageNetC(
                args.imagenet_c_dir,
                distortion=distortion,
                severity=severity,  # ty:ignore[invalid-argument-type]
                transform=transform,
            )
            loader = ds.create_loader(ddp=args.ddp, batch_size=rank_bs, workers=args.workers, collate_fn=collate_fn)

            acc1 = _eval_with_loader(
                device=device,
                args=args,
                loader=loader,
                tracker=tracker,
                model=_model,
                ddp_model=ddp_model,
            )
            err1 = 100.0 - acc1
            errs.append(err1)
        ce_unnorm = torch.tensor(errs).mean().item()
        error_rates.append(ce_unnorm)
        logger.info("distortion: %15s | CE (unnormalized) (%%): %.2f", distortion, ce_unnorm)
    mCE_unnorm = torch.tensor(error_rates).mean().item()
    logger.info("mCE (unnormalized) (%%): %.2f", mCE_unnorm)


def _eval_with_loader(
    device: torch.device,
    args: settings.Eval,
    tracker: track.Tracker,
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    ddp_model: torch.nn.parallel.DistributedDataParallel | None,
) -> float:
    """Evaluate on the validation set. Returns the top-1 accuracy."""
    ctx = context.Eval(
        device=device,
        use_mixed_precision=args.use_mixed_precision,
        ddp=args.ddp,
        world_size=args.world_size,
        is_master_process=args.is_master_process,
        workers=args.workers,
        batch_size=args.batch_size,
        val_loader=loader,
        log_freq=args.log_freq,
        tracker=tracker,
        arch=args.arch,
        _raw_model=model,
        _ddp_model=ddp_model,
    )
    return ctx.evaluate()
