import logging

import torch
from torchvision import models

from crepa import context, dataset, model, settings, utils

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

    match args.arch:
        case "alexnet":
            _model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            transform = models.AlexNet_Weights.DEFAULT.transforms()
        case "resnet18":
            _model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            transform = models.ResNet18_Weights.DEFAULT.transforms()
        case "resnet50":
            _model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            transform = models.ResNet50_Weights.DEFAULT.transforms()
        case "ijepa_vith14_1k" | "ijepa_vith16_1k" | "ijepa_vith14_22k" | "ijepa_vitg16_22k":
            _model = model.IJepaImageClassifier.from_pretrained(
                f"facebook/{args.arch}", num_labels=1000, token=args.hf_token
            )
            transform = _model.transform
        case _:
            msg = f"unsupported architecture {args.arch}"
            raise RuntimeError(msg)

    # switch from the default NCHW layout to channels-last NHWC memory format, improving data locality and unlocking
    # optimized convolution kernels
    _model.to(device, memory_format=torch.channels_last)  # ty:ignore[no-matching-overload]
    if not args.ddp:
        ddp_model = None
        _model.compile()
    else:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            _model,
            device_ids=[args.local_rank],
        )
        ddp_model.compile()

    ds = dataset.ImageNet1kClean(args.imagenet_dir, split="val", transform=transform)
    # each rank only processes a subset of the effective batch
    rank_batch_size = args.batch_size // args.world_size
    loader = ds.create_loader(batch_size=rank_batch_size, workers=args.workers, ddp=args.ddp)
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
        arch=args.arch,
        _raw_model=_model,
        _ddp_model=ddp_model,
    )
    ctx.evaluate()
