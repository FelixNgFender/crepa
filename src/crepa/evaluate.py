import logging
import time

import torch
import trackio
from torch import distributed as dist
from torch import nn
from torch.utils import data
from torchvision import datasets, models, transforms

from crepa import metrics, settings, utils

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
    if args.is_master_process:
        match args.arch:
            case "alexnet":
                net = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
            case "resnet18":
                net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            case "resnet50":
                net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            case _:
                msg = f"unsupported model {args.model}"
                raise RuntimeError(msg)

    if args.ddp:
        dist.barrier()  # wait for master to load the model before other ranks can access it

    net.eval().to(device)
    if not args.ddp:
        net.compile()
    else:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
        )
        net.compile()

    clean_ds = datasets.ImageFolder(
        args.imagenet_dir,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    clean_sampler = (
        data.distributed.DistributedSampler(
            clean_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=True
        )
        if args.ddp
        else None
    )
    # each rank only processes a subset of the effective batch
    rank_batch_size = args.batch_size // args.world_size
    clean_loader = data.DataLoader(
        clean_ds,
        batch_size=rank_batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
        sampler=clean_sampler,
    )

    if args.is_master_process:
        trackio.init(
            project="crepa",
            name=f"imagenet-clean-{args.arch}-{utils.current_dt_human()}",
            config={"arch": args.arch, "batch_size": args.batch_size},
            # TODO: group, name?
        )
    eval_loop(clean_loader, net, device, args)


def eval_loop(
    val_loader: data.DataLoader,
    model: nn.Module,
    device: torch.device,
    args: settings.Eval,
) -> None:
    def run_eval(loader: data.DataLoader, base_progress: int = 0) -> None:
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.use_mixed_precision),
        ):
            end = time.perf_counter()
            for i, (images, target) in enumerate(loader):
                global_i = base_progress + i
                images_d = images.to(device, non_blocking=True)
                target_d = target.to(device, non_blocking=True)

                logits = model(images_d)
                acc1, acc5 = metrics.accuracy(logits, target_d, topk=(1, 5))

                top1.update(acc1, images.size(0))
                top5.update(acc5, images.size(0))
                err1.update(100.0 - acc1, images.size(0))

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                if args.is_master_process:
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
                if global_i % args.log_freq == 0:
                    progress.display(global_i + 1)

    model.eval()

    batch_time = metrics.AverageMeter("time", device, ":6.3f", metrics.Summary.NONE)
    top1 = metrics.AverageMeter("acc@1", device, ":6.2f", metrics.Summary.AVERAGE)
    top5 = metrics.AverageMeter("acc@5", device, ":6.2f", metrics.Summary.AVERAGE)
    err1 = metrics.AverageMeter("err@1", device, ":6.2f", metrics.Summary.AVERAGE)
    progress = metrics.ProgressMeter(
        len(val_loader) + (args.ddp and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),  # ty:ignore[invalid-argument-type]
        [batch_time, top1, top5, err1],
        prefix="test: ",
        is_master_process=args.is_master_process,
    )
    run_eval(val_loader)
    if args.ddp:
        top1.all_reduce()
        top5.all_reduce()
        err1.all_reduce()

    if args.ddp and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):  # ty:ignore[invalid-argument-type]
        aux_val_dataset = data.Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)),  # ty:ignore[invalid-argument-type]
        )
        aux_val_loader = data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )
        run_eval(aux_val_loader, len(val_loader))

    progress.display_summary()
    if args.is_master_process:
        trackio.log(
            {
                "acc@1/final": top1.avg,
                "acc@5/final": top5.avg,
                "error@1/final": err1.avg,
            }
        )
