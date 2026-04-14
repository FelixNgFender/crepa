import atexit
import datetime
import random
from typing import Literal

import numpy as np
import torch

from crepa import constants


def current_dt() -> str:
    return datetime.datetime.now(tz=datetime.UTC).astimezone().strftime(constants.DATEFMT_STR)


def parse_dt(dt_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(dt_str, constants.DATEFMT_STR).astimezone()


def current_dt_human() -> str:
    """Human-readable datetime string for logging and reporting."""
    return datetime.datetime.now(tz=datetime.UTC).astimezone().strftime(constants.DATEFMT_STR_HUMAN)


def parse_dt_human(dt_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(dt_str, constants.DATEFMT_STR_HUMAN).astimezone()


def compute_init(
    *,
    use_accelerator: bool,
    seed: int | None = None,
    fp32_matmul_precision: Literal["highest", "high", "medium"],
    # default to single device
    ddp: bool = False,
    local_rank: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> torch.device:
    """Seed everything and initialize distributed process group if DDP is enabled. Returns the device in use."""
    # device
    if ddp:
        if not torch.cuda.is_available():
            msg = "DDP training requires CUDA, but no CUDA devices are available."
            raise RuntimeError(msg)

        # process affinity: set device before doing any cuda operations
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        # init process group
        backend = torch.distributed.get_default_backend_for_device(device)
        torch.distributed.init_process_group(backend, rank=rank, world_size=world_size, device_id=device)
        # attach cleanup so we don't forget
        atexit.register(torch.distributed.destroy_process_group)
    else:
        device = (
            torch.accelerator.current_accelerator()
            if torch.accelerator.is_available() and use_accelerator
            else torch.device("cpu")
        )
        assert device is not None, "device cannot be None"  # noqa: S101

    # setup reproducibility
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # fire on all cylinders

    # tells pytorch to use different kernels depending on precision
    torch.set_float32_matmul_precision(fp32_matmul_precision)
    return device
