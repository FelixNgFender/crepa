import datetime
import random
from typing import Literal

import torch

from crepa import constants


def current_dt() -> str:
    return (
        datetime.datetime.now(tz=datetime.UTC)
        .astimezone()
        .strftime(constants.DATEFMT_STR)
    )


def parse_dt(dt_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(dt_str, constants.DATEFMT_STR).astimezone()


def current_dt_human() -> str:
    """Human-readable datetime string for logging and reporting."""
    return (
        datetime.datetime.now(tz=datetime.UTC)
        .astimezone()
        .strftime(constants.DATEFMT_STR_HUMAN)
    )


def parse_dt_human(dt_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(dt_str, constants.DATEFMT_STR_HUMAN).astimezone()


def compute_init(
    *,
    use_accelerator: bool,
    seed: int | None = None,
    torch_seed: int | None = None,
    fp32_matmul_precision: Literal["highest", "high", "medium"],
) -> torch.device:
    """Set device and seed and everything else. Returns the device in use."""
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available() and use_accelerator
        else torch.device("cpu")
    )
    assert device is not None, "device cannot be None"  # noqa: S101

    random.seed(seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
    # tells pytorch to use different kernels depending on precision
    torch.set_float32_matmul_precision(fp32_matmul_precision)
    return device
