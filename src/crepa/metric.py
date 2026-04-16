# ruff: noqa: T201
import enum
from collections.abc import Iterable

import torch
import torch.distributed as dist


class Summary(enum.Enum):
    NONE = enum.auto()
    AVERAGE = enum.auto()
    SUM = enum.auto()
    COUNT = enum.auto()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(
        self, name: str, device: torch.device, fmt: str = ":f", summary_type: enum.Enum = Summary.AVERAGE
    ) -> None:
        self.name = name
        self.device = device
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self) -> None:
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self) -> str:
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            msg = f"invalid summary type {self.summary_type!r}"
            raise ValueError(msg)

        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(
        self,
        num_batches: int,
        meters: Iterable[AverageMeter],
        prefix: str = "",
        *,
        is_master_process: bool = False,
    ) -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.is_master_process = is_master_process

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def display(self, batch: int) -> None:
        if not self.is_master_process:
            return
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self) -> None:
        if not self.is_master_process:
            return
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))


def accuracy(logits: torch.Tensor, target: torch.Tensor, *, topk: tuple[int, ...] = (1,)) -> list[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        _, maxk_preds = logits.topk(maxk, -1, largest=True, sorted=True)  # (batch_size, maxk)
        maxk_preds = maxk_preds.t()  # (maxk, batch_size)
        # maxk_preds' columns[j] now contain maxk indices for sample j
        correct = maxk_preds.eq(
            target.view(1, -1).expand_as(maxk_preds)
        )  # broadcast target across rows (1, batch_size) -> (maxk, batch_size)
        # essentialy mean that it's a correct if any of the maxk predictions match the target

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
