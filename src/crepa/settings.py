import functools
import os
import pathlib
from typing import Annotated, Literal

import pydantic
import pydantic_settings as ps

from crepa import constants


class Base(ps.BaseSettings):
    """Base settings all settings inherit from, i.e., pydantic global settings"""

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


################################################################################
#                                PRIMITIVES                                    #
################################################################################


class Seed(Base):
    seed: Annotated[
        int,
        pydantic.Field(description="Random seed for Python"),
    ] = constants.SEED


class Device(Base):
    use_accelerator: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(description="Whether to use accelerator"),
    ] = constants.USE_ACCELERATOR


class Precision(Base):
    fp32_matmul_precision: Annotated[
        Literal["highest", "high", "medium"],
        pydantic.Field(description="FP32 matrix multiplication precision"),
    ] = constants.FP32_MATMUL_PRECISION
    use_mixed_precision: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(description=("Whether to use mixed precision. If enabled, use bfloat16 where applicable.")),
    ] = constants.USE_MIXED_PRECISION


class DDP(Base):
    """DDP settings auto-populated from environment variables set by torchrun."""

    @functools.cached_property[bool]
    def ddp(self) -> bool:
        """DDP is enabled when running with torchrun."""
        return all(var in os.environ for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

    rank: Annotated[
        pydantic.NonNegativeInt,
        pydantic.Field(description="Global process rank. 0 for master process."),
    ] = constants.DDP_RANK
    local_rank: Annotated[
        pydantic.NonNegativeInt,
        pydantic.Field(description="Local process rank used for device assignment."),
    ] = constants.DDP_LOCAL_RANK
    world_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Total number of processes in the process group."),
    ] = constants.DDP_WORLD_SIZE

    @pydantic.computed_field
    @functools.cached_property[bool]
    def is_master_process(self) -> bool:
        return self.rank == 0


class HuggingFace(Base):
    hf_token: Annotated[
        str | None,
        pydantic.Field(
            description="Hugging Face token for authentication",
        ),
    ] = None

    @pydantic.model_validator(mode="after")
    def override_hf_env(self) -> "HuggingFace":
        if self.hf_token is not None:
            os.environ["HF_TOKEN"] = self.hf_token
        return self


class Log(Base):
    log_freq: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("l", "log_freq"),
            description="Frequency of logging",
        ),
    ] = constants.LOG_FREQ
    tracker: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("t", "tracker"),
            description="Whether to log training and evaluation metrics to trackio.",
        ),
    ] = constants.TRACKER


class DataLoading(Base):
    data_dir: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("d", "data", "data_dir"),
            description="Dataset directory",
        ),
    ] = constants.DATA_DIR

    @pydantic.computed_field
    @functools.cached_property[pathlib.Path]
    def imagenet_dir(self) -> pathlib.Path:
        """
        Directory for ImageNet-1k clean dataset.

        Expected structure:
             imagenet/train/
             ├── n01440764
             │   ├── n01440764_10026.JPEG
             │   ├── n01440764_10027.JPEG
             │   ├── ......
             ├── ......
             imagenet/val/
             ├── n01440764
             │   ├── ILSVRC2012_val_00000293.JPEG
             │   ├── ILSVRC2012_val_00002138.JPEG
             │   ├── ......
             ├── ......
        """
        return self.data_dir / "imagenet"

    @pydantic.computed_field
    @functools.cached_property[pathlib.Path]
    def imagenet_c_dir(self) -> pathlib.Path:
        """
        Directory for ImageNet-C (intentionally) corrupted dataset.

        Expected structure:
            imagenet-c
            ├── brightness
            │   ├── 1
            │   │   ├── n01440764
            │   │   │   ├── ILSVRC2012_val_00000293.JPEG
            │   │   │   ├── ILSVRC2012_val_00002138.JPEG
            │   │   │   ├── ILSVRC2012_val_00003014.JPEG
            │   │   │   ├── ILSVRC2012_val_00006697.JPEG
            │   │   │   ├── ILSVRC2012_val_00007197.JPEG
            │   │   │   ├── ILSVRC2012_val_00009111.JPEG
            │   │   │   ├── ......
            │   │   ├── ......
            │   ├── ......
            ├── ......
        """
        return self.data_dir / "imagenet-c"

    workers: Annotated[
        pydantic.NonNegativeInt,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("j", "workers"),
            description="Number of data loading workers",
        ),
    ] = constants.WORKERS
    batch_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("b", "batch_size"),
            description="Total batch size of all GPUs on the current node when using Distributed Data Parallel",
        ),
    ] = constants.BATCH_SIZE


class Hyperparameters(Base):
    epochs: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            description="Number of epochs to train for",
        ),
    ] = constants.EPOCHS
    lr: Annotated[
        float,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("lr", "learning_rate"),
            description="Learning rate for training",
        ),
    ] = constants.LR


class Checkpoint(Base):
    ckpt_dir: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("c", "ckpt", "ckpt_dir"),
            description="Checkpoint directory",
        ),
    ] = constants.CKPT_DIR


################################################################################
#                                COMMANDS                                      #
################################################################################


class Finetune(Checkpoint, Hyperparameters, DataLoading, Log, DDP, Precision, Device, Seed, HuggingFace):
    """Settings for the `finetune` CLI subcommand."""

    @pydantic.model_validator(mode="after")
    def validate_batch_size(self) -> "Finetune":
        if self.batch_size % self.world_size != 0:
            msg = f"batch size {self.batch_size} must be divisible by world size {self.world_size}"
            raise ValueError(msg)
        return self

    arch: Annotated[
        Literal[
            "ijepa_vith14_1k",
            "ijepa_vith16_1k",
            "ijepa_vith14_22k",
            "ijepa_vitg16_22k",
        ],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("a", "arch"),
            description="Model architecture to finetune",
        ),
    ]


class Eval(DataLoading, Log, DDP, Precision, Device, Seed, HuggingFace):
    """Settings for the `eval` CLI subcommand."""

    @pydantic.model_validator(mode="after")
    def validate_batch_size(self) -> "Eval":
        if self.batch_size % self.world_size != 0:
            msg = f"batch size {self.batch_size} must be divisible by world size {self.world_size}"
            raise ValueError(msg)
        return self

    corrupted: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("cr", "corrupted"),
            description="Whether to evaluate on the corrupted ImageNet-C. If false, evaluate on clean ImageNet-1k.",
        ),
    ] = constants.EVAL_CORRUPTED

    arch: Annotated[
        Literal[
            "alexnet",
            # 'squeezenet1.0', 'squeezenet1.1', 'condensenet4', 'condensenet8',
            # 'vgg11', 'vgg', 'vggbn',
            # 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet264',
            "resnet18",
            "resnet50",
            #  "resnet34", "resnet101", "resnet152",
            # 'resnext50', 'resnext101', 'resnext101_64'
            "ijepa_vith14_1k",
            "ijepa_vith16_1k",
            "ijepa_vith14_22k",
            "ijepa_vitg16_22k",
        ],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("a", "arch"),
            description="Model architecture to evaluate",
        ),
    ]
    ckpt: Annotated[
        pathlib.Path | None,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("c", "ckpt"),
            description="Weights-only checkpoint to evaluate on. If none, use pretrained weights if available, "
            "raise RuntimeError otherwise.",
        ),
    ] = None


class Parse(Base):
    """Settings for the `parse` CLI subcommand."""

    mode: Annotated[
        Literal["table", "sheet", "formulas", "long"],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("m", "mode"),
            description=(
                "Output format: table: readable; sheet: 2 TSV lines (categories row + formulas row); "
                "formulas: formulas-only TSV row; long: distortion+formula TSV rows"
            ),
        ),
    ] = "sheet"
    input: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("i", "input"),
            description="Path to one ImageNet-C log file",
        ),
    ]
