import functools
import os
import pathlib
from typing import Annotated, Literal

import pydantic
import pydantic_settings as ps

from crepa import constants


class Base(ps.BaseSettings):
    """Base settings all settings inherit from, i.e., pydantic global settings"""

    verbose: Annotated[
        ps.CliImplicitFlag[bool],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("v", "verbose"),
            description="Logs extra debugging information",
        ),
    ] = False
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


################################################################################
#                                COMMANDS                                      #
################################################################################
class Eval(Seed, Device, Precision, DDP):
    """Settings for the `eval` CLI subcommand."""

    arch: Annotated[
        Literal[
            "alexnet",
            # 'squeezenet1.0', 'squeezenet1.1', 'condensenet4', 'condensenet8',
            # 'vgg11', 'vgg', 'vggbn',
            # 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet264',
            "resnet18",
            "resnet50",
            # "resnet34", "resnet101", "resnet152",
            # 'resnext50', 'resnext101', 'resnext101_64'
        ],
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("a", "arch"),
            description="Model architecture to evaluate",
        ),
    ]
    workers: Annotated[
        pydantic.NonNegativeInt,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("j", "workers"),
            description="Number of data loading workers",
        ),
    ] = constants.NUM_WORKERS
    data_dir: Annotated[
        pathlib.Path,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("d", "data_dir"),
            description="Dataset directory for evaluation",
        ),
    ] = constants.DATA_DIR

    @pydantic.computed_field
    @functools.cached_property[pathlib.Path]
    def imagenet_dir(self) -> pathlib.Path:
        """
        Directory for ImageNet-1k clean dataset.

        Expected structure:
            imagenet/
            └── val
                └── ILSVRC2012_img_val.tar

            2 directories, 1 file
        """
        return self.data_dir / "imagenet"

    batch_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("b", "batch_size"),
            description="Total batch size of all GPUs on the current node when using Distributed Data Parallel",
        ),
    ] = constants.BATCH_SIZE

    @pydantic.model_validator(mode="after")
    def validate_batch_size(self) -> "Eval":
        if self.batch_size % self.world_size != 0:
            msg = f"batch size {self.batch_size} must be divisible by world size {self.world_size}"
            raise ValueError(msg)

        return self

    log_freq: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(
            validation_alias=pydantic.AliasChoices("l", "log_freq"),
            description="Frequency of logging during evaluation",
        ),
    ] = constants.LOG_FREQ
