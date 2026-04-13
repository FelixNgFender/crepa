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
    torch_seed: Annotated[
        int,
        pydantic.Field(description="Random seed for PyTorch"),
    ] = constants.TORCH_SEED


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
        pydantic.Field(
            description=(
                "Whether to use mixed precision. If enabled, use bfloat16 where applicable."
            )
        ),
    ] = constants.USE_MIXED_PRECISION


################################################################################
#                                COMMANDS                                      #
################################################################################
class Eval(Seed, Device, Precision):
    """Settings for the `eval` CLI subcommand."""

    model: Annotated[
        str | None,
        pydantic.Field(
            default=None,
            description="Model with pretrained weights available in timm",
        ),
    ] = None
    test_clean_acc: Annotated[
        int | None,
        pydantic.Field(
            default=None,
            description="Test accuracy of the tested model on clean test set",
        ),
    ] = None
    baseline_clean_acc: Annotated[
        int | None,
        pydantic.Field(
            default=None,
            description="Test accuracy of the baseline model on clean test set",
        ),
    ] = None
    ckpt: Annotated[
        str | None,
        pydantic.Field(
            default=None,
            description="Checkpoint of a model",
        ),
    ] = None
    ckpt_baseline: Annotated[
        str | None,
        pydantic.Field(
            default=None,
            description="Checkpoint of a baseline model",
        ),
    ] = None
    dataset: Annotated[
        str,
        pydantic.Field(
            default="cifar",
            description="Dataset",
        ),
    ] = "cifar"
    data_path: Annotated[
        str,
        pydantic.Field(
            default="./datasets/",
            description="Data path",
        ),
    ] = "./datasets/"
    image_size: Annotated[
        int,
        pydantic.Field(
            default=32,
            description="Size of images in dataset",
        ),
    ] = 32
    difficulty: Annotated[
        int,
        pydantic.Field(
            default=1,
            description="Difficulty level",
            ge=1,
            le=3,
        ),
    ] = 1
