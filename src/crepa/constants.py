# compute
import pathlib

TORCH_SEED = 2_147_483_647
SEED = 42
USE_ACCELERATOR = True
FP32_MATMUL_PRECISION = "high"  # "highest", "high", "medium"
USE_MIXED_PRECISION = True

# data
DATA_DIR = pathlib.Path("data")
CKPT_DIR = pathlib.Path("ckpt")

# misc
DATEFMT_STR_HUMAN = "%Y-%m-%d %H:%M:%S"
DATEFMT_STR = "%Y-%m-%d_%H:%M:%S"
