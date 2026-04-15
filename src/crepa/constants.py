import pathlib

# compute
SEED = 42
USE_ACCELERATOR = True
FP32_MATMUL_PRECISION = "high"  # "highest", "high", "medium"
USE_MIXED_PRECISION = True
DDP_RANK = 0
DDP_LOCAL_RANK = 0
DDP_WORLD_SIZE = 1

# data
NUM_WORKERS = DDP_WORLD_SIZE * 16
BATCH_SIZE = 256
DATA_DIR = pathlib.Path("data")
CKPT_DIR = pathlib.Path("ckpt")

# misc
DATEFMT_STR_HUMAN = "%Y-%m-%d %H:%M:%S"
DATEFMT_STR = "%Y-%m-%d_%H:%M:%S"
LOG_FREQ = 10
