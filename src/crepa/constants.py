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
DATA_DIR = pathlib.Path("data")
CKPT_DIR = pathlib.Path("ckpt")
WORKERS = DDP_WORLD_SIZE * 16
BATCH_SIZE = 256

# log
LOG_FREQ = 10

# hyperparameters
EPOCHS = 5
LR = 3e-4

# checkpoint
EPOCH_CKPT_TMPL = "epoch_{epoch}.pth"
BEST_CKPT = "best.pth"
FINAL_CKPT = "final.pth"

# misc
DATEFMT_STR_HUMAN = "%Y-%m-%d %H:%M:%S"
DATEFMT_STR = "%Y-%m-%d_%H:%M:%S"
