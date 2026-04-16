```bash
# install uv https://docs.astral.sh/uv/getting-started/installation/
# $ uv venv
# $ uv sync
# $ source .venv/bin/activate
# fill in your huggingface token for faster downloads
# $ cp .env.example .env
# register for imagenet dataset (use institution email) and download ILSVRC2012_img_val.tar from https://image-net.org/challenges/LSVRC/2012/2012-downloads.php. put it in your current directory.
# prepare imagenet 1k val set (one-time)
./scripts/extract_imagenet.sh

# single node, 2 GPUs
torchrun --standalone --nproc_per_node=2 -m crepa eval -a alexnet

# serious
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m crepa finetune -a ijepa_vith14_1k -j 32 -b 4096 --epochs 2 --lr 3e-4 --log-freq 1 2>&1 | tee logs/finetune/ijepa.log

# adjust nproc_per_node and batch size according to your system

# view wandb-like live charts
trackio show

# slurm stuff: push/pull work to remote cluster with rsync
./scripts/push.sh # run without args to see how to use it
./scripts/pull.sh
# slurm interactive
ACCOUNT=goat PARTITION=short ./scripts/slurm.sh

# monitor GPU utilization and memory usage during training
watch -n 1 'nvidia-smi \
  --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv && \
  echo && \
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv'
```
