# corruption robustness of vision classifiers

evaluates and finetunes vision classifiers -- classical CNNs (AlexNet, ResNet),
supervised ViTs, and self-supervised models spanning contrastive (DINO/DINOv2),
pixel-reconstruction (MAE, BEiTv2), and latent-prediction (I-JEPA) pretraining
objectives -- against corruption error on ImageNet-C.

main research question: does I-JEPA's latent-prediction pretraining objective
produce more corruption-robust representations than pixel-reconstruction (MAE)
or self-distillation (DINOv2) objectives?

built with pytorch, huggingface `transformers`, and `timm` for the model zoo.
ddp-native via `torchrun`, with `trackio` for experiment tracking and slurm
scripts for remote cluster work.

the `crepa` cli supports:

- `eval` -- evaluate a model on clean ImageNet-1k or corrupted ImageNet-C
- `finetune` -- finetune JEPA-family models (I-JEPA, LeJEPA) on ImageNet-1k
- `parse` -- parse ImageNet-C validation logs into err@1 tables/formulas

### setup

```bash
# install uv https://docs.astral.sh/uv/getting-started/installation/
uv venv
uv sync
source .venv/bin/activate

# fill in your huggingface token for faster downloads
cp .env.example .env

# register for imagenet dataset (use institution email) and download
# ILSVRC2012_img_val.tar from
# https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
# put it in your current directory, then run:
./scripts/extract_imagenet.sh
```

### evaluating

```bash
# single node, 2 GPUs, clean ImageNet-1k
torchrun --standalone --nproc_per_node=2 -m crepa eval -a alexnet

# corrupted ImageNet-C
torchrun --standalone --nproc_per_node=2 -m crepa eval -a resnet50 --corrupted

# view results
crepa parse -i logs/imagenet-c/resnet50.txt
```

### finetuning

```bash
# adjust --nproc_per_node, -b, and -j depending on your system
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m crepa finetune \
  -a ijepa_vith14_1k -j 32 -b 1024 --epochs 2 --lr 3e-4 --log-freq 1 \
  2>&1 | tee logs/finetune/ijepa.log
```

### tracking experiments

```bash
# view wandb-like live charts
trackio show

# slurm stuff: push/pull work to remote cluster with rsync
./scripts/push.sh # run without args to see how to use it
./scripts/pull.sh
ACCOUNT=goat PARTITION=short ./scripts/slurm.sh

# monitor GPU utilization and memory usage during training
watch -n 1 'nvidia-smi \
  --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv && \
  echo && \
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv'
```

## metrics

- **Clean Error** ($E_{clean}$) -- baseline error rate on the uncorrupted
  validation set
- **Corruption Error** ($E^f_{s,c}$) -- error rate on a specific corruption type
  ($c$) at a specific severity level ($s \in \{1, ..., 5\}$)
- **mean Corruption Error** (mCE) -- normalizes error against AlexNet across all
  15 corruptions and 5 severity levels. lower is more robust.
- **relative Corruption Error** (rCE) -- subtracts clean error from corrupted
  error, isolating how much the corruption itself hurt the model, independent of
  baseline accuracy

## results

17 vision classifiers evaluated on ImageNet-C (15 corruptions, 5 severities).

| model           | paradigm    | params (M) | clean err. | mCE ↓     | rCE ↓     |
| --------------- | ----------- | ---------- | ---------- | --------- | --------- |
| AlexNet         | sup. CNN    | 61         | 43.5%      | 1.000     | 1.000     |
| ResNet-18       | sup. CNN    | 12         | 30.3%      | 0.844     | 1.020     |
| ResNet-50       | sup. CNN    | 26         | 19.2%      | 0.635     | 0.872     |
| ConvNeXt-B CLIP | sup. CNN    | 89         | 12.9%      | 0.430     | 0.595     |
| ConvNeXtV2-B    | SSL CNN     | 89         | 13.3%      | 0.389     | 0.496     |
| ViT-B/16 AugReg | sup. ViT    | 87         | 14.9%      | 0.383     | 0.435     |
| ViT-B/16 CLIP   | sup. ViT    | 87         | 14.8%      | 0.463     | 0.612     |
| EVA-G CLIP      | sup. ViT    | 1013       | 11.1%      | 0.263     | 0.728     |
| DeiT-III-B      | sup. ViT    | 87         | 13.3%      | 0.447     | 0.620     |
| ViT-B/16 DINO   | SSL (dist.) | 87         | 22.4%      | 0.585     | 0.585     |
| ViT-B/16 MAE    | SSL (pix.)  | 87         | 48.0%      | 1.260     | 1.546     |
| I-JEPA ViT-H/14 | SSL (lat.)  | 632        | 27.4%      | 0.659     | 0.675     |
| EVA-02-B (MIM)  | SSL (lat.)  | 87         | 11.3%      | 0.319     | 0.391     |
| DINOv2-S/reg    | SSL (dist.) | 23         | 19.2%      | 0.551     | 0.678     |
| DINOv2-B/reg    | SSL (dist.) | 88         | 15.6%      | 0.396     | 0.438     |
| DINOv2-L/reg    | SSL (dist.) | 306        | 13.5%      | 0.291     | 0.267     |
| DINOv2-G/reg    | SSL (dist.) | 1140       | 13.0%      | **0.260** | **0.213** |
| BEiTv2-B        | SSL (pix.)  | 87         | 13.5%      | 0.360     | 0.427     |

takeaways:

- **latent prediction beats pixel reconstruction.** MAE collapses under
  corruption (mCE 1.260, worse than AlexNet), while I-JEPA (mCE 0.659) avoids
  the low-level texture brittleness that plagues pixel-level targets.
- **latent prediction alone doesn't guarantee robustness.** I-JEPA, despite a
  much larger backbone, is only on par with plain ResNet-50 (mCE 0.635) and
  trails DINOv2-B (mCE 0.396) with 7x fewer parameters -- a 40% relative gap.
  the gap widens to 68% against DINOv2-G at comparable scale.
- **likely culprits:** I-JEPA is evaluated here via linear probe (it's designed
  for finetuning, not probing), and was pretrained on ImageNet-1k with block
  masking but limited photometric augmentation, unlike DINOv2's 142M-image
  curated pretraining with heavy augmentation that mimics ImageNet-C-style
  shifts.

see [`knowledge/REPORT.md`](knowledge/REPORT.md) for the full writeup and
[`knowledge/KNOWLEDGE.md`](knowledge/KNOWLEDGE.md) for paper notes on I-JEPA,
MAE, DINOv2, and related work.

## roadmap

- finetune, eval [LeJEPA](https://github.com/galilai-group/lejepa)
- finetune I-JEPA/LeJEPA until SOTA
- extend to ImageNet-R (renditions), ImageNet-A (adversarial), ImageNet-P
  (perturbation), ImageNet-O (out-of-distribution), and LAION-C
- evaluate newer backbones: SAM 3, SigLIP 2, DINOv3
