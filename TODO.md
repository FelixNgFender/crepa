## Datasets

1. CIFAR-10-C (corruption, quick prototyping)
1. ImageNet-C (corruption)

## Metrics

- Clean Error ($E_{clean}$), which is the baseline error rate of the model on
  the uncorrupted validation set
- Corruption Error ($E^f_{s,c}$), which is the error rate of the model on a
  specific corruption type ($c$) at a specific severity level
  ($s \in \{1, 2, 3, 4, 5\}$).
- Mean Corruption Error ($m_{CE}$), normalizes the error of the target model
  ($f$) against a standard reference baseline (traditionally AlexNet or
  ResNet-50) across all 15 corruptions and 5 severity levels. A lower $m_{CE}$
  indicates superior corruption robustness.
- Relative Corruption Error ($r_{CE}$). measures the strict degradation in
  performance. By subtracting the clean error from the corrupted error, it
  effectively tracks how much the corruption itself harmed the model relative to
  the baseline. A model with a high mCE but a low rCE is highly accurate but
  equally fragile, whereas a model with a low rCE maintains its baseline
  performance regardless of the environmental conditions

## Steps

1. establish/reuse strong baselines: classical (ResNet, DenseNet), modern (ViT,
   ResNeXt, Swin Transfomer)
1. control confounding variables: number of parameters, training dataset volume
1. eval metrics: 4 above
1. main contestants: I-JEPA, LeJEPA, MAE

> Both are self-supervised ViT-Base models. However, MAE learns by
> reconstructing raw pixels, while I-JEPA learns by predicting abstract latents.
> Comparing these two isolates the exact mechanism LeCun praises JEPA for.

## May get to

models: SAM 3, SigLIP 2, DINOv3

1. ImageNet-R (renditions): recognize objects in art, cartoons, sketches, and
   video games
1. ImageNet-A (adversarial): natural images that frequently fool standard
   classifiers due to spurious correlations. Spurious correlations in computer
   vision occur when models rely on non-causal, coincidental patterns (like
   backgrounds or textures) rather than core features to make predictions. Often
   caused by simplicity bias in training data, models might identify a "cow" by
   its green background rather than its shape, failing when encountering a cow
   on sand
1. ImageNet-P (perturbation): measures prediction stability under small,
   non-adversarial perturbations (e.g., slight translations, rotations)
1. ImageNet-O (out-of-distribution)

## TODO

- finetune, eval lejepa
  <https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md>
- cancel vastai
- finetune i-jepa/lejepa until SOTA
