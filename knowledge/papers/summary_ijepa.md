# Summary of I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture

## Overview
This is the seminal I-JEPA paper. It introduces a non-generative approach for SSL: predicting the representations of target blocks from a context block within the same image. Crucially, it achieves highly semantic representations *without* relying on hand-crafted data augmentations (like color jitter or cropping used in contrastive learning).

## Relevance to noisy-jepa
This is the primary model for our `noisy-jepa` project. The most critical takeaway for our robustness study is that **I-JEPA does not use hand-crafted data augmentations during pretraining**. Most models gain robustness to ImageNet-C (like color/contrast changes) precisely because they use color jitter augmentations during training. If I-JEPA is robust to these corruptions *without* having seen them as augmentations, it definitively proves Felix's intuition: the latent prediction objective itself inherently forces the model to learn corruption-invariant semantic features.
