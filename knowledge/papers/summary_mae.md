# Summary of Masked Autoencoders Are Scalable Vision Learners (MAE)

## Overview
This highly influential paper introduces the Masked Autoencoder (MAE) approach for self-supervised learning in computer vision. MAE works by masking a high proportion (e.g., 75%) of an image's patches and training a model to reconstruct the missing *pixels*. It uses an asymmetric encoder-decoder architecture where the heavy encoder only processes visible patches, and a lightweight decoder reconstructs the full image from the encoded latent representation and mask tokens. 

## Relevance to noisy-jepa
MAE is the quintessential *pixel-reconstruction* based self-supervised learner, which stands in direct contrast to the *latent-prediction* approach of JEPAs. 

For the `noisy-jepa` project, MAE is an essential conceptual (and potentially empirical) baseline:
1. **The Core Debate:** Felix's intuition is that predicting abstract latent representations (JEPA) yields substantial robustness gains over predicting raw pixels (MAE). Because MAE forces the network to perfectly recreate low-level high-frequency details (like pixel noise or exact textures), it might inadvertently learn to be highly sensitive to input corruptions (like Gaussian noise or JPEG compression in ImageNet-C). JEPA, by skipping pixel reconstruction entirely, might naturally filter out this noise.
2. **Evaluation Baseline:** If we want to truly isolate the effect of the *JEPA objective*, we should compare I-JEPA against an MAE model of the same size (e.g., ViT-Huge) trained on the same data. If I-JEPA achieves a significantly lower mCE on ImageNet-C than MAE, it definitively proves that latent prediction is superior to pixel reconstruction for out-of-distribution robustness.
