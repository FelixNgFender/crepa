# Summary of V-JEPA 2.1: Unlocking Dense Features

## Overview
V-JEPA 2.1 improves upon V-JEPA 2 by introducing a Dense Predictive Loss, forcing all tokens (visible and masked) to contribute to the loss, alongside deep self-supervision (applying the objective at multiple layers). This yields highly structured, semantically coherent dense features for both images and videos.

## Relevance to noisy-jepa
Our proposal currently targets V-JEPA 2. Depending on checkpoint availability, upgrading our evaluation to V-JEPA 2.1 could be highly beneficial. The introduction of the "Dense Predictive Loss" means the model is forced to represent the *entire* spatial structure perfectly in latent space. We can hypothesize that this dense grounding makes V-JEPA 2.1 even more robust to localized spatial corruptions (like pixelation or glass blur in ImageNet-C) compared to earlier JEPA iterations.
