# Summary of DINOv3

## Overview
DINOv3 represents the latest evolution in the DINO family of self-supervised Vision Transformers. It focuses on scaling training without supervision and introduces novel regularizations (like the "Gram" regularization for dense features) to improve the quality of local/dense representations alongside global semantic understanding.

## Relevance to noisy-jepa
Although our proposal focuses on I-JEPA and V-JEPA 2, DINOv3 represents a parallel, highly successful branch of self-supervised ViTs. If we want to establish a robust self-supervised baseline to compare the JEPA models against, DINOv2 or DINOv3 would be the perfect candidates. Comparing a reconstruction-based/contrastive SSL method (like DINO) against a latent-prediction method (JEPA) under corruption could directly answer whether the robustness stems from *self-supervision in general* or the *specific JEPA architecture/objective*.
