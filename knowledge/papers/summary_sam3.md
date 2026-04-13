# Summary of SAM 3: Segment Anything with Concepts

## Overview
This paper introduces the Segment Anything Model (SAM) 3, which dramatically expands the capabilities of the SAM family by introducing Promptable Concept Segmentation (PCS). Unlike previous versions that relied primarily on spatial prompts (points/boxes), SAM 3 allows users to segment and track all instances of a concept using short noun phrases or image exemplars across both images and videos.

## Relevance to noisy-jepa
While SAM 3 is an interactive segmentation model rather than a pure classification backbone, it is highly relevant as a potential downstream evaluation target or a comparative baseline for dense feature robustness:
1. **Dense Feature Quality under Corruption:** SAM 3 relies heavily on high-quality, dense visual features to perform accurate zero-shot segmentation. We could extend our `noisy-jepa` evaluation to test if the dense features of V-JEPA 2 are more robust to ImageNet-C/CIFAR-10-C corruptions than the backbone features of SAM 3 when used for dense prediction tasks.
2. **Video vs. Image Corruptions:** Since SAM 3 operates on both images and videos (like V-JEPA 2), it provides another state-of-the-art reference point for how modern architectures handle spatiotemporal tracking. If we test V-JEPA 2's robustness to temporal corruptions, SAM 3 would be the premier baseline to beat.
