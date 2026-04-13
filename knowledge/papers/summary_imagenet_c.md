# Summary of Benchmarking Neural Network Robustness to Common Corruptions and Perturbations

## Overview
This paper introduces the ImageNet-C and ImageNet-P datasets, which have become the gold standard for evaluating robustness to common corruptions (like noise, blur, weather, and digital artifacts). It establishes the mean Corruption Error (mCE) metric to evaluate how well models generalize to these out-of-distribution shifts compared to clean data.

## Relevance to noisy-jepa
This paper is the foundational dataset and metric paper for our `noisy-jepa` project. Since our goal is to evaluate I-JEPA and V-JEPA 2 on CIFAR-10-C and ImageNet-C, we will directly adopt the methodologies, the 15 corruption types, and the 5 severity levels defined here. We will also compute mCE as our primary evaluation metric to directly compare the JEPA models against the ResNet and ViT baselines discussed in our proposal.
