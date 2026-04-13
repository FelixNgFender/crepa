# Summary of Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)

## Overview
This paper introduces the original V-JEPA, extending the Joint-Embedding Predictive Architecture to video. It proves that feature prediction alone (without image encoders, text, or reconstruction) is sufficient to learn versatile spatiotemporal representations.

## Relevance to noisy-jepa
This provides the foundational understanding of how V-JEPA works. It confirms that the model learns purely by predicting features of masked spacetime regions. When we evaluate on ImageNet-C/CIFAR-10-C, we are testing if this spatiotemporal feature prediction objective naturally filters out spatial corruptions. It will be interesting to see if a model trained on *video* (V-JEPA) is more robust to *static image corruptions* (ImageNet-C) than a model trained only on images (I-JEPA), perhaps due to the temporal consistency learned during pretraining.
