# Summary of V-JEPA 2: Scaling Self-Supervised Video Pretraining

## Overview
This paper introduces V-JEPA 2, scaling up video self-supervised learning to create an action-conditioned world model capable of understanding, prediction, and even zero-shot robot control planning.

## Relevance to noisy-jepa
This is one of the core models we proposed to evaluate! Understanding that V-JEPA 2 acts as a "world model" is crucial. A world model must inherently learn the underlying physics and stable semantics of a scene, ignoring superficial noise. This strongly supports Felix's intuition: a model trained to predict the future state of a world in latent space *must* learn to be invariant to superficial input corruptions (like weather or camera noise), otherwise its world model would collapse.
