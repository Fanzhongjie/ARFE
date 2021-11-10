# Adaptive Region-aware Feature Enhancement for Object Detection
---
This repository re-implements AC-FPN on the base of Detectron-Cascade-RCNN. Please follow Detectron on how to install and use this repo.

This repo has released CEM module without AM module, but we can get higher performance than the implementation of pytorch in paper. Also, thanks to the power of detectron, this repo is faster in training and inference.

The implementation of CEM is very simple, which is less than 200 lines code, but it can boost the performance almost 3% AP in FPN(resnet50).
