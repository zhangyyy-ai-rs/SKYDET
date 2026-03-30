<h2 align="center">
  SKYDET: A End-to-End Multi-Scale Attentive Detection Network from Foundation Models for Small Object in Remote Sensing Images
</h2>

<p align="center">
    <a href="./LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://arxiv.org/abs/XXXX.XXXXX">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-red">
    </a>
    <a href="https://github.com/zhangyyy-ai-rs/SKYDET/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/zhangyyy-ai-rs/SKYDET?color=olive">
    </a>
    <a href="https://github.com/zhangyyy-ai-rs/SKYDET/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/zhangyyy-ai-rs/SKYDET">
    </a>
    <a href="https://github.com/zhangyyy-ai-rs/SKYDET">
        <img alt="stars" src="https://img.shields.io/github/stars/zhangyyy-ai-rs/SKYDET">
    </a>
</p>

<p align="center">
  SKYDET is an end-to-end remote sensing object detector designed for dense small-object detection in complex aerial scenes.
  It builds upon DINOv3-based foundation representations and introduces dedicated modules for multi-scale feature adaptation and cross-scale fusion.
</p>

---

## Highlights

- A remote sensing object detector tailored for **dense small objects**, **large scale variations**, and **complex backgrounds**.
- Supports both **DINOv3-ConvNeXt** and **DINOv3-ViT** based explorations.
- Includes the proposed **Cross Fused Encoder** for enhanced cross-scale interaction.
- Includes a **Semantic Guiding Adapter (SGA)** for ViT-based multi-scale feature generation.
- Built on a DETR-style framework with efficient training, evaluation, inference, visualization, and deployment utilities.

---

## News

- **[2026.xx.xx]** Initial release of SKYDET code.
- Checkpoints and project page will be released progressively.

---

## Table of Contents

- [1. Installation](#1-installation)
- [2. Repository Structure](#2-repository-structure)
- [3. Data Preparation](#3-data-preparation)
- [4. Pretrained Weights](#4-pretrained-weights)
- [5. Training](#5-training)
- [6. Evaluation](#6-evaluation)
- [7. Inference](#7-inference)
- [8. ONNX Export](#8-onnx-export)
- [9. Model Complexity](#9-model-complexity)
- [10. Visualization](#10-visualization)
- [11. Citation](#11-citation)
- [12. Acknowledgement](#12-acknowledgement)

---

## 1. Installation

### Environment

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8
- torchvision
- pycocotools

### Install dependencies

```bash
conda create -n skydet python=3.10 -y
conda activate skydet
pip install -r requirements.txt
