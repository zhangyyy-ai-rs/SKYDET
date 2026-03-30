<h2 align="center">
  SKYDET: A End-to-End Multi-Scale Attentive Detection Network from Foundation Models for Small Object in Remote Sensing Images
</h2>

<p align="center">
    <a href="https://github.com/zhangyyy-ai-rs/SKYDET/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://arxiv.org/abs/XXXX.XXXXX">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-red">
    </a>
    <a href="https://github.com/zhangyyy-ai-rs/SKYDET">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/zhangyyy-ai-rs/SKYDET">
    </a>
    <a href="https://github.com/zhangyyy-ai-rs/SKYDET/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/zhangyyy-ai-rs/SKYDET?color=olive">
    </a>
    <a href="https://github.com/zhangyyy-ai-rs/SKYDET/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/zhangyyy-ai-rs/SKYDET">
    </a>
    <a href="https://your-project-page.github.io/">
        <img alt="Project Page" src="https://img.shields.io/badge/Webpage-SKYDET-purple">
    </a>
</p>

<p align="center">
  SKYDET is a novel end-to-end multi-scale attentive detection framework for remote sensing small-object detection.
  It systematically explores the transferability of <strong>DINOv3</strong> to remote sensing imagery and introduces
  dedicated modules to enhance multi-scale representation learning and cross-scale feature fusion.
</p>

---

<div align="center">
  Yao Zhang,
  Wei Guo,
  Boxiang Xie,
  Lingfeng Lin,
  Jie Zhang,
  Hongwei Yang,
  Yuke Meng,
  Yi Liu*,
  Wei Zhang*
</div>

<p align="center">
<i>
Wuhan University, Northeast Forestry University, Ningde Normal University, KTH Royal Institute of Technology, Harbin Institute of Technology
</i>
</p>

<p align="center">
  <strong>Corresponding authors:</strong> Yi Liu and Wei Zhang
</p>

<p align="center">
<strong>If you find SKYDET useful, please give this repo a ⭐!</strong>
</p>

<p align="center">
  <img src="./figures/teaser.png" alt="teaser" width="90%">
</p>

## 🚀 Updates
- [x] **[2026.03.30]** Initial release of SKYDET code and README.
- [ ] Checkpoints and pretrained models will be released soon.
- [ ] Inference, deployment, and visualization tools will be continuously updated.

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Highlights](#2-highlights)
- [3. Model Zoo](#3-model-zoo)
- [4. Quick Start](#4-quick-start)
- [5. Data Preparation](#5-data-preparation)
- [6. Usage](#6-usage)
- [7. Tools](#7-tools)
- [8. Citation](#8-citation)
- [9. Acknowledgement](#9-acknowledgement)

## 1. Introduction

Remote sensing object detection remains highly challenging due to large scale variations, dense small-object distributions, and complex background interference. Existing detectors often rely on supervised pretraining and may suffer from domain gaps when transferred to remote sensing imagery.

To address this issue, we propose **SKYDET**, a new end-to-end detection framework that leverages the strong representation capability of **vision foundation models (VFMs)** for remote sensing object detection. Specifically, SKYDET systematically evaluates **DINOv3** as a remote sensing backbone and further introduces dedicated modules for multi-scale adaptation and cross-scale fusion.

Extensive experiments on **DOTA-v1.0**, **AI-TOD**, and **NWPU VHR-10** demonstrate that SKYDET achieves strong performance, reaching **72.6 AP50**, **51.6 AP50**, and **95.6 AP50**, respectively.

## 2. Highlights

- **First systematic exploration of DINOv3 for remote sensing object detection.**
- **Semantic-Guided Adapter (SGA)** to convert ViT single-scale features into informative multi-scale representations.
- **Cross Fused Encoder (CFE)** with **Reciprocal Guidance Module (RGM)** for robust cross-scale feature interaction.
- Two variants:
  - **SKYDET-T**: DINOv3-ViT based
  - **SKYDET-C**: DINOv3-ConvNeXt based
- Strong performance on challenging remote sensing benchmarks.

## 3. Model Zoo

> The following table summarizes the main reported results in the paper.
> Please replace checkpoint links with your real release links.

### Main Results on AI-TOD and NWPU VHR-10

| Model | Backbone | AI-TOD AP50 | AI-TOD AP50:95 | NWPU VHR-10 AP50 | NWPU VHR-10 AP50:95 | Params | GFLOPs | Config | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SKYDET-T | DINOv3-ViT | 54.7 | 27.8 | 94.5 | 71.5 | 805.5M | 458.2G | [yml](./configs/skydet/skydet_t.yml) | [ckpt](#) |
| SKYDET-C | DINOv3-ConvNeXt | 56.1 | 26.4 | 95.6 | 69.0 | 20.58M | 237.8G | [yml](./configs/skydet/skydet_c.yml) | [ckpt](#) |

### Best Reported AP50 in the Paper

| Dataset | Metric | Performance |
| :---: | :---: | :---: |
| DOTA-v1.0 | AP50 | 72.6 |
| AI-TOD | AP50 | 51.6 |
| NWPU VHR-10 | AP50 | 95.6 |

## 4. Quick Start

### Setup

```shell
conda create -n skydet python=3.11
conda activate skydet
pip install -r requirements.txt
