<h2 align="center">
  SKYDET: A End-to-End Multi-Scale Attentive Detection Network from Foundation Models for Small Object in Remote Sensing Images
</h2>

<p align="center">
  <a href="./LICENSE">
    <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
  </a>
</p>

<p align="center">
  SKYDET is an end-to-end remote sensing object detector for dense small-object detection in complex aerial scenes.
  It is built on DETR-style detection and explores how DINOv3 foundation representations can be transferred to remote sensing imagery through dedicated multi-scale adaptation and cross-scale fusion modules.
</p>

---

## Highlights

- Designed for remote sensing object detection with **dense distributions**, **small targets**, **large scale variations**, and **complex backgrounds**.
- Supports both **DINOv3-ConvNeXt** and **DINOv3-ViT** explorations.
- Includes the proposed **Cross Fused Encoder** for enhanced cross-scale interaction.
- Includes a **Semantic Guiding Adapter (SGA)** for generating multi-scale features from ViT-based backbones.
- Provides complete training, evaluation, inference, ONNX export, FLOPs computation, and FiftyOne visualization utilities.
- Supports multiple remote sensing datasets, including **DOTA-v1.0**, **DOTA-v2.0**, **AI-TOD**, **NWPU VHR-10**, **DIOR**, and **custom COCO-format datasets**.

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
- [9. Benchmark](#9-benchmark)
- [10. Visualization](#10-visualization)
- [11. Experimental Configurations](#11-experimental-configurations)
- [12. Citation](#12-citation)
- [13. Acknowledgement](#13-acknowledgement)

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
```

### Main dependencies

```text
faster-coco-eval>=1.6.5
PyYAML
tensorboard
scipy
calflops
thop
transformers
pytorch_wavelets==1.3.0
timm==1.0.7
tidecv
einops
prettytable
pycocotools==2.0.8
```

For deployment and visualization, additional packages may be required:

```bash
pip install onnx onnxsim
pip install -r tools/inference/requirements.txt
pip install -r tools/benchmark/requirements.txt
pip install fiftyone
```

---

## 2. Repository Structure

```text
SKYDET/
├── configs/
│   ├── base/
│   ├── dataset/
│   ├── dinov3_convnext_with_dfine/
│   ├── dinov3_vits_with_dfine/
│   ├── runtime.yml
│   └── skydet/
│       ├── skydet_3scale.yml
│       └── skydet_4scale.yml
├── engine/
│   ├── backbone/
│   ├── core/
│   ├── data/
│   ├── misc/
│   ├── optim/
│   ├── skydet/
│   └── solver/
├── tools/
│   ├── benchmark/
│   ├── dataset/
│   ├── deployment/
│   ├── inference/
│   ├── reference/
│   └── visualization/
├── requirements.txt
└── train.py
```

### Key files

- `train.py`: main training and evaluation entry.
- `configs/skydet/skydet_3scale.yml`: SKYDET 3-scale configuration.
- `configs/skydet/skydet_4scale.yml`: SKYDET 4-scale configuration.
- `configs/dataset/*.yml`: dataset definitions.
- `tools/inference/torch_inf.py`: PyTorch inference for images and videos.
- `tools/deployment/export_onnx.py`: ONNX export.
- `tools/benchmark/get_info.py`: FLOPs, MACs, and parameter statistics.
- `tools/visualization/fiftyone_vis.py`: FiftyOne-based visualization.

---

## 3. Data Preparation

This repository provides dataset configs for:

- DOTA-v1.0
- DOTA-v2.0
- AI-TOD
- NWPU VHR-10
- DIOR
- Custom dataset

### Available dataset config files

```text
configs/dataset/dota1.0_detection.yml
configs/dataset/dota2.0_detection.yml
configs/dataset/ai_tod_detection.yml
configs/dataset/nwpu_detection.yml
configs/dataset/dior_detection.yml
configs/dataset/custom_detection.yml
```

All datasets are expected in **COCO-style detection format**.

### Default custom dataset format

The default custom config uses the following structure:

```text
dataset/
├── train/
│   ├── images/ or image files directly under train/
│   └── train.json
└── val/
    ├── images/ or image files directly under val/
    └── val.json
```

The provided custom config currently uses paths like:

```yaml
img_folder: /data/yourdataset/train
ann_file: /data/yourdataset/train/train.json
```

and

```yaml
img_folder: /data/yourdataset/val
ann_file: /data/yourdataset/val/val.json
```

So before training, please edit `configs/dataset/custom_detection.yml` according to your actual dataset location.

### Important fields to modify

```yaml
num_classes
train_dataloader.dataset.img_folder
train_dataloader.dataset.ann_file
val_dataloader.dataset.img_folder
val_dataloader.dataset.ann_file
```

### Important note for SKYDET configs

Both `configs/skydet/skydet_3scale.yml` and `configs/skydet/skydet_4scale.yml` currently include:

```yaml
'../dataset/custom_detection.yml'
```

So before training on DOTA, AI-TOD, NWPU, or DIOR, you should either:

1. Replace the dataset include path in the config, or
2. Copy the config and create your own dataset-specific version.

For example:

```text
configs/skydet/skydet_3scale_dota1.0.yml
configs/skydet/skydet_4scale_aitod.yml
```

---

## 4. Pretrained Weights

SKYDET relies on DINOv3-based pretrained backbones.

### Examples used in configs

For ConvNeXt-based SKYDET:

```yaml
ConvNeXt:
  pretrained: dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
```

For ViT-based experiments, the config files in `configs/dinov3_vits_with_dfine/` use DINOv3 ViT weights such as:

```text
dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

Please download the required pretrained weights and modify the corresponding path in the config file.

---

## 5. Training

### Single-GPU training

```bash
python train.py -c configs/skydet/skydet_3scale.yml --use-amp --seed 0
```

### Multi-GPU training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
train.py -c configs/skydet/skydet_3scale.yml --use-amp --seed 0
```

### Train SKYDET-4scale

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
train.py -c configs/skydet/skydet_4scale.yml --use-amp --seed 0
```

### Resume training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
train.py -c configs/skydet/skydet_3scale.yml --use-amp --seed 0 -r path/to/checkpoint.pth
```

### Fine-tuning from a checkpoint

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
train.py -c configs/skydet/skydet_3scale.yml --use-amp --seed 0 -t path/to/checkpoint.pth
```

### Notes

The SKYDET configs currently use:

- image size: `640 × 640`
- epochs: `120`
- optimizer: `AdamW`
- warmup: `2000` iterations
- mixed precision: supported via `--use-amp`
- EMA and DETR-style criterion settings are included in config

---

## 6. Evaluation

### Single-GPU evaluation

```bash
python train.py -c configs/skydet/skydet_3scale.yml --test-only -r path/to/checkpoint.pth
```

### Multi-GPU evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
train.py -c configs/skydet/skydet_3scale.yml --test-only -r path/to/checkpoint.pth
```

---

## 7. Inference

The repository provides PyTorch, ONNXRuntime, OpenVINO, and TensorRT inference scripts.

### PyTorch inference

```bash
python tools/inference/torch_inf.py \
-c configs/skydet/skydet_3scale.yml \
-r path/to/checkpoint.pth \
-i path/to/image_or_video \
-d cuda:0
```

For images, the result is saved as:

```text
torch_results.jpg
```

For videos, the result is saved as:

```text
torch_results.mp4
```

### Other inference backends

```bash
python tools/inference/onnx_inf.py --onnx model.onnx --input image.jpg
python tools/inference/openvino_inf.py --xml model.xml --input image.jpg
python tools/inference/trt_inf.py --trt model.engine --input image.jpg
```

### Visualized PyTorch inference

```bash
python tools/inference/torch_inf_vis.py \
-c configs/skydet/skydet_3scale.yml \
-r path/to/checkpoint.pth \
-i path/to/image
```

---

## 8. ONNX Export

### Export model to ONNX

```bash
python tools/deployment/export_onnx.py \
-c configs/skydet/skydet_3scale.yml \
-r path/to/checkpoint.pth
```

The export script supports ONNX checking and simplification.

### TensorRT conversion example

```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

---

## 9. Benchmark

### FLOPs, MACs, and Params

```bash
python tools/benchmark/get_info.py -c configs/skydet/skydet_3scale.yml
```

### TensorRT latency benchmark

```bash
python tools/benchmark/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
```

---

## 10. Visualization

### FiftyOne visualization

```bash
python tools/visualization/fiftyone_vis.py \
-c configs/skydet/skydet_3scale.yml \
-r path/to/checkpoint.pth \
-p 5151
```

This is useful for browsing predictions and analyzing dense small-object detection results interactively.

---

## 11. Experimental Configurations

Besides the main SKYDET configs, the repository also includes exploratory configurations for DINOv3-based backbones.

### DINOv3 + ConvNeXt

```text
configs/dinov3_convnext_with_dfine/
├── not_use_feature_fusion_network_or_adapter/
│   ├── convnext_base_dfine.yml
│   ├── convnext_large_dfine.yml
│   ├── convnext_small_dfine.yml
│   └── convnext_tiny_dfine.yml
└── use_feature_fusion_network_or_adapter/
    ├── convnext_base_dfine.yml
    ├── convnext_large_dfine.yml
    ├── convnext_small_dfine.yml
    └── convnext_tiny_dfine.yml
```

### DINOv3 + ViT

```text
configs/dinov3_vits_with_dfine/
├── not_use_feature_fusion_network_or_adapter/
│   ├── vit7b_lvd_dfine.yml
│   ├── vit7b_sat_dfine.yml
│   ├── vitb16_lvd_dfine.yml
│   ├── vith16plus_lvd_dfine.yml
│   ├── vitl16_lvd_dfine.yml
│   ├── vitl16_sat_dfine.yml
│   ├── vits16_lvd_dfine.yml
│   └── vits16plus_lvd_dfine.yml
└── use_feature_fusion_network_or_adapter/
    └── fea/
        ├── vit7b_lvd_dfine.yml
        ├── vit7b_sat_dfine.yml
        ├── vitb16_lvd_dfine.yml
        ├── vith16plus_lvd_dfine.yml
        ├── vitl16_lvd_dfine.yml
        ├── vitl16_sat_dfine.yml
        ├── vits16_lvd_dfine.yml
        └── vits16plus_lvd_dfine.yml
```

These configs are useful for ablation studies, backbone transfer experiments, and reproducing different design choices in the paper.

---

## 12. Citation

Full citations will be published after the article is accepted.

## 13. Acknowledgement

This repository is built upon several excellent open-source projects in the DETR family. We sincerely thank the authors of:

- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- [D-FINE](https://github.com/Peterande/D-FINE)
- [DEIM](https://github.com/ShihuaHuang95/DEIM)

for their inspiring work and open-source contributions.

---

## Contact

If you have any questions, please open an issue in this repository.
