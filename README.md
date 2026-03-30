\<h2 align="center"\>
SKYDET: A End-to-End Multi-Scale Attentive Detection Network from Foundation Models for Small Object in Remote Sensing Images
\</h2\>

\<p align="center"\>
\<a href="[https://github.com/zhangyyy-ai-rs/SKYDET/blob/main/LICENSE](https://www.google.com/search?q=https://github.com/zhangyyy-ai-rs/SKYDET/blob/main/LICENSE)"\>
\<img alt="license" src="[https://img.shields.io/badge/LICENSE-Apache%202.0-blue](https://img.shields.io/badge/LICENSE-Apache%202.0-blue)"\>
\</a\>
\<a href="[https://github.com/zhangyyy-ai-rs/SKYDET](https://github.com/zhangyyy-ai-rs/SKYDET)"\>
\<img alt="stars" src="[https://img.shields.io/github/stars/zhangyyy-ai-rs/SKYDET](https://www.google.com/search?q=https://img.shields.io/github/stars/zhangyyy-ai-rs/SKYDET)"\>
\</a\>
\<a href="mailto:2020302142023@whu.edu.cn"\>
\<img alt="Contact Us" src="[https://img.shields.io/badge/Contact-Email-yellow](https://img.shields.io/badge/Contact-Email-yellow)"\>
\</a\>
\</p\>

\<p align="center"\>
[cite_start]SKYDET is a novel end-to-end multi-scale attention-based detection network designed to effectively transfer the general representation capabilities of Vision Foundation Models (VFMs), specifically DINOv3, to remote sensing imagery[cite: 6, 7, 8]. [cite_start]It tackles drastic scale variations, dense small objects, and complex background interference, achieving state-of-the-art (SOTA) performance across multiple challenging benchmarks[cite: 4, 13].
\</p\>

-----

\<div align="center"\>
Yao Zhang\<sup\>1\</sup\>,
Wei Guo\<sup\>1\</sup\>,
Boxiang Xie\<sup\>2\</sup\>,
Lingfeng Lin\<sup\>3\</sup\>,
Jie Zhang\<sup\>4\</sup\>,
Hongwei Yang\<sup\>1\</sup\>,
Yuke Meng\<sup\>1\</sup\>,
Yi Liu\<sup\>1\*\</sup\>,
Wei Zhang\<sup\>5\*\</sup\>
\</div\>

\<p align="center"\>
\<i\>

1.  Wuhan University \&nbsp; 2. Northeast Forestry University (Aulin College) \&nbsp; 3. Northeast Forestry University (College of Life Science) \&nbsp; 4. Ningde Normal University \&nbsp; 5. KTH Royal Institute of Technology
    \</i\>
    \</p\>
    \<p align="center"\>
    **📧 Contact:** \<a href="mailto:2020302142023@whu.edu.cn"\>2020302142023@whu.edu.cn\</a\> (Yao Zhang) | \<a href="mailto:yliu@sgg.whu.edu.cn"\>yliu@sgg.whu.edu.cn\</a\> (Yi Liu) | [cite_start]\<a href="mailto:wezhan@kth.se"\>wezhan@kth.se\</a\> (Wei Zhang) [cite: 2, 17, 18, 19, 20, 21, 23, 24]
    \</p\>

\<p align="center"\>
\<strong\>If you find our work helpful, please give us a ⭐\!\</strong\>
\</p\>

## 🚀 Highlights

  * [cite_start]**VFM for Remote Sensing**: The first to systematically evaluate and introduce DINOv3 into Transformer-based end-to-end remote sensing object detection[cite: 62, 67].
  * [cite_start]**Semantic-Guided Adapter (SGA)**: Efficiently converts single-scale features from ViT into multi-scale representations, enriching semantic content with fine-grained spatial details[cite: 69].
  * [cite_start]**Cross Fused Encoder & RGM**: A novel Reciprocal Guidance Module (RGM) enables spatial structure and channel semantics to guide each other, effectively suppressing background noise and sharpening responses to small objects[cite: 11, 12, 70].
  * [cite_start]**SOTA Performance**: Achieves exceptional accuracy on DOTA-v1.0 (72.6% AP50), AI-TOD (56.1% AP50), and NWPU VHR-10 (95.6% AP50) datasets[cite: 13, 469].

## Table of Content

  * [1. Model Zoo](https://www.google.com/search?q=%231-model-zoo)
  * [2. Quick Start](https://www.google.com/search?q=%232-quick-start)
  * [3. Usage](https://www.google.com/search?q=%233-usage)
  * [4. Citation](https://www.google.com/search?q=%234-citation)
  * [5. Acknowledgement](https://www.google.com/search?q=%235-acknowledgement)

## 1\. Model Zoo

[cite_start]We provide two primary variants of SKYDET: **SKYDET-C** (built on a robust ConvNeXt backbone) and **SKYDET-T** (built on the ViT backbone with our Semantic-Guided Adapter)[cite: 10, 334].

### Results on DOTA-v1.0 & AI-TOD

| Model | Backbone | Params (M) | GFLOPs | DOTA-v1.0 AP50 | AI-TOD AP50 | NWPU VHR-10 AP50 | config | checkpoint
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
**SKYDET-C** | DINOv3-ConvNeXt | 20.58 | 237.8 | **72.6** | **56.1** | **95.6** | [yml](https://www.google.com/search?q=./configs/skydet/skydet_c.yml) | [ckpt](https://www.google.com/search?q=%23) |
**SKYDET-T** | DINOv3-ViT | 805.5 | 458.2 | **70.8** | **54.7** | **94.5** | [yml](https://www.google.com/search?q=./configs/skydet/skydet_t.yml) | [ckpt](https://www.google.com/search?q=%23) |

[cite_start]*(Note: Data derived from AI-TOD and DOTA-v1.0 evaluation splits[cite: 311, 444, 445, 456, 457, 468, 469, 595, 598]. Checkpoints will be uploaded soon.)*

## 2\. Quick Start

### Setup

```shell
conda create -n skydet python=3.10
conda activate skydet
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Data Preparation

[cite_start]Organize your datasets (e.g., DOTA-v1.0, AI-TOD, NWPU VHR-10) in COCO format[cite: 300, 301, 305, 313]. Update the dataset paths in your configuration files:

```yaml
train_dataloader:
    img_folder: /path/to/DOTA/train/images/
    ann_file: /path/to/DOTA/annotations/instances_train.json
val_dataloader:
    img_folder: /path/to/DOTA/val/images/
    ann_file: /path/to/DOTA/annotations/instances_val.json
```

## 3\. Usage

### Training

Train SKYDET-C on 4 GPUs using distributed data parallel (DDP):

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/skydet/skydet_c.yml --use-amp --seed=0
```

### Testing

Evaluate a trained model checkpoint:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/skydet/skydet_c.yml --test-only -r /path/to/checkpoint.pth
```

## 4\. Citation

If you use `SKYDET` or its methods in your work, please cite the following BibTeX entries:

```latex
@article{zhang2025skydet,
  title={SKYDET: A End-to-End Multi-Scale Attentive Detection Network from Foundation Models for Small Object in Remote Sensing Images},
  author={Zhang, Yao and Guo, Wei and Xie, Boxiang and Lin, Lingfeng and Zhang, Jie and Yang, Hongwei and Meng, Yuke and Liu, Yi and Zhang, Wei},
  journal={arXiv preprint},
  year={2025}
}
```

## 5\. Acknowledgement

Our work is built upon excellent open-source projects, including [DINOv3](https://www.google.com/search?q=https://github.com/facebookresearch/dinov3), [RT-DETR](https://github.com/lyuwenyu/RT-DETR), and [D-FINE](https://github.com/Peterande/D-FINE). [cite_start]We sincerely thank the authors for their contributions to the community[cite: 15, 35, 146].
