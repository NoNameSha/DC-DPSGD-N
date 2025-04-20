# 🛡️ Clip Body and Tail Separately: Differentially Private Non-Convex Stochastic Gradient Descent with Heavy Tails

This repository provides the official implementation of **DC-DPSGD**, a novel framework for training deep models under differential privacy with **heavy-tailed gradient noise**. It introduces *discriminative clipping* to better handle the heavy tail in DPSGD.

## 📁 Repository Structure
models/
├── DP-CNN-based         # CNN models for MNIST, FMNIST
├── DP-Resnet9           # ResNet-9 / ResNeXt for CIFAR and ImageNette
├── Handcrafted-DP-HT    # DP training for heavy-tailed datasets


## 📊 Datasets & Tasks

We evaluate DC-DPSGD on:

### 🖼️ Image Classification
- **MNIST / FMNIST** – using small CNNs
- **CIFAR10 / CIFAR10-HT** – SimCLRv2 + ResNeXt-29
- **ImageNette / ImageNette-HT** – ResNet-9 from scratch

### 📝 Natural Language Generation
- **E2E** – GPT-2 (160M) fine-tuned, evaluated by BLEU

## 🚀 How to Run

Train ResNeXt-29 on CIFAR10 using DC-DPSGD:

```bash
python train.py --dataset cifar10 --model resnext29 --dp_type dc-dpsgd --epsilon 8.0 --lr 1.0 --clip 0.1


Dataset | Model | Batch Size | Clip | LR | Notes
MNIST/FMNIST | CNN | 128 | 0.1 | 1.0 | Simple 2-layer CNN
CIFAR10/HT | ResNeXt-29 | 256 | 0.1 | 1.0 | SimCLR pre-trained
ImageNette/HT | ResNet-9 | 1000 | 0.15 | 0.0001 | Trained from scratch
E2E | GPT-2 (160M) | - | 0.2 | 5e-5 | BLEU-based evaluation



## ⚙️ \textbf{Environment}
This code is tested on Linux system with CUDA version 11.0

To run the source code, please first install the following packages:

```
python>=3.6
numpy>=1.15
torch>=1.3
torchvision>=0.4
scipy
six
backpack-for-pytorch
```

