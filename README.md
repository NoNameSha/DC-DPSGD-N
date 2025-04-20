# 🛡️ Clip Body and Tail Separately: Differentially Private Non-Convex Stochastic Gradient Descent with Heavy Tails

This repository provides the official implementation of **DC-DPSGD**, a novel framework for training deep models under differential privacy with **heavy-tailed gradient noise**. It introduces *discriminative clipping* to better handle the heavy tail in DPSGD.

## 📁 Repository Structure

```
models/
├── DP-CNN-based         # Two-layer CNNs for MNIST, FMNIST
├── DP-Resnet9           # ResNet-9 or ResNeXt-29 for CIFAR, ImageNette
├── Handcrafted-DP-HT    # Heavy-tailed training configurations
```

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
```

Train GPT-2 for E2E text generation:

```bash
python train_gpt2.py --dataset e2e --dp_type dc-dpsgd --clip 0.2 --lr 5e-5
```

## ⚙️ Default Hyperparameters

| Dataset        | Model        | Batch Size | Clip-c2  | Clip c1 | LR     | Notes                     |
|----------------|--------------|------------|-------|-------|--------|---------------------------|
| MNIST/FMNIST   | CNN          | 128        | 0.1   | 1     |0.1    | Simple 2-layer CNN        |
| CIFAR10/HT     | ResNeXt-29   | 256        | 0.1   | 1     | 1.0    | SimCLR pre-trained        |
| ImageNette/HT  | ResNet-9     | 1000       | 0.15  | 1.5   |0.0001 | Trained from scratch      |
| E2E            | GPT-2 (160M) | -          | 0.1   | 1     | 5e-5   | BLEU-based evaluation     |

## 🔍 Baselines

We compare DC-DPSGD against:
- DPSGD (Abadi-style clipping)
- Auto-S/NSGD
- DP-PSAC
- Non-private baseline (ε = ∞)

## 🧪 Experiments

For detailed setup, see Section 5 of our paper. We use:
- **LDAM-DRW loss** for CIFAR-HT/ImageNet-HT
- **BackPACK** for per-sample gradient computation
- Privacy budget ε = {1.0, 2.0, 4.0, 8.0}, δ = 1e-5

## 📚 Citation

```bibtex
@inproceedings{dc-dpsgd2024,
  title     = {Distributionally Concentrated DPSGD under Heavy-tailed Gradients},
  author    = {Your Name and Collaborators},
  booktitle = {To Appear},
  year      = {2024}
}
```

## ⚙️ Environment
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

## 🔗 References

- [SimCLRv2 Pretrain Code](https://github.com/ftamer/Handcrafted-DP)
- [ResNeXt Pretrain](https://github.com/ftamer/Handcrafted-DP)
- [ResNet-9 for ImageNette](https://github.com/cbenitezb21/Resnet9)
- [LDAM-DRW Loss](https://github.com/kaidic/LDAM-DRW)

---


