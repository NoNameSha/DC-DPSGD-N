# 🛡️ Clip Body and Tail Separately: Differentially Private Non-Convex Stochastic Gradient Descent with Heavy Tails

This repository provides the official implementation of **DC-DPSGD**, a novel framework for training deep models under differential privacy with **heavy-tailed gradient noise**. It introduces *discriminative clipping* to better handle the heavy tail in DPSGD.

![framework](./models/overview.jpg)

## 📁 Repository Structure

```
models/
├── DP-CNN-based         # Two-layer CNNs for MNIST, FMNIST
├── DP-Resnet9           # ResNet-9 or ResNeXt-29 for CIFAR, ImageNette
├── Handcrafted-DP-HT    # Heavy-tailed training configurations
├── ...
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

Please refer to the different details in the notes of each code file.

## ⚙️ Default Hyperparameters

| Dataset        | Model        | Batch Size | Clip-c2  | Clip c1 | LR     | Notes                     |
|----------------|--------------|------------|-------|-------|--------|---------------------------|
| MNIST/FMNIST   | CNN          | 128        | 0.1   | 1     |0.1    | Simple 2-layer CNN        |
| CIFAR10/HT     | ResNeXt-29   | 256        | 0.1   | 1     | 1.0    | SimCLR pre-trained        |
| ImageNette/HT  | ResNet-9     | 1000       | 0.15  | 1.5   |0.0001 | Trained from scratch      |
| E2E            | GPT-2 (160M) | 100          | 0.1   | 1     | 2e-3   | BLEU-based evaluation     |

## 🔍 Baselines

We compare DC-DPSGD against:
- DPSGD (Abadi-style clipping)
- Auto-S/NSGD
- DP-PSAC
- Non-private baseline (ε = ∞)

<img src="./models/gradient_weight.jpg" alt="Gradient-weight" width="500"/>


## 🧪 Per-sample Clipping

- **BackPACK** for per-sample gradient computation
- [https://docs.backpack.pt/](https://docs.backpack.pt/en/master/use_cases/example_differential_privacy.html)


## 📚 Citation

```bibtex
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

- [SimCLRv2/ResNeXt Pretrain Code](https://github.com/ftamer/Handcrafted-DP)
- [ResNet-9 for ImageNette](https://github.com/cbenitezb21/Resnet9)
- [LDAM-DRW Loss](https://github.com/kaidic/LDAM-DRW)
- [NLP tasks](https://github.com/lxuechen/private-transformers)

---


