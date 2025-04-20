# Handcrafted-DP

**This repository contains code to train differentially private models 
with handcrafted vision features.**

These models are introduced and analyzed in:

*Differentially Private Learning Needs Better Features (Or Much More Data)*</br>
**Florian Tramèr and Dan Boneh**</br>
[arXiv:2011.11660](http://arxiv.org/abs/2011.11660)

## Installation

The main dependencies are [pytorch](https://github.com/pytorch/pytorch), 
[kymatio](https://github.com/kymatio/kymatio) 
and [opacus](https://github.com/pytorch/opacus).

You can install all requirements with:
```bash
pip install -r requirements.txt
```

The code was tested with `python 3.7`, `torch 1.6` and `CUDA 10.1`.

### Private Transfer Learning
Our paper also contains some results for private transfer learning to CIFAR-10.
For a privacy budget of `(epsilon=2, delta=1e-5)` we get:

 Source Model  | Transfer Accuracy on CIFAR-10 |
| ------------- | ------------- | 
| ResNeXt-29 (CIFAR-100) | 79.6%  | 
| SIMCLR v2 (unlabelled ImageNet) | 92.4%  | 

These results can be reproduced as follows. 
First, you'll need to download the `resnext-8x64d` model from 
[here](https://github.com/bearpaw/pytorch-classification).

Then, we extract features from the source models:
```bash
python3 -m transfer.extract_cifar100
python3 -m transfer.extract_simclr
```
This will create a `transfer/features` directory unless one already exists.

Finally, we train linear models with DP-SGD on top of these features:
```bash
python3 -m transfer.transfer_cifar --feature_path=transfer/features/cifar100_resnext --batch_size=2048 --lr=8 --noise_multiplier=3.32
python3 -m transfer.transfer_cifar --feature_path=transfer/features/simclr_r50_2x_sk1 --batch_size=1024 --lr=4 --noise_multiplier=2.40
```
