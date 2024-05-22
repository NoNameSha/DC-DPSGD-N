# Implementation of Discriminative Clipping (DC)-DPSGD

This code implements the DC-DPSGD algorithm.


# Environment
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
# Example commands

CUDA_VISIBLE_DEVICES=0 python main.py   --private  --eps 8 --delta 1e-5

