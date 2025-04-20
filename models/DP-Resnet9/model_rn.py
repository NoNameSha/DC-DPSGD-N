#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:16:37 2021

@author: fabian
"""
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def __init__(self):
        super(ImageClassificationBase, self).__init__()
        self.loss_f = nn.CrossEntropyLoss(reduction='mean')
        #loss_f = F.cross_entropy()
        self.loss_f = extend(self.loss_f)
    def training_step(self, batch):
        images, labels = batch 
        images = images.cuda()
        #labels = labels.cuda()
        out,_ = self(images)                  # Generate predictions        
        loss = self.loss_f(out, labels) # Calculate loss
        return loss
    
    def get_embedding(self, batch):
        images, labels = batch 
        images = images.cuda()
        #labels = labels.cuda()
        _,embbedding = self(images)                    # Generate predictions                
        return embbedding
    
    def validation_step(self, batch):
        images, labels = batch 
        images = images.cuda()
        #labels = labels.cuda()
        out,_ = self(images)                    # Generate predictions        
        loss = self.loss_f(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_channels, out_channels, pool=False, pool_no=2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              #nn.BatchNorm2d(out_channels), 
              nn.GroupNorm(4, 16, affine=False),
              Mish()
              ]
    if pool: layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True, pool_no=2)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 256, pool=True, pool_no=2)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        
        self.MP = nn.MaxPool2d(2)
        #self.MP = nn.AdaptiveAvgPool2d(2)
        self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(1024,num_classes)
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)        
        out = self.res2(out) + out        
        out = self.MP(out) # classifier(out_emb)
        out_emb = self.FlatFeats(out)
        out = self.classifier(out_emb)
        return out, out_emb

    # net.avgpool = nn.AdaptiveAvgPool2d(1)
        # net.bn1 = nn.GroupNorm(4, 16, affine=False) 
        # net.bn2 = nn.GroupNorm(4, 16, affine=False) 