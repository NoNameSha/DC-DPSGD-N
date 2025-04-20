import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets 
import torchvision.io as io

from functools import partial

import os


import numpy as np
#from rdp_accountant import compute_rdp, get_privacy_spent
#from lanczos import Lanczos
from torch.distributions import Normal, weibull

#from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100


def process_grad_batch_with_rp(params, clipping, s_clipping, sigma):
    g_weibull = weibull.Weibull(1/2, 1)
    d_gauss = Normal(0,1)
    n = params[-1].grad_batch.shape[0]
    eigen_g = torch.ones(n).cuda()
    grad_norm_list = torch.zeros(n).cuda()
    scaling = torch.zeros(n).cuda()
    noise_scaling = torch.zeros(n).cuda()
    grad_summed = []

    idx_layer = 0
    for p in params:  #every layer
        if p.requires_grad == True:
            flat_g = p.grad_batch.reshape(n, -1)
            pd = flat_g.shape[1]
            ### clipping norm
            current_norm_list = torch.norm(flat_g, dim=1)
            grad_norm_list += torch.square(current_norm_list)
            if idx_layer == 0:     
                weibull_rp = g_weibull.sample([pd,100])
                weibull_rp = torch.Tensor(weibull_rp).cuda()
                Vk_layer,_ = torch.linalg.qr(weibull_rp)
                
                #Vk_layer, _ = np.linalg.qr(weibull_rp)
                #Vk_layer = torch.from_numpy(Vk_layer).float().cuda()

                # gauss_rp = d_gauss.sample([pd,int(pd/2)])
                # Vk_layer, _ = np.linalg.qr(gauss_rp)
                
                dir_g = flat_g/torch.norm(flat_g)
                abs_dir_g = torch.abs(dir_g.reshape(n,-1,1))
                cov_g = torch.matmul(torch.matmul(Vk_layer.T,torch.matmul(abs_dir_g, abs_dir_g.reshape(n,1,-1))), Vk_layer)
                for i in range(n):
                    eigen_g[i] = torch.trace(cov_g[i]) + torch.normal(0, sigma/n, size=torch.tensor(1).shape, device=p.grad.device)  #sigma/n
                    #eigen_g[i] = torch.trace(cov_g[i]) 
    
                sorted_eig, indices = torch.sort(eigen_g, descending = True) #true >
            
                threshold_eig = sorted_eig[int(n/100)] #pd/10   10-20-50-100
                #print("threshold_eig:",threshold_eig)
                eigen_g[eigen_g > threshold_eig] = 0
                num_hvt = torch.sum(eigen_g == 0).item()
                #print("num of heavy-tailed samples:", num_hvt)

                ### random
                # for i in range(n):
                #     weibull_rp = g_weibull.sample([p,int(p/10)])
                #     Vk_layer, _ = np.linalg.qr(weibull_rp)
                #     Vk_layer = torch.from_numpy(Vk_layer).float().cuda()
                #     dir_g = flat_g[i]/torch.norm(flat_g[i])
                #     abs_dir_g = torch.abs(dir_g.reshape(-1,1))
                #     cov_g = torch.matmul(torch.matmul(Vk_layer.T,torch.matmul(abs_dir_g, abs_dir_g.reshape(1,-1))), Vk_layer)
                #     eigen_g[i] = torch.trace(cov_g) 
                # eigen_g[eigen_g<0.6] = 0

            idx_layer += 1
    grad_norm_list = torch.sqrt(grad_norm_list)

    #grad_norm_noise_list = grad_norm_list + torch.normal(0, sigma*c, size=grad_norm_list.shape, device=p.grad.device)

    for i in range(n):
        if eigen_g[i] == 0:
            scaling[i] = s_clipping/grad_norm_list[i]
            noise_scaling[i] = s_clipping
        else:
            scaling[i] = clipping/grad_norm_list[i]
            noise_scaling[i] = clipping
    scaling[scaling>1] = 1

    for p in params:
        if p.requires_grad == True:
            p_dim = len(p.shape)
            scaling = scaling.view([n] + [1]*p_dim)
            p.grad_batch *= scaling
            grad_noise_normal = torch.normal(0, sigma*clipping/n, size=p.grad.shape, device=p.grad.device)
            grad_noise_heavy_tailed = torch.normal(0, sigma*s_clipping/n, size=p.grad.shape, device=p.grad.device)
            #for i in range(n):
                # if eigen_g[i] == 0:
                #     p.grad_batch[i] += grad_noise_heavy_tailed
                # else:
                #     p.grad_batch[i] += grad_noise_normal

                # if eigen_g[i] == 0:
                #     p.grad_batch[i] += torch.normal(0, sigma*s_clipping/n, size=p.grad.shape, device=p.grad.device)
                # else:
                #     p.grad_batch[i] += torch.normal(0, sigma*clipping/n, size=p.grad.shape, device=p.grad.device)
                

                # grad_noise = torch.normal(0, sigma*noise_scaling[i]/n, size=p.grad.shape, device=p.grad.device)
                # p.grad_batch[i] += grad_noise
            p.grad = torch.mean(p.grad_batch, dim=0)
            p.grad_batch.mul_(0.)
            grad_summed.append(p.grad)

    return eigen_g, num_hvt, grad_summed


def process_grad_batch(params, clipping=1):
    n = params[-1].grad_batch.shape[0]
    #print("n:", n)
    #print("len(params):", len(params))
    grad_norm_list = torch.zeros(n).cuda()
    grad_summed = []
    for i, p in enumerate(params):  #every layer
        if p.requires_grad == True:
            flat_g = p.grad_batch.reshape(n, -1)
            #print("flat_g:", flat_g.shape)
            current_norm_list = torch.norm(flat_g, dim=1)
            #print("current_norm_list:", current_norm_list[0:10])
            grad_norm_list += torch.square(current_norm_list)
            #print("grad_norm_list",grad_norm_list.shape)
    grad_norm_list = torch.sqrt(grad_norm_list)

    # clipping dp-sgd
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1  

    # auto clip - bu
    # gamma=0.01
    # scaling = clipping/(grad_norm_list + gamma)

    # auto clip - xia
    # gamma=0.01
    # scaling = clipping/(grad_norm_list + gamma / (grad_norm_list + gamma) )

    for p in params:
        if p.requires_grad == True:
            p_dim = len(p.shape)
            #print("scaling:",scaling.shape)
            scaling = scaling.view([n] + [1]*p_dim)
            #print("scaling-a:",scaling.shape)
            p.grad_batch *= scaling
            p.grad = torch.mean(p.grad_batch, dim=0)
            p.grad_batch.mul_(0.)
            grad_summed.append(p.grad)

    return grad_summed

def process_grad_batch_v2(params, clipping=1):
    n = params[-1].grad_sample.shape[0]
    #print("n:", n)
    #print("len(params):", len(params))
    grad_norm_list = torch.zeros(n).cuda()
    for p in params:  #every layer
        if p.requires_grad == True:
            flat_g = p.grad_sample.reshape(n, -1)
            #print("flat_g:", flat_g.shape)
            current_norm_list = torch.norm(flat_g, dim=1)
            #print("current_norm_list:", current_norm_list[0:10])
            grad_norm_list += torch.square(current_norm_list)
            #print("grad_norm_list",grad_norm_list.shape)
    grad_norm_list = torch.sqrt(grad_norm_list)

    # clipping dp-sgd
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1  

    # auto clip - bu
    # gamma=0.01
    # scaling = clipping/(grad_norm_list + gamma)

    # auto clip - xia
    # gamma=0.01
    # scaling = clipping/(grad_norm_list + gamma / (grad_norm_list + gamma) )

    for p in params:
        if p.requires_grad == True:
            p_dim = len(p.shape)
            #print("scaling:",scaling.shape)
            scaling = scaling.view([n] + [1]*p_dim)
            #print("scaling-a:",scaling.shape)
            p.grad_sample *= scaling
            p.grad = torch.mean(p.grad_sample, dim=0)
            p.grad_sample.mul_(0.)
