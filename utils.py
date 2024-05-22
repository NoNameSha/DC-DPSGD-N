
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os


import numpy as np
from rdp_accountant import compute_rdp, get_privacy_spent
#from lanczos import Lanczos
from torch.distributions import Normal, weibull

from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100


def process_grad_batch_with_rp(params, clipping, s_clipping, sigma):
    g_weibull = weibull.Weibull(1.0, 1)
    d_gauss = Normal(0,1)
    n = params[0].grad_batch.shape[0]
    eigen_g = torch.ones(n).cuda()
    grad_norm_list = torch.zeros(n).cuda()
    scaling = torch.zeros(n).cuda()
    noise_scaling = torch.zeros(n).cuda()

    idx_layer = 0
    for p in params:  #every layer
        flat_g = p.grad_batch.reshape(n, -1)
        pd = flat_g.shape[1]
        ### clipping norm
        current_norm_list = torch.norm(flat_g, dim=1)
        grad_norm_list += torch.square(current_norm_list)
        if idx_layer == 0:     
            weibull_rp = g_weibull.sample([pd,200])
            Vk_layer, _ = np.linalg.qr(weibull_rp)

            
            Vk_layer = torch.from_numpy(Vk_layer).float().cuda()
            dir_g = flat_g/torch.norm(flat_g)
            abs_dir_g = torch.abs(dir_g.reshape(n,-1,1))
            cov_g = torch.matmul(torch.matmul(Vk_layer.T,torch.matmul(abs_dir_g, abs_dir_g.reshape(n,1,-1))), Vk_layer)
            for i in range(n):
                eigen_g[i] = torch.trace(cov_g[i]) + torch.normal(0, sigma, size=torch.tensor(1).shape, device=p.grad.device) 
                #eigen_g[i] = torch.trace(cov_g[i]) 
 
            sorted_eig, indices = torch.sort(eigen_g, descending = True) #true >
          
            threshold_eig = sorted_eig[int(n/10)] 
            eigen_g[eigen_g > threshold_eig] = 0
            num_hvt = torch.sum(eigen_g == 0).item()
           
        idx_layer += 1
    grad_norm_list = torch.sqrt(grad_norm_list)

    for i in range(n):
        if eigen_g[i] == 0:
            scaling[i] = s_clipping/grad_norm_list[i]
            noise_scaling[i] = s_clipping
        else:
            scaling[i] = clipping/grad_norm_list[i]
            noise_scaling[i] = clipping
    scaling[scaling>1] = 1

    for p in params:
        p_dim = len(p.shape)
        scaling = scaling.view([n] + [1]*p_dim)
        p.grad_batch *= scaling
        grad_noise_normal = torch.normal(0, sigma*clipping/n, size=p.grad.shape, device=p.grad.device)
        grad_noise_heavy_tailed = torch.normal(0, sigma*s_clipping/n, size=p.grad.shape, device=p.grad.device)
    
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)

    return eigen_g, num_hvt


def process_grad_batch(params, clipping=1):
    n = params[0].grad_batch.shape[0]
    grad_norm_list = torch.zeros(n).cuda()
    for p in params:  #every layer
        flat_g = p.grad_batch.reshape(n, -1)
        current_norm_list = torch.norm(flat_g, dim=1)
        grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list)
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1

    for p in params:
        p_dim = len(p.shape)
        scaling = scaling.view([n] + [1]*p_dim)
        p.grad_batch *= scaling
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)


def process_grad(params, clipping=1):
    for p in params:
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)
        

def process_layer_grad_batch(params, batch_idx, Vk, clipping=1):
    n = params[0].grad_batch.shape[0]
    grad_norm_list = torch.zeros(len(params), n).cuda()
    idx_layer = 0
    for p in params:  #every layer
        flat_g = p.grad_batch.reshape(n, -1)
        mean_batch = torch.mean(flat_g, dim=0)

        if batch_idx == 0:
            print("flat_g - mean_batch:", (flat_g - mean_batch).shape)
            Vk_layer, _, _ = torch.linalg.svd( (flat_g - mean_batch).T, full_matrices=False)
            Vk.append(Vk_layer[:,0:1])
            print("Vk_layer:", Vk_layer[:,0:1].shape)
        Vk_layer = Vk[idx_layer]
        flat_g = torch.matmul(Vk_layer, torch.matmul(Vk_layer.T, flat_g.T)).T + mean_batch
        p.grad_batch = flat_g.reshape(p.grad_batch.shape)
       
        current_norm_list = torch.norm(flat_g, dim=1)
        
        grad_norm_list[idx_layer] += current_norm_list
        idx_layer += 1 
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1
    
    idx_layer = 0
    for p in params:
        p_dim = len(p.shape)
       
        scaling_layer = scaling[idx_layer].view([n] + [1]*p_dim)
       
        idx_layer += 1
        p.grad_batch *= scaling_layer
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)
    return grad_norm_list[15,0], Vk


def sparsify(d, ratio):
    vec_one = np.ones((d,1))
    vec_zero = np.zeros((d,1))
    vec = np.concatenate((vec_one[0:int(d*ratio),:], vec_zero[0:int(d*(1-ratio)),:]))
    idx = np.arange(d)
    np.random.shuffle(idx)
    vec = torch.from_numpy(vec[idx]).float().cuda()
    return vec

def eigen_by_lanczos(mat, proj_dims):
        T, V = Lanczos(mat, 128)
        T_evals, T_evecs = np.linalg.eig(T)
        idx = T_evals.argsort()[-1 : -(proj_dims+1) : -1]
        Vk = np.dot(V.T, T_evecs[:,idx])
        return Vk


def get_data_loader(dataset, batchsize):
    if(dataset == 'svhn'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.SVHN('./data/SVHN',split='train', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=73257, shuffle=True, num_workers=0) #load full btach into memory, to concatenate with extra data

        extraset = torchvision.datasets.SVHN('./data/SVHN',split='extra', download=True, transform=transform)
        extraloader = torch.utils.data.DataLoader(extraset, batch_size=531131, shuffle=True, num_workers=0) #load full btach into memory

        testset = torchvision.datasets.SVHN('./data/SVHN',split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)
        return trainloader, extraloader, testloader, len(trainset)+len(extraset), len(testset)

    if(dataset == 'mnist'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.MNIST(root='./data',train=True, download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2) #load full btach into memory, to concatenate with extra data

        testset = torchvision.datasets.MNIST(root='./data',train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
        return trainloader, testloader, len(trainset), len(testset)

    elif(dataset == 'fmnist'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data',train=True, download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2) #load full btach into memory, to concatenate with extra data

        testset = torchvision.datasets.FashionMNIST(root='./data',train=False, download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
        return trainloader, testloader, len(trainset), len(testset)
        
    elif(dataset == 'tiny_imagenet'):  
        data_dir = "tiny-imagenet-200/"
        num_workers = {"train": 2, "val": 0, "test": 0}
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]
            ),
        }
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]
        }
        dataloaders = {
            x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=num_workers[x])
            for x in ["train", "val", "test"]
        }

        return dataloaders["train"], dataloaders["test"], len(image_datasets["train"]), len(image_datasets["test"])


    elif(dataset == 'cifar10'):
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        #trainset = IMBALANCECIFAR10(root='./data/CIFAR10', imb_type="exp", imb_factor=0.01, rand_number=0, train=True, download=False, transform=transform_train)
        trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=False, transform=transform_train) 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=False, transform=transform_test) 
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

        return trainloader, testloader, len(trainset), len(testset)



def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        if(rgp):
            rdp = compute_rdp(q, cur_sigma, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        if(cur_eps<eps and cur_sigma>interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break    
    return cur_sigma, previous_eps


## interval: init search inerval
## rgp: use residual gradient perturbation or not
def get_sigma(q, T, eps, delta, init_sigma=10, interval=1., rgp=True):
    cur_sigma = init_sigma
    
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps


def restore_param(cur_state, state_dict):
    own_state = cur_state
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)

def sum_list_tensor(tensor_list, dim=0):
    return torch.sum(torch.cat(tensor_list, dim=dim), dim=dim)

def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


def checkpoint(net, acc, epoch, sess):
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
    }
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + sess  + '.ckpt')

def adjust_learning_rate(optimizer, init_lr, epoch, all_epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    decay = 1.0
    if(epoch<all_epoch*0.5):
        decay = 1.
    elif(epoch<all_epoch*0.75):
        decay = 10.
    else:
        decay = 100.

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr / decay
    return init_lr / decay
