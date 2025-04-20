import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

import os
import argparse
import csv
import random
import time
import numpy as np

from models.resnet import resnet18
from models.cnn import CNN, LeNet
from utils import get_data_loader, get_sigma, restore_param, checkpoint, adjust_learning_rate, process_grad_batch, process_grad, process_grad_batch_with_rp
from main_utils import save_pro

#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

#privacy accountants
from opacus.accountants.utils import get_noise_multiplier

parser = argparse.ArgumentParser(description='Differentially Private learning with DP-SGD')

## general arguments
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='resnet20_cifar10', type=str, help='session name')
parser.add_argument('--seed', default=2, type=int, help='random seed')
parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
parser.add_argument('--batchsize', default=100, type=int, help='batch size')
parser.add_argument('--n_epoch', default=40, type=int, help='total number of epochs')
parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')
parser.add_argument('--time', default='03092235', type=str, help='time')
parser.add_argument('--save_dir', default='res', type=str, help='save path')



## arguments for learning with differential privacy
parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
parser.add_argument('--hdp', default=False, type=bool, help='enable hdp-sgd')
parser.add_argument('--clip', default= 0.01, type=float, help='gradient clipping bound')
parser.add_argument('--s_clip', default= 0.1, type=float, help='gradient clipping bound')

parser.add_argument('--lr', default= 1, type=float, help='base learning rate (default=0.1)')
parser.add_argument('--eps', default=0.2, type=float, help='privacy parameter epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')



args = parser.parse_args()

assert args.dataset in ['cifar10', 'cifar10-LT', 'svhn', 'mnist', 'fmnist']

use_cuda = True
best_acc = 0  
accuracy_accountant = []
grad_norm_accountant = []
start_epoch = 0  
batch_size = args.batchsize

if(args.seed != -1): 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

print('==> Preparing data..')
## preparing data for training && testing
if(args.dataset == 'cifar10'):
    trainloader, testloader, n_training, n_test = get_data_loader('cifar10', batchsize = args.batchsize)
    train_samples, train_labels = None, None

elif(args.dataset == 'mnist'):
    trainloader, testloader, n_training, n_test = get_data_loader('mnist', batchsize = args.batchsize)
    train_samples, train_labels = None, None
elif(args.dataset == 'fmnist'):
    trainloader, testloader, n_training, n_test = get_data_loader('fmnist', batchsize = args.batchsize)
    train_samples, train_labels = None, None

print('# of training examples: ', n_training, '# of testing examples: ', n_test)


print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(args.eps, args.delta))
sampling_prob=args.batchsize/n_training
steps = int(args.n_epoch/sampling_prob)

if args.hdp == True:
     noise_multiplier = get_noise_multiplier(target_epsilon= args.eps/2, target_delta=args.delta, 
        sample_rate= sampling_prob, epochs=args.n_epoch, accountant='rdp')
else:
     noise_multiplier = get_noise_multiplier(target_epsilon= args.eps, target_delta=args.delta, 
        sample_rate= sampling_prob, epochs=args.n_epoch, accountant='rdp')    
noise_multiplier = sigma

print('noise scale: ', noise_multiplier, 'privacy guarantee: ', args.eps)

print('\n==> Creating '+ args.sess +' model instance')
if(args.resume):
    try:
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_file = './checkpoint/' + args.sess  + '.ckpt'
        checkpoint = torch.load(checkpoint_file)
        #net = resnet20()
        net = CNN(input_dim=1, output_dim=10)
        net.cuda()
        restore_param(net.state_dict(), checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    except:
        print('resume from checkpoint failed')
else:
    net = CNN(input_dim=1, output_dim=10)
    net.cuda()


net = extend(net)

num_params = 0
for p in net.parameters():
    num_params += p.numel()

print('total number of parameters: ', num_params/(10**6), 'M')

if(args.private):
    loss_func = nn.CrossEntropyLoss(reduction='sum')
else:
    loss_func = nn.CrossEntropyLoss(reduction='mean')

loss_func = extend(loss_func)

num_params = 0
np_list = []
for p in net.parameters():
    num_params += p.numel()
    np_list.append(p.numel())

optimizer = optim.SGD(
        net.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    Vk = []
    t0 = time.time()
    steps = n_training//args.batchsize

    if(train_samples == None): # using pytorch data loader for CIFAR10
        loader = iter(trainloader)
    else: 
        sample_idxes = np.arange(n_training)
        np.random.shuffle(sample_idxes)

    
    true_g = []
    batch_g = []
    noise_norm = []
    for p in net.parameters():
        g_para = torch.zeros((1, p.numel())).cuda()
        true_g.append(g_para)
    

    for batch_idx in range(steps):
        print("batch_idx:", batch_idx)
        if(args.dataset=='svhn'):
            current_batch_idxes = sample_idxes[batch_idx*args.batchsize : (batch_idx+1)*args.batchsize]
            inputs, targets = train_samples[current_batch_idxes], train_labels[current_batch_idxes]
        else:
            inputs, targets = next(loader)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if(args.private):
            logging = batch_idx % 20 == 0
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            with backpack(BatchGrad()):
                loss.backward()

                ### clip
                if args.hdp == True:
                    eigen_g, num_hvt = process_grad_batch_with_rp(list(net.parameters()), args.clip, args.s_clip, sigma)
                else:
                    process_grad_batch(list(net.parameters()), args.clip)
                
                
                ## add noise to gradient
                for p in net.parameters():
                    shape = p.grad.shape

                    numel = p.grad.numel()
                    if args.hdp == True:
                        grad_noise = (num_hvt/args.batchsize)*torch.normal(0, noise_multiplier*args.s_clip/args.batchsize, size=p.grad.shape, device=p.grad.device) + (1-(num_hvt/args.batchsize))*torch.normal(0, noise_multiplier*args.clip/args.batchsize, size=p.grad.shape, device=p.grad.device)
                        p.grad.data += grad_noise
                    else:
                        grad_noise = torch.normal(0, noise_multiplier*args.clip/args.batchsize, size=p.grad.shape, device=p.grad.device)
                        p.grad.data += grad_noise
                             
                   
        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            try:
                for p in net.parameters():
                    del p.grad_batch
            except:
                pass
        optimizer.step()
        step_loss = loss.item()
        if(args.private):
            step_loss /= inputs.shape[0]
            #print("input.shape:",inputs.shape)
        train_loss += step_loss
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).float().cpu().sum()
        acc = 100.*float(correct)/float(total)
    t1 = time.time()
    print('Train loss:%.5f'%(train_loss/(batch_idx+1)), 'time: %d s'%(t1-t0), 'train acc:', acc, end=' ')
    return (train_loss/batch_idx, acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_correct = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            step_loss = loss.item()
            if(args.private):
                step_loss /= inputs.shape[0]

            test_loss += step_loss 
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()
            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()

        acc = 100.*float(correct)/float(total)
        accuracy_accountant.append(acc)
        print('test loss:%.5f'%(test_loss/(batch_idx+1)), 'test acc:', acc)
        ## Save checkpoint.
        if acc > best_acc:
            best_acc = acc
            checkpoint(net, acc, epoch, args.sess)

    return (test_loss/batch_idx, acc)


print('\n==> Strat training')

for epoch in range(start_epoch, args.n_epoch):
    lr = adjust_learning_rate(optimizer, args.lr, epoch, all_epoch=args.n_epoch)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    save_pro.save_progress(args, accuracy_accountant, grad_norm_accountant)
