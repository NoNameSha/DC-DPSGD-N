import time
import os
import argparse
import numpy as np

import torch
import random
import torch.nn as nn
#from opacus import PrivacyEngine

from models import StandardizeLayer
from train_utils import get_device, train, train_private, test
from data import get_data
from dp_utils import ORDERS, get_privacy_spent, get_renyi_divergence
from log import Logger

#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

#privacy accountants
from opacus.accountants.utils import get_noise_multiplier


def main(feature_path=None, batch_size=2048, mini_batch_size=256,
         lr=1, optim="SGD", momentum=0.9, nesterov=False, seed=1, noise_multiplier=1,
         max_grad_norm=0.1, hdp=False, clip=0.1, s_clip = 1, max_epsilon=None, eps=1, delta=0.00001, epochs=40, logdir=None):


    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    current_time = current_time + 'hdp' + str(hdp) + 'lr' + str(lr) + 'c' + str(clip) + 's_c' + str(s_clip) + 's8-2&6' + 'sha-7-wei-1&1-k200'
    logdir = os.path.join(logdir, current_time)
    logger = Logger(logdir)
   
    print("current_time:",current_time)
    print("hdp:",hdp)
    print("clip:",clip)
    print("s_clip:",s_clip)
    print("lr:",lr)

    device = get_device()

    # get pre-computed features
    x_train = np.load(f"{feature_path}_train_lt.npy")
    x_test = np.load(f"{feature_path}_test_lt.npy")

    train_data, test_data = get_data("cifar10", augment=False)
    cls_num_list = train_data.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list

    train_sampler = None
    idx = epochs // 160
    betas = [0, 0.9999]
    effective_num = 1.0 - np.power(betas[idx], cls_num_list)
    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    y_train = np.asarray(train_data.targets)
    y_test = np.asarray(test_data.targets)
    
    print("len",len(x_train))
    print("len",len(y_train))
    trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=False, num_workers=1, pin_memory=True)

    n_features = x_train.shape[-1]
    try:
        mean = np.load(f"{feature_path}_mean_lt.npy")
        var = np.load(f"{feature_path}_var_lt.npy")
    except FileNotFoundError:
        mean = np.zeros(n_features, dtype=np.float32)
        var = np.ones(n_features, dtype=np.float32)

    bn_stats = (torch.from_numpy(mean).to(device), torch.from_numpy(var).to(device))

    model = nn.Sequential(StandardizeLayer(bn_stats), nn.Linear(n_features, 10)).to(device)
    model = extend(model)

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sampling_prob = bs / len(train_data)

    if hdp == True:
         noise_multiplier = get_noise_multiplier(target_epsilon= eps/2, target_delta=delta, 
           sample_rate= sampling_prob, epochs=epochs, accountant='rdp')
    else:
         noise_multiplier = get_noise_multiplier(target_epsilon= eps, target_delta=delta, 
           sample_rate= sampling_prob, epochs=epochs, accountant='rdp')    
    sigma = noise_multiplier

    for epoch in range(0, epochs):
        print(f"\nEpoch: {epoch}")

        train_loss, train_acc = train_private(model, cls_num_list, per_cls_weights, train_loader, optimizer, hdp=hdp, clip=clip, s_clip=s_clip, noise_multiplier = sigma, n_acc_steps=n_acc_steps)
        test_loss, test_acc = test(model, test_loader)
         
        logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc, eps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optim', type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action="store_true")
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument('--noise_multiplier', type=float, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--feature_path', default=None)
    parser.add_argument('--max_epsilon', type=float, default=None)
    parser.add_argument('--logdir', default='log')
    parser.add_argument('--hdp', default=False, type=bool, help='enable hdp-sgd')
    parser.add_argument('--eps', default=1, type=float, help='privacy budget-epsilon')
    parser.add_argument('--delta', default=0.00001, type=float, help='privacy budget-delta')
    parser.add_argument('--clip', default= 0.01, type=float, help='gradient clipping bound') #0.01-1
    parser.add_argument('--s_clip', default= 0.1, type=float, help='gradient clipping bound')    
    args = parser.parse_args()
    if(args.seed != -1): 
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    main(**vars(args))
