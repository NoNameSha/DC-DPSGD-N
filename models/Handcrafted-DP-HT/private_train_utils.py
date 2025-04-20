import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

from torch.distributions import Normal, weibull

from losses_LDAW import LDAMLoss
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100

def get_device():
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def train_private(model, train_loader, optimizer, hdp, clip, s_clip, noise_multiplier, n_acc_steps=1):
    device = next(model.parameters()).device
    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0

    loss_f = nn.CrossEntropyLoss(reduction='sum')
    loss_f = extend(loss_f)

    rem = len(train_loader) % n_acc_steps
    num_batches = len(train_loader)
    num_batches -= rem

    bs = train_loader.batch_size if train_loader.batch_size is not None else train_loader.batch_sampler.batch_size
    print(f"training on {num_batches} batches of size {bs}")

    for batch_idx, (data, target) in enumerate(train_loader):

        if batch_idx > num_batches - 1:
            break

        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = loss_f(output, target)
        
        with backpack(BatchGrad()):
            loss.backward()
            
            ### clip
            if hdp == True:
                eigen_g, num_hvt = process_grad_batch_with_rp(list(model.parameters()), clip, s_clip, noise_multiplier)
            else:
                process_grad_batch(list(model.parameters()), clip)
            
            ## add noise to gradient
            for p in model.parameters():
                if p.requires_grad == True:

                    if hdp == True:
                        grad_noise = (num_hvt/bs)*torch.normal(0, noise_multiplier*s_clip/bs, size=p.grad.shape, device=p.grad.device) + (1-(num_hvt/bs))*torch.normal(0, noise_multiplier*clip/bs, size=p.grad.shape, device=p.grad.device)
                        p.grad.data += grad_noise
                    else:
                        grad_noise = torch.normal(0, noise_multiplier*clip/bs, size=p.grad.shape, device=p.grad.device)
                        p.grad.data += grad_noise
        optimizer.step()

        # if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(train_loader)):
        #     optimizer.step()
        #     optimizer.zero_grad()
        # else:
        #     with torch.no_grad():
        #         # accumulate per-example gradients but don't take a step yet
        #         optimizer.virtual_step()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += loss_f(output, target).item()
        num_examples += len(data)

    train_loss /= num_examples
    train_acc = 100. * correct / num_examples

    print(f'Train set: Average loss: {train_loss:.4f}, '
            f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')

    return train_loss, train_acc


def train(model, cls_num_list, per_cls_weights, train_loader, optimizer, n_acc_steps=1):
    device = next(model.parameters()).device
    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0

    #loss_f = nn.CrossEntropyLoss(reduction='sum')
    loss_f = LDAMLoss(cls_num_list=cls_num_list, weight=per_cls_weights)
    loss_f = extend(loss_f)

    rem = len(train_loader) % n_acc_steps
    num_batches = len(train_loader)
    num_batches -= rem

    bs = train_loader.batch_size if train_loader.batch_size is not None else train_loader.batch_sampler.batch_size
    print(f"training on {num_batches} batches of size {bs}")

    for batch_idx, (data, target) in enumerate(train_loader):

        if batch_idx > num_batches - 1:
            break

        data, target = data.to(device), target.to(device)

        output = model(data)

        #loss = F.cross_entropy(output, target)
        loss = loss_f(output, target)
        loss.backward()

        if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                # accumulate per-example gradients but don't take a step yet
                optimizer.virtual_step()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.cross_entropy(output, target).item()
        num_examples += len(data)

    train_loss /= num_examples
    train_acc = 100. * correct / num_examples

    print(f'Train set: Average loss: {train_loss:.4f}, '
            f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')

    return train_loss, train_acc


def test(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    loss_f = nn.CrossEntropyLoss(reduction='sum')
    loss_f = extend(loss_f)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_f(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_loss, test_acc


def process_grad_batch_with_rp(params, clipping, s_clipping, sigma):
    g_weibull = weibull.Weibull(1.0, 1)
    d_gauss = Normal(0,1)
    n = params[-1].grad_batch.shape[0]
    eigen_g = torch.ones(n).cuda()
    grad_norm_list = torch.zeros(n).cuda()
    scaling = torch.zeros(n).cuda()
    noise_scaling = torch.zeros(n).cuda()

    idx_layer = 0
    for p in params:  #every layer
        if p.requires_grad == True:
            flat_g = p.grad_batch.reshape(n, -1)
            pd = flat_g.shape[1]
            ### clipping norm
            current_norm_list = torch.norm(flat_g, dim=1)
            grad_norm_list += torch.square(current_norm_list)
            if idx_layer == 1:     
                weibull_rp = g_weibull.sample([pd,1000])
                Vk_layer, _ = np.linalg.qr(weibull_rp)
                #Vk_layer, _ = torch.linalg.qr(weibull_rp).float().cuda()
                
                # gauss_rp = d_gauss.sample([pd,int(pd/2)])
                # Vk_layer, _ = np.linalg.qr(gauss_rp)

                Vk_layer = torch.from_numpy(Vk_layer).float().cuda()
                dir_g = flat_g/torch.norm(flat_g)
                abs_dir_g = torch.abs(dir_g.reshape(n,-1,1))
                cov_g = torch.matmul(torch.matmul(Vk_layer.T,torch.matmul(abs_dir_g, abs_dir_g.reshape(n,1,-1))), Vk_layer)
                for i in range(n):
                    eigen_g[i] = torch.trace(cov_g[i]) + torch.normal(0, sigma/n, size=torch.tensor(1).shape, device=p.grad.device) 
                    #eigen_g[i] = torch.trace(cov_g[i]) 
    
                sorted_eig, indices = torch.sort(eigen_g, descending = True) #true >
            
                threshold_eig = sorted_eig[int(n/10)] #pd/10
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

    return eigen_g, num_hvt


def process_grad_batch(params, clipping=1):
    n = params[-1].grad_batch.shape[0]
    #print("n:", n)
    #print("len(params):", len(params))
    grad_norm_list = torch.zeros(n).cuda()
    for p in params:  #every layer
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
