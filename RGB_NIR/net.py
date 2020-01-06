#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 10:50:30 2018

@author: lps
 
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch
from tqdm import tqdm 
import numpy as np
import math 
import os
import time
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc

import torch.optim as optim
import torch.backends.cudnn as cudnn
from VIS_NIR_Data import create_loaders, args
from utils import L2Norm, ErrorRateAt95Recall 
from Losses import find_hard_pair
from spp_layer import spatial_pyramid_pool


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
triplet_flag = False


if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
 
if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

LOG_DIR =  args.log_dir + '/run-optim_{}-n{}-lr{}-wd{}-m{}-S{}-tanh' \
             .format(args.optimizer, args.n_triplets, args.lr, args.wd, args.margin, args.seed)   

       
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
       
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False), 
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),            
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
      
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=2,bias = False),   # stride = 2
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
       
            nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1, dilation=2, bias = False),   # stride = 2  wrong:10368
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3,  stride=1 ,padding=1,  bias = False),          # bs, 128,8,8
            nn.BatchNorm2d(128, affine=False), 
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1 ,padding=1,  bias = False),
            nn.BatchNorm2d(128, affine=False), 
            )

        self.output_num = [8,4,2,1]
         
        self.fc1 = nn.Sequential(                                     
                nn.Linear(10880,128),  
                )
        return
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input1):

        bs = input1.size(0)
        feat = self.block(self.input_norm(input1))
        spp_a = spatial_pyramid_pool(feat, bs, [int(feat.size(2)), int(feat.size(3))], self.output_num)

        feature_a = self.fc1(spp_a).view(bs, -1)   
        
        return  L2Norm()(feature_a)      


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.3)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return


def train(train_loader, model, optimizer, epoch, load_triplets, suffix):
    # load pairwise as default
    model.train()
    for group in optimizer.param_groups:
        if group['lr']>0:
             print(group['lr'])
    
    pbar = tqdm(enumerate(train_loader))

    for batch_idx, data in pbar:

        if load_triplets:
            data_a, data_p, data_n = data
        else:
            data_a, data_p = data

        if args.cuda:
            data_a, data_p  = data_a.cuda(), data_p.cuda()
            data_a, data_p = Variable(data_a), Variable(data_p)

            feat_a = model(data_a)
            feat_p = model(data_p)

        loss = find_hard_pair(feat_a, feat_p, epoch, True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if (batch_idx) % (args.log_interval) == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.item()))
    try:
        os.stat('{}{}'.format(args.model_dir,suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir,suffix))
    try:
      torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,epoch))
    except:
      pass


def test(test_loader, model, epoch, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)
        out_a = model(data_a)
        out_p = model(data_p)

        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy().reshape(-1,1))
        ll = label.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)

        if batch_idx % args.log_interval == 0:
            pbar.set_description(logger_test_name+' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    num_tests = len(test_loader.dataset)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))


    return fpr95


model = Model().cuda()
model = nn.DataParallel(model, [0,1,2])
optimizer1 = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=0.1)


def main(train_loader, test_loaders, model, suffix):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
            
    start = args.start_epoch
    end = start + args.epochs
    idx = 0
    
    for epoch in range(start, end):
        fff = []
        # iterate over test loaders and test results
        train(train_loader, model, optimizer1, epoch,  triplet_flag, suffix)
        with open('net1.txt', 'a') as f:
            f.write('\n')
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(now)
        for test_loader in test_loaders:
            fpr = test(test_loader['dataloader'], model, epoch, test_loader['name'])
            with open('net1.txt', 'a') as f:                        
                f.write('  ' +test_loader['name']+':'+ '{:6f}'.format(fpr)+' ')   
            fff.append(fpr)
        print('>>>',np.mean(np.array(fff)))   
        idx += 1
        train_loader = create_loaders(training_set, load_random_triplets=triplet_flag, idx=idx)
        scheduler.step()


if __name__ == '__main__':

    training_set = 'country'
    suffix = '{}_{}'.format(args.experiment_name, training_set)
    train_loader, test_loaders = create_loaders(training_set, load_random_triplets = False, idx=0)
    main(train_loader, test_loaders, model, suffix)
       
