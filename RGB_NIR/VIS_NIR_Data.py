# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:47:11 2018

@author: lenovo
"""
import torch 
import torch.nn as nn
from tqdm import tqdm 
import numpy as np
import random
import os
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description="The model")

 
parser.add_argument('--dataroot', type=str, default = '../nirscene',
                    help = 'path to dataset')
parser.add_argument('--log-dir', default='./logs',
                    help = 'flolder to out logs')
parser.add_argument('--model-dir', default = './models/',
                    help = 'flolder to output model checkpoints')
parser.add_argument('--enable-logging',type=bool, default=False,
                    help='output to tensorlogger')
parser.add_argument('--imageSize', type=int, default=64,
                    help = 'the height / width of the input image to network')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help = 'path to latest checkpoint (default:none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar = 'N',
                    help = 'manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                    help = 'number of epochs to train(default=20)')
parser.add_argument('--experiment-name', default= 'country_train/',
                    help='experiment path')
parser.add_argument('--training-set', default= 'country',
                    help='Other options: notredame, yosemite')
parser.add_argument('--mean-image', type=float, default=0.443728476019,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209,
                    help='std of train dataset for normalization')
parser.add_argument('--loss', default= 'triplet_margin',               # default:triplet_margin
                    help='Other options: softmax, contrastive,triplet_margin')

# Training options
parser.add_argument('--batch-size', type=int, default=128+64, metavar='BS',
                    help = 'input batch size for training')
parser.add_argument('--test-batch-size', type=int,default=128, metavar='BST',
                    help = 'input batch size for testing')
parser.add_argument('--n-triplets', type=int,default=500000, metavar = 'N',
                    help = 'how mant triplet will generate from dataset')
parser.add_argument('--lr', type=float, default=0.1, metavar = 'LR',
                    help = 'learning default')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar = 'LRD',
                    help='learning rate decay ratio')
parser.add_argument('--wd', default='1e-4', type=float, metavar='W',
                    help = 'weight decay')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 2.0')
parser.add_argument('--optimizer', default='sgd', type=str, metavar = 'OPT',
                    help = 'The optimizer to use')
parser.add_argument('--anchorswap', type=bool, default=False,
                    help = 'turns on anchor swap')
parser.add_argument('--anchorave', type=bool, default=False,
                    help='anchorave')
parser.add_argument('--batch-reduce', default= 'min',
                    help='Other options: average, random, random_global, L2Net')
parser.add_argument('--augmentation', type=bool, default=False,
                    help='turns on shift and small scale rotation augmentation')
parser.add_argument('--fliprot', type=bool, default=True,
                    help='turns on flip and 90deg rotation augmentation')

# Device options
parser.add_argument('--no-cuda',action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0,1,2,3', type=str,
                    help = 'id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-workers', default= 10, type=int,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')

args = parser.parse_args()


class VIS_NIR_Hard(nn.Module):
    """
    This is the version that output only pos or triplets.
    A like-HardNet version
    """
    def __init__(self, name=None, root=None, train=True, transform=None, batch_size = None,out_triplets=False,*arg, **kw):
        super(VIS_NIR_Hard, self).__init__()
        self.name = name
        self.root = root
        self.transform = transform
        self.train = train
        self.out_triplets = out_triplets
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size
        self.data = np.load(os.path.join(self.root,self.name+'.npz'))
        self.data_pair = self.data['arr_0']    
        self.label = self.data['arr_1']            
        self.num = self.data_pair.shape[0]     
         
        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.pair_wise = self.generate_data(self.data_pair, self.n_triplets)
            
            
    def generate_data(self, data_pair, num_triplets):
        pairwise = []
        idx = np.where(self.label==1)[0]
        already_idxs = set()
        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= 120000:
                already_idxs = set()
            c1 = np.random.choice(idx, 1)[0]
            while c1 in already_idxs:
                c1 = np.random.choice(idx, 1)[0]
            already_idxs.add(c1)
            
            c2 = np.random.randint(0, self.num)
            while c1 == c2:
                c2 = np.random.randint(0, self.num)
            if np.random.random()>0.5:
                pairwise.append([data_pair[c1,:,:64], data_pair[c1,:,64:], data_pair[c2,:,:64]])
            else:
                pairwise.append([data_pair[c1,:,:64], data_pair[c1,:,64:], data_pair[c2,:,64:]])
            
        return pairwise    
    
    
    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img)
            return img

        if not self.train:
            m = self.data_pair[index]           
            img1 = transform_img(m[:,:64])
            img2 = transform_img(m[:,64:])
            label = self.label[index]
            return img1, img2, label
        
        m = self.pair_wise[index]
        a, p, n  = m[0], m[1], m[2]
        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = None
        
        if self.out_triplets:
            img_n = transform_img(n)
    
        if args.fliprot:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0,2,1)
                img_p = img_p.permute(0,2,1)
                if self.out_triplets:
                    img_n = img_n.permute(0,2,1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
                if self.out_triplets:
                    img_n = torch.from_numpy(deepcopy(img_n.numpy()[:,:,::-1]))
                    
        if self.out_triplets:
            return (img_a, img_p, img_n)            
        else:        
            return (img_a, img_p)


    def __len__(self):
        if self.train:
            return  len(self.pair_wise)
        else:
            return self.num

    
import cv2, copy, PIL
import torchvision.transforms as transforms
dataset_names = ['country','field','forest','indoor','mountain','oldbuilding','street','urban','water']

def create_loaders(training_set, load_random_triplets = True, idx=0):

    test_dataset_names = copy.copy(dataset_names)
    test_dataset_names.remove(training_set)
    
    cv2_scale = lambda x: cv2.resize(x, dsize=(args.imageSize, args.imageSize),  
                                 interpolation=cv2.INTER_LINEAR)
    cv2_crop = lambda x:x[16:48,16:48]
    np_reshape = lambda x: np.reshape(x, (args.imageSize, args.imageSize, 1))
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    np_reshape64 = lambda x: np.reshape(x, (args.imageSize, args.imageSize, 1))
    
    cv2_scale32 = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)
    np_reshape32 = lambda x: np.reshape(x, (32, 32, 1))
    
    transform_test = transforms.Compose([
            transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor()])
    transform_train = transforms.Compose([
            transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.RandomRotation(5,PIL.Image.BILINEAR),
            transforms.RandomResizedCrop(32, scale = (0.9,1.0),ratio = (0.9,1.1)),
            transforms.Resize(32),
            transforms.ToTensor()])
    
    if idx == 10000:
        transform = transforms.Compose([          
                transforms.Lambda(cv2_scale32),     # default 32
                transforms.Lambda(np_reshape),
                transforms.ToTensor(),
                transforms.Normalize((args.mean_image,), (args.std_image,))])
        
    else:
        
        transform = transforms.Compose([         
                transforms.Lambda(cv2_scale),     # default 32
                transforms.Lambda(np_reshape),
                transforms.ToTensor(),
                transforms.Normalize((args.mean_image,), (args.std_image,))])
    
    if not args.augmentation:
        transform_train = transform
        transform_test = transform
    train_loader = torch.utils.data.DataLoader(
            VIS_NIR_Hard(train=True,
                             batch_size=args.batch_size,
                             root=args.dataroot,
                             name=training_set,
                             download=True,
                             transform=transform_train),
                             batch_size=args.batch_size,
                             shuffle=False, **kwargs)
        
    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             VIS_NIR_Hard(train=False,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     download=True,
                     transform=transform_test),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names[:8]]
    if idx ==0: 
        return train_loader, test_loaders
    else:
        return train_loader
