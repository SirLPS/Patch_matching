# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 12:01:03 2018

@author: lps
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision as tv
from tqdm import tqdm 
from copy import deepcopy
import numpy as np
import cv2
import os
import math 
import random
import copy
import PIL
from sklearn.metrics import roc_curve, auc
import argparse
#from utils import L2Norm, ErrorRateAt95Recall 
#from Losses import triplet_simi


parser = argparse.ArgumentParser(description="The model")

# Training settings
parser.add_argument('--dataroot', type=str, default = '../Brown_datasets/',
                    help = 'path to dataset')
parser.add_argument('--log-dir', default='./logs',
                    help = 'flolder to output model checkpoints')
parser.add_argument('--model-dir', default='./models',
                    help = 'flolder to output model checkpoints')
parser.add_argument('--enable-logging',type=bool, default=True,
                    help='output to tensorlogger')
parser.add_argument('--imageSize', type=int, default=64,
                    help = 'the height / width of the input image to network')
parser.add_argument('--resume', default='', type=str, metavar='PATH',)
#                /home/sw/Desktop/fr/tongyuan/ty/modelsliberty_train/_liberty/checkpoint_11.pth
parser.add_argument('--resumeF', default='', type=str, metavar='PATH',)
#                /home/sw/Desktop/fr/tongyuan/ty/modelsliberty_train/_liberty/checkpoint_11.pth                    
parser.add_argument('--resumeL', default='', type=str, metavar='PATH',)
#                /home/sw/Desktop/fr/tongyuan/ty/modelsliberty_train/_liberty/checkpoint_11.pth
                    
               
parser.add_argument('--start-epoch', default=0, type=int, metavar = 'N',
                    help = 'manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=20, metavar='E',
                    help = 'number of epochs to train(default=20)')
parser.add_argument('--experiment-name', default= 'model1/',
                    help='experiment path')
#parser.add_argument('--training-set', default= 'notredame',
#                    help='Other options: notredame, yosemite')
parser.add_argument('--mean-image', type=float, default=0.443728476019,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209,
                    help='std of train dataset for normalization')
parser.add_argument('--loss', default= 'triplet_margin',       # triplet_devide   triplet_margin
                    help='Other options: softmax, contrastive')

# Training options
parser.add_argument('--batch-size', type=int, default=64+128, metavar='BS',    # 160 best
                    help = 'input batch size for training')
parser.add_argument('--test-batch-size', type=int,default=64, metavar='BST',
                    help = 'input batch size for testing')
parser.add_argument('--n-triplets', type=int,default=500000, metavar = 'N',
                    help = 'how mant triplet will generate from dataset')
parser.add_argument('--lr', type=float, default=0.1, metavar = 'LR',         # 0.1 best
                    help = 'learning default')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar = 'LRD',
                    help='learning rate decay ratio')
parser.add_argument('--wd', default='1e-5', type=float, metavar='W',        #1e－4
                    help = 'weight decay')
parser.add_argument('--margin', type=float, default=2.0, metavar='MARGIN',
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
parser.add_argument('--fliprot', type=bool, default=False,
                    help='turns on flip and 90deg rotation augmentation')

# Device options
parser.add_argument('--no-cuda',action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='3', type=str,
                    help = 'id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-workers', default= 10, type=int,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
parser.add_argument('--which', default=0, type=int,
                    help = 'id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
dataset_names = ['liberty','yosemite','notredame']  #, 'notredame', 'yosemite'
triplet_flag = True

#suffix = '{}_{}'.format(args.experiment_name, args.training_set)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
#cv2.setRNGSeed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
 
if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

LOG_DIR =  args.log_dir + '/run-optim_{}-n{}-lr{}-wd{}-m{}-S{}-tanh' \
             .format(args.optimizer, args.n_triplets, args.lr, args.wd, args.margin, args.seed)   


class TripletPhotoTour(dset.PhotoTour):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, batch_size = None,load_random_triplets = False,  *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)

    @staticmethod
    def generate_triplets(labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels.numpy())
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= 120000:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]))
                n2 = np.random.randint(0, len(indices[c1]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]))
            n3 = np.random.randint(0, len(indices[c2]))
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        # transform images if required
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
            return self.triplets.size(0)
        else:
            return self.matches.size(0)
        

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
            TripletPhotoTour(train=True,
                             load_random_triplets = load_random_triplets,    
                             batch_size=args.batch_size,
                             root=args.dataroot,
                             name=training_set,
                             download=True,
                             transform=transform_train),
                             batch_size=args.batch_size,
                             shuffle=False, **kwargs)
        
    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             TripletPhotoTour(train=False,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     download=True,
                     transform=transform_test),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names[:2]]
    if idx ==0: 
        return train_loader, test_loaders
    else:
        return train_loader
    
