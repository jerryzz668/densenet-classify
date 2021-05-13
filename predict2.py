import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import cv2
import numpy as np
import densenet as dn
import pdb
# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value
parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=510, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=500, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: False)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='C:\\Users\\fs\\Desktop\\410unet+densenet\\densenet-pytorch-master_new\\densenet-pytorch-master\\densenet-pytorch-master\\CP0.pth', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_Unet_fs', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=False)
global args, best_prec1
if __name__ == "__main__":
        args = parser.parse_args([])
        val_dirs = 'C:\\Users\\fs\\Desktop\\densenet-pytorch-master\\testimg'
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([ 
                normalize
                    ])
        # model = dn.DenseNet3(100, 4, 24, bottleneck=args.bottleneck,  small_inputs = False)
        # model = model.cuda()
        # if args.resume:
        #     if os.path.isfile(args.resume):
        #         print("=> loading checkpoint '{}'".format(args.resume))
        #         checkpoint = torch.load(args.resume)
        #         # args.start_epoch = checkpoint['epoch']
        #         #best_prec1 = checkpoint['best_prec1']
        #         # model.load_state_dict(checkpoint['state_dict'])
        #         model.load(checkpoint['state_dict'])
        #         print("=> loaded checkpoint '{}' (epoch {})"
        #                 .format(args.resume, checkpoint['epoch']))
        #     else:
        #         print("=> no checkpoint found at '{}'".format(args.resume))
        model=torch.load('C:\\Users\\fs\\Desktop\\410unet+densenet\\densenet-pytorch-master_new\\densenet-pytorch-master\\densenet-pytorch-master\\CP0.pth')
        img = cv2.imread('flaw (6).bmp').astype(np.float32)
        #img = cv2.imread('bubble(2).bmp').astype(np.float32)
        img.shape
        img = np.transpose(img, axes=[2, 0, 1])
        model = model.eval()
        model = model.cuda()
        #img = np.expand_dims(img, axis=0)
        print(type(img))
        transform_t = transforms.Compose([    
                normalize,
                    ])
        #type(input)
        inputim = torch.from_numpy(img)
        inputim1 = transform_t(inputim/255)
        inputim1 = inputim1.unsqueeze(0)
        inputim1 = inputim1.cuda()
        inputim1 = torch.autograd.Variable(inputim1)
        print(inputim1.shape)

        #model.eval()
        output = model(inputim1)
        print(output)
        x= torch.max(output,1)
        # classes = []
        # with open('classes.txt', 'r') as list_:
        #     for line in list_:
        #         classes.append(line.rstrip('\n'))

        print('Prediction:{} '.format(x))