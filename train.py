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
import torchvision.models as models2
import adabound
import models.densenet as dn
import torchvision.models as models2
models=models2.resnet18()
import models.datasets as da
import PIL
import torchvision.models as models
from torch.optim import SGD, Adam, lr_scheduler
import pdb
# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=330, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=20, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=16, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0.2, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: False)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_Unet_fs', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=False)

best_prec1 = 0
slr=0.1

train_dirs = '/home/adt/Desktop/aug_heidian/train'
val_dirs = '/home/adt/Desktop/aug_heidian/val'
def main():
    global args, best_prec1
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.name))
    
    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    if args.augment:
        transform_train = transforms.Compose([
            transforms.Pad(padding=32,fill=0,padding_mode='reflect'),
            # transforms.Resize([1408,1024]),
            transforms.CenterCrop(64),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.RandomAffine(3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            # transforms.Pad(padding=32, fill=0, padding_mode='reflect'),
            # # transforms.Resize([1408,1024]),
            # transforms.CenterCrop(96),
            transforms.Resize([128,128]),
            # transforms.RandomResizedCrop(64),
            # transforms.RandomHorizontalFlip(),            # transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(3),
            # transforms.ColorJitter(brightness=0.3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.RandomAffine(3),
            transforms.RandomHorizontalFlip(),            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        # transforms.CenterCrop(96),
        transforms.Resize([128,128]),
        # transforms.Pad(padding=32, fill=0, padding_mode='reflect'),
        # transforms.Resize([1408,1024]),

        transforms.ToTensor(),
        normalize
        ])
    #normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    train_dataset = datasets.ImageFolder(
        train_dirs,
        transform_train
        )
    print('train_dataset:{}'.format(train_dataset))     
    #pdb.set_trace()
    val_dataset = datasets.ImageFolder(
        val_dirs,
        transform_test)
             
    kwargs = {'num_workers': 16, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        #datasets.CIFAR10('../data', train=True, download=True,transform=transform_train),
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        #datasets.CIFAR10('../data', train=False, transform=transform_test),
        val_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # create model
    print("args.layers",args.layers)
    print("args.growth",args.growth)
    print("args.reduce",args.reduce)
    print("args.bottleneck",args.bottleneck)
    print("args.droprate",args.droprate)
    print("args.lr",args.lr)
    model = dn.DenseNet3(args.layers, 2, args.growth, reduction=args.reduce,
                         bottleneck=args.bottleneck, dropRate=args.droprate, small_inputs = False)
    # kwargs = {'num_classes': 2}
    # model = models2.densenet121(**kwargs)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    print ("Start training..")
    # from adabound import AdaBound
    # optimizer = AdaBound(model.parameters(), lr=args.lr, betas=(0.9, 0.999,),final_lr=0.1, gamma=1e-3, weight_decay=5e-4, amsbound=True)
    # optimizer = adabound.AdaBound(model.parameters(), lr=0.001, final_lr=0.1)
    # lambda1 = lambda epoch: pow((1 - ((epoch - 1) / args.epochs)), 0.99)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)    #  learning rate changed every epoch
    for epoch in range(args.start_epoch, args.epochs):
        FS_lr=adjust_learning_rate(optimizer, epoch,args.epochs)
        print("lr='{}'".format(FS_lr))
        # train for one epoch
        # scheduler.step(epoch)
        # for param_group in optimizer.param_groups:
        #     lr=param_group['lr']
        # print("lr='{}'".format(lr))
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        },model, is_best)
    print('Best accuracy: ', best_prec1)

def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        #pdb.set_trace()
        target = target.cuda()
        input = input.cuda()
        # print('input:{}'.format(input))
        # print('input.size:{}'.format(input.size(2)))
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # print('input_var:{}'.format(input_var))
        # print('input_var.size:{}'.format(input_var.size(2)))
        # print('target:{}'.format(target))
        # print('target_var:{}'.format(target_var))
        # print('target_var.size:{}'.format(target_var.size(0)))
        # compute output
        output = model(input_var)
        # print("output>shape:",output.shape)
        # print('output:{}'.format(output.size()))
        #pdb.set_trace()
        loss = criterion(output, target_var)
        # print('loss:{}'.format(loss))
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, model,is_best, filename='D_20200323_checkpoint.pth',filename2='D_20200323_CP0.pth'):
    """Saves checkpoint to disk"""
    directory = "runs2_C_aug_heidian/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    filename2 = directory + filename2
    torch.save(state, filename)
   
    if is_best:
        torch.save(model, filename2)
        shutil.copyfile(filename, 'runs2_C_aug_heidian/%s/'%(args.name) + 'D_20200323_model_best.pth')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch,epochs):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 60))  * (0.1 ** (epoch // 100))
    if epoch <= 30:
        lr = args.lr*(1.27**epoch)
        global slr
        slr=lr
    else:
        lr = slr * (0.993 ** (epoch-epochs*0.1))

    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # print('topk:{}'.format(topk))
    # print('maxk:{}'.format(maxk))
    batch_size = target.size(0)
    
    
    _, pred = output.topk(maxk, 1, True, True)
    # print('_:{}'.format(_))
    # print('pred:{}'.format(pred))
    pred = pred.t()
    # print('pred2:{}'.format(pred))
    # print('target:{}'.format(target))
    # print('target.view(1, -1):{}'.format(target.view(1, -1)))

    # target.view(1, -1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print('correct:{}'.format(correct))
    res = []
    for k in topk:
        # print('k:{}'.format(k))
        # print('correct[:k]:{}'.format(correct[:k].view(-1)))
        # print('correct[:k].size():{}'.format(correct[:k].view(-1).size()))
        # a.view(-1),
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
