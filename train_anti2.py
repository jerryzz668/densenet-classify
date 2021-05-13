import argparse
import os
import shutil
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#import densenet as dn
import anti_densenet as dn
import pdb
# used for logging to TensorBoard

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=250, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=52, type=int,
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
# train_dirs = 'C:\\Users\\fs\\Desktop\\410unet+densenet\\densedataset\\new\\train'
# val_dirs = 'C:\\Users\\fs\\Desktop\\410unet+densenet\\densedataset\\new\\val\\'
train_dirs = '/home/wkx/data/datas/huawei_class/train_new/train8/'
val_dirs = '/home/wkx/data/datas/huawei_class/train_new/val8/'

img_width = 200
img_height = 200

class_dir_name = ["guojian", "yaojian"]

def PreProcessImage(train_dir, val_dir):
    #cope the train dir
    new_size = (200, 200)
    for index in range(len(class_dir_name)):
        train_path_name = train_dirs + class_dir_name[index] + "/"
        val_path_name = val_dirs + class_dir_name[index] + "/"
        
        for file_name in os.listdir(train_path_name):
            name = train_path_name + file_name
            print (name)
            img = cv2.imread(name)
            height = img.shape[0]
            width = img.shape[1]
            
            if(height> 200 or width > 200):
                img = cv2.resize(img, new_size)
                cv2.imwrite(name, img)
            
            #image = cv2.cvtColor(img_new,cv2.COLOR_BGR2GRAY)
            #cv2.imwrite(name, image)
        # cope the val dir 
        for file_name in os.listdir(val_path_name):
            name = val_path_name + file_name
            print (name)
            img = cv2.imread(name)

            height = img.shape[0]
            width = img.shape[1]
            
            if(height> 200 or width > 200):
                img = cv2.resize(img, new_size)
                cv2.imwrite(name, img)
            #image = cv2.cvtColor(img_new,cv2.COLOR_BGR2GRAY)
            #cv2.imwrite(name, image)

def main():
    global args, best_prec1

    #PreProcessImage(train_dirs, val_dirs)

    args = parser.parse_args()
    if args.tensorboard: configure("runs/DenseNet_Unet_fs/")
    
    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    if args.augment:
        transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            #transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            #transforms.RandomAffine(3),
            #transforms.RandomHorizontalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3),
            #transforms.RandomCrop(200, padding = 4),
            transforms.ToTensor(),
            normalize
            ])
    
    transform_test = transforms.Compose([
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        #transforms.RandomAffine(3),
        transforms.ToTensor(),
        normalize
        ])
    #normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    train_dataset = datasets.ImageFolder(
        train_dirs,
        transform_train
        )
    #pdb.set_trace()
    val_dataset = datasets.ImageFolder(
        val_dirs,
        transform_test)
    
    kwargs = {'num_workers': 0, 'pin_memory': True}
    
    train_loader = torch.utils.data.DataLoader(
        #datasets.CIFAR10('../data', train=True, download=True,transform=transform_train),
        train_dataset,
        batch_size=64, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        #datasets.CIFAR10('../data', train=False, transform=transform_test),
        val_dataset,
        batch_size=24, shuffle=True, **kwargs)

    # create model
    model = dn.densenet121(num_class = 2, pretrained = False, filter_size = 3, pool_only = True)
    
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load("./runs/DenseNet_Unet_fs/checkpoint.pth")
        args.start_epoch = checkpoint['epoch']
        best_prec1 = 0.0
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    #criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    print ("Start training..")

    for epoch in range(args.start_epoch, args.epochs):

        print ()
        FS_lr=adjust_learning_rate(optimizer, epoch,args.epochs)
        print("lr='{}'".format(FS_lr))
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
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
        },{
            'epoch': epoch + 1,
            'state_dict': model,
            'best_prec1': best_prec1,
        }, 
        is_best)
        
        torch.save(model.state_dict(), "model_dict.pth")
        torch.save(model, "model.pth")
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
        target = target.cuda(async=True)
        input = input.cuda()

        #print(input.size())
        #print (target.size())
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        #pdb.set_trace()

        #print(output.size())
        #print (target_var.size())
        loss = criterion(output, target_var)

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
        target = target.cuda(async=True)
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


def save_checkpoint(state, model_state, is_best, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    directory = "./runs_anti8/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    model_file_name = filename
    model_file_name = model_file_name.replace(".pth", "_model.pth")
    torch.save(state, filename)
    
    torch.save(model_state, model_file_name)

    if is_best:
        print ("############################")
        print ("Get the current best model")
        print ("############################")
        shutil.copyfile(filename, 'runs_anti/%s/'%(args.name) + 'model_best.pth')
        shutil.copyfile(model_file_name, 'runs_anti8/%s/'%(args.name) + 'model_best_model.pth')

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
#    if epoch <= 5:
#        lr = args.lr
#    else:
#        lr = args.lr * (0.977 ** (epoch-5))
    if epoch <= epochs*0.1:
        lr = args.lr*(1.25**epoch)
        global slr
        slr=lr
    else:
        lr = slr * (0.977 ** (epoch-epochs*0.1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
