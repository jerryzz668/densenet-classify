import argparse
import os
import shutil
import time
import numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
from PIL import Image
import xlsxwriter

import models.densenet as dn
import pdb
# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value
parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--model', '-m', default='/home/jerry/Documents/code/densenet-pytorch-master/runs2_C_aug_heidian/DenseNet_Unet_fs/D_20200323_CP0.pth',
                    metavar='FILE',
                    help="Specify the file in which is stored the model"
                        " (default : 'model_best.pth')")
parser.add_argument('--cpu', '-c', action='store_true',
                    help="Do not use the cuda version of the net",
                    default=False)
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
parser.add_argument('--resume', default='/home/wkx/Downloads/densenet-pytorch-master/runs_new_datas/DenseNet_Unet_fs/D_20200323_CP0.pth', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_Unet_fs', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=False)
global args, best_prec1
args = parser.parse_args([])
# class_to_idx={0:'bai', 1:'hongliang', 2:'hongmie', 3:'huangliang', 4:'huangmie', 5:'luliang', 6:'lumie'}
class_to_idx={0:'NG', 1:'OK'}
val_dirs = '/home/jerry/Desktop/val_daowen/NG/'
output_path='/home/jerry/Desktop/test_result/'

# granule_path='C:\\Users\\fs\\Desktop\\410unet+densenet\\densedataset\\test\\result\\granule\\'
# other_path='C:\\Users\\fs\\Desktop\\410unet+densenet\\densedataset\\test\\result\\other\\'
files = os.listdir(val_dirs)
# for index,value in enumerate(files):
#     files[index] = val_dirs + files[index]
if os.path.exists(output_path):
    shutil.rmtree(output_path)  # delete output folder
os.makedirs(output_path)  # make new output folder




def predict_img(net,full_img):
    if full_img.mode != 'RGB':
        print("RGB")
        full_img = full_img.convert("RGB")
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.Resize([128,128]),
        transforms.ToTensor(),
        normalize,
            ])
    # full_img=numpy.asarray(full_img)
    #
    # # flag,full_img = adaptivehistogram_enhance(full_img)
    # # outimg=full_img
    # full_img = Image.fromarray(full_img)
    input = transform_train(full_img)

    input = input.unsqueeze(0)
    input = input.cuda()
    input = torch.autograd.Variable(input)
    # print("input",input.shape)
    output = net(input)
    print("output:",output)
    logit = F.softmax(output, dim=1)

    a=output.data
    _, pre = a.topk(1, dim=1, largest=True)

    # x=0
    print("label:",int(pre.cpu().numpy()))
    return pre.cpu().numpy(),output.data.cpu().numpy(),logit.data.cpu().numpy()



if __name__ == "__main__":
    # args = get_args()
    in_files = files

    net = torch.load(args.model)

    # print(list(net))
    # net=net['state_dict']
    net.cuda()
    net.eval()
    # net = dn.DenseNet3(100, 4, 24, bottleneck=args.bottleneck,  small_inputs = False)
    #
    # print("Loading model {}".format(args.model))
    #
    # if not args.cpu:
    #     print("Using CUDA version of the net, prepare your GPU !")
    #     net.cuda()
    #     net.load_state_dict(torch.load(args.model))
    # else:
    #     net.cpu()
    #     net.load_state_dict(torch.load(args.model, map_location='cpu'))
    #     print("Using CPU version of the net, this may be very slow")

    print("Model loaded :",args.model)
    # start2 = time.clock()
    workbook = xlsxwriter.Workbook('Expenses0_NG.xlsx')
    worksheet = workbook.add_worksheet()
    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
        fn=val_dirs + in_files[i]
        img = Image.open(fn)
        # img = img.convert("BGR")
        img2=img
        image = cv2.imread(fn)
        # image2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        flag,result,log = predict_img(net=net,full_img=img2)
        # print("result:",result[0][0])
        print("log:::",log[0][1])
        worksheet.write(i, 0, in_files[i])
        worksheet.write(i, 1, result[0][0])
        worksheet.write(i, 2, result[0][1])
        worksheet.write(i, 3, result[0][1]+result[0][0])
        worksheet.write(i, 4, log[0][0])
        worksheet.write(i, 5, log[0][1])
        output_img_path=os.path.join(output_path,class_to_idx[int(flag)])
        if not os.path.exists(output_img_path):
            os.makedirs(output_img_path)
        out_files=os.path.join(output_img_path,in_files[i])
        shutil.copyfile(fn, out_files)
        # if log[0][0]>=0.5:
        #     out_files=bubble_path + in_files[i]
        #     shutil.copyfile(fn, out_files)
        #     # image2.save(out_files)
        #     # cv2.imwrite(out_files,image)
        #
        # else:
        #     out_files=flaw_path + in_files[i]
        #     shutil.copyfile(fn, out_files)
        #     # image2.save(out_files)
        #     # cv2.imwrite(out_files, image)


        print("Mask saved to {}".format(out_files))
    # end2 = time.clock()
    workbook.close()
    # print('Total running time:%s Seconds'%(end2-start2))
    # global totaltime1
    # print('Totalmodel running time:%s Seconds'%(totaltime1))
