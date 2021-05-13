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
import densenet as dn
import pdb
# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value
parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--model', '-m', default='./runs_attention_aotuhen/DenseNet_Unet_fs/D_20200323_CP0.pth',
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
val_dirs = '/home/wkx/datas/aotuhen2/test_ok2/'
bubble_path='/home/wkx/datas/huanxingdaowen_fenlei2/test_ng2/'
flaw_path='/home/wkx/datas/huanxingdaowen_fenlei2/test_ok2/'
# granule_path='C:\\Users\\fs\\Desktop\\410unet+densenet\\densedataset\\test\\result\\granule\\'
# other_path='C:\\Users\\fs\\Desktop\\410unet+densenet\\densedataset\\test\\result\\other\\'
files = os.listdir(val_dirs)
# for index,value in enumerate(files):
#     files[index] = val_dirs + files[index]


def adaptivehistogram_enhance(img, clipLimit=5, tileGridSize=(9, 9)):
    flag = 0
    if img is None:
        print('Í¼ÏñÎª¿Õ')
        return flag, img
    if img.sum() == 0:
        print('warning:È«ÎªÁã')

    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    if len(img.shape) == 2:
        flag = 1
        cl1 = clahe.apply(img)
    if len(img.shape) == 3:
        flag = 3
        b, g, r = cv2.split(img)
        img1 = clahe.apply(b)
        img2 = clahe.apply(g)
        img3 = clahe.apply(r)
        cl1 = cv2.merge([img1, img2, img3])

    return flag, cl1

def inference(full_img,transform_train):

    full_img = Image.fromarray(full_img)
    input = transform_train(full_img)

    input = input.unsqueeze(0)
    input = input.cuda()
    input = torch.autograd.Variable(input)

    output = net(input)
    # print("output:",output)
    logit = F.softmax(output, dim=1)
    a=output.data
    _, pre = a.topk(1, dim=1, largest=True)
    return pre,output,logit


def predict_img(net,full_img):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.Resize([1408,1200]),
        transforms.ToTensor(),
        normalize,
            ])
    full_img=numpy.asarray(full_img)
    
    flag,full_img = adaptivehistogram_enhance(full_img)
    # print('full_img',full_img.shape)
    if (full_img.shape[1]+full_img.shape[0])/2>3500:
        height, width = full_img.shape[0], full_img.shape[1]
        #     print(image_path)
        #     print(image.shape)
        crop_width = full_img.shape[1] // 2 + 200
        crop_height = full_img.shape[0] // 2 + 200
        height_len = height // crop_height + 1
        width_len = width // crop_width + 1
        start_x, start_y = 0, 0
        count = 0
        # print(height_len)
        # print(width_len)

        for y_index in range(height_len):
            end_y = min(start_y + crop_height, height)
            if end_y - start_y < crop_height:
                start_y = end_y - crop_height
            for x_index in range(width_len):
                end_x = min(start_x + crop_width, width)
                if end_x - start_x < crop_width:
                    start_x = end_x - crop_width
                crop_image = full_img[start_y:end_y, start_x:end_x]
                print('crop_image', crop_image.shape)
                pre1,output1,logit1=inference(crop_image, transform_train)
                print("pre1.cpu().numpy():",pre1.cpu().numpy())
                if pre1.cpu().numpy()==0:
                    pre=0
                    output, logit=output1,logit1
                    break
                else:
                    pre=1
                    output, logit = output1, logit1
                count = count + 1
                # out_image_path = out_image_path[:-4] + "_" + str(count) + ".jpg"
                # print(out_image_path)
                # cv2.imwrite(out_image_path, crop_image)
                start_x = start_x + crop_width - 100
            if pre==0:
                break
            start_x = 0
            start_y = start_y + crop_height
    else:
        pre1, output1, logit1 = inference(full_img, transform_train)
        print("pre1.cpu().numpy():", pre1.cpu().numpy())
        if pre1.cpu().numpy() == 0:
            pre = 0
            output, logit = output1, logit1
        else:
            pre = 1
            output, logit = output1, logit1

    # # outimg=full_img
    # full_img = Image.fromarray(full_img)
    # input = transform_train(full_img)
    #
    # input = input.unsqueeze(0)
    # input = input.cuda()
    # input = torch.autograd.Variable(input)
    #
    # output = net(input)
    # print("output:",output)
    # logit = F.softmax(output, dim=1)
    # a=output.data
    # _, pre = a.topk(1, dim=1, largest=True)
    #
    # # x=0
    # print("label:",pre.cpu().numpy())
    return pre,output.data.cpu().numpy(),logit.data.cpu().numpy()



if __name__ == "__main__":
    # args = get_args()
    in_files = files

    net = torch.load(args.model)
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
    start2 = time.clock()
    workbook = xlsxwriter.Workbook('Expenses0_ok.xlsx')
    worksheet = workbook.add_worksheet()
    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
        fn=val_dirs + in_files[i]
        img = Image.open(fn)
        if img.mode!='RGB':
            img=img.convert("RGB")


        flag,result,log = predict_img(net=net,full_img=img)
        print("result:",result[0][0])
        worksheet.write(i, 0, in_files[i])
        worksheet.write(i, 1, result[0][0])
        worksheet.write(i, 2, result[0][1])
        worksheet.write(i, 3, result[0][1]+result[0][0])
        worksheet.write(i, 4, log[0][0])
        worksheet.write(i, 5, log[0][1])
        if flag==0:
            out_files=bubble_path + in_files[i]
            img.save(out_files)
        elif flag==1:
            out_files=flaw_path + in_files[i]
            img.save(out_files)
        # elif flag==2:
        #     out_files=granule_path + in_files[i]
        #     img.save(out_files)
        # elif flag==3:
        #     out_files=other_path + in_files[i]
        #     img.save(out_files)

        print("Mask saved to {}".format(out_files))
    end2 = time.clock()
    workbook.close()
    print('Total running time:%s Seconds'%(end2-start2))
    # global totaltime1
    # print('Totalmodel running time:%s Seconds'%(totaltime1))
