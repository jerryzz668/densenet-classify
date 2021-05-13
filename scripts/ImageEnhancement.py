import numpy as np
import cv2
from skimage import io, data,util

import random
import os

imgPath = 'C:\\Users\\fs\\Desktop\\410unet+densenet\\densedataset\\new\\train_source\\new_bubble\\'
outPath = 'C:\\Users\\fs\\Desktop\\410unet+densenet\\densedataset\\new\\train_sharpen\\new_sharpen_bubble\\'


def sharpen(img):
# 锐化函数
    kernel_sharpen_1 = np.array([
            [-1,-1,-1],
            [-1,9,-1],
            [-1,-1,-1]])
    # kernel_sharpen_2 = np.array([
    #         [1,1,1],
    #         [1,-7,1],
    #         [1,1,1]])
    # kernel_sharpen_3 = np.array([
    #         [-1,-1,-1,-1,-1],
    #         [-1,2,2,2,-1],
    #         [-1,2,8,2,-1],
    #         [-1,2,2,2,-1], 
    #         [-1,-1,-1,-1,-1]])/8.0
    #卷积
    output_1 = cv2.filter2D(img,-1,kernel_sharpen_1)
    # output_2 = cv2.filter2D(img,-1,kernel_sharpen_2)
    # output_3 = cv2.filter2D(img,-1,kernel_sharpen_3)
    return output_1


def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

def gamma(img, c, v):
    # 伽马变换
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
        if lut[i]>=254:
           lut[i]=254
    output_img = cv2.LUT(img, lut) #像素灰度值的映射

    output_img = np.uint8(output_img+0.5)  
    return output_img


def readImage(path):
    img = cv2.imread(os.path.join(imgPath, path))

    return img

def writeImage(img,imgName):
    cv2.imwrite(os.path.join(outPath, imgName), img)

def RandomSelectionFile(fileDir):
    # 随机选择某些文件名字
        # pathDir = os.listdir(fileDir)    #取图片的原始路径
        filenumber=len(fileDir)
        rate=1   #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
        sample = random.sample(fileDir, picknumber)  #随机选取picknumber数量的样本图片

        return sample

def main():
    imgFiles = sorted(os.listdir(imgPath))
    RandomImgFiles=RandomSelectionFile(imgFiles)
    for imgName in RandomImgFiles:
        img = readImage(imgName )
        print('processing ...:', imgName)
        # 图像高斯模糊
        Gaus_BlurImg = cv2.GaussianBlur(img, (9, 9), 0)

        # 图像锐化
        # sharpenimg=sharpen(img)
        # 直方图均衡化
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # equalizeHisImg = cv2.equalizeHist(gray)
        # equalizeHisImg2 = cv2.cvtColor(equalizeHisImg, cv2.COLOR_GRAY2BGR)
        # 高斯噪声
        # gauss_noiseImg= gasuss_noise(img,0,0.005)
        # sp_noiseImg=sp_noise(img,0.015)

        # gammaImg=gamma(img,0.004, 2.0)
        sharpen_img_name = imgName.split('.')[0]+'Gaus_Blur.bmp'

        writeImage(Gaus_BlurImg, sharpen_img_name )
    
main()