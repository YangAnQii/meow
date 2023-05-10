import torch
from swd import swd
import os
import numpy as np
import torchvision.transforms as transforms
import cv2 as cv
import random

img1list = [] ##存放真实数据
img2list = [] ##存放生成数据
def extract64img(dirPath):
    """
    :param dirPath: 文件夹路径
    :return:
    """
    # 对目录下的文件进行遍历
    tensor_list=list()
    pathDir = os.listdir(dirPath)
    imgnum = len(pathDir)

    if imgnum>64:##如果是训练集，则只取其中64张图片
        sample = random.sample(pathDir, 64)
        for j in sample:
            # 判断是否是文件
            if os.path.isfile(os.path.join(dirPath, j)) == True:
                a = os.path.basename(j)
                name = dirPath + '\\' + a
                img = cv.imread(name)  ##此时img.shape为(64,64,3)
                transf = transforms.ToTensor()
                img_tensor = transf(img)##img_tensor.size()=torch.Size([3, 64, 64])
                tensor_list.append(img_tensor)

    elif imgnum==64:##fake图片共有64张
        for j in pathDir:
            # 判断是否是文件
            if os.path.isfile(os.path.join(dirPath, j)) == True:
                a = os.path.basename(j)
                name = dirPath + '\\' + a
                img = cv.imread(name)
                transf = transforms.ToTensor()
                img_tensor = transf(img)
                # img=cv2.resize(img,(100,100))#使尺寸大小一样
                tensor_list.append(img_tensor)

    final_list = torch.stack(tensor_list)
    #print(final_list.size()),torch.Size([64, 3, 64, 64])
    return final_list

path1='E:\pythonProject\LIDC-IDRI-train\\5'
img1list=extract64img(path1) ##img1list.shape=(64,64,64,3),272个训练数据中随机提取64个
print('img1list.size()=',img1list.size())
# img1 = cv.imread('E:\pythonProject\LIDC-IDRI-train\\5')
# print('img1.shape=',img1)   # numpy数组格式为（H,W,C）

# transf = transforms.ToTensor()
# img1_tensor = transf(img1)  # tensor数据格式是torch(C,H,W)
# # x1=torch.unsqueeze(img_tensor, dim=0)  # 在第一维度添加维度
# print('img1_tensor.size()=',img1_tensor.size())


path2='E:\pythonProject\\test\lidc-idri-1\\fake'
img2list=extract64img(path2) ##img2list.shape=(64,64,64,3),64个生成的fake图片
print('img2list.size()=',img2list.size())
# img2 = cv.imread('E:\pythonProject\\test\lidc-idri-1\\fake')
# print('img2.shape=',img2.size)   # numpy数组格式为（H,W,C）
#
# transf = transforms.ToTensor()
# img2_tensor = transf(img2)  # tensor数据格式是torch(C,H,W)
# # x2=torch.unsqueeze(img2_tensor, dim=0)  # 在第一维度添加维度
# print('img2_tensor.size()=',img2_tensor.size())

path3='E:\pythonProject\\test\lidc-idri-2\\fake'
img3list=extract64img(path3)

#torch.manual_seed(123) # fix seed

out1 = swd(img1list, img2list, n_pyramids=1,device="cpu") # Fast estimation if device="cuda"
out2 = swd(img1list, img3list, n_pyramids=1,device="cpu") # Fast estimation if device="cuda"
print(out1,out2)

'''
FID:python -m pytorch_fid+两个文件夹
'''