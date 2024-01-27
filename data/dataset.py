from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np
import torch
import glob
import random
import torch.nn.functional as F

class Data(data.Dataset):
    def __init__(self, root, args, train=False):
        # 返回指定路径下的文件和文件夹列表。
        self.args = args
        self.train = train
        self.imgs_HR_path = os.path.join(root, 'HR')


        if self.args.scale == 8:
            self.imgs_LR_path = os.path.join(root, 'LR_x8_bicubic')
        elif self.args.scale == 4:
            self.imgs_LR_path = os.path.join(root, 'LR_x4_bicubic')
        elif self.args.scale == 16:
            self.imgs_LR_path = os.path.join(root, 'LR_x16_bicubic')


        self.imgs_LR = sorted(
            glob.glob(os.path.join(self.imgs_LR_path, '*.png'))#获取指定目录下的所有图片
        )
        # print('排序后目录下的图片：',self.imgs_LR)
        self.imgs_HR = sorted(
                glob.glob(os.path.join(self.imgs_HR_path, '*.png')))

        self.transform = transforms.ToTensor()
        self.train = train

        print(self.imgs_LR_path, self.imgs_HR_path)


    def __getitem__(self, item): #将对象做成数组  item为对象数组下标

        img_path_LR = self.imgs_LR[item]               #os.path.join(self.imgs_LR_path, self.imgs_LR[item])
        img_path_HR = self.imgs_HR[item]               #os.path.join(self.imgs_HR_path, self.imgs_HR[item])
        #print(img_path_LR,img_path_HR)
        LR = Image.open(img_path_LR)
        #print(LR.size)
        LR = LR.resize((self.args.scale*16, self.args.scale*16), Image.BICUBIC)
        HR = Image.open(img_path_HR)
        #print(LR.size,HR.size)
        HR = np.array(HR)
        LR = np.array(LR)

        LR = np.ascontiguousarray(LR)
        HR = np.ascontiguousarray(HR)
        HR = ToTensor()(HR)
        LR = ToTensor()(LR)

        #print('HR LR shape',HR.shape,LR.shape) 128*128   128*128
        filename = os.path.basename(img_path_HR)
        # print("lr up:",LR.size(),'img-gt:',HR.size(),'img-name:',filename)
        return {'lr_up': LR, 'img_gt': HR, 'img_name': filename}   #这里应该要变大的？？

    def __len__(self):
        return len(self.imgs_HR)

