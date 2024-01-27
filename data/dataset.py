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
            glob.glob(os.path.join(self.imgs_LR_path, '*.png'))
        )

        self.imgs_HR = sorted(
                glob.glob(os.path.join(self.imgs_HR_path, '*.png')))

        self.transform = transforms.ToTensor()
        self.train = train

        print(self.imgs_LR_path, self.imgs_HR_path)


    def __getitem__(self, item): #将对象做成数组  item为对象数组下标

        img_path_LR = self.imgs_LR[item]
        img_path_HR = self.imgs_HR[item]
        LR = Image.open(img_path_LR)
        LR = LR.resize((self.args.scale*16, self.args.scale*16), Image.BICUBIC)
        HR = Image.open(img_path_HR)
        HR = np.array(HR)
        LR = np.array(LR)

        LR = np.ascontiguousarray(LR)
        HR = np.ascontiguousarray(HR)
        HR = ToTensor()(HR)
        LR = ToTensor()(LR)

        filename = os.path.basename(img_path_HR)
        return {'lr_up': LR, 'img_gt': HR, 'img_name': filename}

    def __len__(self):
        return len(self.imgs_HR)

