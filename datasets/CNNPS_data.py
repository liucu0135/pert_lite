from __future__ import division
import os
import numpy as np
#from scipy.ndimage import imread
from imageio import imread
import scipy.io as sio

import time

import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
# np.random.seed()

class CNNPS_data(data.Dataset):
    def __init__(self, args, split='train'):
        self.root   = 'data/datasets/cnnps'
        self.split  = split
        self.args   = args
        self.objs   = os.listdir(self.root)
        self.names=[n for n in os.listdir(os.path.join(self.root, self.objs[0])) if '0' in n]

    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    def __getitem__(self, index):
        obj = self.objs[index]
        if self.args.in_img_num==100:
            select_idx = np.arange(0,100)
        else:
            np.random.seed()
            select_idx = np.random.permutation(len(self.names))[:self.args.in_img_num]

        img_list   = [os.path.join(self.root, obj, self.names[i]) for i in select_idx]

        normal_path = os.path.join(self.root, obj, 'Normal_gt.png')
        normal = imread(normal_path).astype(np.float32) / 255.0*2-1

        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            imgs.append(img)
        img = np.concatenate(imgs, 2)

        # print(np.histogram(img))


        mask = self._getMask(obj)
        down = 4
        if mask.shape[0] % down != 0 or mask.shape[1] % down != 0:
            pad_h = down - mask.shape[0] % down
            pad_w = down - mask.shape[1] % down
            img = np.pad(img, ((0,pad_h), (0,pad_w), (0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            mask = np.pad(mask, ((0,pad_h), (0,pad_w), (0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            normal = np.pad(normal, ((0,pad_h), (0,pad_w), (0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        img  = img * mask.repeat(img.shape[2], 2)
        item = {'N': normal, 'img': img, 'mask': mask}

        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light:
            lights=np.genfromtxt(os.path.join(self.root, obj, 'lights.txt'))
            item['light'] = torch.from_numpy(lights[select_idx,:]).view(-1, 1, 1).float()
        item['obj'] = obj
        item['id']=index
        return item

    def __len__(self):
        return len(self.objs)
