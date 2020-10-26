from __future__ import division
import os
import numpy as np
#from scipy.ndimage import imread
from imageio import imread
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import scipy.io as sio
import time

import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
# import util
# np.random.seed()

class Light_stage_dataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.root   = 'data/datasets/light_stage'
        self.args   = args
        self.objs   = util.readList(os.path.join(self.root, 'objects.txt'), sort=False)
        self.l_dir   = -util.read_light_dir(os.path.join(self.root, 'light_directions.txt'))
        self.l_intens   = util.read_light_intens(os.path.join(self.root, 'light_intensities.txt'))
        pass


    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    def __getitem__(self, index):
        # np.random.seed(index)
        obj = self.objs[index]
        # select_idx=np.arrange(0,self.args.in_img_num)
        if index==0 or index==3:
            self.names=[ obj+'_{:04d}.png'.format(i) for i in range(252)]
        else:
            self.names = [obj + '_{:03d}.png'.format(i) for i in range(252)]

        select_idx= np.arange(252)
        select_idx =[s for s in select_idx if self.l_dir[s,2]>0.2] #192
        # select_idx =select_idx[::2]
        # args.in_img_num=len(select_idx)
        # if self.args.in_img_num==250:
        #     select_idx = np.arange(250)
        # else:
        #     select_idx = np.arange(self.args.in_img_num)*(250//self.args.in_img_num)
        img_list   = [os.path.join(self.root, obj, self.names[i]) for i in select_idx]



        imgs = []
        # intents=self.l_intens[select_idx,:]
        intents = [np.diag(1 / self.l_intens[i]) for i in select_idx]



        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32)[:,:,:3] / 255.0#/10
            # img = img**5
            img = np.dot(img, intents[idx])
            # img=img/np.max(img)
            # print('loading, {}'.format(img_name))
            # temp=np.zeros_like(img)
            # temp[img>0.1]=img[img>0.1]
            # img=temp

            imgs.append(img)

        # img = np.concatenate(imgs, 2)
        # # img=img/np.max(img)
        # img=np.clip(img,0,1)
        # imgs = np.split(img, img.shape[2] // 3, 2)
        # imgs = pms_transforms.normalize(imgs)
        img = np.concatenate(imgs, 2)
        img=img**2

        # [798.0, 981.0, 10.0, 0.0, 0.0, 0.0, 0.0]
        print(list(np.histogram(img[img>0], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1])[0] // (np.sum(img>0)/1000)))
        # print(np.histogram(img, bins=[0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        item = {'img': img}

        for k in item.keys():
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light:
            item['light'] = torch.from_numpy(self.l_dir[select_idx]).view(-1, 1, 1).float()
        item['obj'] = obj
        item['id']=index
        item['N']=img[:,:,:3]
        item['mask']=np.ones_like(img[:,:,0])
        return item

    def __len__(self):
        return len(self.objs)

