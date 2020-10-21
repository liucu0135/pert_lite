from __future__ import division
import os
import numpy as np
#from scipy.ndimage import imread
from imageio import imread

import PIL
from PIL import Image
import scipy.io as sio
import time

import torch
import torch.utils.data as data
import matplotlib.pyplot as plt


from datasets import pms_transforms
from . import util


class Apple_gourd_dataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.root   = 'data/datasets/gourd_apple'
        self.args   = args
        self.objs   = util.readList(os.path.join(self.root, 'objects.txt'), sort=False)

        pass


    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    def read_light_dir(self, list_path, ignore_head=False, sort=True):
        with open(list_path) as f:
            lists = f.read().splitlines()
        fromstring = lambda n: [np.fromstring(nn, dtype=float, sep=' ') for nn in n]
        light_dir = fromstring(lists)
        return np.stack(light_dir)

    def read_light_intens(self, list_path, ignore_head=False, sort=True):
        with open(list_path) as f:
            lists = f.read().splitlines()
        fromstring = lambda n: [np.fromstring(nn, dtype=float, sep=' ') for nn in n]
        light_intens = fromstring(lists)
        return np.stack(light_intens)

    def __getitem__(self, index):
        # np.random.seed(index)
        obj = self.objs[index]
        self.l_dir   = self.read_light_dir(os.path.join(self.root, obj,'light_directions.txt'))
        self.l_intens   = self.read_light_intens(os.path.join(self.root, obj,'light_intensities.txt'))
        # select_idx=np.arrange(0,self.args.in_img_num)
        self.names = ['I_{:04d}.png'.format(i) for i in range(self.args.in_img_num)]

        select_idx = np.arange(self.args.in_img_num)

        img_list   = [os.path.join(self.root, obj, self.names[i]) for i in select_idx]



        imgs = []
        intents=self.l_intens[select_idx,0]
        for idx, img_name in enumerate(img_list):
            # img = Image.open(img_name)
            # img=img.resize((256,256))
            # img=np.array(img)/255/intents[idx]
                  # / 255.0/intents[idx]
            img = imread(img_name).astype(np.float32) / 255.0/intents[idx]
            plt.imshow(img[:,:,:3])
            plt.show()
            imgs.append(img[:,:,:3])

        img = np.concatenate(imgs, 2)
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

