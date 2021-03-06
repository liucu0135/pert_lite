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

class DiLiGenT_main(data.Dataset):
    def __init__(self, args, split='train'):
        self.root   = 'data/datasets/DiLiGenT-MV/mvpmsData'
        self.split  = split
        self.args   = args
        self.objs   = util.readList(os.path.join(self.root, 'objects.txt'), sort=False)
        self.names  = util.readList(os.path.join(self.root, 'filenames.txt'), sort=False)

        print('[%s Data] \t%d objs %d lights. Root: %s' % 
                (split, len(self.objs), len(self.names), self.root))
        self.intens = {}
        self.l_dir = {}
        intens_name = 'light_intensities.txt'
        dirs_name = 'light_directions.txt'
        for obj in self.objs:
            for v in range(20):
                view = 'view_%02d' % (v + 1)
                self.intens[obj+view] = np.genfromtxt(os.path.join(self.root, obj, view,intens_name))
                self.l_dir[obj+view]  = self.read_lights(os.path.join(self.root, obj, view,dirs_name))

    def _getMask(self, obj, view):
        mask = imread(os.path.join(self.root, obj,view,  'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    def read_lights(self, path):
        # f=open(path)
        # l=f.readlines()
        l=np.loadtxt(path, dtype=np.float32)
        # needs more work
        return l

    def __getitem__(self, index):
        # np.random.seed(index)
        view_num=index%20
        view='view_%02d'% (view_num+1)
        index=index//20
        obj = self.objs[index]
        intens=self.intens[obj+view]
        l_dir=self.l_dir[obj+view]
        if self.args.in_img_num==96:
            select_idx = np.arange(0,96)
        else:
            np.random.seed()
            select_idx = np.random.permutation(len(self.names))[:self.args.in_img_num]
            # print(select_idx)

        img_list   = [os.path.join(self.root, obj,view, self.names[i]) for i in select_idx]
        intens     = [np.diag(1 / intens[i]) for i in select_idx]

        normal_path = os.path.join(self.root, obj,view, 'Normal_gt.mat')
        normal = sio.loadmat(normal_path)
        normal = normal['Normal_gt']

        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            # shadow = imread(img_name[:-4]+'s'+img_name[-4:]).astype(np.float32) / 255.0#for shadowing only
            img = np.dot(img, intens[idx])
            # img = np.concatenate([img,shadow], 2)[:,:,:-2]#for shadowing only
            imgs.append(img)
        img = np.concatenate(imgs, 2)

        mask = self._getMask(obj, view)
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
            item['light'] = torch.from_numpy(l_dir[select_idx]).view(-1, 1, 1).float()
        item['obj'] = obj
        item['id']=index
        return item

    def __len__(self):
        return len(self.objs)*20
