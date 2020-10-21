from __future__ import division
import os
import numpy as np
from imageio import imread

import torch
import torch.utils.data as data
from datasets import pms_transforms
from . import util
import scipy.io as io
import h5py
import matplotlib.pyplot as plt
import numpy as np
# np.random.seed(0)





class MERL_Dataset(data.Dataset):
    def __init__(self, args, root, split='train'):
        self.root   = os.path.join(root)
        self.split  = split
        self.args   = args
        self.normal_path=os.path.join(self.args.bm_dir,'blob01_n.png.png')
        dir=os.listdir(self.args.bm_dir)
        self.shape_list = [os.path.join(self.args.bm_dir,d) for d in dir if '.' not in d]
        self.lights=np.genfromtxt(os.path.join(self.args.bm_dir, 'light_directions.txt'), dtype='str', delimiter=' ').astype(np.float)
        # self.lights=util.readList(os.path.join(self.args.bm_dir, 'light_directions.txt'))
    def _getInputPath(self, index):
        select_idx = np.random.permutation(96)[:self.args.in_img_num]
        # lights=np.genfromtxt(self.lights[select_idx], dtype='str', delimiter=' ')
        lights=self.lights[select_idx,:]
        # lights = [np.array(l.split(' '))for l in self.lights[select_idx]]
        imgs=[os.path.join(self.shape_list[index],'{}.png'.format(i)) for i in select_idx]
        return self.normal_path, imgs, lights

    def __getitem__(self, index):
        obj = self.shape_list[index]
        if self.args.in_img_num == 96:
            select_idx = np.arange(0, 96)
        else:
            np.random.seed()
            select_idx = np.random.permutation(96)[:self.args.in_img_num]

        img_list = [os.path.join(self.shape_list[index],'{}.png'.format(i)) for i in select_idx]

        normal = imread(self.normal_path).astype(np.float32)*2-1
        normal =normal[180:340,230:390,:]

        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            imgs.append(img[180:340,230:390,:])
        img = np.concatenate(imgs, 2)

        mask = np.sum(normal,axis=2)>0
        mask = np.expand_dims(mask, 2)
        img = img * mask.repeat(img.shape[2], 2)
        item = {'N': normal, 'img': img, 'mask': mask}

        for k in item.keys():
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light:
            item['light'] = torch.from_numpy(self.lights[select_idx,:]).view(-1, 1, 1).float()
        item['obj'] = obj
        return item

    def __len__(self):
        return len(self.shape_list)
    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0