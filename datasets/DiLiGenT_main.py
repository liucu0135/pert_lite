from __future__ import division
import os
import numpy as np
#from scipy.ndimage import imread
import matplotlib.pyplot as plt
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
        self.root   = os.path.join(args.bm_dir)
        self.split  = split
        self.args   = args
        self.objs   = util.readList(os.path.join(self.root, 'objects.txt'), sort=False)
        self.names  = util.readList(os.path.join(self.root, 'filenames.txt'), sort=False)
        self.l_dir  = util.light_source_directions()


        print('[%s Data] \t%d objs %d lights. Root: %s' % 
                (split, len(self.objs), len(self.names), self.root))
        self.intens = {}
        intens_name = 'light_intensities.txt'
        print('Files for intensity: %s' % (intens_name))
        for obj in self.objs:
            self.intens[obj] = np.genfromtxt(os.path.join(self.root, obj, intens_name))

        if self.args.in_img_num < 0:
            # light_num_set=[12,24,48,96]
            self.selected_idx=self.get_light_sets(-self.args.in_img_num)
            self.args.in_img_num =  len(self.selected_idx)
            # for testing
            lights=self.l_dir[self.selected_idx]
            plt.clf()
            plt.scatter(lights[:,0],lights[:,1])
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.savefig('result/figs/sparse_light_distribution/{}.png'.format(self.args.in_img_num))

    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    def get_light_sets(self, set):
        if set==1:
            return range(96)
        if set==2:
            return range(0,96,2)

        l=np.array((range(96)))
        l=np.reshape(l,(12,8))
        ll=np.concatenate([l[6:,:],l[:6,:]])
        lm=ll.transpose()
        ll=lm[::2,::2].reshape(-1)
        if set==3:
            return ll
        return ll[::2]

    def __getitem__(self, index):
        # np.random.seed(index)
        obj = self.objs[index]
        if self.args.in_img_num<0:
            select_idx=self.selected_idx
        else:
            if self.args.in_img_num==96:
                select_idx = np.arange(0,96)
            else:
                np.random.seed()
                select_idx = np.random.permutation(len(self.names))[:self.args.in_img_num]

        select_idx = np.random.permutation(len(self.names))
        img_list   = [os.path.join(self.root, obj, self.names[i]) for i in select_idx]
        intens     = [np.diag(1 / self.intens[obj][i]) for i in select_idx]
        print(np.histogram(1 / np.sum(self.intens[obj][select_idx], axis=1)))
        normal_path = os.path.join(self.root, obj, 'Normal_gt.mat')
        normal = sio.loadmat(normal_path)
        normal = normal['Normal_gt']

        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0*10.0
            # shadow = imread(img_name[:-4]+'s'+img_name[-4:]).astype(np.float32) / 255.0#for shadowing only
            # print(np.max(img))
            # img=np.clip(img*5,0,1)
            img = np.dot(img, intens[idx])
            # img=img/np.max(img)
            # img = np.concatenate([img,shadow], 2)[:,:,:-2]#for shadowing only
            imgs.append(img)


        # normalize images
        # imgs = np.split(img, img.shape[2] // 3, 2)
        # imgs = pms_transforms.normalize(imgs)


        img = np.concatenate(imgs, 2)
        # img=img**2
        mask = self._getMask(obj)
        # [798.0, 981.0, 10.0, 0.0, 0.0, 0.0, 0.0]


        # img = img * 5
        # img = np.clip(img, 0, 1)
        print(list(np.histogram(img[img > 0], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])[0]) / (
            np.sum(img > 0)))
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
            item['light'] = torch.from_numpy(self.l_dir[select_idx]).view(-1, 1, 1).float()
        item['obj'] = obj
        item['id']=index
        return item

    def __len__(self):
        return len(self.objs)
