import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import kaiming_normal_
from . import model_utils
import math
import matplotlib.pyplot as plt

class Attention_layer(nn.Module):
    def __init__(self, ch_in, batch=False, factor=1, bias=True, extra=False):
        super(Attention_layer, self).__init__()
        # self.atten_factor=nn.Parameter(torch.zeros(1).cuda())
        self.atten_factor=1
        self.extra=extra
        self.shurink=ch_in//2
        self.atten_k = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0, bias=bias)
        self.atten_q = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0, bias=bias)
        self.atten_v = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0, bias=bias)
        self.atten = nn.Conv3d(self.shurink, ch_in, kernel_size=1, stride=1, padding=0, bias=bias)

    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print('click:  {},{}'.format(ix,iy))
        return ix,iy

    def forward(self, x, visual=False):
        shape=x.shape
        if self.extra:
            extra=nn.functional.avg_pool3d(x,kernel_size=[shape[2],1,1], stride=[1,1,1], padding=[0,0,0])
            x=torch.cat([extra,x], dim=2)
            shape = x.shape
        atten_k = self.atten_k(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        atten_q = self.atten_q(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        atten_v = self.atten_v(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        a=nn.functional.softmax(torch.bmm(atten_k.transpose(1,2),atten_q)/math.sqrt(self.shurink), dim=1)
        if visual:
            im2show=a.detach().cpu().view(shape[0],shape[3],shape[4],shape[2],shape[2]).numpy()
            # xx=30
            # yy=30
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.imshow(im2show[0,xx,yy,:,:])
            # cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
            # plt.show()


        a=torch.bmm(atten_v,a)

        a=self.atten(a.view(shape[0],shape[3],shape[4],self.shurink,shape[2]).permute(0,3,4,1,2))
        x=x+self.atten_factor*a
        if self.extra:
            return x[:,:,1:,:,:]
        else:
            if visual:
                return x,im2show
            else:
                return x

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.ln=32
        # more atten
        # self.conv1 = model_utils.conv3d(batchNorm, c_in, 64,  k=[1,1,1], stride=1, pad=[0,0,0])
        # self.at1=Attention_layer(64)
        # self.bn1=nn.BatchNorm3d(64, affine=False)
        # self.conv2 = model_utils.conv3d(batchNorm, 64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        # self.conv3 = model_utils.conv3d(batchNorm, 128,  128, k=[1,1,1], stride=1, pad=[0,0,0])
        # self.at1=Attention_layer(128)
        # self.bn1=nn.BatchNorm3d(128, affine=False)
        # self.conv5 = model_utils.conv3d(batchNorm, 128,  128, k=[1,1,1], stride=1, pad=[0,0,0])
        # self.at1=Attention_layer(128)
        # self.bn1=nn.BatchNorm3d(128, affine=False)
        # self.conv7 = model_utils.conv3d(batchNorm, 128, 128, k=[1,3,3], stride=1, pad=[0,1,1])

        # less atten
        self.conv1 = model_utils.conv3d(batchNorm, 6, 64,  k=[1,1,1], stride=1, pad=[0,0,0])
        self.at1=Attention_layer(64)
        self.bn1=nn.BatchNorm3d(64, affine=False)
        self.conv2 = model_utils.conv3d(batchNorm, 64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        self.conv3 = model_utils.conv3d(batchNorm, 128,  128, k=[1,1,1], stride=1, pad=[0,0,0])
        self.conv31 = model_utils.conv3d(batchNorm, 128,  128, k=[1,3,3], stride=1, pad=[0,1,1])
        self.at2=Attention_layer(128)
        self.bn2=nn.BatchNorm3d(128, affine=False)
        self.conv5 = model_utils.conv3d(batchNorm, 128,  128, k=[1,1,1], stride=1, pad=[0,0,0])
        self.conv51 = model_utils.conv3d(batchNorm, 128,  128, k=[1,3,3], stride=1, pad=[0,1,1])
        self.at3=Attention_layer(128)
        self.bn3=nn.BatchNorm3d(128, affine=False)
        self.conv7 = model_utils.conv3d(batchNorm, 128, 128, k=[1,3,3], stride=1, pad=[0,1,1])


        # the one actually work
        # self.conv1 = model_utils.conv3d(batchNorm, c_in, 64,  k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv2 = model_utils.conv3d(batchNorm, 64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1], atten=True)
        # self.conv3 = model_utils.conv3d(batchNorm, 128,  128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv5 = model_utils.conv3d(batchNorm, 128,  128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv7 = model_utils.conv3d(batchNorm, 128, 128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)

    def forward(self, x, visual=False, crop=True):
        out = self.conv1(x)
        out = self.at1(out,visual)
        if visual:
            out,im2show=out
        out = self.bn1(out)
        out = self.conv2(out)
        out0 = self.conv3(out)
        out = out0+self.conv31(out)
        out = self.at2(out)

        out = self.bn2(out)
        out0 = self.conv5(out)
        out = out0+self.conv51(out)
        out = self.at3(out)


        out = self.bn3(out)
        out_feat = self.conv7(out)
        if visual:
            return out_feat,im2show
        else:
            return out_feat





class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other   = other
        self.deconv1 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal= self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other   = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x):
        shape=x.shape
        x=nn.functional.max_pool3d(x,kernel_size=[shape[2],1,1], stride=[1,1,1], padding=[0,0,0]).squeeze(2)
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class SPS_Net(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(SPS_Net, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

    def forward(self, x,visual=False,id=0, normal=None):

        if visual:
            img=x[0]
            net_in=torch.cat((x[0],x[1]),dim=1)
            # light_id = range(96)
            light_id = np.concatenate(([43],np.arange(8)+77))
            # light_id = np.concatenate((np.arange(8)+40,np.arange(8)+77))
            # feat, im2show = self.extractor(net_in, visual)
            net_in=net_in[:,:,light_id,:,:]
            feat, im2show = self.extractor(net_in, visual)

            # im2show=im2show[:,:,:,light_id,:]
            # im2show=im2show[:,:,:,:,light_id]
            frow=3
            fcol=5
            # kpx=[30,100,30,100]
            # kpy=[30,30,100,100]
            # kpx = [30, 40, 50, 100]
            # kpy = [70, 70, 70, 90]

            kpx=[100,100]
            kpy=[30,100]
            # kpx = [ 40,  100]
            # kpy = [ 70,  90]

            # kpx=[40,  100,100,100]
            # kpy=[70,  90,30,100]



            plt.figure( figsize=(10, 6), dpi=700, facecolor='w', edgecolor='k')
            # lights=np.zeros(9,16,3)
            plt.subplot(frow, fcol, 1)
            plt.axis('off')
            plt.imshow(normal[0, :, :, :].detach().cpu().permute(1, 2, 0) /2+0.5)
            for tt in range(2):
                xx = kpx[tt]
                yy = kpy[tt]
                plt.scatter(x=yy, y=xx, c='r', s=4)
            for t in range(9):
                plt.subplot(frow, fcol, t+2)
                plt.axis('off')
                plt.imshow(img[0, :,light_id[t], :, :].detach().cpu().permute(1, 2, 0)*4)
                for tt in range(2):
                    xx = kpx[tt]
                    yy = kpy[tt]
                    plt.scatter(x=yy, y=xx, c='r', s=2)
            for t in range(2):
                xx=kpx[t]//1
                yy=kpy[t]//1
                plt.subplot(frow, fcol,t+fcol+fcol+1)
                plt.axis('off')
                plt.imshow(im2show[0, xx, yy, :, :],vmin=0,vmax=0.5)
                # if t==3:
                #     plt.colorbar()
                #     plt.show()
                # for tt in range(16):
                #     lights[tt,t,:]=img[0, :,t+77, xx, yy]
                if id==6:
                    plt.show()
            plt.subplot(frow, fcol, fcol+fcol+4+1)
            plt.axis('off')
            img2show=[img[0, :, lid, kpx, kpy].detach().cpu() * 4 for lid in light_id]
            img2show=torch.stack(img2show,dim=2)
            plt.imshow(img2show.permute(1,2,0))

            plt.savefig('draw_figures/attenmap/{}.png'.format(id), bbox_inches=0.1)
        else:
            img   = x[0]
            img_split = torch.split(img, 3, 1)
            # if len(x) > 1: # Have mask
            #     mask = x[1]
            #     mask_split = torch.split(mask, 3, 1)
            if len(x) > 1: # Have lighting
                light = x[1]
                light_split = torch.split(light, 3, 1)
            net_in =[torch.cat([img,light], dim=1) for img, light in zip(img_split,light_split)]
            # net_in =[torch.cat([img,mask,light], dim=1) for img, mask, light in zip(img_split,mask_split,light_split)]
            net_in=torch.stack(net_in, dim=2)
            feat = self.extractor(net_in, visual)

            normal = self.regressor(feat)
            return normal

