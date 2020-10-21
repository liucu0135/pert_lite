import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils


class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.ln=32

        # the one actually work
        self.conv1 = model_utils.conv3d(c_in, 64,  k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        self.conv2 = model_utils.conv3d(64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        self.conv3 = model_utils.conv3d(128,  128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        self.conv5 = model_utils.conv3d(128,  128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        self.conv7 = model_utils.conv3d(128, 128, k=[1,3,3], stride=1, pad=[0,1,1])




    def forward(self, x, shadow=False):
        out1 = self.conv1(x)
        out = self.conv2(out1)
        out = self.conv3(out)
        # out = self.conv4(out)
        out = self.conv5(out)
        # out = self.conv6(out)
        out_feat = self.conv7(out)
        if shadow:
            return out_feat,out1
        else:
            return out_feat





class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}, shadow=False):
        super(Regressor, self).__init__()
        self.other   = other
        self.shadow=shadow
        if shadow:
            self.deconv0 = model_utils.conv3d(6, 128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1], atten=True)
            self.deconv1 = model_utils.conv3d(256, 128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
            self.deconv2 = model_utils.conv3d(128, 128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
            self.deconv3 = model_utils.deconv3d(128, 64)
            self.est_normal= nn.Conv3d(64, 1, kernel_size=[1,3,3], stride=1, padding=[0,1,1], bias=False)
            self.other   = other

        else:
            self.deconv1 = model_utils.conv(128, 128, k=3, stride=1, pad=1)
            self.deconv2 = model_utils.conv(128, 128, k=3, stride=1, pad=1)
            self.deconv3 = model_utils.deconv(128, 64)
            self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)
            self.other   = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x):
        if self.shadow:
            img=x[1]
            img=self.deconv0(img)

            x=torch.cat([x[0],img],dim=1)

        else:
            shape = x.shape
            x=nn.functional.max_pool3d(x,kernel_size=[shape[2],1,1], stride=[1,1,1], padding=[0,0,0]).squeeze(2)

        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        if self.shadow:
            normal = torch.nn.functional.sigmoid(normal)
        else:
            normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class PS_FCN(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(PS_FCN, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.s = Regressor(batchNorm, other,shadow=True)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

    def forward(self, x, s=False):
        img   = x[0]
        img_split = torch.split(img, self.c_in-3, 1)
        if len(x) > 1: # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)
        net_in =[torch.cat([img,light], dim=1) for img, light in zip(img_split,light_split)]
        net_in=torch.stack(net_in, dim=2)
        feat = self.extractor(net_in, shadow=False)
        if s:
            normal = self.s([feat,net_in])
            return normal
        else:
            normal = self.regressor(feat)
            return normal

