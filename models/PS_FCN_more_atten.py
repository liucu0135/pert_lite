import torch
import torch.nn as nn
import math
from torch.nn.init import kaiming_normal_
from . import model_utils


class Attention_layer(nn.Module):
    def __init__(self, ch_in, batch=False, factor=1):
        super(Attention_layer, self).__init__()
        # self.atten_factor=nn.Parameter(torch.zeros(1).cuda())
        self.atten_factor=1
        self.shurink=ch_in//2
        self.atten_k = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0)
        self.atten_q = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0)
        self.atten_v = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0)
        self.atten = nn.Conv3d(self.shurink, ch_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        shape=x.shape
        atten_k = self.atten_k(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        atten_q = self.atten_q(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        atten_v = self.atten_v(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        a=nn.functional.softmax(torch.bmm(atten_k.transpose(1,2),atten_q)/math.sqrt(self.shurink), dim=1)
        a=torch.bmm(atten_v,a)
        a=self.atten(a.view(shape[0],shape[3],shape[4],self.shurink,shape[2]).permute(0,3,4,1,2))
        x=x+self.atten_factor*a
        return x

class Attention_block(nn.Module):
    def __init__(self, ch_in, chi_out, norm=None, fuse=False):
        super(Attention_block, self).__init__()
        self.ff1 = nn.Conv3d(ch_in, chi_out, kernel_size=1, stride=1, padding=0)
        if fuse:
            self.ff2 = nn.Conv3d(chi_out, chi_out, kernel_size=1, stride=[1, 2, 2], padding=[0, 1, 1])
        else:
            self.ff2 = nn.Conv3d(chi_out, chi_out, kernel_size=1, stride=1, padding=0)
        self.atten=Attention_layer(ch_in)
        if norm is not None:
            self.norm=nn.GroupNorm(ch_in//8,ch_in)

    def forward(self, x):
        x=self.atten(x)
        x=self.norm(x)
        x=self.ff1(x)
        x=nn.functional.relu(x)
        x=self.ff2(x)



class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.ln=32


        # the one actually work
        self.conv1 = model_utils.conv3d(c_in, 64,  k=[1,1,1], stride=1, pad=[0,0,0], atten=True)
        self.conv2 = model_utils.conv3d(64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        self.conv3 = model_utils.conv3d(128,  128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        self.conv5 = model_utils.conv3d(128,  128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        self.conv7 = model_utils.conv3d(128, 128, k=[1,3,3], stride=1, pad=[0,1,1])



    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out_feat = self.conv5(out)
        return out_feat





class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other   = other
        self.deconv1 = model_utils.conv( 128, 128,  k=3, stride=1, pad=1,Norm=batchNorm)
        self.deconv2 = model_utils.conv(128, 128,  k=3, stride=1, pad=1, Norm=batchNorm)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal= self._make_output(64, 3, k=1, stride=1, pad=0)
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

class PS_FCN(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(PS_FCN, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

    def forward(self, x):
        img   = x[0]
        img_split = torch.split(img, self.c_in-3, 1)
        if len(x) > 1: # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)
        net_in =[torch.cat([img,light], dim=1) for img, light in zip(img_split,light_split)]
        net_in=torch.stack(net_in, dim=2)
        feat = self.extractor(net_in)
        normal = self.regressor(feat)
        return normal

