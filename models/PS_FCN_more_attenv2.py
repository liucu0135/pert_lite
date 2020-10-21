import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
import math

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.ln=32
        # the one actually work
        self.conv0 = nn.Conv3d(9, 64,  kernel_size=[1,1,1], stride=1, padding=[0,0,0], bias=False)
        self.at1=Attention_layer(64)
        self.conv1 = model_utils.conv3d(64, 128,  k=[1,3,3], stride=1, pad=[0,1,1])
        self.at2=Attention_layer(128)
        self.conv2 = model_utils.conv3d(128,   256, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        self.at3=Attention_layer(256)
        self.conv3 = model_utils.conv3d(256,  128, k=[1,3,3], stride=1, pad=[0,1,1])
        self.at4=Attention_layer(128)
        self.conv4 = model_utils.conv3d(128, 128, k=[1,1,1], stride=1, pad=[0,0,0])

    def forward(self, x):
        x = self.conv0(x)
        x = self.at1(x)
        x = self.conv1(x)
        x = self.at2(x)
        x = self.conv2(x)
        x = self.at3(x)
        x = self.conv3(x)
        x = self.at4(x)
        out_feat = self.conv4(x)
        return out_feat

class Attention_layer(nn.Module):
    def __init__(self, ch_in, shrink=0, batch=True):
        super(Attention_layer, self).__init__()
        # self.atten_factor=nn.Parameter(torch.zeros(1).cuda())
        self.atten_factor=1
        if shrink>0:
            self.shrink = shrink
            self.out=shrink
        else:
            self.shrink=ch_in//2
            self.out=ch_in
        self.atten_k = nn.Conv3d(ch_in, self.shrink, kernel_size=1, stride=1, padding=0, bias=False)
        self.atten_q = nn.Conv3d(ch_in, self.shrink, kernel_size=1, stride=1, padding=0, bias=False)
        self.atten_v = nn.Conv3d(ch_in, self.shrink, kernel_size=1, stride=1, padding=0, bias=False)
        self.atten = nn.Conv3d(self.shrink, self.out, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch=nn.BatchNorm3d(ch_in, momentum=0.05)
    def forward(self, x):
        input=x
        shape=x.shape
        x=self.batch(x)
        atten_k = self.atten_k(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shrink,shape[2])
        atten_q = self.atten_q(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shrink,shape[2])
        atten_v = self.atten_v(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shrink,shape[2])
        a=nn.functional.softmax(torch.bmm(atten_k.transpose(1,2),atten_q)/math.sqrt(self.shrink), dim=1)
        a=torch.bmm(atten_v,a)
        a=self.atten(a.view(shape[0],shape[3],shape[4],self.shrink,shape[2]).permute(0,3,4,1,2))
        x=input+self.atten_factor*a
        return x



class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other   = other
        self.deconv1 = model_utils.conv(128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal= self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other   = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x):
        shape=x.shape
        x=nn.functional.avg_pool3d(x,kernel_size=[shape[2],1,1], stride=[1,1,1], padding=[0,0,0]).squeeze(2)
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
        self.c_in      = c_in+3#for mask +3
        self.fuse_type = fuse_type
        self.other = other

    def forward(self, x):
        img   = x[0]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1: # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)
        if len(x) > 2: # Have lighting
            mask = x[2]
        net_in =[torch.cat([img,light,mask[:]], dim=1) for img, light in zip(img_split,light_split)]
        net_in=torch.stack(net_in, dim=2)
        feat = self.extractor(net_in)
        normal = self.regressor(feat)
        return normal

