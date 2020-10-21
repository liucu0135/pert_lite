import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
import math

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

    def forward(self, x):
        shape=x.shape
        if self.extra:
            extra=nn.functional.avg_pool3d(x,kernel_size=[shape[2],1,1], stride=[1,1,1], padding=[0,0,0])
            x=torch.cat([extra,x], dim=2)
            shape = x.shape
        atten_k = self.atten_k(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        atten_q = self.atten_q(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        atten_v = self.atten_v(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        a=nn.functional.softmax(torch.bmm(atten_k.transpose(1,2),atten_q)/math.sqrt(self.shurink), dim=1)
        a=torch.bmm(atten_v,a)
        a=self.atten(a.view(shape[0],shape[3],shape[4],self.shurink,shape[2]).permute(0,3,4,1,2))
        x=x+self.atten_factor*a
        if self.extra:
            return x[:,:,1:,:,:]
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
        self.conv1 = model_utils.conv3d(batchNorm, 6, 32,  k=[1,1,1], stride=1, pad=[0,0,0])
        self.at1=Attention_layer(32)
        self.bn1=nn.BatchNorm3d(32, affine=False)
        self.conv2 = model_utils.conv3d(batchNorm, 32,   128, k=[3,3,3], stride=[2,2,2], pad=[1,1,1])
        self.conv31 = model_utils.conv3d(batchNorm, 128,  128, k=[1,1,1], stride=1, pad=[0,0,0])
        self.conv33 = model_utils.conv3d(batchNorm, 128,  128, k=[1,3,3], stride=1, pad=[0,1,1])
        # self.conv3 = model_utils.conv3d(batchNorm, 256,  128, k=[1,1,1], stride=1, pad=[0,0,0])
        self.at3=Attention_layer(256)
        self.bn3=nn.BatchNorm3d(256, affine=False)
        self.conv3 = model_utils.conv3d(batchNorm, 256, 256, k=[3, 3, 3], stride=[2, 1, 1],
                                        pad=[1, 1, 1])
        self.conv41 = model_utils.conv3d(batchNorm, 256,  256, k=[1,1,1], stride=1, pad=[0,0,0])
        self.conv43 = model_utils.conv3d(batchNorm, 256,  256, k=[1,3,3], stride=1, pad=[0,1,1])
        # self.conv4 = model_utils.conv3d(batchNorm, 256,  128, k=[1,1,1], stride=1, pad=[0,0,0])
        self.at4=Attention_layer(512)
        self.bn4=nn.BatchNorm3d(512, affine=False)
        self.conv5 = model_utils.conv3d(batchNorm, 512, 128, k=[3,3,3], stride=[1, 1, 1], pad=[1,1,1])


        # the one actually work
        # self.conv1 = model_utils.conv3d(batchNorm, c_in, 64,  k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv2 = model_utils.conv3d(batchNorm, 64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1], atten=True)
        # self.conv3 = model_utils.conv3d(batchNorm, 128,  128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv5 = model_utils.conv3d(batchNorm, 128,  128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv7 = model_utils.conv3d(batchNorm, 128, 128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.at1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out1 = self.conv33(out)
        out3 = self.conv31(out)
        out = torch.cat((out1,out3),dim=1)
        out = self.at3(out)
        out = self.bn3(out)
        out1 = self.conv41(out)
        out3 = self.conv43(out)
        out = torch.cat((out1,out3),dim=1)
        out = self.at4(out)
        out = self.bn4(out)
        out_feat = self.conv5(out)
        return out_feat

        # out = self.conv1(x)
        # out = self.conv2(out)
        # out = self.conv3(out)
        # out = self.conv5(out)
        # out_feat = self.conv7(out)
        # return out_feat





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
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

    def forward(self, x):
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
        feat = self.extractor(net_in)
        normal = self.regressor(feat)
        return normal

