import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
import math

class Attention_layer(nn.Module):
    def __init__(self, ch_in, batch=False, factor=1, bias=True, extra=True):
        super(Attention_layer, self).__init__()
        # self.atten_factor=nn.Parameter(torch.zeros(1).cuda())
        self.atten_factor=1
        self.extra=extra
        self.shurink=ch_in//2
        self.atten_k = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0, bias=bias)
        self.atten_q = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0, bias=bias)
        self.atten_v = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0, bias=bias)
        self.atten = nn.Conv3d(self.shurink, ch_in-3, kernel_size=1, stride=1, padding=0, bias=bias)
        light_layers=[
        nn.Conv2d(3, self.shurink//4, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(self.shurink//4, self.shurink//2, kernel_size=1, stride=1, padding=0),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(self.shurink//2, self.shurink, kernel_size=1, stride=1, padding=0)]
        self.light_emb=nn.Sequential(*light_layers)


    def forward(self, x, light):
        shape=x.shape
        #make light
        cop=torch.ones_like(light)*-1
        light=torch.cat((light,cop[:,:,0].unsqueeze(2)), dim=2).unsqueeze(3)
        l=light.repeat(1,1,1,shape[2]+1)
        light=l-l.transpose(2,3)
        light=self.light_emb(light)
        if self.extra:
            extra=nn.functional.max_pool3d(x,kernel_size=[shape[2],1,1], stride=[1,1,1], padding=[0,0,0])
            x=torch.cat([extra,x], dim=2)
            shape = x.shape
        atten_k = self.atten_k(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        atten_q = self.atten_q(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        atten_v = self.atten_v(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        # light=light.permute(0,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        # light=light.unsqueeze(1).unsqueeze(1).repeat(1,shape[3],shape[4],1,1,1)
        # light=light.view(shape[0]*shape[3]*shape[4],self.shurink,shape[2],shape[2])
        a=nn.functional.softmax(torch.bmm(atten_k.transpose(1,2),atten_q)/math.sqrt(self.shurink), dim=1)
        a=a+torch.einsum('iuvjk,ikg->iuvkg',(atten_q,light))
        a=torch.bmm(atten_v,a)

        a=self.atten(a.view(shape[0],shape[3],shape[4],self.shurink,shape[2]).permute(0,3,4,1,2))

        x=x+self.atten_factor*a
        return x[:,:,1:,:,:]

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.ln=32
        # more atten
        self.conv1 = model_utils.conv3d(batchNorm, 6, 64,  k=[1,1,1], stride=1, pad=[0,0,0])
        self.at1=Attention_layer(64)
        self.bn1=nn.BatchNorm3d(64, affine=False)
        self.conv2 = model_utils.conv3d(batchNorm, 64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        self.conv3 = model_utils.conv3d(batchNorm, 128,  128, k=[1,1,1], stride=1, pad=[0,0,0])
        self.at2=Attention_layer(128)
        self.bn1=nn.BatchNorm3d(128, affine=False)
        self.conv5 = model_utils.conv3d(batchNorm, 128,  128, k=[1,1,1], stride=1, pad=[0,0,0])
        self.at3=Attention_layer(128)
        self.bn1=nn.BatchNorm3d(128, affine=False)
        self.conv7 = model_utils.conv3d(batchNorm, 128, 128, k=[1,3,3], stride=1, pad=[0,1,1])

        # less atten
        # self.conv1 = model_utils.conv3d(batchNorm, c_in, 64,  k=[1,1,1], stride=1, pad=[0,0,0], atten=True)
        # self.conv2 = model_utils.conv3d(batchNorm, 64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        # self.conv3 = model_utils.conv3d(batchNorm, 128,  128, k=[1,1,1], stride=1, pad=[0,0,0], atten=True)
        # self.conv5 = model_utils.conv3d(batchNorm, 128,  128, k=[1,1,1], stride=1, pad=[0,0,0], atten=True)
        # self.conv7 = model_utils.conv3d(batchNorm, 128, 128, k=[1,3,3], stride=1, pad=[0,1,1])

        # the one actually work
        # self.conv1 = model_utils.conv3d(batchNorm, c_in, 64,  k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv2 = model_utils.conv3d(batchNorm, 64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1], atten=True)
        # self.conv3 = model_utils.conv3d(batchNorm, 128,  128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv5 = model_utils.conv3d(batchNorm, 128,  128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv7 = model_utils.conv3d(batchNorm, 128, 128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)

    def forward(self, x):
        light=x[:,3:,:,0,0]
        # x=x[:,:3,:,:,:]
        out = self.conv1(x)
        out=self.at1(out,light)
        out = self.conv2(out)
        out = self.conv3(out)
        out=self.at2(out,light)
        out = self.conv5(out)
        out=self.at3(out,light)
        out_feat = self.conv7(out)
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

