import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils

# class FeatExtractor(nn.Module):
#     def __init__(self, batchNorm=False, c_in=3, other={}):
#         super(FeatExtractor, self).__init__()
#         self.other = other
#         self.conv1 = model_utils.conv(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
#         self.conv2 = model_utils.conv(batchNorm, 64,   128, k=3, stride=2, pad=1)
#         self.conv3 = model_utils.conv(batchNorm, 128,  128, k=3, stride=1, pad=1)
#         self.conv4 = model_utils.conv(batchNorm, 128,  256, k=3, stride=2, pad=1)
#         self.conv5 = model_utils.conv(batchNorm, 256,  256, k=3, stride=1, pad=1)
#         self.conv6 = model_utils.deconv(256, 128)
#         self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
#     def forward(self, x):
#         out0 = self.conv1(x)
#         out = self.conv2(out0)
#         out = self.conv3(out)
#         out1 = self.conv4(out)
#         out2 = self.conv5(out1)
#         out3 = self.conv6(out2)
#         out_feat = self.conv7(out3)
#         # out_feat = self.conv8(out)
#         n, c, h, w = out_feat.data.shape
#         out_feat   = [out0,out_feat]
#         return out_feat

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.ln=32
        self.conv1 = model_utils.conv3d(batchNorm, c_in, 64,  k=[1,3,3], stride=1, pad=[0,1,1])
        self.conv2 = model_utils.conv3d(batchNorm, 64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        self.conv3 = model_utils.conv3d(batchNorm, 128,  128, k=[1,3,3], stride=1, pad=[0,1,1])
        self.conv4 = model_utils.conv3d(batchNorm, 128,  256, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        self.conv5 = model_utils.conv3d(batchNorm, 256,  256, k=[1,3,3], stride=1, pad=[0,1,1])
        self.conv6 = model_utils.deconv3d(256, 128, k=[1,4,4], stride=[1,2,2], pad=[0,1,1])
        self.conv7 = model_utils.conv3d(batchNorm, 128, 128, k=[1,3,3], stride=1, pad=[0,1,1])
    def forward(self, x):
        out0 = self.conv1(x)
        out = self.conv2(out0)
        out = self.conv3(out)
        out1 = self.conv4(out)
        out2 = self.conv5(out1)
        out3 = self.conv6(out2)
        out_feat = self.conv7(out3)
        return out_feat



class Attention_layer(nn.Module):
    def __init__(self, ch_in, batch=False, factor=1):
        super(Attention_layer, self).__init__()
        self.atten_factor=nn.Parameter(torch.zeros(1).cuda())
        self.shurink=ch_in//2
        self.atten_k = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0)
        self.atten_q = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0)
        self.atten_v = nn.Conv3d(ch_in, self.shurink, kernel_size=1, stride=1, padding=0)
        self.atten = nn.Conv3d(self.shurink, ch_in, kernel_size=1, stride=1, padding=0)
        if batch:
            self.atten_bn = nn.BatchNorm2d(ch_in)
        else:
            self.atten_bn=None

    def forward(self, x):
        shape=x.shape
        atten_k = self.atten_k(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        atten_q = self.atten_q(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        atten_v = self.atten_v(x).permute(0,3,4,1,2).contiguous().view(shape[0]*shape[3]*shape[4],self.shurink,shape[2])
        a=nn.functional.softmax(torch.bmm(atten_k.transpose(1,2),atten_q)/torch.sqrt(self.shurink), dim=1)
        a=torch.bmm(atten_v,a)
        a=self.atten(a.view(shape[0],shape[3],shape[4],self.shurink,shape[2]).permute(0,3,4,1,2))
        x=x+self.atten_factor*a
        # x=nn.functional.softmax(a,dim=2)*x
        x=torch.mean(x,dim=2)
        if self.atten_bn is not None:
            x=self.atten_bn(x)
        return x

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
        # shape=x.shape
        # x=nn.functional.max_pool3d(x,kernel_size=[shape[2],1,1], stride=[1,1,1], padding=[0,0,0]).squeeze()
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

        feats = torch.Tensor()
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            feat= self.extractor(net_in.unsqueeze(2)).squeeze(2)
            if i == 0:
                feats = feat
            else:
                if self.fuse_type == 'mean':
                    feats = torch.stack([feats, feat], 1).sum(1)
                elif self.fuse_type == 'max':
                    feats, _ = torch.stack([feats, feat], 1).max(1)
        if self.fuse_type == 'mean':
            feats = feats / len(img_split)
        feat_fused = feats

        normal = self.regressor(feat_fused)
        return normal

