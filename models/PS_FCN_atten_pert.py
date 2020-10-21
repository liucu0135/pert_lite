import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
import matplotlib.pyplot as plt

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.ln=32
        # self.conv1 = model_utils.conv3d(batchNorm, c_in, 64,  k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv2 = model_utils.conv3d(batchNorm, 64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        # self.conv3 = model_utils.conv3d(batchNorm, 128,  128, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv4 = model_utils.conv3d(batchNorm, 128,  256, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        # self.conv5 = model_utils.conv3d(batchNorm, 256,  256, k=[1,3,3], stride=1, pad=[0,1,1], atten=True)
        # self.conv6 = model_utils.deconv3d(256, 128, k=[1,4,4], stride=[1,2,2], pad=[0,1,1])
        # self.conv7 = model_utils.conv3d(batchNorm, 128, 128, k=[1,3,3], stride=1, pad=[0,1,1])


        # the one actually work
        self.conv1 = model_utils.conv3d(c_in, 64,  k=[1,1,1], stride=1, pad=[0,0,0], atten=True)
        self.conv2 = model_utils.conv3d(64,   128, k=[1,3,3], stride=[1,2,2], pad=[0,1,1])
        self.conv3 = model_utils.conv3d(128,  128, k=[1,1,1], stride=1, pad=[0,0,0], atten=True)
        self.conv5 = model_utils.conv3d(128,  128, k=[1,1,1], stride=1, pad=[0,0,0], atten=True)
        self.conv7 = model_utils.conv3d(128, 128, k=[1,3,3], stride=1, pad=[0,1,1])




    def forward(self, x):
        out = self.conv1(x)
        out1 = self.conv2(out)
        out = self.conv3(out1)
        # out = self.conv4(out)
        out = self.conv5(out)
        # out = self.conv6(out)
        out_feat = self.conv7(out)
        return out_feat, out1





class Regressor(nn.Module):
    def __init__(self, decoder=False, batchnorm=True, other={}):
        super(Regressor, self).__init__()
        self.decoder=decoder
        self.other   = other
        if decoder:
            self.deconv1 = model_utils.conv(256, 128,  k=3, stride=1, pad=1)
        else:
            self.deconv1 = model_utils.conv(128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal= self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other   = other
    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x):
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        if self.decoder:
            return nn.functional.sigmoid(normal)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class Light_est(nn.Module):
    def __init__(self):
        super(Light_est, self).__init__()
        self.conv1 = model_utils.conv(256, 128,  k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(128, 256,  k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(256, 256,  k=3, stride=2, pad=1)
        self.conv4 = model_utils.conv(256, 256,  k=3, stride=2, pad=1)
        self.fc=nn.Linear(256,3)
    def forward(self, x):

        out    = self.conv1(x)
        out    = self.conv2(out)
        out    = self.conv3(out)
        out    = self.conv4(out)
        shape = out.shape
        out=nn.functional.avg_pool2d(out, kernel_size=[shape[2],shape[3]])
        normal = self.fc(out.squeeze())
        normal = torch.nn.functional.normalize(normal, 2, 1).unsqueeze(2).unsqueeze(2)
        normal = normal.expand(x.shape[0], 3,x.shape[2]*2,x.shape[3]*2)
        return normal

class PS_FCN(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(PS_FCN, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor()
        self.decoder = Regressor(decoder=True)
        # self.decoder = Light_est()
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

    def forward(self, x, require_gen=False):
        ridx=torch.randperm(x[0].shape[1]//3+1)

        img   = x[0]
        img_split = torch.split(img, self.c_in-3, 1)
        if len(x) > 1: # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)
        if len(x) > 2: # Have lighting
            mask = x[2].repeat(1,3,1,1)
        net_in =[torch.cat([img,light], dim=1) for img, light in zip(img_split,light_split)]
        net_in.insert(0,torch.cat([mask,torch.zeros_like(mask)], dim=1).cuda())
        net_in=torch.stack(net_in, dim=2)[:,:,ridx,:,:]
        feats, shallow = self.extractor(net_in)
        shape=feats.shape
        fused_feat=nn.functional.max_pool3d(feats,kernel_size=[shape[2],1,1], stride=[1,1,1], padding=[0,0,0]).squeeze(2)
        normal = self.regressor(fused_feat)*mask
        if require_gen:
            gen=[]
            im_num=len(ridx)
            for i in range(im_num):
                if ridx[i] in range(im_num-5,im_num-1):
                    input=torch.cat([fused_feat,shallow[:,:,ridx[-i],:,:]], dim=1)
                    gen.append(self.decoder(input)*mask)
                    # print(torch.min(net_in[0,:3,ridx,:,:]))
                    # plt.imshow(net_in[0,:3,ridx[-i],:,:].detach().cpu().permute(1,2,0))
                    # plt.show()
            gen=torch.cat(gen,dim=1)
            return normal, gen
        else:
            return normal

