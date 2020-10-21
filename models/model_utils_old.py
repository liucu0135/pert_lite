import os
import torch
import torch.nn as nn
import math



def getInput(args, data):
    input_list = [data['input']]
    # input_list.append(data['m'])
    if args.in_light: input_list.append(data['l'])
    return input_list

def parseData(args, sample, timer=None, split='train'):
    input, target, mask = sample['img'], sample['N'], sample['mask'] 
    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        input  = input.cuda(); target = target.cuda(); mask = mask.cuda(); 

    input_var  = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    mask_var   = torch.autograd.Variable(mask, requires_grad=False);

    if timer: timer.updateTime('ToGPU')
    if args.pshadow:
        if len(input_var.shape)==4:
            data = {'input': input_var[:,:args.in_img_num*3,:,:], 'tar': input_var[:,args.in_img_num*3:,:,:], 'm': mask_var}
        else:
            data = {'input': input_var[:args.in_img_num*3,:,:], 'tar': input_var[args.in_img_num*3:,:,:], 'm': mask_var}

    else:
        data = {'input': input_var, 'tar': target_var, 'm': mask_var}

    if args.in_light:
        right_size=torch.tensor(input.shape).numpy()
        right_size[-3]=args.in_img_num*3
        light = sample['light'].expand(tuple(right_size))
        if args.cuda: light = light.cuda()
        light_var = torch.autograd.Variable(light);
        data['l'] = light_var
    return data 

def getInputChanel(args):
    print('[Network Input] Color image as input')
    c_in = 3
    if args.shadow:
        c_in+=1
        if args.pshadow:
            c_in += -1
    if args.in_light:
        print('[Network Input] Adding Light direction as input')
        c_in += 3

    print('[Network Input] Input channel: {}'.format(c_in))
    return c_in

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'],strict=False)

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records, 
            'args': args}
    torch.save(state, os.path.join(save_path, 'checkp_%d.pth.tar' % (epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_%d_rec.pth.tar' % (epoch)))

def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = (k - 1) // 2 if pad < 0 else pad
    print('Conv pad = %d' % (pad))
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(cout, momentum=0.05),
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )

def conv3d(batchNorm, cin, cout, k=3, stride=1, pad=-1, atten=None):

    layers=[nn.Conv3d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False)]
    layers.append(nn.LeakyReLU(0.1, inplace=True))
    if atten is not None:
        layers.append(Attention_layer(cout))
    if batchNorm:
        layers.append(nn.BatchNorm3d(cout, momentum=0.05))

    return nn.Sequential(*layers)

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )

def deconv3d(cin, cout, k=3, stride=1, pad=1):
    return nn.Sequential(
            nn.ConvTranspose3d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )

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