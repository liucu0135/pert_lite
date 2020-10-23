import os
import torch
import torchvision.utils as vutils
import numpy as np
from models import model_utils
from utils import eval_utils, time_utils
import matplotlib.pyplot as plt
import torch.nn.functional as F


def get_itervals(args, split):
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    return disp_intv, save_intv

def test(args, split, loader, model, log, epoch, recorder):
    model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv = get_itervals(args, split)
    with torch.no_grad():

        errors=[]
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer, split)
            input = model_utils.getInput(args, data)
            out_var = model(input); timer.updateTime('Forward')
            acc = eval_utils.calNormalAcc(data['tar'].cpu().data, out_var.cpu().data, data['m'].cpu().data)
            recorder.updateIter(split, acc.keys(), acc.values())

            iters = i + 1
            if args.benchmark == 'MERL':
                set = []
                str=data['obj'][0].split('/')[-1].split('_')[1]
                set.append(str)
                set.append(acc['n_err_mean'])
                errors.append(set)

            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader),
                       'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)

            if iters % save_intv == 0:
                pred = (out_var.data + 1) / 2
                masked_pred = pred.cpu() * data['m'].data.expand_as(out_var.data)
                log.saveNormalResults(masked_pred, split, epoch, iters)

    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)
    return errors

def test_split(args, split, loader, model, log, epoch, recorder, padding=8, stride=32):

    model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv = get_itervals(args, split)
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer, split)
            input = model_utils.getInput(args, data)

            #split_img:
            out_rows=[]
            shape = input[0].shape
            im_w=shape[2]
            im_h=shape[3]
            input[0] = F.pad(input=input[0], pad=(padding, padding, padding, padding), mode='constant', value=0)
            input[1] = F.pad(input=input[1], pad=(padding, padding, padding, padding), mode='constant', value=0)
            # input[2] = F.pad(input=input[2].repeat(1,3,1,1), pad=(padding, padding, padding, padding), mode='constant', value=0)

            if i==1 :
                continue

                # input[2]=input[2][:,60:,:,:]
            model.eval()

            # if stride < 1:
            #     stride =shape[]

            for row_idx in range (im_w//stride):
                out_cols=[]
                for col_idx in range(im_h//stride):
                    r = row_idx * stride + padding
                    c = col_idx * stride + padding
                    in1=input[0][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
                    in2=input[1][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
                    # in3=input[2][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
                    in_iter=([in1,in2])
                    # in_iter=([in1,in2, in3])
                    # out_cols.append(in_iter[0][:,0:3,16:-16,16:-16])
                    if padding==0:
                        out_cols.append(model(in_iter))
                    else:
                        out_cols.append(model(in_iter)[:,:,padding:-padding,padding:-padding])
                # out_cols.append((torch.ones(out_cols[0].shape[0],out_cols[0].shape[1],out_cols[0].shape[2],4)/np.sqrt(3)).cuda())
                out_rows.append(torch.cat(out_cols,dim=3))
            out_var=torch.cat(out_rows,dim=2)
            # print('out shape:{}'.format(out_var.shape))
            out_var=F.pad(out_var, pad=(0, im_h%stride,0 ,im_w%stride), mode='constant', value=0)

            timer.updateTime('Forward')
            acc = eval_utils.calNormalAcc(data['tar'].cpu().data, out_var.cpu().data, data['m'].cpu().data)

            emap = eval_utils.calNormalAngularMap(data['tar'].cpu().data, out_var.cpu().data, data['m'].cpu().data)

            emap[0,0]=90

            # plt.clf()
            # plt.imshow(emap.detach().cpu(), cmap='jet',vmin=0,vmax=90)
            # plt.savefig('./result/sparse/{}_{}_{}.png'.format(args.model,i, args.in_img_num))
            # plt.imshow(data['m'].detach().cpu().squeeze(), cmap='gray',vmin=0,vmax=1)
            # # plt.savefig('./result/{}_{}m.png'.format(args.model,i))

            recorder.updateIter(split, acc.keys(), acc.values())

            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader),
                        'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)

            if iters % save_intv == 0:
                pred = (out_var.data + 1) / 2
                masked_pred = pred.cpu() * data['m'].cpu().data.expand_as(out_var.cpu().data)
                log.saveNormalResults(masked_pred, split, epoch, iters)
                # log.saveErrorResults(emap, split, epoch, iters)

    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    return recorder.records['test']['n_err_mean'][1]

def estimate(args, iteration, loader, model, log, epoch, recorder, padding=8, stride=32, split=True, dataset='light_stage'):

    # model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer, split)
            input = model_utils.getInput(args, data)

            #split_img:
            out_rows=[]
            shape = input[0].shape
            im_w=shape[2]
            im_h=shape[3]
            input[0] = F.pad(input=input[0], pad=(padding, padding, padding, padding), mode='constant', value=0)
            input[1] = F.pad(input=input[1], pad=(padding, padding, padding, padding), mode='constant', value=0)
            # input[2] = F.pad(input=input[2].repeat(1,3,1,1), pad=(padding, padding, padding, padding), mode='constant', value=0)


                # input[2]=input[2][:,60:,:,:]
            model.eval()

            # if stride < 1:
            #     stride =shape[]
            if split:
                for row_idx in range (im_w//stride):
                    out_cols=[]
                    for col_idx in range(im_h//stride):
                        r = row_idx * stride + padding
                        c = col_idx * stride + padding
                        in1=input[0][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
                        in2=input[1][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
                        # in3=input[2][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
                        in_iter=([in1,in2])
                        # in_iter=([in1,in2, in3])
                        # out_cols.append(in_iter[0][:,0:3,16:-16,16:-16])
                        if padding==0:
                            out_cols.append(model(in_iter))
                        else:
                            out_cols.append(model(in_iter)[:,:,padding:-padding,padding:-padding])
                    # out_cols.append((torch.ones(out_cols[0].shape[0],out_cols[0].shape[1],out_cols[0].shape[2],4)/np.sqrt(3)).cuda())
                    out_rows.append(torch.cat(out_cols,dim=3))
                out_var=torch.cat(out_rows,dim=2)
                # print('out shape:{}'.format(out_var.shape))
                out_var=F.pad(out_var, pad=(0, im_h%stride,0 ,im_w%stride), mode='constant', value=0)
            else:
                out_var=model(input)
            timer.updateTime('Forward')
            vutils.save_image(out_var[0]/2+0.5, './result/'+dataset+'/{}_{}_{}.png'.format(args.model,i,iteration))
            print('image　saved, {}_{}.png'.format(args.model,i))
            # emap = out_var.permute(2,3,1,0).squeeze()

            # emap[0,0]=90
            # plt.imshow(emap.detach().cpu()/2+0.5)
            # # plt.colorbar()
            # plt.ion()
            # plt.savefig('./result/light_stage/{}_{}.png'.format(args.model,i))
            # plt.imshow(data['m'].detach().cpu().squeeze(), cmap='gray',vmin=0,vmax=1)
            # plt.savefig('./result/{}_{}m.png'.format(args.model,i))



def test_split_rob(args, split, loader, model, log, epoch, recorder, padding=8, stride=32, noise_level=1):

    model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);
    ie_set=[0.039,0.048,0.095,0.073,0.067,0.082,0.058,0.048,0.105]
    le_set=[3.27,4.34,4.08,4.52,10.36,6.32,5.44,2.87,4.50]
    # , Bal., Bud., Cat, Cow, Gob., Har., Pt.1, Pt.2, Rea., Avg., Inc.


    disp_intv, save_intv = get_itervals(args, split)
    with torch.no_grad():
        for i, sample in enumerate(loader):
            s_int = np.sqrt(3.1415926 / 2) * 0.0068*noise_level/5#*ie_set[sample['id']-1]
            s_l = np.sqrt(3.1415926 / 2) * 0.05*noise_level/5#*np.sin(le_set[sample['id']-1]/180*3.1415926)
            dev_int = np.random.normal(0, s_int, (args.in_img_num))
            dev_l = np.random.normal(0, s_l, (args.in_img_num, 3))
            lights=torch.split(sample['light'].squeeze(),3)
            lights=torch.stack(lights,dim=0)
            # imgs=torch.split(sample['img'].squeeze(),3,dim=0)
            # imgs=torch.stack(imgs,dim=0)

            #add noise:
            for l in range(args.in_img_num):
                lights[l,0]=lights[l,0]+dev_l[l,0]
                lights[l,1]=lights[l,1]+dev_l[l,1]
                lights[l,2]=lights[l,2]+dev_l[l,2]
                lights[l, :]=lights[l,:]/np.sqrt(np.dot(lights[l,:],lights[l,:]))
                # sample['img'][0,l*3:l*3+3,:,:]=sample['img'][0,l*3:l*3+3,:,:]+dev_int[l]
                sample['img'][0,l*3:l*3+3,:,:]=sample['img'][0,l*3:l*3+3,:,:]*(1+dev_int[l])
            sample['light']=lights.view(-1).unsqueeze(0).unsqueeze(2).unsqueeze(2)
            # sample['imgs']=lights.view(-1).unsqueeze(0).unsqueeze(2).unsqueeze(2)
            data = model_utils.parseData(args, sample, timer, split)


            input = model_utils.getInput(args, data)



            #split_img:
            out_rows=[]
            input[0] = F.pad(input=input[0], pad=(padding, padding, padding, padding), mode='constant', value=0)
            input[1] = F.pad(input=input[1], pad=(padding, padding, padding, padding), mode='constant', value=0)
            # input[2] = F.pad(input=input[2].repeat(1,3,1,1), pad=(padding, padding, padding, padding), mode='constant', value=0)





            if i==1 :
                if args.in_img_num == 97:# abondon bear
                    print('doing bear, ignoring first 20 images')
                    input[0]=input[0][:,60:,:,:]
                    input[1]=input[1][:,60:,:,:]
                else:
                    continue

                # input[2]=input[2][:,60:,:,:]
            model.eval()
            shape=input[0].shape
            # if stride < 1:
            #     stride =shape[]
            for row_idx in range (512//stride):
                out_cols=[]
                for col_idx in range(612//stride):
                    r = row_idx * stride + padding
                    c = col_idx * stride + padding

                    in1=input[0][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
                    in2=input[1][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
                    # in3=input[2][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
                    in_iter=([in1,in2])
                    # in_iter=([in1,in2, in3])
                    # out_cols.append(in_iter[0][:,0:3,16:-16,16:-16])
                    out_cols.append(model(in_iter)[:,:,padding:-padding,padding:-padding])
                # out_cols.append((torch.ones(out_cols[0].shape[0],out_cols[0].shape[1],out_cols[0].shape[2],4)/np.sqrt(3)).cuda())
                out_rows.append(torch.cat(out_cols,dim=3))
            out_var=torch.cat(out_rows,dim=2)
            out_var=F.pad(out_var, pad=(0, 612%stride,0 ,512%stride), mode='constant', value=0)

            timer.updateTime('Forward')
            acc = eval_utils.calNormalAcc(data['tar'].data, out_var.data, data['m'].data)
            emap = eval_utils.calNormalAngularMap(data['tar'].data, out_var.data, data['m'].data)
            plt.clf()
            plt.imshow(emap.detach().cpu(), cmap='jet',vmin=0,vmax=90)
            plt.savefig('./result/robust/{}_{}_{}.png'.format(args.model,i, noise_level))
            recorder.updateIter(split, acc.keys(), acc.values())

            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader),
                        'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)

            if iters % save_intv == 0:
                pred = (out_var.data + 1) / 2
                masked_pred = pred * data['m'].data.expand_as(out_var.data)
                log.saveNormalResults(masked_pred, split, epoch, iters)
                # log.saveErrorResults(emap, split, epoch, iters)

    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    return recorder.records['test']['n_err_mean'][1]

def test2c(args, split, loader, model, log, epoch, recorder):
    model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv = get_itervals(args, split)
    with torch.no_grad():

        errors=[]
        for i, sample in enumerate(loader):
            print('loading{}'.format(i))
            # if i<54:
            #     continue
            data = model_utils.parseData(args, sample, timer, split)
            input = model_utils.getInput(args, data)
            model(input,visual=True,id=i); timer.updateTime('Forward')

def test_split2c(args, split, loader, model, log, epoch, recorder, padding=8, stride=32):
    model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv = get_itervals(args, split)
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer, split)
            input = model_utils.getInput(args, data)

            #split_img:
            out_rows=[]
            # input[0] = F.pad(input=input[0], pad=(padding, padding, padding, padding), mode='constant', value=0)
            # input[1] = F.pad(input=input[1], pad=(padding, padding, padding, padding), mode='constant', value=0)
            # input[2] = F.pad(input=input[2].repeat(1,3,1,1), pad=(padding, padding, padding, padding), mode='constant', value=0)


            # light_id=[47,7,95,0]
            # light_id=[47,7,95,0]
            input[0]=torch.split(input[0],3,dim=1)
            input[1]=torch.split(input[1],3,dim=1)
            input[0]=torch.stack((input[0]),dim=2)
            input[1]=torch.stack((input[1]),dim=2)
            # input[0]=input[0][0,:,light_id,:,:].unsqueeze(0)
            # input[1]=input[1][0,:,light_id,:,:].unsqueeze(0)




                # input[2]=input[2][:,60:,:,:]
            model.eval()
            r=256
            c=306
            padding=64
            stride=0
            norm=data['tar']
            norm=norm[:, :, r - padding:r + padding + stride, c - padding:c + padding + stride]
            in1 = input[0][:, :,:, r - padding:r + padding + stride, c - padding:c + padding + stride]
            in2 = input[1][:, :,:, r - padding:r + padding + stride, c - padding:c + padding + stride]
            in_iter = ([in1, in2])
            model(in_iter, visual=True, id='obj{}'.format(i), normal=norm)
            # for row_idx in range (512//stride):
            #     out_cols=[]
            #     for col_idx in range(612//stride):
            #         r = row_idx * stride + padding
            #         c = col_idx * stride + padding
            #         in1=input[0][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
            #         in2=input[1][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
            #         # in3=input[2][:, :, r-padding:r + padding+stride, c-padding:c + padding+stride]
            #         in_iter=([in1,in2])
            #         model(in_iter, visual=True, id='obj{}_r{}_c{}'.format(i,row_idx,col_idx))
