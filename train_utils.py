from models import model_utils
from utils  import time_utils
import matplotlib.pyplot as plt
import torch
def train(args, loader, model, criterion, optimizer, log, epoch, recorder):
    model.train()

    print('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    for i, sample in enumerate(loader):
        # args.in_img_num=i%32+32
        data  = model_utils.parseData(args, sample, timer, 'train')
        input = model_utils.getInput(args, data)

        # if args.bert:
        #     gen_gt=input[0][:,-4*3:,:,:].clone()
        #     input[0][:,-4*3:,:,:]=input[0][:,-4*3:,:,:]*0-1
        #     # only blacked out the images in input[0], the lights in input[1] are untouched
        #     out_var, gen_pre = model(input); timer.updateTime('Forward')
        #     optimizer.zero_grad()
        #     loss = criterion[0].forward(out_var, data['tar'])
        #     loss.update(criterion[1].forward(gen_pre*100, gen_gt*100));timer.updateTime('Crit');
        #     # criterion[1].backward();timer.updateTime('Backward')
        #     criterion[0].backward(retain_graph=True);criterion[1].backward();timer.updateTime('Backward')
        # else:
        out_var = model(input).squeeze(); timer.updateTime('Forward')
        optimizer.zero_grad()
        out_var=out_var * (data['m'].cuda())
        loss = criterion.forward(out_var, data['tar'].cuda()); timer.updateTime('Crit');
        criterion.backward(); timer.updateTime('Backward')





        recorder.updateIter('train', loss.keys(), loss.values())

        optimizer.step(); timer.updateTime('Solver')

        iters = i + 1
        if iters % args.train_disp == 0:
            # plt.ion()
            # for u in range(10):
            #     plt.subplot(3,10,u+1)
            #     im2show1=out_var[u,:,:,:].detach().cpu()/2+0.5
            #     plt.imshow(im2show1.permute(1,2,0))
            #     plt.subplot(3,10,u+11)
            #     im2show2=data['tar'][u,:,:,:].detach().cpu()/2+0.5
            #     plt.imshow(im2show2.permute(1,2,0))
            #     plt.subplot(3, 10, u + 21)
            #     plt.imshow(torch.sum(torch.abs((im2show1.permute(1,2,0)-im2show2.permute(1,2,0))), dim=2))
            # plt.draw()
            # plt.pause(0.01)
            opt = {'split':'train', 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                    'timer':timer, 'recorder': recorder}
            log.printItersSummary(opt)
            # print('r of 2 attens: {}, {}'.format(model.at1.atten_factor,model.at2.atten_factor))
    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)
