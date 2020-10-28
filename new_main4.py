import torch
from options  import train_opts
from utils    import logger, recorders
from datasets import custom_data_loader
from models   import custom_model, solver_utils, model_utils

import train_utils
import test_utils





def main(args):
    train_loader, val_loader = custom_data_loader.customDataloader(args)
    model = custom_model.buildModel(args)
    optimizer, scheduler, records = solver_utils.configOptimizer(args, model)
    criterion   = solver_utils.Criterion(args)
    recorder  = recorders.Records(args.log_dir, records)
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        recorder.insertRecord('train', 'lr', epoch, scheduler.get_lr()[0])
        # test_utils.test(args, 'val', val_loader, model, log, epoch, recorder)

        train_utils.train(args, train_loader, model, criterion, optimizer, log, epoch, recorder)
        if epoch % args.save_intv == 0:
            model_utils.saveCheckpoint(args.cp_dir, epoch, model, optimizer, recorder.records, args)

        if epoch % args.val_intv == 0:
            test_utils.test(args, 'val', val_loader, model, log, epoch, recorder)

        # if model.regressor.atten_factor


if __name__ == '__main__':
    args = train_opts.TrainOpts().parse()
    log = logger.Logger(args)
    # torch.manual_seed(args.seed)

    # args.resume='/home/lhy/PycharmProjects/PS-FCN/data/models/PS-FCN_B_S_32.pth.tar'
    args.model='PS_FCN_atten'
    args.use_BN=False
    # args.retrain = "data/checkpoints/legacy.pth.tar"
    # args.init_lr=1e
    # -5
    args.lr_decay=0.2
    args.batch=64
    args.in_img_num=32
    args.val_batch=16
    args.save_root= 'data/checkpoints/atten_nobias'
    args.cp_dir= 'data/checkpoints/atten_nobias'
    args.bert = False
    args.in_light = True
    args.in_mask = False
    args.seed=None
    args.benchmark='none'
    torch.cuda.set_device(0)
    args.crop_h=32
    args.crop_w=32
    args.train_disp=400
    # args.retrain = "data/Training5shadow/checkp_20.pth.tar"
    # args.fuse_type='mean'
    main(args)
