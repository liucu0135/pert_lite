import sys, os, shutil
import torch
import xlwt
sys.path.append('.')

import test_utils
from datasets import custom_data_loader
from options  import run_model_opts
from models import custom_model
from utils  import logger, recorders
import numpy as np

args = run_model_opts.RunModelOpts().parse()

# args.retrain = "data/Training4shadow/checkp_6.pth.tar"#@6:7.99 wo bear
# args.retrain = "data/legacy/less_atten/checkp_28.pth.tar"

# args.model = 'PS_FCN_atten'
# args.retrain = "data/Training5shadow/checkp_5.pth.tar"#7.5@ epoch 15
# args.retrain = "data/legacy/0227res/checkp_15.pth.tar"#7.5@ epoch 15
# args.use_BN=True

args.pert=False

args.model = 'PS_FCN'
args.retrain = "data/models/PS-FCN_B_S_32.pth.tar.1"
args.use_BN=False




args.test_batch=1
args.benchmark = 'DiLiGenT_main'
args.workers=1
args.bm_dir='data/datasets/DiLiGenT/pmsData'
args.in_img_num = 96
torch.cuda.set_device(1)
log = logger.Logger(args)
repeat=5

def main(args):
    rs=[]
    for i in range(1,repeat):
        print('doing{}'.format(i))
        test_loader = custom_data_loader.benchmarkLoader(args)
        model    = custom_model.buildModel(args)
        # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        recorder = recorders.Records(args.log_dir)
        # r=test_utils.test_split2c(args, 'test', test_loader, model, log, 1, recorder, padding=4, stride=128)
        r=test_utils.test_split_rob(args, 'test', test_loader, model, log, 1, recorder, padding=16, stride=96, noise_level=i)
        # r=test_utils.test(args, 'test', test_loader, model, log, 1, recorder)
        r.append(np.mean(r))
        rs.append(np.stack(r))
    # test_utils.test(args, 'test', test_loader, model, log, 1, recorder)
    # rs=np.stack(rs)
    # every_mean=np.mean(rs,axis=0)
    # mean=np.mean(rs)
    # print(every_mean,mean)

    print(rs)
    # rs=np.stack(rs,axis=0)
    # rs=np.mean(rs,axis=0)

    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('My Worksheet', cell_overwrite_ok=True)
    for i,r in zip(range(repeat),rs):
        for ii,rr in zip(range(10),r):
            worksheet.write(i + 1, ii, label=rr)
    workbook.save('result/robust/pro.xls')

if __name__ == '__main__':
    # torch.manual_seed(args.seed)
    main(args)

