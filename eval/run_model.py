import sys, os, shutil
import torch
sys.path.append('.')
import xlwt
import test_utils
from datasets import custom_data_loader
from options  import run_model_opts
from models import custom_model
from utils  import logger, recorders
import numpy as np
from visdom import Visdom
viz = Visdom()
assert viz.check_connection()

args = run_model_opts.RunModelOpts().parse()


args.model = 'PS_FCN_atten'
# args.retrain = "data/legacy/0227res/checkp_15.pth.tar"#7.5@ epoch 15
# args.retrain = "data/Training5shadow/checkp_23.pth.tar"#7.5@ epoch 15
args.retrain = 'data/checkpoints/atten_no_norm/checkp_25.pth.tar'
# args.retrain = "data/checkpoints/checkp_15.pth.tar"
# args.retrain = "data/checkpoints/legacy.pth.tar"
args.use_BN=False

# # args.retrain = "data/Training4shadow/checkp_6.pth.tar"#@6:7.99 wo bear
# # args.retrain = "data/legacy/less_atten/checkp_28.pth.tar"
#
# args.pert=False

# args.model = 'PS_FCN'
# args.retrain = "data/models/PS-FCN_B_S_32.pth.tar"
# args.use_BN=False
# #44.85


args.in_img_num=96
args.test_batch=1
args.benchmark = 'DiLiGenT_main'
# args.benchmark = 'CNNPS_data'
# args.benchmark = 'Light_stage_dataset'
args.workers=1
# args.retrain = "data/Training4shadow/checkp_5.pth.tar"
# args.retrain = "data/Training/history/2_10/calib/train/checkp_16.pth.tar"
# args.in_img_num = 'set1'
torch.cuda.set_device(0)
repeat=1


def main(args):
    rs=[]
    for i in range(repeat):
        # path="data/Training5shadow/checkp_{}.pth.tar".format(i)
        # args.retrain=path
        log = logger.Logger(args)
        test_loader = custom_data_loader.benchmarkLoader(args)
        model    = custom_model.buildModel(args)
        # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        recorder = recorders.Records(args.log_dir)
        # r=test_utils.test_split2c(args, 'test', test_loader, model, log, 1, recorder, padding=4, stride=128)
        r=test_utils.test_split(args, 'test', test_loader, model, log, 1, recorder, padding=20, stride=100, viz=viz)
        rs.append(r)
    # test_utils.test(args, 'test', test_loader, model, log, 1, recorder)
    # rs=np.stack(rs)
    # every_mean=np.mean(rs,axis=0)
    # mean=np.mean(rs)
    # print(every_mean,mean)

    print(rs)
    # workbook = xlwt.Workbook(encoding='ascii')
    # worksheet = workbook.add_sheet('My Worksheet', cell_overwrite_ok=True)
    # for i,r in zip(range(repeat),rs):
    #     for ii,rr in zip(range(9),r):
    #         worksheet.write(i + 1, ii, label=rr)
    # workbook.save('sparse_pro10.xls')


if __name__ == '__main__':
    # torch.manual_seed(args.seed)
    main(args)

# [9.153817918565538, 8.916984770033094, 8.950070566601223, 9.427849743101332, 10.223750485314262, 9.165475633409288, 9.244231833351982, 9.007659912109375, 9.156129148271349]
# [13.2, 13.2, 13.1, 13.1, 13.0, 12.8, 12.9, 13.2, 13.0]

# mv  [10.5334505551978, 10.267972396898873, 10.442034775697731, 11.042774635025218, 11.20679395410079, 10.772286439243752, 10.997662501999095, 10.627840802639346, 11.456960774675201]