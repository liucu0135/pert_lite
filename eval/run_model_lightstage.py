import sys, os, shutil
import torch
sys.path.append('.')

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
# args.retrain = "data/Training5shadow/checkp_25.pth.tar"#7.5@ epoch 15
# args.retrain = "data/checkpoints/legacy.pth.tar"
args.retrain = 'data/checkpoints/atten_no_norm/checkp_11.pth.tar'
args.use_BN=False
#
# args.pert=False

# args.model = 'PS_FCN_run'
# args.retrain = "data/models/PS-FCN_B_S_32.pth.tar"
# args.use_BN=False

#44.85



args.test_batch=1
# args.benchmark = 'DiLiGenT_main'
args.benchmark = 'Light_stage_dataset'
# args.benchmark = 'CNNPS_data'
args.workers=1
args.in_img_num = 119
torch.cuda.set_device(0)



def main(args):
    rs=[]
    for i in range(5,6):
        # path="data/Training5shadow/checkp_{}.pth.tar".format(i)
        # args.retrain=path
        log = logger.Logger(args)
        test_loader = custom_data_loader.benchmarkLoader(args)
        model    = custom_model.buildModel(args)
        # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        recorder = recorders.Records(args.log_dir)
        # r=test_utils.test_split2c(args, 'test', test_loader, model, log, 1, recorder, padding=4, stride=128)
        test_utils.estimate(args, i, test_loader, model, log, 1, recorder, padding=10, stride=100, split=True, vis=viz)


if __name__ == '__main__':
    # torch.manual_seed(args.seed)
    main(args)

# [9.153817918565538, 8.916984770033094, 8.950070566601223, 9.427849743101332, 10.223750485314262, 9.165475633409288, 9.244231833351982, 9.007659912109375, 9.156129148271349]
# [13.2, 13.2, 13.1, 13.1, 13.0, 12.8, 12.9, 13.2, 13.0]

# mv  [10.5334505551978, 10.267972396898873, 10.442034775697731, 11.042774635025218, 11.20679395410079, 10.772286439243752, 10.997662501999095, 10.627840802639346, 11.456960774675201]