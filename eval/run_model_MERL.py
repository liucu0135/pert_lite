import sys, os, shutil
import torch
import csv
sys.path.append('.')

import test_utils
from datasets import custom_data_loader
from options  import run_model_opts
from models import custom_model
from utils  import logger, recorders



# persons=[('Lata',22,45),('Anil',21,56),('John',20,60)]
# csvfile=open('persons.csv','w', newline='')
# obj=csv.writer(csvfile)
# for person in persons:
# 	obj.writerow(person)
# csvfile.close()




args = run_model_opts.RunModelOpts().parse()



args.benchmark = 'MERL'
args.bm_dir = 'data/datasets/merl_test'
args.pert=False

# args.retrain = "data/Training4shadow/calib/train/checkp_30.pth.tar"
# args.retrain = "data/legacy/less_atten/checkp_28.pth.tar"

# args.model = 'PS_FCN_run'
# args.retrain = "data/models/PS-FCN_B_S_32.pth.tar.1"
# args.use_BN=False

args.model = 'PS_FCN_atten'
args.retrain = "data/Training4shadow/checkp_15.pth.tar"#7.5@ epoch 15
args.use_BN=True



args.test_batch=1
args.workers=1
# args.retrain = "data/Training4shadow/checkp_5.pth.tar"
# args.retrain = "data/Training/history/2_10/calib/train/checkp_16.pth.tar"
args.in_img_num = 96
torch.cuda.set_device(1)
log = logger.Logger(args)


def main(args):
    rs=[]
    # path="data/Training4shadow/checkp_{}.pth.tar".format(i)
    # args.retrain=path
    test_loader = custom_data_loader.benchmarkLoader(args)
    model    = custom_model.buildModel(args)
    recorder = recorders.Records(args.log_dir)
    # r=test_utils.test_split(args, 'test', test_loader, model, log, 1, recorder, padding=4, stride=160)
    # errors=test_utils.test(args, 'test', test_loader, model, log, 1, recorder)
    errors=test_utils.test2c(args, 'test', test_loader, model, log, 1, recorder)

    csvfile = open('names.csv', 'w', newline='')
    obj = csv.writer(csvfile)
    for person in errors:
        obj.writerow(person)
    csvfile.close()


if __name__ == '__main__':
    # torch.manual_seed(args.seed)
    main(args)

