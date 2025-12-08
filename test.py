import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
from evaluation.online_tester import OnlineTester
from evaluation.evaler import Evaler
from utils.config import cfg, cfg_from_file


class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args
        self.device = torch.device("cuda")

        self.setup_logging()
        self.setup_network()
        # 测试集
        self.evaler = Evaler(
            eval_ids=cfg.DATA_LOADER.TEST_ID,
            gv_feat=cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats=cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile=cfg.INFERENCE.TEST_ANNFILE,
            extract_feature_files='./mscoco/feature/CLIP-L-14/clip_feature_test'
        )
        # 验证集
        # self.evaler = Evaler(
        #     eval_ids = cfg.DATA_LOADER.VAL_ID,  # 图像id文件  './mscoco/txt/coco_val_image_id.txt'
        #     gv_feat = cfg.DATA_LOADER.VAL_GV_FEAT,
        #     att_feats = cfg.DATA_LOADER.VAL_ATT_FEATS,
        #     eval_annfile = cfg.INFERENCE.VAL_ANNFILE,
        #     extract_feature_files='./mscoco/feature/CLIP-L-14/clip_feature_val'   # 原始的pure不需要这个特征,直接从s-t中
        # )
        #  在线评估生成
        # self.evaler = OnlineTester(
        #     eval_ids=cfg.DATA_LOADER.TEST_4W_ID, # './mscoco/misc/ids2path_json/coco_val4w_ids2path.json'
        #     gv_feat=cfg.DATA_LOADER.TEST_GV_FEAT,
        #     att_feats=cfg.DATA_LOADER.TEST_ATT_FEATS,
        #     # eval_annfile=cfg.INFERENCE.TEST_ANNFILE,
        #     extract_feature_files='/mnt/usb/CLIP_features/VIT-L-14/VIT-L-test4w') # '/mnt/usb/CLIP_features/VIT-L-14/VIT-L-test4w'

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, 'OfflineTest_' + cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    # 构建网络
    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE)
        # print(model)
        self.model = torch.nn.DataParallel(model).cuda()
        # 加载网络参数
        if self.args.resume > 0:
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
                           map_location=lambda storage, loc: storage)
            )

    def eval(self, epoch):
        res = self.evaler(self.model, 'test' + str(epoch))
        # res = self.evaler(self.model, 'val_' + str(epoch))
        self.logger.info('######## Epoch ' + str(epoch) + ' ########')
        self.logger.info(str(res))

    # 快照文件夹的路径
    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    # parser.add_argument('--folder', dest='folder', default="experiment_MyModels/CLIP_XE_ViTb16", type=str)
    parser.add_argument('--folder', dest='folder', default="experiment_MyModels/AFE/CPMA_AFE_SCST", type=str)
    parser.add_argument("--resume", type=int, default=40)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''这里的代码是用来进行模型的测试的,会输出对应的模型测试评估分数 
    '''
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    tester = Tester(args)
    tester.eval(args.resume)
