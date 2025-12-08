import skimage.io as io
import json
import os
import sys
import tempfile
import json
from json import encoder

import torch
from PIL import Image
from torchvision import transforms

from matplotlib import pyplot as plt


from clip import clip
from coco_caption.pycocoevalcap.cider.cider import Cider
# sys.path.append(cfg.INFERENCE.COCO_PATH)
# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap




def eval_all():
    with open("../mscoco/evl/gts.json","r") as f:
        gts = json.load(f)

    with open("../mscoco/evl/res.json","r") as f:
        res = json.load(f)


    coco = COCO("../mscoco/misc/captions_test5k.json")

    cocoRes = coco.loadRes(r"../mscoco/evl/res.json")

    cocoEval = COCOEvalCap(coco, cocoRes)
    # cocoEval.evaluate()
    cocoEval.evaluate_no_spice()

    # 可视化所有的CIDEr分数
    # plot score histogram
    ciderScores = [eva['CIDEr'] for eva in cocoEval.evalImgs]
    plt.hist(ciderScores)
    plt.title('Histogram of CIDEr Scores', fontsize=20)
    plt.xlabel('CIDEr score', fontsize=20)
    plt.ylabel('result counts', fontsize=20)
    plt.show()


    # 进行低分数的结果过滤
    evals = [eva for eva in cocoEval.evalImgs if eva['CIDEr'] < 0.2]
    print('ground truth captions')

    for evl_img in evals:
        imgId = evl_img['image_id']
        annIds = coco.getAnnIds(imgIds=imgId)
        anns = coco.loadAnns(annIds)
        print("图像真实描述")
        print("图像ID为:",imgId)
        for caption in anns:
            print(caption["caption"])

        print('该图像的 CIDEr score %0.3f)' % (evl_img['CIDEr']))
        annIds = cocoRes.getAnnIds(imgIds=imgId)
        anns = cocoRes.loadAnns(annIds)
        for caption in anns:
            print("模型生成的描述:",caption["caption"])
            print('\n')

        img = coco.loadImgs(imgId)[0]
        I = io.imread('%s/%s/%s' % ("../mscoco/feature/coco2014", "val2014", img['file_name']))
        plt.imshow(I)
        plt.axis('off')
        plt.show()


# 单独进行cider值的计算,这里我们可以进行不同图片的分数的探索,看看究竟是哪些图片的描述不符合
def compute_CIDER():
    # 这里的gts和res是测试后Token后的句子
    with open("../mscoco/evl/cider/gts.json", "r") as f:
        gts = json.load(f)

    with open("../mscoco/evl/cider/res.json", "r") as f:
        res = json.load(f)

    cider = Cider()


    gts_sample = {}
    res_sample = {}
    for index, image_id in enumerate(gts.keys()):
        # print("原始图像id : ", image_id)
        gts_sample[image_id] = gts[image_id]
        res_sample[image_id] = res[image_id]

        # for _,caption in enumerate(gts[image_id]):
            # print("gts",str(_+1),":", caption)

        # print("ours: ",res[image_id][0])

        # if (index+1)%3 == 0 :
        #     score, scores = cider.compute_score(gts_sample, res_sample)
        #     gts_sample = {}
        #     res_sample = {}
        #


    # 计算5000个样本的分数
    score, scores = cider.compute_score(gts, res)
    print("ok")



if __name__ == '__main__':
    eval_all()
    # compute_CIDER()


