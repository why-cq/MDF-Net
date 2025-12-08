import os
import random

import PIL.Image
import numpy as np
import torch
import torch.utils.data as data
from utils import utils
import pickle

import json
import cv2
from PIL import Image

# 图像读取预处理单元
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch
from torchvision.datasets import CocoCaptions
# from transformers import CLIPProcessor, CLIPModel
import clip


# 加载CLIP模型和处理器
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


# from timm.data.transforms import _pil_interp

class ClipCocoDataset(data.Dataset):
    def __init__(
            self,
            image_ids_path,
            input_seq,
            target_seq,
            gv_feat_path,
            att_feats_folder,
            seq_per_img,
            max_feat_num,
            extract_feature_files,

    ):
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        # self.image_ids = utils.load_lines(image_ids_path)
        # 此处image_ids_path为ids2path的映射dict
        with open(image_ids_path, 'r') as f:
            self.ids2path = json.load(f)  # dict {image_id: image_path}
            self.image_ids = list(self.ids2path.keys())  # list of str

        self.att_feats_folder = att_feats_folder if len(att_feats_folder) > 0 else None
        self.gv_feat = pickle.load(open(gv_feat_path, 'rb'), encoding='bytes') if len(gv_feat_path) > 0 else None

        self.extract_feature_folder = extract_feature_files if len(extract_feature_files) > 0 else '../mscoco/feature/CLIP-B-16-196/clip_feature_train'

        # 构建图像预处理单元
        # transforms.Resize((384, 384), interpolation=_pil_interp('bicubic')),
        self.transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
        )

        # 把输入的tokens存在了文件里面
        if input_seq is not None and target_seq is not None:
            self.input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')
            self.target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')
            self.seq_len = len(self.input_seq[self.image_ids[0]][0, :])
        else:
            self.seq_len = -1
            self.input_seq = None
            self.target_seq = None

    def set_seq_per_img(self, seq_per_img):
        self.seq_per_img = seq_per_img

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(
        #     torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        image_id = self.image_ids[index]
        image_path = self.ids2path[image_id]
        indices = np.array([index]).astype('int')

        # 获取全局特征
        # if self.gv_feat is not None:
        #     gv_feat = self.gv_feat[image_id]
        #     gv_feat = np.array(gv_feat).astype('float32')
        # else:
        #     gv_feat = np.zeros((1, 1))

        # 此处att_feats_folder为coco数据集源图像保存路径，而非预训练特征保存路径
        if not self.extract_feature_folder: # 使用原始的CLIPAPI进行特征提取(时间开销太大了)
            # att_feats = np.load(os.path.join(self.att_feats_folder, str(image_id) + '.npz'))['feat']
            # att_feats = np.array(att_feats).astype('float32')

            # 读取图像，并进行预处理
            model, transform = clip.load("ViT-B/16", jit=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            image_path = self.ids2path[image_id]

            image = transform(Image.open(os.path.join(self.att_feats_folder, image_path)).convert("RGB")).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            # 获取局部特征和全局特征
            att_feats, gv_feat = model.encode_image(image)
            att_feats = np.array(att_feats).astype('float32')
            gv_feat = np.array(gv_feat).astype('float32')
        else:
            # todo： 读取文件夹，然后把特征提取出来
            feats = np.load(os.path.join(self.extract_feature_folder, str(image_id) + '.npz'))
            att_feats = feats['features']
            att_feats = np.array(att_feats).astype('float32')  # vit-b/16  shape:(49,768)
            gv_feat = feats['g_feature']
            gv_feat = np.array(gv_feat).astype('float32')   ## vit-b/16  shape:(1,512)

        if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num:
            att_feats = att_feats[:self.max_feat_num, :]

        if self.seq_len < 0:
            return indices, gv_feat, att_feats

        input_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        target_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')

        n = len(self.input_seq[image_id])
        if n >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_img)
        else:
            sid = n
            ixs = random.sample(range(n), self.seq_per_img - n)
            input_seq[0:n, :] = self.input_seq[image_id]
            target_seq[0:n, :] = self.target_seq[image_id]

        for i, ix in enumerate(ixs):
            input_seq[sid + i] = self.input_seq[image_id][ix, :]
            target_seq[sid + i] = self.target_seq[image_id][ix, :]
            # indices 为个体item的下标  input_seq,target_seq是否从文件中加载
            # att_feats 是transforms.Resize((384, 384) 后的图像3，384，384
        return indices, input_seq, target_seq, gv_feat, att_feats