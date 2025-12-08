import os
import sys
import torch.nn.functional as F
import cv2
import numpy as np
import torch
import tqdm
import json
from torchvision.transforms import Compose, Normalize, ToTensor
from matplotlib import pyplot as plt
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

import evaluation
import dataloder.data_loader as data_loader
import dataloder.Clip_data_loader as clip_data_lodar
import losses
from losses import CrossEntropy
from utils import utils
from utils.config import cfg
from utils.imageCaption_grad_cam import GradCAM


# model evaluation
class Evaler(object):
    def __init__(
        self,
        eval_ids,
        gv_feat,
        att_feats,
        eval_annfile,
        extract_feature_files
    ):
        super(Evaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        # './mscoco/txt/coco_val_image_id.txt'  Karpathy验证集  5K张图像
        # './mscoco/txt/coco_test_image_id.txt' Karpathy测试集  5K张图像
        # './mscoco/txt/coco_test4w_image_id.txt' MSCOCO在线测试集 4W张图像
        
        # 读取txt文件，读取的为image_ids的list
        # self.eval_ids = np.array(utils.load_ids(eval_ids))
        
        # 端到端训练时，直接读取annotation的json文件，其中包含了图像id和路径
        # 读取json文件，读取的为{image_id: image_path}的dict
        with open(eval_ids, 'r') as f:
            self.ids2path = json.load(f)           # dict {image_id: image_path}
            self.eval_ids = np.array(list(self.ids2path.keys()))  # array of str

        # todo: 进行在线测试的文件id
        
        if extract_feature_files == None :

            self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats)
        else:
            self.eval_loader = clip_data_lodar.load_val(eval_ids, gv_feat, att_feats,extract_feature_files)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)

    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs
        
    def __call__(self, model, rname):
        model.eval()
        
        results = []
        with torch.no_grad():
            for _, (indices, gv_feat, att_feats, att_mask) in enumerate(tqdm.tqdm(self.eval_loader, desc=rname)):
                ids = self.eval_ids[indices]
                gv_feat = gv_feat.cuda()
                att_feats = att_feats.cuda()
                att_mask = att_mask.cuda()
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask)
                if kwargs['BEAM_SIZE'] > 1:
                    seq, _ = model.module.decode_beam(**kwargs)
                else:
                    seq, _ = model.module.decode(**kwargs)
                # 这个使用来把ids转化为单词
                sents = utils.decode_sequence(self.vocab, seq.data)
                for sid, sent in enumerate(sents):
                    # result {'image_id': ***, 'caption': 'word1 word2 word3 ...'}
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent}
                    # print(result)
                    results.append(result)
        # COCO evaluation
        eval_res = self.evaler.eval(results)
        # w/o spice
        # eval_res = self.evaler.eval_no_spice(results)

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))

        model.train()
        return eval_res


class CAM(object):
    def __init__(
            self,
            eval_ids,
            gv_feat,
            att_feats,
            eval_annfile,
            extract_feature_files
    ):
        super(CAM, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        # './mscoco/txt/coco_val_image_id.txt'  Karpathy验证集  5K张图像
        # './mscoco/txt/coco_test_image_id.txt' Karpathy测试集  5K张图像
        # './mscoco/txt/coco_test4w_image_id.txt' MSCOCO在线测试集 4W张图像

        # 读取txt文件，读取的为image_ids的list
        # self.eval_ids = np.array(utils.load_ids(eval_ids))

        # 端到端训练时，直接读取annotation的json文件，其中包含了图像id和路径
        # 读取json文件，读取的为{image_id: image_path}的dict
        with open(eval_ids, 'r') as f:
            self.ids2path = json.load(f)  # dict {image_id: image_path}
            self.eval_ids = np.array(list(self.ids2path.keys()))  # array of str

        if extract_feature_files == None:

            self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats)
        else:
            self.eval_loader = clip_data_lodar.load_val(eval_ids, gv_feat, att_feats, extract_feature_files)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)

        self.feature_map = []
        self.grad = []

    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs

    def __call__(self, model, rname):
        # todo: 这里已经有了整个模型的调用,我们需要注意力机制热力图,将模型进行一次前向传播和梯度回传
        # todo: 1.1 根据得到的图像id显示原始图像
        # todo: 1.2 得到生成每个词的feature_map和grad
        # todo: 1.3 将得到的热力图通过grad_cam算法进行实现
        # todo: 1.4 显示每个单词的注意力图



        for _, (indices, gv_feat, att_feats, att_mask) in enumerate(tqdm.tqdm(self.eval_loader, desc=rname)):
            ids = self.eval_ids[indices]
            gv_feat = gv_feat.cuda()
            att_feats = att_feats.cuda()
            att_mask = att_mask.cuda()
            kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask)
            model.eval()

        # todo:1.1 根据得到的图像id显示原始图像
            image_paths  = [self.ids2path[id] for id in ids]
            image_dir_path = r"mscoco/feature/coco2014"
            for index, image_path in enumerate(image_paths):
                rgb_img = cv2.imread(os.path.join(image_dir_path, image_path), 1)[:, :, ::-1]
                rgb_img = cv2.resize(rgb_img, (224, 224))
                # 显示原图
                plt.figure("Image {}".format(index))  # 图像窗口名称
                plt.imshow(rgb_img)
                plt.title('origin_image {}'.format(index))  # 图像题目
                plt.show()

            # 1.2 得到生成每个词的feature_map和grad
            # target_layers = [model.module.decoder.layers[-1].layer_norm3]
            # target_layers = [model.module.encoder.layers[-1].encoder_attn]
            target_layers = [model.module.decoder.layers[-1].cross_att]
            target_layers[0].register_forward_hook(self.forward_hook)
            target_layers[0].register_full_backward_hook(self.backward_hook)


            # if kwargs['BEAM_SIZE'] > 1:
            #     seq, _ = model.module.decode_beam(**kwargs)
            # else:
            seq, logit = model.module.decode(**kwargs)
            loss = logit
            loss.sum().backward()
            model.zero_grad()

            # todo: 1.3 已经得到了feature_map和相对应的grad


                # 计算梯度的强度
            grad_list = []
            for i in range(len(self.grad)):
                grad_list = grad_list + [grad for grad in self.grad[i]]

            feature_map = torch.stack(self.feature_map).squeeze(1).squeeze(1)
            grad_avgs = torch.mean(torch.stack(grad_list).squeeze(1).squeeze(1), dim=-1)


            heatmap = feature_map.permute(1,0) * grad_avgs
            heatmap = heatmap.permute(1,0)

            heatmap = heatmap.detach().cpu().numpy()
            heatmap = np.mean(heatmap, axis=1)  # 10

            heatmap = np.maximum(heatmap, 0)
            heatmap /= (np.max(heatmap) + 1e-8)


            # todo:1.4 把ids转化为单词, 并显示热力图(每个单词对应一张热力图)
            sents = utils.decode_sequence(self.vocab, seq.data)
            for sid, sent in enumerate(sents):
                # result {'image_id': ***, 'caption': 'word1 word2 word3 ...'}
                # result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent}
                img = cv2.imread(os.path.join(image_dir_path, image_path), 1)[:, :, ::-1]
                img = cv2.resize(img, (224,224))
                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
                # cv2.imshow('1', superimposed_img)
                # cv2.waitKey(0)
                plt.figure("Image")  # 图像窗口名称
                plt.imshow(superimposed_img)
                plt.title('origin_image')  # 图像题目
                plt.show()
                # 单张图片的id和响应描述都在这里
                plt.figure("image_id:{}".format(int(ids[sid])))  # 图像窗口名称
                # plt.imshow(heatmap_img)
                plt.title('caption:{}'.format(sent))  # 图像题目
                # plt.show()




        return "success"


    def forward_hook(self, module, inp, outp):  # 定义hook
        self.feature_map.append(outp)  # 把输出装入字典feature_map


    def backward_hook(self, module, inp, outp):  # 定义hook
        self.grad.append(outp)  # 把输出装入列表grad



class CAM_emb(object):
    def __init__(
            self,
            eval_ids,
            gv_feat,
            att_feats,
            eval_annfile,
            extract_feature_files
    ):
        super(CAM_emb, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)

        # './mscoco/txt/coco_val_image_id.txt'  Karpathy验证集  5K张图像
        # './mscoco/txt/coco_test_image_id.txt' Karpathy测试集  5K张图像
        # './mscoco/txt/coco_test4w_image_id.txt' MSCOCO在线测试集 4W张图像

        # 读取txt文件，读取的为image_ids的list
        # self.eval_ids = np.array(utils.load_ids(eval_ids))

        # 端到端训练时，直接读取annotation的json文件，其中包含了图像id和路径
        # 读取json文件，读取的为{image_id: image_path}的dict
        with open(eval_ids, 'r') as f:
            self.ids2path = json.load(f)  # dict {image_id: image_path}
            self.eval_ids = np.array(list(self.ids2path.keys()))  # array of str

        if extract_feature_files == None:

            self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats)
        else:
            self.eval_loader = clip_data_lodar.load_val_cam(eval_ids, gv_feat, att_feats, extract_feature_files)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)

        self.feature_map = []
        self.grad = []

        self.preprocess_image = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        ])


    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs

    def __call__(self, model, rname):
        # todo: 这里已经有了整个模型的调用,我们需要注意力机制热力图,将模型进行一次前向传播和梯度回传
        # todo: 1.1 根据得到的图像id显示原始图像
        # todo: 1.2 得到生成每个词的feature_map和grad
        # todo: 1.3 将得到的热力图通过grad_cam算法进行实现
        # todo: 1.4 显示每个单词的注意力图



        for _, (indices, gv_feat, att_feats, att_mask) in enumerate(tqdm.tqdm(self.eval_loader, desc=rname)):
            ids = self.eval_ids[indices]
            gv_feat = gv_feat.cuda()
            att_feats = att_feats.cuda()
            att_mask = att_mask.cuda()
            kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask)
            model.eval()

        # todo:1.1 根据得到的图像id显示原始图像
            image_paths  = [self.ids2path[id] for id in ids]
            image_dir_path = r"mscoco/feature/coco2014"
            for index, image_path in enumerate(image_paths):
                rgb_img = cv2.imread(os.path.join(image_dir_path, image_path), 1)[:, :, ::-1]
                rgb_img = cv2.resize(rgb_img, (224, 224))
                rgb_img = np.float32(rgb_img) / 255
                input_tensor = self.preprocess_image(rgb_img)

                # 显示原图
                plt.figure("Image {}".format(index))  # 图像窗口名称
                plt.imshow(rgb_img)
                plt.title('origin_image {}'.format(index))  # 图像题目
                plt.show()

            # 1.2 得到生成每个词的feature_map和grad
            target_layers = [model.module.decoder.cross_layers[-1].layer_norm2]
            # target_layers = [model.module.encoder.layers[-1].layer_norm1]
            # target_layers = [model.module.encoder.layers[-1].layer_norm1,model.module.decoder.layers[-1].cross_att]

            # 初始化cam
            cam = GradCAM(model=model, target_layers=target_layers,
                          # use_cuda=args.use_cuda,
                          reshape_transform=self.reshape_transform_decoder)

            targets = None

            seq , grayscale_cam_list = cam(input_tensor=input_tensor,
                                targets=targets,
                                eigen_smooth=False,
                                aug_smooth=False,
                                **kwargs)

            # 去掉头和尾部的标记
            grayscale_cam_list = [i for i in grayscale_cam_list[0:-3]] + [grayscale_cam_list[-1]]
            sents = utils.decode_sequence(self.vocab, seq.data)
            # 这里拿到每个token的cam
            for i in range(len(grayscale_cam_list)):

                grayscale_cam = grayscale_cam_list[i][0,0,:,:]

                cam_image = show_cam_on_image(rgb_img, grayscale_cam,image_weight=0.6)
                # cv2.imwrite(f'E:\\base-caption\\images\\{image_paths[0]}.jpg', cam_image)

                b,g,r = cv2.split(cam_image)
                cam_image = cv2.merge((r,g,b))
                plt.figure("Image")  # 图像窗口名称
                plt.imshow(cam_image)
                cam_folder = os.path.join(cfg.ROOT_DIR, 'cam')
                if not os.path.exists(cam_folder):
                    os.mkdir(cam_folder)
                if i == len(grayscale_cam_list)-1:
                    plt.title('caption:{}'.format(sents))
                    plt.savefig(os.path.join(cam_folder, image_path.split('/')[-1].split('.')[0]+'.jpg'))

                else:
                    plt.title('token:{}'.format(sents[0].split(' ')[i]))  # 图像题目
                    plt.savefig(os.path.join(cam_folder, image_path.split('/')[-1].split('.')[0]+'_'+sents[0].split(' ')[i]+'_'+str(i+1)+'.jpg'))

                plt.show()


        return "success"

    # 这个Transform是针对于编码器中的层,他将输出的feature_map(B,197,512)-->(B, 512, 14,14)
    def reshape_transform(self,x, height=14, width=14):
        result = x[:, 1:, :].reshape(x.size(0),
                                          height, width, x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    # todo: 这里的transform是针对于解码器而言,解码输出的feature是正对于每一个token进行的注意力
    # 输入x为(B,1,512)应该将其转换为(B,512,1,1)
    def reshape_transform_decoder(self,x,height=1, width=1):
        result = x[:, :, :].reshape(x.size(0),
                                     height, width, x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result



