import os

import skimage
from torchvision import transforms
import matplotlib.lines as mlines
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from clip import clip
from models import ACT_Transformer_adaptive_336
from utils.config import cfg_from_file, cfg

import torch.utils.data.dataloader

def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def visualize_heatmap(image, mask):
    '''
    Save the heatmap of ones
    '''
    masks = norm_image(mask).astype(np.uint8)
    # mask->heatmap
    heatmap = cv2.applyColorMap(masks, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # same shape

    # merge heatmap to original image
    cam = 0.4 * heatmap + 0.6 * np.float32(image)
    return cam


def heat_show(img_path):
    img = Image.open(img_path).convert('RGB')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
    ])
    data = transforms(img).unsqueeze(0)

    # 加载预训练模型
    model = torchvision.models.vgg11_bn(pretrained=True)
    # model = torchvision.models.resnet50()
    model.eval()

    features = model.features(data)
    features.retain_grad()  # 保留特征层的梯度 或者用register_hook也可以取到 但比较麻烦
    t = model.avgpool(features)
    t = t.reshape(1, -1)
    # 经过分类头后得到1000个类别的概率
    output = model.classifier(t)[0]

    # 预测得分最高的那一类对应的输出值
    pred = torch.argmax(output).item()
    pred_class = output[pred]

    pred_class.backward()  # 计算梯度
    grads = features.grad  # 获取梯度
    '''
    计算每层特征图的平均梯度 每层特征图乘上该层的平均梯度 最后所有层再平均成一层原始热力图 经ReLu激活后 压缩至(0,1)
    features:(1, 512, h, w) grad:(1, 512, h, w) avg_grads:(512)
    为了不用循环 将avg_grads扩充成(h, w, 512)->(512, h, w) 与 features直接相乘
    '''
    # features = torch.rand((512,224,224))
    # grads = torch.rand((512,224,224))
    features = features[0]  # (512,7,7)
    avg_grads = torch.mean(grads[0], dim=(1, 2))  # (512)
    avg_grads = avg_grads.expand(features.shape[1], features.shape[2], features.shape[0]).permute(2, 0, 1)
    features *= avg_grads  # 512*7*7

    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)  # 7*7

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    # 将热力图的大小调整为与原始图像相同 乘255转成灰度图 映射成彩图 最后和原图按比例叠加
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
    # cv2.imshow('1', superimposed_img)
    # cv2.waitKey(0)
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(superimposed_img)
    plt.title('origin_image')  # 图像题目title
    plt.show()


def image_to_caprion(image_path):
    # 1.首先使用CLIP进行特征提取
    device = "cuda"
    model, transform = clip.load("ViT-L/14@336px", jit=False, device=device)
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device=device)
    model.eval()
    local_feat, global_feat = model.encode_image(image)
    # 显示图像
    # 将tensor转换为PIL图像,进过增强后的图像
    # to_pil = transforms.ToPILImage()
    # pil_image = to_pil(image.squeeze(0).cpu())
    # plt.imshow(pil_image)
    plt.imshow(Image.open(image_path))
    plt.axis('off')
    plt.show()

    # 2.加载模型
    caption_model = torch.nn.DataParallel(ACT_Transformer_adaptive_336()).cuda()
    caption_model.load_state_dict(torch.load(r"E:\base-caption\experiment_MyModels\ACT\snapshot\caption_model_34.pth"))

    # 3.进行描述生成
    att_mask = torch.ones(1, 576).to(device)
    kwargs = make_kwargs(global_feat.type(torch.float32).cuda(), local_feat.type(torch.float32).cuda(), att_mask.cuda())

    if kwargs['BEAM_SIZE'] > 1:
        seq, _ = caption_model.module.decode_beam(**kwargs)
    else:
        seq, _ = caption_model.module.decode(**kwargs)
    # 这个使用来把ids转化为单词
    vocab = load_vocab(r"E:\base-caption\mscoco\txt\coco_vocabulary.txt")
    sents = decode_sequence(vocab, seq.data)

    print("图像描述模型的输出为:", sents[0])


def visualize_attention(image_path, alpha_weights):
    # load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get Height, Width of image
    H, W = img.shape[:2]
    dH, dW = H // 4, W // 4

    alpha_weights = alpha_weights
    # keep the top-k weights
    k = 20
    _tmp = alpha_weights.reshape(-1)
    top_k = _tmp[_tmp.argsort()[-k]]
    alpha_weights = alpha_weights * (alpha_weights >= top_k)

    # resize the weights from (12, 12) to (H/4, W/4)
    alpha_weights = skimage.transform.resize(alpha_weights, (dH, dW))
    # expand the weights to the raw size of image
    alpha_weights = skimage.transform.pyramid_expand(alpha_weights, upscale=4, sigma=20)

    #  draw image and weights
    plt.plot()
    plt.imshow(img)
    plt.imshow(alpha_weights, alpha=0.75, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()


def make_kwargs(gv_feat, att_feats, att_mask):
    kwargs = {}
    kwargs['INDICES'] = [0]
    kwargs['GV_FEAT'] = gv_feat
    kwargs['ATT_FEATS'] = att_feats
    kwargs['ATT_FEATS_MASK'] = att_mask
    kwargs['BEAM_SIZE'] = 1
    kwargs['GREEDY_DECODE'] = True
    return kwargs


def load_vocab(path):
    vocab = ['.']
    with open(path, 'r') as fid:
        for line in fid:
            vocab.append(line.strip())
    return vocab


def decode_sequence(vocab, seq):
    N, T = seq.size()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t]
            if ix == 0:
                break
            words.append(vocab[ix])
        sent = ' '.join(words)
        sents.append(sent)
    return sents


def draw_bar_cider():
    colors = ['#1E90FF', '#ff7f0e', '#fae768', 'r', '#FF7F50']
    # colors = ['#fae768', '#87e885', '#3cb9fc', '#73abf5', '#cb9bff']
    # colors = ['#fae768', '#87e885', '#3cb9fc', '#73abf5','#cb9bff']
    # colors = ['#fae768', '#87e885', '#3cb9fc', '#73abf5','#cb9bff']
    # colors = ['#fae768', '#87e885', '#3cb9fc', '#73abf5','#cb9bff']
    labels = ['0~50', '50~100', '100~150', '150~200', '200~500']
    data = [434, 1249, 1350, 879, 1088]

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'r','b']
    # labels = ['0~20', '20~40', '40~60', '60~80','80~100']
    # data = [3, 394, 2019, 1948, 636]

    # 生成柱状图
    fig, ax = plt.subplots(figsize=(16, 16))
    x = np.arange(len(labels))
    bars1 = plt.bar(x, data, align='center', alpha=0.6, color=colors)

    # 设置刻度标签
    plt.xticks(x, labels, fontsize=30, weight='bold')
    plt.yticks(fontsize=30, weight='bold')

    # 给每个柱子上添加标注
    for i, b in enumerate(bars1):
        height = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, height, '{}'.format(height), ha='center', va='bottom', fontsize=40,
                weight='bold')

    ax.set_title("CIDEr Score (%)", fontsize=45, y=1.04, weight='bold')
    ax.set_xlabel("Score Range", fontsize=45, y=-0.5, labelpad=20, weight='bold')
    ax.set_ylabel("Counts", fontsize=45, weight='bold')
    plt.savefig('./images/zhu/CIDER.svg', format='svg')
    plt.show()
    print("ok")


def draw_bar_rouge():
    colors = ['#1E90FF', '#ff7f0e', '#fae768', 'r', '#FF7F50']
    labels = ['0~20', '20~40', '40~60', '60~80', '80~100']
    data = [3, 394, 2019, 1948, 636]

    # 生成柱状图
    fig, ax = plt.subplots(figsize=(15, 15))
    x = np.arange(len(labels))
    bars1 = plt.bar(x, data, align='center', alpha=0.6, color=colors)

    # 设置刻度标签
    plt.xticks(x, labels, fontsize=30, weight='bold')
    plt.yticks(fontsize=30, weight='bold')

    # 给每个柱子上添加标注
    for i, b in enumerate(bars1):
        height = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, height, '{}'.format(height), ha='center', va='bottom', fontsize=30,
                weight='bold')

    ax.set_title("ROUGE_L Score (%)", fontsize=35, weight='bold')
    ax.set_xlabel("Score Range", fontsize=35, weight='bold')
    ax.set_ylabel("Counts", fontsize=35, weight='bold')
    plt.savefig('./images/rouge.png')
    plt.show()
    print("ok")


def draw_bar_B1():
    colors = ['#1E90FF', '#ff7f0e', '#2ca02c', 'r', '#FF7F50']
    labels = ['0~20', '20~40', '40~60', '60~80', '80~100']
    data = [1, 39, 442, 1859, 2659]

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'r','b']
    # labels = ['0~20', '20~40', '40~60', '60~80','80~100']
    # data = [3, 394, 2019, 1948, 636]

    # 生成柱状图
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.arange(len(labels))
    bars1 = plt.bar(x, data, align='center', alpha=0.7, color=colors)

    # 设置刻度标签
    plt.xticks(x, labels, fontsize=15)
    plt.yticks(fontsize=15)

    # 给每个柱子上添加标注
    for i, b in enumerate(bars1):
        height = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, height, '{}'.format(height), ha='center', va='bottom', fontsize=15)

    ax.set_title("Bleu_1 Score (%)", fontsize=20)
    ax.set_xlabel("Score Range", fontsize=20)
    ax.set_ylabel("Counts", fontsize=20)
    plt.savefig('./images/B1.png')
    plt.show()
    print("ok")


def draw_bar_B4():
    colors = ['#1E90FF', '#ff7f0e', '#2ca02c', 'r', '#FF7F50']
    labels = ['0~20', '20~40', '40~60', '60~80', '80~100']
    data = [2269, 587, 1087, 678, 379]

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'r','b']
    # labels = ['0~20', '20~40', '40~60', '60~80','80~100']
    # data = [3, 394, 2019, 1948, 636]

    # 生成柱状图
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.arange(len(labels))
    bars1 = plt.bar(x, data, align='center', alpha=0.7, color=colors)

    # 设置刻度标签
    plt.xticks(x, labels, fontsize=15)
    plt.yticks(fontsize=15)

    # 给每个柱子上添加标注
    for i, b in enumerate(bars1):
        height = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, height, '{}'.format(height), ha='center', va='bottom', fontsize=15)

    ax.set_title("Bleu_4 Score (%)", fontsize=20)
    ax.set_xlabel("Score Range", fontsize=20)
    ax.set_ylabel("Counts", fontsize=20)
    plt.savefig('./images/B4.png')
    plt.show()
    print("ok")


def draw_bar_Meteor():
    colors = ['#1E90FF', '#ff7f0e', '#2ca02c', 'r', '#FF7F50']
    labels = ['0~20', '20~40', '40~60', '60~80', '80~100']
    data = [540, 3491, 877, 0, 92]

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'r','b']
    # labels = ['0~20', '20~40', '40~60', '60~80','80~100']
    # data = [3, 394, 2019, 1948, 636]

    # 生成柱状图
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.arange(len(labels))
    bars1 = plt.bar(x, data, align='center', alpha=0.7, color=colors)

    # 设置刻度标签
    plt.xticks(x, labels, fontsize=15)
    plt.yticks(fontsize=15)

    # 给每个柱子上添加标注
    for i, b in enumerate(bars1):
        height = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, height, '{}'.format(height), ha='center', va='bottom', fontsize=15)

    ax.set_title("METEOR Score (%)", fontsize=20)
    ax.set_xlabel("Score Range", fontsize=20)
    ax.set_ylabel("Counts", fontsize=20)
    plt.savefig('./images/METEOR.png')
    plt.show()
    print("ok")


def draw_bar_SPICE():
    colors = ['#1E90FF', '#ff7f0e', '#2ca02c', 'r', '#FF7F50']
    labels = ['0~20', '20~40', '40~60', '60~80', '80~100']
    data = [1691, 2853, 442, 14, 0]

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'r','b']
    # labels = ['0~20', '20~40', '40~60', '60~80','80~100']
    # data = [3, 394, 2019, 1948, 636]

    # 生成柱状图
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.arange(len(labels))
    bars1 = plt.bar(x, data, align='center', alpha=0.7, color=colors)

    # 设置刻度标签
    plt.xticks(x, labels, fontsize=15)
    plt.yticks(fontsize=15)

    # 给每个柱子上添加标注
    for i, b in enumerate(bars1):
        height = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, height, '{}'.format(height), ha='center', va='bottom', fontsize=15)

    ax.set_title("SPICE Score (%)", fontsize=20)
    ax.set_xlabel("Score Range", fontsize=20)
    ax.set_ylabel("Counts", fontsize=20)
    plt.savefig('./images/SPICE.png')
    plt.show()
    print("ok")


def draw_zhexiantu():
    # 数据
    x = np.linspace(0, 10, 5)  # x轴数据
    # y = np.array([83.533, 83.556, 83.630, 83.474, 83.028])  # off_B1
    # y = np.array([42.727, 42.739, 42.414, 42.427, 42.198])  # off_B4
    # y = np.array([30.937, 30.882, 30.873, 30.710, 30.760])  # off_M
    # y = np.array([61.346, 61.393, 61.319, 61.180, 61.079])  # off_R
    # y = np.array([143.680, 143.360, 143.282, 143.205, 142.572])  # off_C
    # y = np.array([83.700, 83.400, 83.200, 83.200, 83.100])  # on_B1
    # y = np.array([42.400, 42.300, 41.800, 41.900, 41.900])  # on_B4
    # y = np.array([30.800, 30.600, 30.600, 30.500, 30.700])  # on_M
    # y = np.array([61.100, 61.000, 60.900, 60.800, 60.900])  # on_R
    # y = np.array([138.700, 138.100, 138.100, 138.000, 138.000])  # on_C
    # y = np.array([97.000, 97.000, 96.900, 96.900, 96.800])  # on_B1_c40
    y = np.array([75.700, 75.300, 75.000, 75.000, 74.800])  # on_B4_c40
    # y = np.array([40.700, 40.400, 40.400, 40.200, 40.500])  # on_M_c40
    # y = np.array([76.900, 76.600, 76.300, 76.300, 76.500])  # on_R_c40
    # y = np.array([140.200, 140.200, 140.500, 140.200, 139.700])  # on_C_c40

    # 创建图形和子图
    fig, ax = plt.subplots()

    # 绘制红色五角星标记的折线
    ax.plot(x, y, 'r-', marker='*', markersize=30, linewidth=4)

    # 添加虚线
    ax.axhline(y=75.3, color='k', linestyle='--', linewidth=4, label='CAST')  # CAST COS-Net

    # 设置y轴刻度
    ax.set_yticks([75.0,75.5,76.0])
    ax.set_yticklabels([75.0,75.5,76.0], fontsize=17, weight='bold')

    # 隐藏x轴刻度标签
    ax.set_xticks([])

    # 添加图例并调整位置和大小
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=25)

    # 调整图例框线粗细
    legend.get_frame().set_linewidth(1.5)

    # 显示图形
    plt.savefig('./images/five/on_B4_c40.svg', format='svg')
    plt.show()


def draw_all(ax, labels, data, title):
    colors = ['#1E90FF', '#ff7f0e', '#fae768', 'r', '#FF7F50']

    x = np.arange(len(labels))
    bars = ax.bar(x, data, align='center', alpha=0.6, color=colors)

    ax.set_title(title, fontsize=40, weight='bold', y=1.04)
    ax.set_xlabel("Score Range", fontsize=35, weight='bold', y=-0.7, labelpad=20)
    ax.set_ylabel("Counts", fontsize=35, weight='bold')

    # 设置刻度标签
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=30, weight='bold')
    ax.set_yticklabels([int(y) for y in ax.get_yticks()], fontsize=30, weight='bold')

    # 给每个柱子上添加标注
    for i, b in enumerate(bars):
        height = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, height, '{}'.format(height), ha='center', va='bottom', fontsize=35,
                weight='bold')


if __name__ == '__main__':
    # image = np.array(Image.open(r"E:\base-caption\images\dog.jpg"))
    # heat_show(r"E:\base-caption\images\dog.jpg")

    # 显示原图
    # plt.figure("Image")  # 图像窗口名称
    # plt.imshow(image)
    # plt.title('origin_image')  # 图像题目
    # plt.show()

    # # 单张图像调用模型,生成描述,获取中间层注意力权重
    # cfg_from_file(os.path.join(r"E:\base-caption\experiment_MyModels\ACT", 'config.yml'))
    # cfg.ROOT_DIR = r"E:\base-caption\experiment_MyModels\ACT"
    # image_path = r"E:\base-caption\mscoco\feature\coco2014\val2014\COCO_val2014_000000355228.jpg"
    # image_to_caprion(image_path)

    # 5000张图像的分数range
    # draw_bar_Meteor()
    # draw_bar_rouge()
    # draw_bar_B1()
    # draw_bar_B4()
    # draw_bar_cider()
    # draw_bar_SPICE()

    # draw_zhexiantu()

    # ---数据和标签列表
    # labels_list = [
    #     ['0~20', '20~40', '40~60', '60~80', '80~100'],
    #     ['0~20', '20~40', '40~60', '60~80', '80~100'],
    #     ['0~20', '20~40', '40~60', '60~80', '80~100'],
    #     ['0~20', '20~40', '40~60', '60~80', '80~100'],
    #     ['0~50', '50~100', '100~150', '150~200', '200~500'],
    #     ['0~20', '20~40', '40~60', '60~80', '80~100']]
    #
    # data_list = [[1, 39, 442, 1859, 2659],
    #              [2269, 587, 1087, 678, 379],
    #              [540, 3491, 877, 0, 92],
    #              [3, 394, 2019, 1948, 636],
    #              [434, 1249, 1350, 879, 1088],
    #              [434, 1249, 1350, 879, 1088]]
    #
    # titles = ["Bleu_1 Score (%)", "Bleu_4 Score (%)", "Meteor Score (%)", "ROUGE_L Score (%)", "CIDEr Score (%)",
    #           "SPICE Score (%)"]
    # # 创建子图
    # fig, axs = plt.subplots(2, 3, figsize=(55, 35))
    # # 设置每个子图的大小
    # axs[0, 0].set_position([0.05, 0.55, 0.31, 0.45])
    # axs[0, 1].set_position([0.4, 0.55, 0.31, 0.45])
    # axs[0, 2].set_position([0.75, 0.55, 0.31, 0.45])
    # axs[1, 0].set_position([0.05, 0.05, 0.31, 0.45])
    # axs[1, 1].set_position([0.4, 0.05, 0.31, 0.45])
    # axs[1, 2].set_position([0.75, 0.05, 0.31, 0.45])
    # # plt.subplots_adjust(hspace=10, wspace=2)
    #
    # # 遍历每个子图并画柱状图
    # for i, ax in enumerate(axs.flat):
    #     draw_all(ax, labels_list[i], data_list[i], titles[i])
    #
    # # 调整子图之间的间距
    # plt.tight_layout()
    #
    # # 保存图像
    # plt.savefig('./images/All_metric_1.pdf')
    #
    # # 显示图像
    # plt.show()
    # print("ok")

    draw_zhexiantu()
