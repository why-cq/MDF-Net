# MDF-Net
Paper: "Rethinking the Interaction between Multimodal Features for Image Captioning" — Implementation Code

![architecture](./imgs/MDF-Net.jpg)

## 环境要求 (Our Main Enviroment)

+ Python 3.7.4
+ PyTorch 1.5.1
+ TorchVision 0.6.0
+ [coco-caption](https://github.com/tylin/coco-caption)
+ numpy
+ tqdm

## 预处理

### 1. coco-caption

参考coco-caption的 [README.md](./coco_caption/README.md), 主要是需要下载SPICE指标需要使用的[Stanford CoreNLP 3.6.0](http://stanfordnlp.github.io/CoreNLP/index.html)代码和模型。 直接使用脚本下载即可:

```bash
cd coco_caption
bash get_stanford_models.sh
```

### 2. 数据准备

训练和验证过程所需要的重要数据都存储在 __`mscoco`__ 路径下，文件夹组织结构如下：

```
mscoco/
|--feature/
    |--coco2014/
       |--train2014/
       |--val2014/
       |--test2014/
       |--annotations/
|--misc/
|--sent/
|--txt/
```

[MSCOCO 2014](https://cocodataset.org/#download) 数据集的所有源图像和注释文件置于`mscoco/feature/coco2014`路径下。

__注意:__ 为了进一步加快训练速度，也可以将数据集中所有图像的特征提取出来并保存为npz文件，可以在`mscoco/feature`路径下新建目录存储特征文件，训练和验证时需要将数据集读取改为[coco_dataset.py](datasets/coco_dataset.py)和[data_loader.py](datasets/data_loader.py)中的方式。

## 模型训练

*注意: 代码实现主要基于[PureT](https://github.com/232525/PureT)，直接复用了他们的配置文件没做太多修改，所以里面会有一些对我们模型无用的超参数设置。（需要进一步整理删除）*

### 1. XE损失下训练


在训练前，可能还需要检查和修改`config.yml`和`train.sh`文件以适应你的运行环境。然后直接开训：

```
# for XE training

bash MDF-Net/MDF-Net_XE/train.sh
```

### 2. SCST训练

将XE训练后相对较好的模型复制并存储于`experiment_MyModels/MDF-Net_SCST/snapshot/`中。然后继续训练：

```bash
# for SCST training
bash experiment_MyModels/MDF-Net_SCST/train.sh
```

## 模型测试

选择最好的模型进行测试即可
```bash
CUDA_VISIBLE_DEVICES=0 python main_test.py --folder experiment_MyModels/MDF-Net_SCST/ --resume 27
```

| BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|-------:|-------:|-------:|--------:|------:|------:|
|   83.0 |   42.1 |   30.7 |    61.0 | 141.6 |  24.6 |