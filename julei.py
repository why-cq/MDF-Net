import json
import math
import os
from math import sqrt

from scipy.special import kl_div
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import genextreme, gennorm, gaussian_kde
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances


## 将5000张验证集的clip-L特征全局平均池化,得到相应的特征梯度值保存为json文件
def feature_gradient_to_json():
    # 文件路径  5000 张val集图像的clip特征
    folder_path = r"E:\base-caption\mscoco\feature\CLIP-L-14\clip_feature_val"
    file_list = os.listdir(folder_path)
    save_file_path = "mscoco/gradient/avg_1/val_gradient_norm.json"  # 保存特征梯度先验json文件路径

    val_feature_gradient = {}
    for file_name in file_list:
        features = np.load(os.path.join(folder_path,file_name))

        grid_f = torch.tensor(np.array(features['features']).astype('float32'))
        g_f = np.array(features['g_feature']).astype('float32')


        # input = torch.randn(5, 256, 512)
        norm = nn.LayerNorm(1024)
        AvgPoll = nn.AdaptiveAvgPool2d(1)
        output = AvgPoll(norm(grid_f)).item()

        val_feature_gradient[file_name.split('.')[0]] = output

    with open(save_file_path, 'w') as json_file:
        json.dump(val_feature_gradient, json_file,indent=4)

    print("特征梯度先验json文件保存成功,保存地址为:" + save_file_path)


# 可视化对通道这个维度进行全局平均池化后的特征图
def show_clip_feature_map():
    # 文件路径  5000 张val集图像的clip特征
    folder_path = r"E:\base-caption\mscoco\feature\CLIP-L-14\clip_feature_val"
    file_list = os.listdir(folder_path)

    # 得到文件夹中的所有特征文件
    for file_name in file_list:
        features = np.load(os.path.join(folder_path, file_name))

        grid_f = torch.tensor(np.array(features['features']).astype('float32'))
        # g_f = np.array(features['g_feature']).astype('float32')
        # 可视化16*16特征图, 对通道这个维度进行全局平均池化
        # norm = nn.LayerNorm(1024)
        avg = nn.AdaptiveAvgPool2d((256, 1))
        # output = avg(norm(grid_f.unsqueeze(0))).reshape(256)
        output = avg(grid_f.unsqueeze(0)).reshape(256)
        # numpy_output = output.view(16, 16).detach().numpy()
        numpy_output = output.view(16,16).numpy()




    # 特征图可视化  interpolation='bicubic'   表示使用的差值方法
        norm = mcolors.Normalize(vmin=numpy_output.min(), vmax=numpy_output.max())
        # plt.imshow(numpy_output,cmap='hot',norm=norm)
        plt.imshow(numpy_output,cmap='plasma',norm=norm,interpolation='bicubic')
        cbar = plt.colorbar()
        cbar.set_label('Data Value', rotation=270, labelpad=20)
        plt.title('file_name:' +  file_name)
        plt.show()
    print("ok")

def show_feature_map(feature_map, HW, channel = 1024):
    H = W = int(sqrt(HW))
    print(feature_map.shape)
    grid_f = feature_map.detach().cpu()

    # 可视化16*16特征图, 对通道这个维度进行全局平均池化
    avg = nn.AdaptiveAvgPool1d(1)
    output = avg(grid_f).reshape(HW)
    numpy_output = output.view(H, W).numpy()

    # 特图可视化  interpolation='bicubic'   表示使用的差值方法
    norm = mcolors.Normalize(vmin=numpy_output.min(), vmax=numpy_output.max())
    # plt.imshow(numpy_output,cmap='hot',norm=norm)
    plt.imshow(numpy_output, cmap='viridis', norm=norm, interpolation='bicubic')
    cbar = plt.colorbar()
    cbar.set_label('Data Value', rotation=270, labelpad=20)
    plt.title('Attention map')
    plt.show()
    print("ok")
# 将中间层特征图可视化
def show_encoder_feature_map(feature_map):
    print(feature_map.shape)
    grid_f = feature_map.cpu()


    # 可视化16*16特征图, 对通道这个维度进行全局平均池化
    avg = nn.AdaptiveAvgPool2d((256, 1))
    output = avg(grid_f.unsqueeze(0)).reshape(256)
    numpy_output = output.view(16,16).numpy()


    # 特图可视化  interpolation='bicubic'   表示使用的差值方法
    norm = mcolors.Normalize(vmin=numpy_output.min(), vmax=numpy_output.max())
    # plt.imshow(numpy_output,cmap='hot',norm=norm)
    plt.imshow(numpy_output,cmap='plasma',norm=norm,interpolation='bicubic')
    cbar = plt.colorbar()
    cbar.set_label('Data Value', rotation=270, labelpad=20)
    plt.title('Attention map')
    plt.show()
    print("ok")

# 提取出来的CLIP-L的原始图像特征池化后的点
def clip_point_graph():
    save_file_path = "mscoco/gradient/avg_1/val_gradient_norm.json"  # 保存特征梯度先验json文件路径
    with open(save_file_path, 'r') as json_file:
        val_feature_gradient = json.load(json_file)
    feature_gradient_list = list(val_feature_gradient.values())

    np_data = np.array(feature_gradient_list)
    scaled_data = (np_data-np_data.min()) / (np_data.max() - np_data.min())

    # 散点图
    plt.figure()
    plt.scatter(range(len(scaled_data)), scaled_data, color='blue', marker='o',s=1)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Scatter Plot of feature_gradient_list')

    # 曲线图
    plt.figure()
    plt.hist(scaled_data, bins=50, color='blue', alpha=1,density=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency-Value')
    plt.show()


    print("ok")

# 分别对3层编码器中出来的特征进行池化后的点
def Encoder_point_graph():
    save_file_path = "mscoco/gradient/avg_1/3_layers_features_norm.json"  # 保存特征梯度先验json文件路径
    with open(save_file_path, 'r') as json_file:
        layers_dic = json.load(json_file)
    layer1 = layers_dic["Layer1"]
    layer2 = layers_dic["Layer2"]
    layer3 = layers_dic["Layer3"]

    layer1_np_data = np.array(layer1)
    layer1_scaled_data = (layer1_np_data - layer1_np_data.min()) / (layer1_np_data.max() - layer1_np_data.min())
    layer2_np_data = np.array(layer2)
    layer2_scaled_data = (layer2_np_data - layer2_np_data.min()) / (layer2_np_data.max() - layer2_np_data.min())
    layer3_np_data = np.array(layer3)
    layer3_scaled_data = (layer3_np_data - layer3_np_data.min()) / (layer3_np_data.max() - layer3_np_data.min())





    # 散点图
    plt.figure()
    plt.scatter(range(len(layer1_scaled_data)), layer1_scaled_data, color='blue', marker='o', s=1)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Scatter Plot of layer1')

    plt.figure()
    plt.scatter(range(len(layer2_scaled_data)), layer2_scaled_data, color='blue', marker='o', s=1)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Scatter Plot of layer2')

    plt.figure()
    plt.scatter(range(len(layer3_scaled_data)), layer3_scaled_data, color='blue', marker='o', s=1)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Scatter Plot of layer3')





    # 曲线图
    plt.figure()
    plt.hist(layer1_scaled_data, bins=50, color='blue', alpha=0.9, density=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency-Value-1')


    plt.figure()
    plt.hist(layer2_scaled_data, bins=50, color='blue', alpha=0.9, density=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency-Value-2')


    plt.figure()
    plt.hist(layer3_scaled_data, bins=50, color='blue', alpha=0.9, density=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency-Value-3')

    plt.show()

    print("ok")

# 使用GGD来拟合clip中提取出来的点的数据
def clip_GGD():
    with open("mscoco/gradient/avg_1/val_gradient_norm.json", "r") as file:
        date_dic = json.load(file)
    origin_data = list(date_dic.values())

    # 真实数据的均值和标准差
    mean = np.mean(origin_data)
    std_dev = np.std(origin_data)

    # 生成一些真实数据
   # real_data = np.random.normal(loc=mean, scale=std_dev, size=1000)

    real_data = np.array(origin_data)

    # 拟合广义高斯分布模型   params[0]:分布形状参数 1: 位置参数 2:尺度参数
    params = gennorm.fit(real_data)

    # 定义广义高斯分布的PDF函数
    ggd_pdf = lambda x: gennorm.pdf(x, *params)

    # 定义真实数据的PDF函数
    # real_data_pdf = lambda x: np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))

    # 生成 x 范围
    x = np.linspace(real_data.min(), real_data.max(), 1000)

    # 绘制拟合的广义高斯分布函数和真实数据的概率密度函数
    plt.plot(x, ggd_pdf(x),color='r', label='Fitted GGD')
    plt.hist(real_data, bins=150, alpha=0.9, color='b', label='Real Data', density=True)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.title('Fitted GGD vs Real Data')
    plt.show()


# 使用GGD来拟合clip中提取出来的点的数据(进行了平滑处理) 有问题
def clip_GGD_smooth():
    with open("mscoco/gradient/avg_1/val_gradient.json", "r") as file:
        date_dic = json.load(file)
    origin_data = list(date_dic.values())
    real_data = np.array(origin_data)

    shape, loc, scale = gennorm.fit(real_data)

    # 生成模拟数据的 x 值
    x = np.linspace(gennorm.ppf(0.01, shape, loc=loc, scale=scale),
                    gennorm.ppf(0.99, shape, loc=loc, scale=scale), 1000)

    # 计算分布函数曲线的 y 值
    pdf = gennorm.pdf(x, shape, loc=loc, scale=scale)

    # 获取直方图的频数和边界值
    hist, bins = np.histogram(real_data, bins=30, density=True)

    # 计算直方图的中点坐标
    hist_midpoints = (bins[1:] + bins[:-1]) / 2

    # 使用高斯核密度估计做平滑曲线
    kde = gaussian_kde(real_data)
    kde_smoothed = kde(hist_midpoints)

    # 绘制真实数据的直方图
    plt.bar(hist_midpoints, hist, width=0.07, alpha=0.5, color='b', label='Real Data')
    plt.plot(hist_midpoints, kde_smoothed, color='g', label='Smoothed Curve')

    # 绘制模拟数据的分布函数曲线
    plt.plot(x, pdf, color='r', label='Simulated Distribution')

    plt.legend()
    plt.show()

# 使用GGD来拟合每一层提取出来的点的数据  choose_layer: 1,2,3
def layers_GGD(choose_layer):

    save_file_path = "mscoco/gradient/avg_1/3_layers_features_norm.json"  # 保存特征梯度先验json文件路径
    with open(save_file_path, 'r') as json_file:
        layers_dic = json.load(json_file)
    layer1 = layers_dic["Layer1"]
    layer2 = layers_dic["Layer2"]
    layer3 = layers_dic["Layer3"]
    choose_dic = {1: layer1, 2 : layer2, 3 : layer3}

    # 真实数据的均值和标准差
    origin_data = choose_dic[choose_layer]
    mean = np.mean(choose_dic[choose_layer])
    std_dev = np.std(choose_dic[choose_layer])


    # 生成一些真实数据
    # real_data = np.random.normal(loc=mean, scale=std_dev, size=1000)

    real_data = np.array(origin_data)

    # 拟合广义高斯分布模型   params[0]:分布形状参数 1: 位置参数 2:尺度参数
    params = gennorm.fit(real_data)


    # params = list(params)
    # params[0] = 1.5
    # 定义广义高斯分布的PDF函数
    ggd_pdf = lambda x: gennorm.pdf(x, *params)

    # 定义真实数据的PDF函数
    # real_data_pdf = lambda x: np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))

    # 生成 x 范围
    x = np.linspace(real_data.min(), real_data.max(), 1000)

    # 绘制拟合的广义高斯分布函数和真实数据的概率密度函数
    plt.plot(x, ggd_pdf(x), color='r', label='Fitted GGD ' + str(choose_layer) + "_layer")
    plt.hist(real_data, bins=50, alpha=0.9, color='b', label='Real Data', density=True)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.title('Fitted GGD ' + str(choose_layer) + '_layer' + 'vs Real Data')
    plt.show()
    print("形状参数: " + str(params[0]))
    print("位置参数: " + str(params[1]))
    print("尺度参数: " + str(params[2]))


def julei_512(choose_layer):
    save_file_path = "mscoco/gradient/avg_512_no_norm/3_layers_features.json"  # 保存特征梯度先验json文件路径
    with open(save_file_path, 'r') as json_file:
        layers_dic = json.load(json_file)
    layer1 = layers_dic["Layer1"]
    layer2 = layers_dic["Layer2"]
    layer3 = layers_dic["Layer3"]
    choose_dic = {1: layer1, 2: layer2, 3: layer3}

    origin_data = choose_dic[choose_layer]
    mse_list = []
    for data_512 in origin_data:
        np_data = np.array(data_512).reshape(512, -1)
        n_clusters = 10

        # julei = AgglomerativeClustering(n_clusters=n_clusters)
        julei = KMeans(n_clusters=n_clusters,n_init=3)

        # 对数据进行聚类
        julei.fit(np_data)
        labels = julei.labels_
        cluster_centers = julei.cluster_centers_

        #计算kl散度和均方误差
        # 计算每个样本到其所属簇中心的均方距离
        # mse = np.mean(pairwise_distances(np_data, cluster_centers[labels]) ** 2)
        mse = np.sqrt(np.sum((np_data - cluster_centers[labels]) ** 2, axis=0))
        mse_list.append(mse)

        # 计算每个样本的分布与簇中心的KL散度
        # kl_divergence = np.mean(kl_div(np_data, cluster_centers[labels]))


        # 输出均方距离和KL散度
        print("Mean Squared Distance: ", mse)
        print("KL Divergence: ", kl_divergence)

        # 可视化聚类结果
        plt.figure(figsize=(8, 6))
        plt.scatter(range(512), np_data.flatten(), c=labels, cmap='viridis', label='Data')
        plt.scatter(range(len(cluster_centers)), cluster_centers.flatten(), marker='x', color='red', label='Cluster Centers')

        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        plt.title('Clustering Results')
        plt.legend()
        plt.show()
    print("5000张图片均方误差为:",np.mean(mse_list))
def test():
    # 生成500个假数据，分为3个类别
    X, y = make_blobs(n_samples=500, centers=3, random_state=42)

    # 使用K均值进行聚类
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    distances_to_assigned_centers = np.sqrt(np.sum((X - cluster_centers[labels]) ** 2, axis=1))

    # 计算均方误差
    mse = np.mean(distances_to_assigned_centers ** 2)

    print("mse:", mse)
    # 可视化
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.7)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.9)
    plt.title('K-Means Clustering')
    plt.show()
    print("ok")
def julei_512_global():
    save_file_path = "mscoco/gradient/avg_512_no_norm/global_feature.pth"  # 保存特征梯度先验json文件路径


    origin_data = torch.load(save_file_path)
    mse_list = []
    for data_512 in origin_data:
        np_data = np.array(data_512.to("cpu")).reshape(512, -1)
        n_clusters = 8

        # julei = AgglomerativeClustering(n_clusters=n_clusters)
        julei = KMeans(n_clusters=n_clusters,n_init=3)

        # 对数据进行聚类
        julei.fit(np_data)
        labels = julei.labels_
        cluster_centers = julei.cluster_centers_

        #计算kl散度和均方误差
        # 计算每个样本到其所属簇中心的均方距离
        # mse = np.mean(pairwise_distances(np_data, cluster_centers[labels]) ** 2)
        mse = np.sqrt(np.sum((np_data - cluster_centers[labels]) ** 2, axis=0))
        mse_list.append(mse)

        # 计算每个样本的分布与簇中心的KL散度
        # kl_divergence = np.mean(kl_div(np_data, cluster_centers[labels]))


        # 输出均方距离和KL散度
        # print("Mean Squared Distance: ", mse)
        # print("KL Divergence: ", kl_divergence)

        # 可视化聚类结果
        plt.figure(figsize=(8, 6))
        plt.scatter(range(512), np_data.flatten(), c=labels, cmap='viridis', label='Data')
        plt.scatter(range(len(cluster_centers)), cluster_centers.flatten(), marker='x', color='red', label='Cluster Centers')

        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        plt.title('Clustering Results')
        plt.legend()
        plt.show()
    print("5000张图片均方误差为:",np.mean(mse_list))

if __name__ == '__main__':
    julei_512(3)
    # test()
    # show_encoder_feature_map(torch.rand(1,256,512).to('cuda'))
    # show_feature_map(torch.randn(1,576,1024),576,1024)
