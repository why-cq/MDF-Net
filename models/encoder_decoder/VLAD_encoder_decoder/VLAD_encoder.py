from torch.nn import functional as F, Linear, Dropout, LayerNorm
# from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn

"""
新的编码器怎增强网络,首先经过VLADnet得到聚类后的grid_feature,
"""


class VLAD_encoder(nn.Module):
    def __init__(self, N, d_model=512, h=8, d_ff=2048, dropout=.1, num_clusters = 9,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(VLAD_encoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering)
                                     for _ in range(N)])
        self.num_clusters = num_clusters

    # todo: 编码器的输入应该给我两个特征,一个为global_feature,一个是经过VLAD网络的grid_feature
    def forward(self, global_feature, grid_feature, attention_weights=None):

        # todo: 这里出问题了,在前面的9个为聚类的特征个数,后面才是网格特征
        region_f = grid_feature[:, :self.num_clusters]
        grid_f = grid_feature[:, self.num_clusters:]

        # todo: 这个mask有点蒙,解码器的时候需要注意,编码器中我并不需要进行mask
        # attention_mask_grid = (torch.sum(grid_f, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        # attention_mask_grid = (torch.sum(grid_f, -1) == self.padding_idx)  # (b_s, 1, 1, seq_len)

        # todo:进入的时候要构建两个不同特征的输出
        out_grid = grid_f
        out_global = global_feature.unsqueeze(1)
        for l in self.layers:
            out_grid,out_global = l(out_grid, out_global, region_f)

            # outs.append(out.unsqueeze(1))

        # outs = torch.cat(outs, 1)
        out_global = out_global.squeeze(1)

        return out_grid, out_global


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        # 第一次交叉注意力是对gv_feature,att_feature和聚类的特征进行注意力
        # self.self_attn = MultiHeadSelfAttention(embed_dim = d_model,num_heads=h)
        self.att1 = EncoderLayer_att1()
        self.att2 = EncoderLayer_att2()

    def forward(self, att_feature, gv_feature, region_feature, attention_mask=None):
        # 第一次注意力
        att_feature = self.att1(att_feature, region_feature, region_feature, attention_mask)
        gv_feature = self.att1(gv_feature, region_feature, region_feature, attention_mask)

        # 第二次注意力是两者之间的  todo:这里的注意力有问题改成g和grid做
        att_feature , gv_feature = self.att2(gv_feature,att_feature)




        return att_feature, gv_feature


class EncoderLayer_att1(nn.Module):
        def __init__(self, d_model=512, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False):
            super(EncoderLayer_att1, self).__init__()
            self.identity_map_reordering = identity_map_reordering
            # 第一次交叉注意力是对gv_feature,att_feature和聚类的特征进行注意力
            # self.self_attn = MultiHeadSelfAttention(embed_dim = d_model,num_heads=h)
            self.self_attn_1 = MultiHeadSelfAttention(embed_dim=d_model, num_heads=h)
            self.dropout1 = Dropout(dropout)
            self.norm1 = LayerNorm(d_model)
            # Feedforward model
            self.ff_layer_1 = FeedForward(d_model, d_ff)
            self.dropout2 = Dropout(dropout)
            self.norm2 = LayerNorm(d_model)

        def forward(self, queries, keys, values, attention_mask=None):
            # todo: 进行注意力计算,然后dropout,残差,norm
            att = self.self_attn_1(queries, keys, values, attention_mask)
            att = self.norm1(queries + self.dropout1(att))

            # todo: 进行第一次ffn
            # out = self.linear2(self.dropout(self.activation(self.linear1(att))))
            # out = self.dropout2(out)
            # att = self.norm2(out + att)
            out = self.ff_layer_1(att)
            att = self.norm2(self.dropout2(out) + att)

            return att


# 第二次注意力att_feature和global_feature
class EncoderLayer_att2(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(EncoderLayer_att2, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        # 第一次交叉注意力是对gv_feature,att_feature和聚类的特征进行注意力
        # self.self_attn = MultiHeadSelfAttention(embed_dim = d_model,num_heads=h)
        self.self_attn_1 = MultiHeadSelfAttention(embed_dim=d_model, num_heads=h)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        # Feedforward model
        self.ff_layer_1 = FeedForward(d_model, d_ff)
        self.dropout2 = Dropout(dropout)
        self.norm2 = LayerNorm(d_model)

    def forward(self, gv_feature, grid_feature, attention_mask=None):
        # todo: 这里的输入是
        #print(gv_feature.shape)
        #print(grid_feature.shape)
        x = torch.cat([grid_feature,gv_feature],dim=1)
        #print(x.shape)

        att = self.self_attn_1(x, x, x, attention_mask)


        att = self.norm1(x + self.dropout1(att))

        # todo: 进行第一次ffn
        # out = self.linear2(self.dropout(self.activation(self.linear1(att))))
        # out = self.dropout2(out)
        # att = self.norm2(out + att)
        out = self.ff_layer_1(att)
        att = self.norm2(self.dropout2(out) + att)

        grid_feature = att[:,:-1,:]
        gv_feature = att[:,-1,:]

        return grid_feature, gv_feature.unsqueeze(1)

    # 主要是和region_feature做注意力




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.o_linear = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(-1)

        self.clear_buffer()

    def init_buffer(self, batch_size):
        # [B, nH, 0, C/nH]
        self.buffer_key = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device='cuda')
        self.buffer_value = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device='cuda')

    def clear_buffer(self):
        self.buffer_key = None
        self.buffer_value = None

    def apply_to_states(self, fn):
        self.buffer_key = fn(self.buffer_key)
        self.buffer_value = fn(self.buffer_value)

    def forward(self, q, k, v, mask):
        """
        Decoder部分有两部分进行注意力：
            1）单词嵌入自注意力，q/k/v大小均为[B, L, D]
            2）单词嵌入与图像特征（包含全局特征）的cross attention，q的大小为[B, L, D]
               k/v的大小为[B, M+1, D]
        输出的维度大小只与q的维度大小相关
        """
        B_, N, C = q.size()
        # 线性变换
        q = self.q_linear(q).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 存储buffer，用于inference时单词嵌入自注意力
        if self.buffer_key is not None and self.buffer_value is not None:
            self.buffer_key = torch.cat([self.buffer_key, k], dim=2)
            self.buffer_value = torch.cat([self.buffer_value, v], dim=2)
            k = self.buffer_key
            v = self.buffer_value

        # 注意力核心操作
        # [B, nH, L, L] or [B, nH, L, M+1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 计算注意力权重
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == True, -1e9)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.o_linear(out)
        return out


# 不包含残差连接和LayerNorm
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.act = nn.ReLU()  # ReLU / GELU / CELU
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = nn.Dropout(relu_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, alpha=10.0,
                 normalize_input=True, input_dim = 768, vlad_trigger=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float 100.0
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = 9
        self.dim = 512
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(1e-1 * torch.rand(self.num_clusters, self.dim))
        self.encoder_change = nn.Linear(input_dim, 512)

        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    # 输入(B,L,768) --> 输出(B,L+9,512)  9表示的是聚类的个数
    def forward(self, x):
        x = x.unsqueeze(-1).permute(0, 2, 1, 3)  # [N, M, dim, 1] -> [N, dim, M, 1]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        x_change = self.encoder_change(x.transpose(1, 3)).transpose(1, 3)

        N, C = x_change.shape[:2]
        soft_assign = self.conv(x_change).view(N, self.num_clusters, -1)  # [N, K, M]
        soft_assign = F.softmax(soft_assign, dim=1)
        vis_temp = soft_assign.permute(0, 2, 1)
        # plt.figure(figsize=(6, 8))

        # vis_mat = torch.randn(soft_assign.size(0),49)
        # for i in range(vis_temp.size(0)):
        #     for j in range(49):
        #         vis_mat[i][j] = torch.argmax(vis_temp[i][j])
        # vis_mat = vis_mat/torch.sum(vis_mat,dim=1,keepdim=True)
        # vis_mat = vis_mat.reshape(10,7,7)

        x_flatten = x_change.reshape(N, C, -1)

        # calculate residuals to each clusters   [300, num_cluster, 1024, M]
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # vlad = vlad.view(x.size(0), -1)       # flatten
        # vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        # concat
        # output = torch.cat((vlad.unsqueeze(-1).transpose(1,2),x_change),dim=2)
        # output_view = output.squeeze(-1).transpose(1,2)
        # attention_mask = (torch.sum(output_view, -1) == 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        # return output_view,attention_mask

        # 换成一个attention
        output = torch.cat((vlad, x_flatten.transpose(1, 2)), dim=1)
        attention_mask = None
        return output, attention_mask
        # return output,attention_mask,vis_mat


if __name__ == '__main__':
    grid_feature = torch.randn(10, 256, 1024)
    g_feature = torch.randn(10, 768)
    net = NetVLAD(input_dim=1024)
    model = VLAD_encoder(N=3)

    # grid_feature先进入VLADnet

    grid_feature, _ = net(grid_feature)

    out_grid, out_global = model(g_feature, grid_feature)
    print('success')
