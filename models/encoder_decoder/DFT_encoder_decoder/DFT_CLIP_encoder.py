import torch
from torch import nn
from torch.nn import functional as F

from utils.attention import ScaledDotProductAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(EncoderLayer, self).__init__()

        self.self_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout)

        self.self_region = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.global_grid = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)
        self.global_region = MultiHeadAttention(d_model, d_k, d_v, h, dropout, shortcut=False)

        self.cls_grid = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        self.cls_region = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

        self.pwff_grid = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.pwff_region = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, gird_features, region_features, attention_mask):
        # 这里实现了原文编码器层的操作
        b_s = region_features.shape[0]
        # 初始化两个特征的分类头
        cls_grid = self.cls_grid.expand(b_s, 1, -1)
        cls_region = self.cls_region.expand(b_s, 1, -1)

        # 分类头和其余特征做注意力,得到一个可以代表全局的向量  todo: 第一次注意力
        cls_grid = self.global_grid(cls_grid, gird_features, gird_features)
        cls_region = self.global_region(cls_region, region_features, region_features, attention_mask=attention_mask)

        # 拼接分类头(不同的全局特征和不同的特征之间进行拼接)
        gird_features = torch.cat([cls_region, gird_features], dim=1)
        region_features = torch.cat([cls_grid, region_features], dim=1)

        # 多了一个维度,那么mask也要多一位
        add_mask = torch.zeros(b_s, 1, 1, 1).bool().to(region_features.device)
        attention_mask = torch.cat([add_mask, attention_mask], dim=-1)

        # todo: 第二次注意力的地方
        grid_att = self.self_grid(gird_features, gird_features, gird_features)
        region_att = self.self_region(region_features, region_features, region_features, attention_mask=attention_mask)

        gird_ff = self.pwff_grid(grid_att)
        region_ff = self.pwff_region(region_att)

        gird_ff = gird_ff[:, 1:]
        region_ff = region_ff[:, 1:]

        return gird_ff, region_ff


class DEF_Encoder(nn.Module):
    def __init__(self, N, device='cuda', d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(DEF_Encoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.device = device

        # self.grid_proj = nn.Sequential(
        #     nn.Linear(2560, self.d_model),
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout),
        #     nn.LayerNorm(self.d_model)
        # )
        #
        # self.region_proj = nn.Sequential(
        #     nn.Linear(2048, self.d_model),
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout),
        #     nn.LayerNorm(self.d_model)
        # )

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])

    def forward(self, grid_features, region_features):
        # 输入grid_features(5*B,49,512) ,region_features(5*B,9,512)
        b_s = region_features.shape[0]
        attention_mask = (torch.sum(torch.abs(region_features), -1) == 0).unsqueeze(1).unsqueeze(1)
        # 进入两个全连接层将特征维度降为512, CLIP的话不需要了
        # grid_features = self.grid_proj(grid_features)
        # region_features = self.region_proj(region_features)

        # 进入N个编码器中
        for l in self.layers:
            grid_features, region_features = l(grid_features, region_features, attention_mask)

        return grid_features, region_features, attention_mask


class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 shortcut=True, attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.shortcut = shortcut
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = self.dropout(torch.relu(out))
            if self.shortcut:
                out = queries + out
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            if self.shortcut:
                out = queries + out
            out = self.layer_norm(out)
        return out


class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, act_fn='ReLU', identity_map_reordering=False, local=False):
        super(PositionWiseFeedForward, self).__init__()
        self.local = local
        self.identity_map_reordering = identity_map_reordering
        # if local:
        #     self.dwconv = DWConv(d_ff, gird_size=(9, 9))
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.act = getattr(nn, act_fn)()

    def forward(self, input):
        if self.identity_map_reordering:
            x = self.layer_norm(input)
            x = self.fc1(x)
            if self.local:
                x = x + self.dwconv(x)
            x = self.act(x)
            x = self.dropout_2(x)
            x = self.fc2(x)
            x = input + self.dropout(self.act(x))
        else:
            x = self.fc1(input)
            if self.local:
                x = self.dwconv(x)
            x = self.act(x)
            x = self.dropout_2(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.layer_norm(input + x)
        return x


def build_encoder(N, device='cuda', d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
    Encoder = DEF_Encoder(N, device, d_model, d_k, d_v, h, d_ff, dropout)

    return Encoder


if __name__ == '__main__':
    region_feature = torch.rand(5, 9, 512)
    grid_feature = torch.rand(5, 49, 512)

    encoder = build_encoder(3)

    att_feature, reg_feature, attention_mask = encoder(grid_feature, region_feature)