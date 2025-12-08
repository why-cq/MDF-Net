from collections import OrderedDict

import torch
import torch.nn as nn
from clip.model import LayerNorm, QuickGELU
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(
            self,
            embed_dim=512,
            depth=3,
            num_heads=8,
    ):
        super(Encoder, self).__init__()
        scale = embed_dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(512))
        self.embed_dim = embed_dim
        self.depth = depth

        # 构建 W-MSA / SW-MSA 层
        # 输入特征尺寸为 49 = 7 x 7，如果构建 SW-MSA 层，
        # 则需要将 window_size 设置得更小，比如设置为 4，且shift_size > 0
        # SW-MSA仅在偶数层被构造，W-MSA在奇数层构造
        # 如：W-MSA，SW-MSA，W-MSA，SW-MSA ......
        self.transformer = Transformer(embed_dim, depth, num_heads)


    def forward(self, x, att_mask=None):
        # todo:这里输入的是原始的图像网格特征(B,256,512)
        # 1.加入分类头  B,256,512 -> B,257,512
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),x],dim=1)

        # 核心操作层
        x = self.transformer(x)

        # 返回网格特征个全局特征
        g_feature = x[:, 0, :]
        grid_feature = x[:,1:,:]

        return g_feature, grid_feature 

    def flops(self):
        flops = 0
        for _l in self.layers:
            flops += _l.flops()
        return flops


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


if __name__ == '__main__':
    x = torch.rand(5,256,512)
    encoder = Encoder()
    out = encoder(x)
    print('success')