import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()


    def forwoard(self):
        pass


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