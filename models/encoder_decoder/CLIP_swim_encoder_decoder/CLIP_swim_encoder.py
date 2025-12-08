import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_, Mlp

from models.Decoder_DIY.decoder_v1 import MultiHeadSelfAttention, FeedForward


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# 构建多个Swim_Transformer块
class WSA_Bolock(nn.Module):

    def __init__(
            self,
            embed_dim=512,  # 嵌入层维度
            input_resolution=(12, 12),  # 输入的H,W
            depth=4,  # 构建几个
            num_heads=8,
            window_size=12,  # 窗口大小
            shift_size=6,  # 滑动的距离
            mlp_ratio=4,
            dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.depth = depth

        # 构建 WSA / S-WSA 技术层为WSA, 偶数层为S-WSA
        self.layers = nn.ModuleList([
            WSA_Layer(
                embed_dim=embed_dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for i in range(self.depth)
        ])

    def forward(self, x):
        # 输入的x的形状判定
        B, HW, C = x.shape
        assert HW == self.input_resolution[0] ** 2, "输入的HW维度和定义的resolution不匹配"
        assert C == self.embed_dim, "嵌入层维度不匹配"

        # 进入WSA / S-WSA
        for layer in self.layers:
            x = layer(x)
        return x

# 利用全局特征
class WSA_Bolock_v2(nn.Module):

    def __init__(
            self,
            embed_dim=512,  # 嵌入层维度
            input_resolution=(12, 12),  # 输入的H,W
            depth=4,  # 构建几个
            num_heads=8,
            window_size=12,  # 窗口大小
            shift_size=6,  # 滑动的距离
            mlp_ratio=4,
            dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.depth = depth



        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * depth, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 构建 WSA / S-WSA 技术层为WSA, 偶数层为S-WSA
        self.layers = nn.ModuleList([
            WSA_Layer_v2(
                embed_dim=embed_dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for i in range(self.depth)
        ])

    def forward(self, x, gv):
        # 输入的x的形状判定
        B, HW, C = x.shape
        assert HW == self.input_resolution[0] ** 2, "输入的HW维度和定义的resolution不匹配"
        assert C == self.embed_dim, "嵌入层维度不匹配"

        # todo: 这里需要一个列表来接受每一层输出的grid_feature, 权重可以自动调节
        grid_feature_list = []
        weigth = 0.1
        # 进入WSA / S-WSA
        for layer in self.layers:
            x, gv = layer(x, gv)
            grid_feature_list.append(x)
        # 这里对所有的特征进行一个权重分配
        short_cut = grid_feature_list[-1]   # 最后一层的输出
        x = torch.cat(grid_feature_list, -1)
        x = self.norm(short_cut + weigth * self.ffn(x))
        return x, gv

# 单个块的代码
class WSA_Layer(nn.Module):

    def __init__(
            self,
            embed_dim=512,  # 嵌入层维度
            input_resolution=(12, 12),  # 输入的H,W
            num_heads=8,
            window_size=12,  # 窗口大小
            shift_size=6,  # 滑动的距离  为0的话退变为W-MSA
            mlp_ratio=4,
            dropout=0.1,
            act_layer=nn.GELU
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim  # 512
        self.input_resolution = input_resolution  # (12， 12)
        self.num_heads = num_heads  # 8
        self.window_size = window_size  # 12 / 6
        self.shift_size = shift_size  # shift_size可用于区分SW-MSA / W-MSA
        self.mlp_ratio = mlp_ratio  # 4
        self.nW = (input_resolution[0] // window_size) ** 2  # 有多少个窗口

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # 初始化网络结构
        self.attn = WindowAttention(dim=embed_dim, window_size=(window_size, window_size), num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # feedforward中的全连接层
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), act_layer=act_layer,
                       drop=dropout)

        # 形成移动后的Mask矩阵
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征HW两个维度不匹配"

        shortcut = x

        x = x.view(B, H, W, C)

        # 下面的步骤不变
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # 注意力后的残差
        x = shortcut + self.dropout(x)

        # FFN
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x

class WSA_Layer_v2(nn.Module):

    def __init__(
            self,
            embed_dim=512,  # 嵌入层维度
            input_resolution=(12, 12),  # 输入的H,W
            num_heads=8,
            window_size=12,  # 窗口大小
            shift_size=6,  # 滑动的距离  为0的话退变为W-MSA
            mlp_ratio=4,
            dropout=0.1,
            act_layer=nn.GELU
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim  # 512
        self.input_resolution = input_resolution  # (12， 12)
        self.num_heads = num_heads  # 8
        self.window_size = window_size  # 12 / 6
        self.shift_size = shift_size  # shift_size可用于区分SW-MSA / W-MSA
        self.mlp_ratio = mlp_ratio  # 4
        self.nW = (input_resolution[0] // window_size) ** 2  # 有多少个窗口

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # 初始化网络结构
        self.attn = WindowAttention(dim=embed_dim, window_size=(window_size, window_size), num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn_gv = MultiHeadSelfAttention(embed_dim=embed_dim,num_heads=num_heads)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.gv_ff = FeedForward(embed_dim=embed_dim,ffn_embed_dim=embed_dim * 4, relu_dropout=dropout)
        self.norm4 = nn.LayerNorm(embed_dim)
        # feedforward中的全连接层
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), act_layer=act_layer,
                       drop=dropout)

        # 形成移动后的Mask矩阵
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x , gv):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征HW两个维度不匹配"

        gv = gv.unsqueeze(1)
        gv_shortcut = gv
        # todo:这里对全局特征做一下注意力,细化全局特征
        gv = self.attn_gv(gv, x, x, mask = None)
        gv = self.dropout(gv)
        gv = self.norm3(gv + gv_shortcut)

        gv_shortcut = gv
        gv = self.gv_ff(gv)
        gv = self.dropout(gv)
        gv = self.norm4(gv + gv_shortcut)

        gv = gv.squeeze(1)



        shortcut = x
        x = x.view(B, H, W, C)

        # 下面的步骤不变
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # 注意力后的残差
        x = shortcut + self.dropout(x)

        # FFN
        x = x + self.dropout(self.mlp(self.norm2(x)))



        return x , gv

if __name__ == '__main__':

    # SWB = WSA_Bolock(embed_dim=512, input_resolution=(16, 16), window_size=4, shift_size=2,depth=6)


    SWB = WSA_Layer(embed_dim=512, input_resolution=(16, 16), window_size=4, shift_size=2)


    input = torch.randn(2, 512, 16, 16)
    # 需要修改的代码
    B, C, H, W = input.shape

    input = input.permute(0, 2, 3, 1).contiguous().reshape(B, -1, C)
    # SWB的输入为(B, H*W, C)
    output = SWB(input)

    output = output.permute(0, 2, 1).reshape(B, C, H, W)

    print('ok')
