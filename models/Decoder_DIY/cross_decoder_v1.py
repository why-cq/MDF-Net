import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from utils.config import cfg


# 位置嵌入矩阵
def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out

# 不使用图像的全局信息
# 第一个版本只是将整体的结构分离,文本编码器和交叉注意力分离

class Cross_Decoder_v1(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim=512,
            text_encoder_depth=3,  # 文本编码器的层数
            cross_depth=3,  # 后续融合模块的层数
            num_heads=8,
            dropout=0.1,
            ff_dropout=0.1,
    ):
        super(Cross_Decoder_v1, self).__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.text_layers = nn.ModuleList([])
        self.cross_layers = nn.ModuleList([])
        self.embed_dim = embed_dim

        # self.use_gx = use_gx
        for i in range(text_encoder_depth):
            sublayer = TextDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                ff_dropout=ff_dropout,
            )
            self.text_layers.append(sublayer)

        for i in range(cross_depth):
            sublayer = CrossDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                ff_dropout=ff_dropout,
            )
            self.cross_layers.append(sublayer)

        # self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED)
        self.dropout = nn.Dropout()

        self.word_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_scale = math.sqrt(self.embed_dim)
        self.pos_embed = nn.Embedding.from_pretrained(
            sinusoid_encoding_table(100, self.embed_dim, 0), freeze=True
        )

        self.generator = nn.Linear(self.embed_dim, self.vocab_size, bias=True)

        # 定义一个全局分类头,用来进行全局信息的获取
        self.cls_text = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)

        self.clear_buffer()

    def init_buffer(self, batch_size):
        self.seq_len = 0
        for layer in self.text_layers:
            layer.init_buffer(batch_size)

    def clear_buffer(self):
        self.seq_len = None
        for layer in self.text_layers:
            layer.clear_buffer()


    def apply_to_states(self, fn):
        for layer in self.text_layers:
            layer.apply_to_states(fn)


    def precompute(self, encoder_out):
        p_att_feats = []
        for layer in self.text_layers:
            key, value2 = layer.precompute(encoder_out)
            p_att_feats.append((key, value2))


        return p_att_feats

    # 解码器需要给我解码器的输出,以及相应文本的词向量输入
    def forward(self, gx, seq, encoder_out, seq_mask=None, att_mask=None):
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)  # [B, 1, M]

        seq_len = seq.size()[1]
        pos_indx = torch.arange(1, seq_len + 1, device='cuda').view(1, -1)
        if self.seq_len is not None:
            seq_len = self.seq_len + seq_len
            self.seq_len = seq_len
            pos_indx = torch.arange(seq_len, seq_len + 1, device='cuda').view(1, -1)

        # 词汇嵌入 + 位置嵌入
        # [B, seq_len, C] for training or [B, 1, C] for inference
        x = self.embed_scale * self.word_embed(seq) + self.pos_embed(pos_indx)

        # todo: 1.先进入文本自注意力编码
        # 进行自注意力操作, 获取全局信息, 将全局信息和文本信息加在一起
        B = x.shape[0]
        cls_text = self.cls_text.expand(B, 1, -1)
        x = torch.cat((cls_text, x), dim = 1)
        for layer in self.text_layers:
            x = layer(x, seq_mask)

        # todo: 2.在进行图像和文本的交叉注意力 ,不要全局特征  (itc阶段需要两个分类头)

        x = x[:,1:,:]
        #cls_text = x[:,1,:]   # 测试时不需要

        for layer in self.cross_layers:
            x = layer(x, encoder_out,  att_mask)


        x = self.dropout(x)
        out = self.generator(x)
        
        return out#, cls_text
        # return out  # 测试的时候只需要这个



# 文本编码器中并没有进行预融合,是否融合的效果还不知道,应为是直接进行的模块分离
class TextDecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, ff_dropout=0.1):
        super(TextDecoderLayer, self).__init__()


        self.global_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.ff_cls = FeedForward(
            embed_dim=embed_dim,
            ffn_embed_dim=embed_dim * 4,
            relu_dropout=ff_dropout
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.word_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.layer_norm3 = nn.LayerNorm(embed_dim)

        self.ff_layer = FeedForward(
            embed_dim=embed_dim,
            ffn_embed_dim=embed_dim * 4,
            relu_dropout=ff_dropout
        )
        self.layer_norm4 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)


    def apply_to_states(self, fn):
        self.word_attn.apply_to_states(fn)

    def init_buffer(self, batch_size):
        self.word_attn.init_buffer(batch_size)

    def clear_buffer(self):
        self.word_attn.clear_buffer()

    def precompute(self, encoder_out):
        # key, value2 = self.cross_att.precompute(encoder_out, encoder_out)
        # return key, value2
        pass

    def forward(self, x, seq_mask):
        # 单词嵌入自注意力
        # short_cut = x
        # 在单词嵌入自注意力阶段，嵌入图像的全局特征
        # 方式2:concat接Linear+GLU / Linear
        short_cut = x[:,1:,:]
        short_cut_cls = x[:,1,:].unsqueeze(1)

        cls_text = x[:,1,:].unsqueeze(1)
        cls_text = self.global_attn(cls_text,short_cut,short_cut, mask = None)
        cls_text = self.layer_norm1(self.dropout(cls_text) + short_cut_cls)

        # ffn
        short_cut_cls = cls_text
        cls_text = self.ff_cls(cls_text)
        cls_text = self.dropout(cls_text)
        cls_text = self.layer_norm2(cls_text + short_cut_cls)


        x = self.word_attn(
            q=short_cut,
            k=short_cut,
            v=short_cut,
            mask=seq_mask
        )
        x = self.dropout(x)
        x = self.layer_norm3(x + short_cut)


        # Feedforward
        short_cut = x
        x = self.ff_layer(x)
        x = self.dropout(x)
        x = self.layer_norm4(x + short_cut)

        return torch.cat((cls_text,x), dim = 1)

class CrossDecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, ff_dropout=0.1):
        super(CrossDecoderLayer, self).__init__()

        self.cross_att = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.ff_layer = FeedForward(
            embed_dim=embed_dim,
            ffn_embed_dim=embed_dim * 4,
            relu_dropout=ff_dropout
        )
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, encoder_out,  att_mask=None):
        # 单词嵌入自注意力
        # short_cut = x
        # 在单词嵌入自注意力阶段，嵌入图像的全局特征
        # 方式2:concat接Linear+GLU / Linear

        # 单词嵌入与图像特征（可包含全局特征）cross 注意力
        short_cut = x

        kv = encoder_out
        _att_mask = att_mask

        x = self.cross_att(
            q=x,
            k=kv,
            v=kv,
            mask=_att_mask,
            # precompute=False
        )
        x = self.dropout(x)
        x = self.layer_norm1(x + short_cut)

        # Feedforward
        short_cut = x
        x = self.ff_layer(x)
        x = self.dropout(x)
        x = self.layer_norm2(x + short_cut)

        return x


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
            attn = attn.masked_fill(mask == 0, -1e9)
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
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = nn.Dropout(relu_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

if __name__ == '__main__':
    decoder = Cross_Decoder_v1(vocab_size=9999)
    decoder.to(device='cuda')
    encoder_out = torch.randn((5,169,512),device='cuda')
    seq_input = torch.zeros((5,17),dtype=torch.long,device='cuda')
    gx = 0
    out = decoder(gx ,seq_input, encoder_out)
    print('out_shape:' , out.shape)