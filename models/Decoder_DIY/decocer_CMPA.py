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

class Decoder_CMPA(nn.Module):
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
        super(Decoder_CMPA, self).__init__()
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


        self.dropout = nn.Dropout()

        self.word_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_scale = math.sqrt(self.embed_dim)
        self.pos_embed = nn.Embedding.from_pretrained(
            sinusoid_encoding_table(100, self.embed_dim, 0), freeze=True
        )

        self.generator = nn.Linear(self.embed_dim, self.vocab_size, bias=True)

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
        for layer in self.text_layers:
            x = layer(x,gx ,seq_mask)
        
    

        # todo: 2.在进行图像和文本的交叉注意力
        for layer in self.cross_layers:
            x = layer(x, encoder_out,  att_mask)


        x = self.dropout(x)
        out = self.generator(x)
        return out


#最新版的解码器改进
class Decoder_CMPA_2(nn.Module):
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
        super(Decoder_CMPA_2, self).__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.text_layers = nn.ModuleList([])
        self.cross_layers = nn.ModuleList([])
        self.embed_dim = embed_dim

        # self.use_gx = use_gx
        for i in range(text_encoder_depth):
            sublayer = TextDecoderLayer_CMPA2(
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

        self.dropout = nn.Dropout()

        self.word_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_scale = math.sqrt(self.embed_dim)
        self.pos_embed = nn.Embedding.from_pretrained(
            sinusoid_encoding_table(100, self.embed_dim, 0), freeze=True
        )

        self.generator = nn.Linear(self.embed_dim, self.vocab_size, bias=True)

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
        for layer in self.text_layers:
            x = layer(x, gx,encoder_out,seq_mask)

        # todo: 2.在进行图像和文本的交叉注意力
        for layer in self.cross_layers:
            x = layer(x, encoder_out, att_mask)

        x = self.dropout(x)
        out = self.generator(x)
        return out
class Decoder_CMPA_ONE(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_dim=512, 
        depth=3,
        num_heads=8,
        dropout=0.1, 
        ff_dropout=0.1, 
        use_gx=False
    ):
        super(Decoder_CMPA_ONE, self).__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.layers = nn.ModuleList([])
        self.embed_dim = embed_dim
        self.use_gx = use_gx
        for i in range(depth):
            sublayer = DecoderLayer_CMPA( 
                embed_dim = embed_dim, 
                num_heads = num_heads, 
                dropout = dropout, 
                ff_dropout = ff_dropout,
                use_gx = use_gx
            )
            self.layers.append(sublayer)
            
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED)
        
        self.word_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_scale = math.sqrt(self.embed_dim)
        self.pos_embed = nn.Embedding.from_pretrained(
            sinusoid_encoding_table(100, self.embed_dim, 0), freeze=True
        )
        
        self.generator = nn.Linear(self.embed_dim, self.vocab_size, bias=True)
                
        self.clear_buffer()

    def init_buffer(self, batch_size):
        self.seq_len = 0
        for layer in self.layers:
            layer.init_buffer(batch_size)

    def clear_buffer(self):
        self.seq_len = None
        for layer in self.layers:
            layer.clear_buffer()

    def apply_to_states(self, fn):
        for layer in self.layers:
            layer.apply_to_states(fn)

    def precompute(self, encoder_out):
        p_att_feats = []
        for layer in self.layers:
            key, value2 = layer.precompute(encoder_out)
            p_att_feats.append((key, value2))
        return p_att_feats

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
        
        for layer in self.layers:
            x = layer(gx, x, encoder_out, seq_mask, att_mask)

        x = self.dropout(x)
        out = self.generator(x)
        return out
# 文本编码器中并没有进行预融合,是否融合的效果还不知道,应为是直接进行的模块分离
class TextDecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, ff_dropout=0.1):
        super(TextDecoderLayer, self).__init__()
        self.word_attn = MultiHeadSelfAttention(
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

        self.cmpa = CMPA()

        


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

    def forward(self, x, gx, seq_mask):
        # 单词嵌入自注意力
        # short_cut = x
        # 在单词嵌入自注意力阶段，嵌入图像的全局特征
        # 方式2:concat接Linear+GLU / Linear
        short_cut = x

        # todo: 加入CMPA模块 x (B,17,512) gx (B,512)
        gx = gx.unsqueeze(1)
        x = self.cmpa(x,gx)

        x = self.word_attn(
            q=x,
            k=x,
            v=x,
            mask=seq_mask
        )
        x = self.dropout(x)
        x = self.layer_norm1(x + short_cut)

        
        
        # Feedforward
        short_cut = x
        x = self.ff_layer(x)
        x = self.dropout(x)
        x = self.layer_norm2(x + short_cut)

        



        return x


class TextDecoderLayer_CMPA2(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, ff_dropout=0.1):
        super(TextDecoderLayer_CMPA2, self).__init__()

        self.word_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.ff_layer = FeedForward(
            embed_dim=embed_dim,
            ffn_embed_dim=embed_dim * 4,
            relu_dropout=ff_dropout
        )

        self.cmpa = CMPA_Align()

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

    def forward(self, x, gx,gv, seq_mask):
        # 单词嵌入自注意力
        # short_cut = x
        # 在单词嵌入自注意力阶段，嵌入图像的全局特征
        # 方式2:concat接Linear+GLU / Linear


        # todo: 加入CMPA模块 x (B,17,512) gx (B,512)
        gx = gx.unsqueeze(1)
        x = self.cmpa(x, gx,gv)

        # 自注意力操作
        short_cut = x
        x = self.word_attn(
            q=x,
            k=x,
            v=x,
            mask=seq_mask
        )
        x = self.dropout(x)
        x = self.layer_norm1(x + short_cut)

        # Feedforward
        short_cut = x
        x = self.ff_layer(x)
        x = self.dropout(x)
        x = self.layer_norm2(x + short_cut)

        return x
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

class DecoderLayer_CMPA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, ff_dropout=0.1, use_gx=False):
        super(DecoderLayer_CMPA, self).__init__()
        self.word_attn = MultiHeadSelfAttention(
            embed_dim = embed_dim,
            num_heads = num_heads
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.cross_att = MultiHeadSelfAttention(
            embed_dim = embed_dim,
            num_heads = num_heads
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ff_layer = FeedForward(
            embed_dim = embed_dim,
            ffn_embed_dim = embed_dim * 4,
            relu_dropout = ff_dropout
        )
        self.layer_norm3 = torch.nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.cmpa = CMPA()

        self.use_gx = use_gx
        if self.use_gx:
            # 方式2，concat接Linear / Linear+GLU
            self.fuse_layer = nn.Sequential(
                nn.Linear(embed_dim*2, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.fuse_layer_norm = nn.LayerNorm(embed_dim)


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

    def forward(self, gx, x, encoder_out, seq_mask, att_mask=None):
        # 先进行pre-fusion

        if self.use_gx:
            x_cat = torch.cat([x, gx.unsqueeze(1).expand_as(x)], dim=-1)
            x = self.fuse_layer(x_cat) + x
            x = self.fuse_layer_norm(x)
        short_cut = x

        x = self.word_attn(
            q = x,
            k = x,
            v = x,
            mask = seq_mask
        )
        x = self.dropout(x)
        x = self.layer_norm1(x + short_cut)

        # 单词嵌入与图像特征（可包含全局特征）cross 注意力
        short_cut = x
        if self.use_gx:
            kv = torch.cat([encoder_out, gx.unsqueeze(1)], 1)
            if att_mask is not None:
                # [B, 1, M+1]，对于grid特征，直接设置为None亦可
                _att_mask = torch.cat(
                    [att_mask, torch.ones(att_mask.size(0), device='cuda').unsqueeze(1).unsqueeze(1)], 2
                ).long()
            else:
                _att_mask = None
        else:
            kv = encoder_out
            _att_mask = att_mask

        x = self.cross_att(
            q = x,
            k = kv,
            v = kv,
            mask = _att_mask,
            # precompute=False
        )
        x = self.dropout(x)
        x = self.layer_norm2(x + short_cut)

        # Feedforward
        short_cut = x
        x = self.ff_layer(x)
        x = self.dropout(x)
        x = self.layer_norm3(x + short_cut)

        return x
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



class CMPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_liner = nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            )

        self.fuse_liner = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.1),
            )
        
        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(512)
        # self.softmax = nn.Softmax(2)
        


    def forward(self,text_feature, gx):  # B,17,512  B,1,512
        short_cut = text_feature
        text_fn = self.text_liner(text_feature)
        text_fn = self.norm1(text_fn)

        g_weight = torch.cat((short_cut,gx.expand_as(short_cut)),-1)

        g_weight = g_weight + text_fn
        
        g_weight = self.fuse_liner(g_weight)

        short_cut = self.norm2(g_weight + short_cut)




        return short_cut


class CMPA_Align(nn.Module):
    #   224 的话 window_size = 4,HW = 224
    def __init__(self,window_size = 6,HW = 576):
        super().__init__()
        self.HW = HW
        self.window_size = window_size
        self.window_nums = int(self.HW/(self.window_size * self.window_size))
        self.cov_nums = self.window_nums

        self.cov_list = nn.ModuleList([])
        for i in range(self.cov_nums):
            subCov = nn.Conv2d(512,1,1)
            self.cov_list.append(subCov)
        #[nn.Conv2d(512,1,1).to('cuda') for _ in range(self.cov_nums)]
        # self.window_f_i = []
        self.softmax = nn.Softmax(1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.guid_attn = MultiHeadSelfAttention(
            embed_dim=512,
            num_heads=8
        )
        self.layer_norm1 = nn.LayerNorm(512)

        self.ff_layer = FeedForward(
            embed_dim=512,
            ffn_embed_dim=512 * 4,
            relu_dropout=0.1
        )
        self.layer_norm2 = torch.nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.1)





    def forward(self, text_feature, gx, gv):  # B,17,512  B,1,512 B,256,512
        B,HW,C = gv.size()
        window_nums = int(HW / (self.window_size * self.window_size))  # 16个窗口  336: 576的话
        self.cov_nums = window_nums
        w_size = self.window_size
        window_f = []
        for i in range(window_nums):
            window_f.append(gv[:,i*w_size*w_size:(i+1)*w_size*w_size,:])

        # 将每一个窗口的特征进行空间注意力操作
        feature_map_list = []
        for i in range(window_nums):
            window_f_i = window_f[i]
            cov_i = self.cov_list[i]

            feature_map_i  = cov_i(window_f_i.permute(0,2,1).unsqueeze(2)).squeeze(1).permute(0,2,1)  # B,16,1

            window_f_i = self.softmax(feature_map_i) * window_f_i
            # 池化
            feature_map_list.append(self.pool(window_f_i.permute(0,2,1)).permute(0,2,1)) 

        local_feature = torch.cat(feature_map_list,1)  # B,window_nums+1,512
        # 这里是concata改为sum做消融
        # local_feature = gx + local_feature
        local_feature = torch.cat((gx, local_feature), 1)
        
        # 开始和文本进行交互
        short_cut = text_feature
        text_feature = self.guid_attn(
            q = text_feature,
            k = local_feature,
            v = local_feature,
            mask = None
        )

        text_feature = self.dropout(text_feature)
        text_feature = self.layer_norm1(short_cut + text_feature)

        # FFN
        short_cut = text_feature
        text_feature = self.ff_layer(text_feature)
        text_feature = self.dropout(text_feature)
        text_feature = self.layer_norm2(text_feature + short_cut)

        return text_feature



if __name__ == '__main__':
    # decoder = Decoder_CMPA(vocab_size=9999)
    # decoder.to(device='cuda')
    # encoder_out = torch.randn((5,169,512),device='cuda')
    # seq_input = torch.zeros((5,17),dtype=torch.long,device='cuda')

    gx = torch.rand((5,1,512))
    text_feature = torch.rand(5,17,512)
    gv = torch.rand(5,256,512)
    model= CMPA_Align()
    model(text_feature,gx,gv)
    # out = decoder(gx,seq_input, encoder_out)
    print("ok")