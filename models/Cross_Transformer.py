import os

import clip
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.Decoder_DIY.cross_decoder_v1 import  Cross_Decoder_v1
from models.basic_model import BasicModel
from models.backbone.encoder_decoder_easy import DecoderWithAttention
from torch.nn.functional import softmax, dropout, cross_entropy, normalize
# from models.encoder_decoder.PureT_encoder import Encoder
from models.encoder_decoder.CLIP_encoder_decoder.CLIP_encoder import Encoder, Encoder_M
from models.encoder_decoder.Cross_encoder_decoder.Cross_encoder import Encoder as Cross_encoder
from models.encoder_decoder.Cross_encoder_decoder.Cross_decoder import Cross_Decoder_v2 as Cross_decoder

from models.encoder_decoder.PureT_decoder import Decoder
from utils import utils
from utils.config import cfg

# For masked MSA
"""
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
"""


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
    return subsequent_mask == 0



class Cross_Transformer_196_decoder_v1(BasicModel):
    def __init__(self):
        super(Cross_Transformer_196_decoder_v1, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1

        # 这里使用的是swin_large我们已经使用clip提取特征


        # raw Dimension to Model Dimension
        if cfg.MODEL.ATT_FEATS_DIM == cfg.MODEL.ATT_FEATS_EMBED_DIM:
            self.att_embed = nn.Identity()
        else:
            self.att_embed = nn.Sequential(
                nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM),
                utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT),
                nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM) if cfg.MODEL.ATT_FEATS_NORM == True else nn.Identity(),
                nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED)

            )

        use_gx = True
        self.encoder = Encoder(
            embed_dim=cfg.MODEL.ATT_FEATS_EMBED_DIM,
            input_resolution=(16, 16),   # 输入的grid特征
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            window_size=8,    # 窗口大小
            shift_size=4,     #移动距离
            mlp_ratio=4, 
            dropout=0.1,
            use_gx=use_gx
        )

        self.decoder = Cross_Decoder_v1(
            vocab_size=self.vocab_size,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            cross_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            text_encoder_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            ff_dropout=cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
        )

        # 12.17 todo:对比学习需要的参数,参照ALBEF代码
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.queue_size = 1024
        self.momentum = 0.995

        # 创建动量模型
        self.encoder_m = Encoder(
            embed_dim=cfg.MODEL.ATT_FEATS_EMBED_DIM,
            input_resolution=(16, 16),  # 输入的grid特征
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            window_size=8,  # 改为14的一半
            shift_size=4,  # 每次移动的距离 改为4
            mlp_ratio=4,
            dropout=0.1,
            use_gx=use_gx
        )

        self.decoder_m = Cross_Decoder_v1(
            vocab_size=self.vocab_size,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            cross_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            text_encoder_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            ff_dropout=cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
        )

        self.model_pairs = [[self.encoder, self.encoder_m],
                            [self.decoder, self.decoder_m],
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("v_queue", torch.randn(self.queue_size, 512))
        self.register_buffer("t_queue", torch.randn(self.queue_size, 512))
        self.v_queue = nn.functional.normalize(self.v_queue, dim=1)
        self.t_queue = nn.functional.normalize(self.t_queue, dim=1)

    def forward(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]

        # att_mask for features
        # todo:怎么找不到这个att_mask的出处????,先自己定义一个  // 找到了在dataloder中的sample_collate中进行的返回处理
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        # att_mask = torch.ones(att_feats.shape[0],att_feats.shape[1])
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        # words mask [B, L, L] 构建seq_mask
        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:, 0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################

        # att_feats就是从CLip中出来的特征  5B,49,768--> 60,49,512 为batch_size*5,144 全为1
        att_feats = self.att_embed(att_feats)

        # 编码器的输入为 batch_size*5 , 144 ,512
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # decoder_out = self.decoder(gx, seq, encoder_out, seq_mask, att_mask)  # 第二阶段返回一个值
        
        decoder_out, cls_text = self.decoder(gx, seq, encoder_out, seq_mask, att_mask)   #  第一阶段返回两个值解码器需要全局特征
        
        
        # SCST阶段暂时不要这个损失函数
        expand_size = 1
        v_embeds = gx
        t_embeds = cls_text
        with torch.no_grad():
            self.temp.clamp_(min=0.01, max=0.5)
            self._momentum_update()
            v_embeds_m, _ = self.encoder_m(att_feats,att_mask)
            _, t_embeds_m = self.decoder_m(gx, seq, encoder_out, seq_mask, att_mask)

            v_embeds_all = torch.cat([v_embeds_m, self.v_queue.clone().detach()], dim=0)
            t_embeds_all = torch.cat([t_embeds_m, self.t_queue.clone().detach()], dim=0)

        sim_i2t = torch.div(torch.matmul(v_embeds, t_embeds_all.t()), self.temp)
        sim_t2i = torch.div(torch.matmul(t_embeds, v_embeds_all.t()), self.temp)
        sim_i2t_target = torch.zeros_like(sim_i2t, device=sim_i2t.device)
        sim_t2i_target = torch.zeros_like(sim_t2i, device=sim_t2i.device)
        for i in range(len(sim_i2t)):
            sim_i2t_target[i, i * expand_size:(i + 1) * expand_size] = 1 / expand_size
            sim_t2i_target[i * expand_size:(i + 1) * expand_size, i] = 1
        co_loss = (cross_entropy(sim_i2t, sim_i2t_target, label_smoothing=0) +
                   cross_entropy(sim_t2i, sim_t2i_target, label_smoothing=0)) / 2

        self._dequeue_and_enqueue(v_embeds_m, t_embeds_m)



        return F.log_softmax(decoder_out, dim=-1), co_loss  # 第二阶段只返回第一个损失

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]

        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gx = kwargs[cfg.PARAM.GLOBAL_FEAT]
        # p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        # state[0][0]: [B, seq_len-1]，previously generated words
        # ys: [B, seq_len]
        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)

        seq_mask = subsequent_mask(ys.size(1)).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(
            1)

        # [B, 1, Vocab_Size] --> [B, Vocab_Size]
        decoder_out  = self.decoder(gx, ys[:, -1].unsqueeze(-1), encoder_out, seq_mask, att_mask).squeeze(1)

        logprobs = F.log_softmax(decoder_out, dim=-1)
        return logprobs, [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        # att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        # 初始化解码器中的缓存
        self.decoder.init_buffer(batch_size)
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            # [B*cur_beam_size, Vocab_size] --> [B, cur_beam_size, Vocab_size]
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            # sum of logprob
            # [B, cur_beam_size, Vocab_size]
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # [B, beam_size], [B, beam_size]
            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # 更行解码器中的缓存
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                                             selected_beam.unsqueeze(-1).expand(batch_size, beam_size,
                                                                                word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                # expand input
                encoder_out = utils.expand_tensor(encoder_out, beam_size)
                gx = utils.expand_tensor(gx, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                # p_att_feats_tmp = []
                # for p_feat in p_att_feats:
                #     p_key, p_value2 = p_feat
                #     p_key = utils.expand_tensor(p_key, beam_size)
                #     p_value2 = utils.expand_tensor(p_value2, beam_size)
                #     p_att_feats_tmp.append((p_key, p_value2))

                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats_tmp

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.decoder.clear_buffer()
        return outputs, log_probs

    def decode(self, **kwargs):
        beam_size = kwargs['BEAM_SIZE']
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        batch_size = att_feats.size(0)
        # att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)
        self.decoder.init_buffer(batch_size)

        state = None
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        # inference word by word
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            logprobs_t, state = self.get_logprobs_state(**kwargs)

            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:, t] = wt
            logprobs[:, t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        self.decoder.clear_buffer()
        return sents, logprobs

    def flops(self):
        flops = 0
        flops += self.backbone.flops()
        # self.att_embed
        flops += 1536 * 512
        # encoder decoder
        flops += self.encoder.flops()
        flops += self.encoder.flops()
        # flops += self.decoder.flops()
        return flops

    
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    # 出入队列的操作需要进行修改
    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_embeds, t_embeds):
        # 单卡无需将其他的GPU上的张量拿过来
        # v_embeds = self.concat_all_gather(v_embeds)
        # t_embeds = self.concat_all_gather(t_embeds)
        # 队头入队,队尾出队
        v_embeds = torch.cat((v_embeds, self.v_queue.clone().detach()), dim=0)
        t_embeds = torch.cat((t_embeds, self.t_queue.clone().detach()), dim=0)
        self.v_queue = v_embeds[:len(self.v_queue)]
        self.t_queue = t_embeds[:len(self.t_queue)]

class Cross_Transformer_196_decoder_v2(BasicModel):
    def __init__(self):
        super(Cross_Transformer_196_decoder_v2, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1

        # 这里使用的是swin_large我们已经使用clip提取特征


        # raw Dimension to Model Dimension
        if cfg.MODEL.ATT_FEATS_DIM == cfg.MODEL.ATT_FEATS_EMBED_DIM:
            self.att_embed = nn.Identity()
        else:
            self.att_embed = nn.Sequential(
                nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM),
                utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT),
                nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM) if cfg.MODEL.ATT_FEATS_NORM == True else nn.Identity(),
                nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED)

            )

        use_gx = True
        self.encoder = Encoder(
            embed_dim=cfg.MODEL.ATT_FEATS_EMBED_DIM,
            input_resolution=(16, 16),   # 输入的grid特征
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            window_size=8,    # 窗口大小
            shift_size=4,     #移动距离
            mlp_ratio=4, 
            dropout=0.1,
            use_gx=use_gx
        )

        self.decoder = Cross_Decoder_v1(
            vocab_size=self.vocab_size,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            cross_depth=4,  # 这个增大一点试试
            text_encoder_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            ff_dropout=cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
        )

        # 12.17 todo:对比学习需要的参数,参照ALBEF代码
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.queue_size = 4096
        self.momentum = 0.995

        # 创建动量模型
        self.encoder_m = Encoder(
            embed_dim=cfg.MODEL.ATT_FEATS_EMBED_DIM,
            input_resolution=(16, 16),  # 输入的grid特征
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            window_size=8,  # 改为14的一半
            shift_size=4,  # 每次移动的距离 改为4
            mlp_ratio=4,
            dropout=0.1,
            use_gx=use_gx
        )

        self.decoder_m = Cross_Decoder_v1(
            vocab_size=self.vocab_size,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            cross_depth=4,
            text_encoder_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            ff_dropout=cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
        )

        self.model_pairs = [[self.encoder, self.encoder_m],
                            [self.decoder, self.decoder_m],
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("v_queue", torch.randn(self.queue_size, 512))
        self.register_buffer("t_queue", torch.randn(self.queue_size, 512))
        self.v_queue = nn.functional.normalize(self.v_queue, dim=1)
        self.t_queue = nn.functional.normalize(self.t_queue, dim=1)

    def forward(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]

        # att_mask for features
        # todo:怎么找不到这个att_mask的出处????,先自己定义一个  // 找到了在dataloder中的sample_collate中进行的返回处理
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        # att_mask = torch.ones(att_feats.shape[0],att_feats.shape[1])
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        # words mask [B, L, L] 构建seq_mask
        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:, 0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################

        # att_feats就是从CLip中出来的特征  5B,49,768--> 60,49,512 为batch_size*5,144 全为1
        att_feats = self.att_embed(att_feats)

        # 编码器的输入为 batch_size*5 , 144 ,512
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # decoder_out = self.decoder(gx, seq, encoder_out, seq_mask, att_mask)  # 第二阶段返回一个值
        
        decoder_out, cls_text = self.decoder(gx, seq, encoder_out, seq_mask, att_mask)   #  第一阶段返回两个值解码器需要全局特征
        
        
        # SCST阶段暂时不要这个损失函数
        expand_size = 1
        v_embeds = gx
        t_embeds = cls_text
        with torch.no_grad():
            self.temp.clamp_(min=0.01, max=0.5)
            self._momentum_update()
            v_embeds_m, _ = self.encoder_m(att_feats,att_mask)
            _, t_embeds_m = self.decoder_m(gx, seq, encoder_out, seq_mask, att_mask)

            v_embeds_all = torch.cat([v_embeds_m, self.v_queue.clone().detach()], dim=0)
            t_embeds_all = torch.cat([t_embeds_m, self.t_queue.clone().detach()], dim=0)

        sim_i2t = torch.div(torch.matmul(v_embeds, t_embeds_all.t()), self.temp)
        sim_t2i = torch.div(torch.matmul(t_embeds, v_embeds_all.t()), self.temp)
        sim_i2t_target = torch.zeros_like(sim_i2t, device=sim_i2t.device)
        sim_t2i_target = torch.zeros_like(sim_t2i, device=sim_t2i.device)
        for i in range(len(sim_i2t)):
            sim_i2t_target[i, i * expand_size:(i + 1) * expand_size] = 1 / expand_size
            sim_t2i_target[i * expand_size:(i + 1) * expand_size, i] = 1
        co_loss = (cross_entropy(sim_i2t, sim_i2t_target, label_smoothing=0) +
                   cross_entropy(sim_t2i, sim_t2i_target, label_smoothing=0)) / 2

        self._dequeue_and_enqueue(v_embeds_m, t_embeds_m)



        return F.log_softmax(decoder_out, dim=-1), co_loss  # 第二阶段只返回第一个损失

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]

        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gx = kwargs[cfg.PARAM.GLOBAL_FEAT]
        # p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        # state[0][0]: [B, seq_len-1]，previously generated words
        # ys: [B, seq_len]
        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)

        seq_mask = subsequent_mask(ys.size(1)).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(
            1)

        # [B, 1, Vocab_Size] --> [B, Vocab_Size]
        decoder_out  = self.decoder(gx, ys[:, -1].unsqueeze(-1), encoder_out, seq_mask, att_mask).squeeze(1)

        logprobs = F.log_softmax(decoder_out, dim=-1)
        return logprobs, [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        # att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        # 初始化解码器中的缓存
        self.decoder.init_buffer(batch_size)
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            # [B*cur_beam_size, Vocab_size] --> [B, cur_beam_size, Vocab_size]
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            # sum of logprob
            # [B, cur_beam_size, Vocab_size]
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # [B, beam_size], [B, beam_size]
            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # 更行解码器中的缓存
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                                             selected_beam.unsqueeze(-1).expand(batch_size, beam_size,
                                                                                word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                # expand input
                encoder_out = utils.expand_tensor(encoder_out, beam_size)
                gx = utils.expand_tensor(gx, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                # p_att_feats_tmp = []
                # for p_feat in p_att_feats:
                #     p_key, p_value2 = p_feat
                #     p_key = utils.expand_tensor(p_key, beam_size)
                #     p_value2 = utils.expand_tensor(p_value2, beam_size)
                #     p_att_feats_tmp.append((p_key, p_value2))

                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats_tmp

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.decoder.clear_buffer()
        return outputs, log_probs

    def decode(self, **kwargs):
        beam_size = kwargs['BEAM_SIZE']
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        batch_size = att_feats.size(0)
        # att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)
        self.decoder.init_buffer(batch_size)

        state = None
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        # inference word by word
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            logprobs_t, state = self.get_logprobs_state(**kwargs)

            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:, t] = wt
            logprobs[:, t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        self.decoder.clear_buffer()
        return sents, logprobs

    def flops(self):
        flops = 0
        flops += self.backbone.flops()
        # self.att_embed
        flops += 1536 * 512
        # encoder decoder
        flops += self.encoder.flops()
        flops += self.encoder.flops()
        # flops += self.decoder.flops()
        return flops

    
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    # 出入队列的操作需要进行修改
    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_embeds, t_embeds):
        # 单卡无需将其他的GPU上的张量拿过来
        # v_embeds = self.concat_all_gather(v_embeds)
        # t_embeds = self.concat_all_gather(t_embeds)
        # 队头入队,队尾出队
        v_embeds = torch.cat((v_embeds, self.v_queue.clone().detach()), dim=0)
        t_embeds = torch.cat((t_embeds, self.t_queue.clone().detach()), dim=0)
        self.v_queue = v_embeds[:len(self.v_queue)]
        self.t_queue = t_embeds[:len(self.t_queue)]

class Cross_Transformer_Dual(BasicModel):
    def __init__(self):
        super(Cross_Transformer_Dual, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1


        # 初始化视觉编码器的分类头
        # visual = clip.load("ViT-L/14")[0].visual
        # self.visual_class_embedding = visual.class_embedding
        self.att_embed = nn.Sequential(
                nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM),
                utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT),
                nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM) if cfg.MODEL.ATT_FEATS_NORM == True else nn.Identity(),
                nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED)
            )

        use_gx = True
        self.encoder = Encoder(
            embed_dim=cfg.MODEL.ATT_FEATS_EMBED_DIM,
            input_resolution=(16, 16),   # 输入的grid特征
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            window_size=8,    # 改为14的一半
            shift_size=4,     #每次移动的距离 改为4
            mlp_ratio=4,
            dropout=0.1,
            use_gx=use_gx
        )

        self.decoder = Cross_decoder(
            vocab_size=self.vocab_size,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            cross_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            text_encoder_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            ff_dropout=cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
        )

        # 12.17 todo:对比学习需要的参数,参照ALBEF代码
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.queue_size = 1024
        self.momentum = 0.995

        # 创建动量模型
        self.encoder_m = Encoder(
            embed_dim=cfg.MODEL.ATT_FEATS_EMBED_DIM,
            input_resolution=(16, 16),   # 输入的grid特征
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            window_size=8,    # 改为14的一半
            shift_size=4,     #每次移动的距离 改为4
            mlp_ratio=4,
            dropout=0.1,
            use_gx=use_gx
        )

        self.decoder_m = Cross_decoder(
            vocab_size=self.vocab_size,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            cross_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            text_encoder_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            ff_dropout=cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
        )

        self.model_pairs = [[self.encoder, self.encoder_m],
                            [self.decoder, self.decoder_m],
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("v_queue", torch.randn(self.queue_size, 512))
        self.register_buffer("t_queue", torch.randn(self.queue_size, 512))
        self.v_queue = nn.functional.normalize(self.v_queue, dim=1)
        self.t_queue = nn.functional.normalize(self.t_queue, dim=1)

    def forward(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]

        # att_mask for features
        # todo:怎么找不到这个att_mask的出处????,先自己定义一个  // 找到了在dataloder中的sample_collate中进行的返回处理
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        # att_mask = torch.ones(att_feats.shape[0],att_feats.shape[1])
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        # words mask [B, L, L] 构建seq_mask
        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:, 0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################

        #5B,256,1024--> 5B,256,512
        att_feats = self.att_embed(att_feats)

        # 编码器的输入为 batch_size*5 , 256 ,1024
        gx, encoder_out = self.encoder(att_feats, att_mask)
        decoder_out= self.decoder(gx, seq, encoder_out, seq_mask, att_mask)
        decoder_out, cls_text = self.decoder(gx, seq, encoder_out, seq_mask, att_mask)   # 解码器需要全局特征

        # expand_size = 1
        # v_embeds = gx
        # t_embeds = cls_text
        # with torch.no_grad():
        #     self.temp.clamp_(min=0.01, max=0.5)
        #     self._momentum_update()
        #     v_embeds_m, _ = self.encoder_m(att_feats,att_mask)
        #     _, t_embeds_m = self.decoder_m(gx, seq, encoder_out, seq_mask, att_mask)

        #     v_embeds_all = torch.cat([v_embeds_m, self.v_queue.clone().detach()], dim=0)
        #     t_embeds_all = torch.cat([t_embeds_m, self.t_queue.clone().detach()], dim=0)

        # sim_i2t = torch.div(torch.matmul(v_embeds, t_embeds_all.t()), self.temp)
        # sim_t2i = torch.div(torch.matmul(t_embeds, v_embeds_all.t()), self.temp)
        # sim_i2t_target = torch.zeros_like(sim_i2t, device=sim_i2t.device)
        # sim_t2i_target = torch.zeros_like(sim_t2i, device=sim_t2i.device)
        # todo: 伪标签的生成??
        # for i in range(len(sim_i2t)):
        #     sim_i2t_target[i, i * expand_size:(i + 1) * expand_size] = 1 / expand_size
        #     sim_t2i_target[i * expand_size:(i + 1) * expand_size, i] = 1
        # co_loss = (cross_entropy(sim_i2t, sim_i2t_target, label_smoothing=0) +
        #            cross_entropy(sim_t2i, sim_t2i_target, label_smoothing=0)) / 2

        # self._dequeue_and_enqueue(v_embeds_m, t_embeds_m)



        return F.log_softmax(decoder_out, dim=-1) #, co_loss

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]

        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gx = kwargs[cfg.PARAM.GLOBAL_FEAT]
        # p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        # state[0][0]: [B, seq_len-1]，previously generated words
        # ys: [B, seq_len]
        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)

        seq_mask = subsequent_mask(ys.size(1)).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(
            1)

        # [B, 1, Vocab_Size] --> [B, Vocab_Size]
        decoder_out = self.decoder(gx, ys[:, -1].unsqueeze(-1), encoder_out, seq_mask, att_mask).squeeze(1)

        logprobs = F.log_softmax(decoder_out, dim=-1)
        return logprobs, [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        # att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        # 初始化解码器中的缓存
        self.decoder.init_buffer(batch_size)
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            # [B*cur_beam_size, Vocab_size] --> [B, cur_beam_size, Vocab_size]
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            # sum of logprob
            # [B, cur_beam_size, Vocab_size]
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # [B, beam_size], [B, beam_size]
            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # 更行解码器中的缓存
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                                             selected_beam.unsqueeze(-1).expand(batch_size, beam_size,
                                                                                word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                # expand input
                encoder_out = utils.expand_tensor(encoder_out, beam_size)
                gx = utils.expand_tensor(gx, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                # p_att_feats_tmp = []
                # for p_feat in p_att_feats:
                #     p_key, p_value2 = p_feat
                #     p_key = utils.expand_tensor(p_key, beam_size)
                #     p_value2 = utils.expand_tensor(p_value2, beam_size)
                #     p_att_feats_tmp.append((p_key, p_value2))

                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats_tmp

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.decoder.clear_buffer()
        return outputs, log_probs

    def decode(self, **kwargs):
        beam_size = kwargs['BEAM_SIZE']
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        batch_size = att_feats.size(0)
        # att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)
        self.decoder.init_buffer(batch_size)

        state = None
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        # inference word by word
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            logprobs_t, state = self.get_logprobs_state(**kwargs)

            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:, t] = wt
            logprobs[:, t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        self.decoder.clear_buffer()
        return sents, logprobs

    def flops(self):
        flops = 0
        flops += self.backbone.flops()
        # self.att_embed
        flops += 1536 * 512
        # encoder decoder
        flops += self.encoder.flops()
        flops += self.encoder.flops()
        # flops += self.decoder.flops()
        return flops

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    # 出入队列的操作需要进行修改
    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_embeds, t_embeds):
        # 单卡无需将其他的GPU上的张量拿过来
        # v_embeds = self.concat_all_gather(v_embeds)
        # t_embeds = self.concat_all_gather(t_embeds)
        # 队头入队,队尾出队
        v_embeds = torch.cat((v_embeds, self.v_queue.clone().detach()), dim=0)
        t_embeds = torch.cat((t_embeds, self.t_queue.clone().detach()), dim=0)
        self.v_queue = v_embeds[:len(self.v_queue)]
        self.t_queue = t_embeds[:len(self.t_queue)]

class Cross_Transformer_576_decoder_v1(BasicModel):
    def __init__(self):
        super(Cross_Transformer_576_decoder_v1, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1

        # 这里使用的是swin_large我们已经使用clip提取特征


        # raw Dimension to Model Dimension
        if cfg.MODEL.ATT_FEATS_DIM == cfg.MODEL.ATT_FEATS_EMBED_DIM:
            self.att_embed = nn.Identity()
        else:
            self.att_embed = nn.Sequential(
                nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM),
                utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT),
                nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM) if cfg.MODEL.ATT_FEATS_NORM == True else nn.Identity(),
                nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED)

            )

        use_gx = True
        self.encoder = Encoder(
            embed_dim=cfg.MODEL.ATT_FEATS_EMBED_DIM,
            input_resolution=(24, 24),   # 输入的grid特征
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            window_size=6,    # 窗口大小
            shift_size=3,     #移动距离
            mlp_ratio=4, 
            dropout=0.1,
            use_gx=use_gx
        )

        self.decoder = Cross_Decoder_v1(
            vocab_size=self.vocab_size,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            cross_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            text_encoder_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            ff_dropout=cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
        )

        # 12.17 todo:对比学习需要的参数,参照ALBEF代码
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.queue_size = 4096
        self.momentum = 0.995

        # 创建动量模型
        self.encoder_m = Encoder(
            embed_dim=cfg.MODEL.ATT_FEATS_EMBED_DIM,
            input_resolution=(24, 24),  # 输入的grid特征
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            window_size=6,  # 改为14的一半 
            shift_size=3,  # 每次移动的距离 改为4
            mlp_ratio=4,
            dropout=0.1,
            use_gx=use_gx
        )

        self.decoder_m = Cross_Decoder_v1(
            vocab_size=self.vocab_size,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            cross_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            text_encoder_depth=cfg.MODEL.BILINEAR.DECODE_LAYERS,
            num_heads=cfg.MODEL.BILINEAR.HEAD,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            ff_dropout=cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
        )

        self.model_pairs = [[self.encoder, self.encoder_m],
                            [self.decoder, self.decoder_m],
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("v_queue", torch.randn(self.queue_size, 512))
        self.register_buffer("t_queue", torch.randn(self.queue_size, 512))
        self.v_queue = nn.functional.normalize(self.v_queue, dim=1)
        self.t_queue = nn.functional.normalize(self.t_queue, dim=1)

    def forward(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]

        # att_mask for features
        # todo:怎么找不到这个att_mask的出处????,先自己定义一个  // 找到了在dataloder中的sample_collate中进行的返回处理
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        # att_mask = torch.ones(att_feats.shape[0],att_feats.shape[1])
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        # words mask [B, L, L] 构建seq_mask
        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:, 0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################

        # att_feats就是从CLip中出来的特征  5B,49,768--> 60,49,512 为batch_size*5,144 全为1
        att_feats = self.att_embed(att_feats)

        # 编码器的输入为 batch_size*5 , 144 ,512
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # decoder_out = self.decoder(gx, seq, encoder_out, seq_mask, att_mask)  # 第二阶段返回一个值
        
        decoder_out, cls_text = self.decoder(gx, seq, encoder_out, seq_mask, att_mask)   #  第一阶段返回两个值解码器需要全局特征
        
        
        # SCST阶段暂时不要这个损失函数
        expand_size = 1
        v_embeds = gx
        t_embeds = cls_text
        with torch.no_grad():
            self.temp.clamp_(min=0.01, max=0.5)
            self._momentum_update()
            v_embeds_m, _ = self.encoder_m(att_feats,att_mask)
            _, t_embeds_m = self.decoder_m(gx, seq, encoder_out, seq_mask, att_mask)

            v_embeds_all = torch.cat([v_embeds_m, self.v_queue.clone().detach()], dim=0)
            t_embeds_all = torch.cat([t_embeds_m, self.t_queue.clone().detach()], dim=0)

        sim_i2t = torch.div(torch.matmul(v_embeds, t_embeds_all.t()), self.temp)
        sim_t2i = torch.div(torch.matmul(t_embeds, v_embeds_all.t()), self.temp)
        sim_i2t_target = torch.zeros_like(sim_i2t, device=sim_i2t.device)
        sim_t2i_target = torch.zeros_like(sim_t2i, device=sim_t2i.device)
        for i in range(len(sim_i2t)):
            sim_i2t_target[i, i * expand_size:(i + 1) * expand_size] = 1 / expand_size
            sim_t2i_target[i * expand_size:(i + 1) * expand_size, i] = 1
        co_loss = (cross_entropy(sim_i2t, sim_i2t_target, label_smoothing=0) +
                   cross_entropy(sim_t2i, sim_t2i_target, label_smoothing=0)) / 2

        self._dequeue_and_enqueue(v_embeds_m, t_embeds_m)



        return F.log_softmax(decoder_out, dim=-1), co_loss  # 第二阶段只返回第一个损失

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]

        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gx = kwargs[cfg.PARAM.GLOBAL_FEAT]
        # p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        # state[0][0]: [B, seq_len-1]，previously generated words
        # ys: [B, seq_len]
        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)

        seq_mask = subsequent_mask(ys.size(1)).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(
            1)

        # [B, 1, Vocab_Size] --> [B, Vocab_Size]
        decoder_out  = self.decoder(gx, ys[:, -1].unsqueeze(-1), encoder_out, seq_mask, att_mask).squeeze(1)

        logprobs = F.log_softmax(decoder_out, dim=-1)
        return logprobs, [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        # att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        # 初始化解码器中的缓存
        self.decoder.init_buffer(batch_size)
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            # [B*cur_beam_size, Vocab_size] --> [B, cur_beam_size, Vocab_size]
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            # sum of logprob
            # [B, cur_beam_size, Vocab_size]
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # [B, beam_size], [B, beam_size]
            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # 更行解码器中的缓存
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                                             selected_beam.unsqueeze(-1).expand(batch_size, beam_size,
                                                                                word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                # expand input
                encoder_out = utils.expand_tensor(encoder_out, beam_size)
                gx = utils.expand_tensor(gx, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                # p_att_feats_tmp = []
                # for p_feat in p_att_feats:
                #     p_key, p_value2 = p_feat
                #     p_key = utils.expand_tensor(p_key, beam_size)
                #     p_value2 = utils.expand_tensor(p_value2, beam_size)
                #     p_att_feats_tmp.append((p_key, p_value2))

                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats_tmp

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.decoder.clear_buffer()
        return outputs, log_probs

    def decode(self, **kwargs):
        beam_size = kwargs['BEAM_SIZE']
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        batch_size = att_feats.size(0)
        # att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)
        self.decoder.init_buffer(batch_size)

        state = None
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        # inference word by word
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            logprobs_t, state = self.get_logprobs_state(**kwargs)

            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:, t] = wt
            logprobs[:, t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        self.decoder.clear_buffer()
        return sents, logprobs

    def flops(self):
        flops = 0
        flops += self.backbone.flops()
        # self.att_embed
        flops += 1536 * 512
        # encoder decoder
        flops += self.encoder.flops()
        flops += self.encoder.flops()
        # flops += self.decoder.flops()
        return flops

    
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    # 出入队列的操作需要进行修改
    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_embeds, t_embeds):
        # 单卡无需将其他的GPU上的张量拿过来
        # v_embeds = self.concat_all_gather(v_embeds)
        # t_embeds = self.concat_all_gather(t_embeds)
        # 队头入队,队尾出队
        v_embeds = torch.cat((v_embeds, self.v_queue.clone().detach()), dim=0)
        t_embeds = torch.cat((t_embeds, self.t_queue.clone().detach()), dim=0)
        self.v_queue = v_embeds[:len(self.v_queue)]
        self.t_queue = t_embeds[:len(self.t_queue)]

if __name__ == '__main__':
    model = Test_model()
    for i in range(10):
        image = torch.randn(5,256,512)
        text = torch.randn(5,17)

        loss = model(image,text)
        loss.backward()
        print('success')

