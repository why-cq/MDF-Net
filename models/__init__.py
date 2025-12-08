from models.ACT_Transformer import ACT_Transformer, ACT_Transformer_adaptive, ACT_Transformer_adaptive_336
from models.CLIP_swim_transform import CLIP_SWM_196, CLIP_SWM_196_v2
from models.Cross_Transformer import Cross_Transformer_196_decoder_v1, Cross_Transformer_196_decoder_v2, Cross_Transformer_576_decoder_v1, Cross_Transformer_Dual
from models.pure_transformer import PureT
from models.pure_transformer import PureT_Base
from models.pure_transformer import PureT_Base_22K
from models.CLIP_Transformer import CLIP_Transformer_49, CLIP_Transformer_AEF, CLIP_Transformer_CMPA, CLIP_Transformer_L_256, CLIP_Transformer_196, \
    CLIP_Transformer_196_decoder_v1, CLIP_Transformer_196_decoder_v2, CLIP_Transformer_196_encoder_M, CLIP_Transformer_L_256_M
from models.VLAD_Transformer import VLAD_Transformer_49, VLAD_Transformer_196, VLAD_Transformer_L_256
from models.DEF_transformer import DEF_Transformer_49, DEF_Transformer_196

__factory = {
    'PureT': PureT,
    'PureT_Base': PureT_Base,
    'PureT_Base_22K': PureT_Base_22K,
    'CLIP_Transformer_49': CLIP_Transformer_49,
    'CLIP_Transformer_196': CLIP_Transformer_196,
    'CLIP_Transformer_196_decoder_v1' : CLIP_Transformer_196_decoder_v1,
    'CLIP_Transformer_196_decoder_v2' : CLIP_Transformer_196_decoder_v2,
    'CLIP_Transformer_196_encoder_M' : CLIP_Transformer_196_encoder_M,
    'VLAD_Transformer_49': VLAD_Transformer_49,
    'VLAD_Transformer_196': VLAD_Transformer_196,
    'VLAD_Transformer_L_256': VLAD_Transformer_L_256,
    'DEF_Transformer_49': DEF_Transformer_49,
    'DEF_Transformer_196': DEF_Transformer_196,
    'CLIP_Transformer_L_256': CLIP_Transformer_L_256,
    'CLIP_Transformer_L_256_M':CLIP_Transformer_L_256_M,
    'CLIP_SWM_196' : CLIP_SWM_196,
    'CLIP_SWM_196_v2' : CLIP_SWM_196_v2,
    'Cross_Transformer_196_decoder_v1' :Cross_Transformer_196_decoder_v1,
    'Cross_Transformer_Dual' :Cross_Transformer_Dual,
    'Cross_Transformer_196_decoder_v2': Cross_Transformer_196_decoder_v2,
    'Cross_Transformer_576_decoder_v1': Cross_Transformer_576_decoder_v1,
    'CLIP_Transformer_AEF':CLIP_Transformer_AEF,
    'CLIP_Transformer_CMPA':CLIP_Transformer_CMPA,
    'ACT_Transformer' : ACT_Transformer,
    'ACT_Transformer_adaptive' : ACT_Transformer_adaptive,
    'ACT_Transformer_adaptive_336' : ACT_Transformer_adaptive_336


}


# 这里是模型构建的工厂,以后自己写的模型一定要在这里进行注册

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)
