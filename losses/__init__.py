from losses.cross_entropy import CrossEntropy
from losses.label_smoothing import LabelSmoothing
from losses.reward_criterion import RewardCriterion

__factory = {
    'CrossEntropy': CrossEntropy,  # 交叉熵损失
    'LabelSmoothing': LabelSmoothing,   # 标签平滑损失
    'RewardCriterion': RewardCriterion,     # 强化学习
}

def names():
    return sorted(__factory.keys())

def create(name):
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]()