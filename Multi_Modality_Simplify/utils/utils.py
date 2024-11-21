# Time : 2023/2/16 11:30
# Author : 小霸奔
# FileName: utils.p
import torch
import numpy as np
import random


def set_requires_grad(model, requires_grad=True):
    """
    :param model: Instance of Part of Net
    :param requires_grad: Whether Need Gradient
    :return:
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def fix_randomness(SEED):
    """
    :param SEED:  Random SEED
    :return:
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

