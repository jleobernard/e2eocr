from typing import Union

import math
import torch
from torch.nn import Module

from utils.data_utils import get_last_model_params

cuda_available = torch.cuda.is_available()


def to_best_device(tensor: torch.Tensor) -> Union[torch.Tensor, Module]:
    if cuda_available:
        tensor = tensor.cuda()
    return tensor


def do_load_model(models_rep: str, model: Module, exit_on_error: bool = False):
    last_model_file = get_last_model_params(models_rep)
    if not last_model_file:
        print(f"No parameters to load from {models_rep}")
        if exit_on_error:
            exit(-1)
        else:
            return
    print(f"Loading parameters from {last_model_file}")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(last_model_file))
    else:
        model.load_state_dict(torch.load(last_model_file, map_location=torch.device('cpu')))


def get_dim_out(height: int, width: int, kernel: (int, int) = (3, 3), max_pool_kernel: (int, int) = (2, 2),
                padding=0, dilatation=1):
    """
    Applies the rules described at https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=convolution#torch.nn.Conv2d
    :param height:
    :param width:
    :param kernel:
    :param max_pool_kernel:
    :param padding:
    :param dilatation:
    :return:
    """
    hout = math.floor((height + 2 * padding - dilatation * (kernel[0] - 1) - 1) + 1)
    wout = math.floor((width + 2 * padding - dilatation * (kernel[1] - 1) - 1) + 1)
    h = math.floor((hout + 2 * padding - dilatation * (max_pool_kernel[0] - 1) - 1) / max_pool_kernel[0] + 1)
    w = math.floor((wout + 2 * padding - dilatation * (max_pool_kernel[1] - 1) - 1) / max_pool_kernel[1] + 1)
    return h, w
