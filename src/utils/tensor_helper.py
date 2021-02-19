from typing import Union

import torch
from torch.nn import Module

cuda_available = torch.cuda.is_available()


def to_best_device(tensor: torch.Tensor) -> Union[torch.Tensor, Module]:
    if cuda_available:
        tensor = tensor.cuda()
    return tensor