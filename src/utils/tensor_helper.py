from typing import Union

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