import os
from typing import Union


def get_last_model_params(models_rep) -> Union[str, None]:
    file_list = os.listdir(models_rep)
    file_list = [f for f in file_list if f[-3:] == '.pt']
    file_list.sort(reverse=True)
    if len(file_list) > 0:
        return f"{models_rep}/{file_list[0]}"
    return None