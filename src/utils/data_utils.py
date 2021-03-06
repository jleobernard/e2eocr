import argparse
import os
from typing import Union


def get_last_model_params(models_rep) -> Union[str, None]:
    if not os.path.exists(models_rep):
        os.makedirs(models_rep)
    else:
        file_list = os.listdir(models_rep)
        file_list = [f for f in file_list if f[-3:] == '.pt']
        file_list.sort(reverse=True)
        if len(file_list) > 0:
            return f"{models_rep}/{file_list[0]}"
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--data', dest='data_path',
                        help='Path to the folder containing training data', required=True)
    parser.add_argument('--models', dest='models_path',
                        help='Path to the folder containing the models (load and save)', required=True)
    parser.add_argument('--epoch', dest='epoch', default=10,
                        help='Path to the folder containing training data')
    parser.add_argument('--batch', dest='batch', default=10,
                        help='Number of images per batch')
    parser.add_argument('--sentence', dest='sentence', default=10,
                        help='Max length of sentences')
    parser.add_argument('--lr', dest='lr', default=0.0001,
                        help='Learning rate')
    parser.add_argument('--max-lr', dest='max_lr', default=0.1,
                        help='Max learning rate')
    parser.add_argument('--load', dest='load', default=False,
                        help='Load model if true')
    parser.add_argument('--feat-mul', dest='feat_mul', default=15,
                        help='Load model if true')
    parser.add_argument('--val-freq', dest='val_freq', default=5,
                        help='Validation frequency')
    parser.add_argument('--val-patience', dest='val_patience', default=5,
                        help='Validation patience')
    return parser.parse_args()