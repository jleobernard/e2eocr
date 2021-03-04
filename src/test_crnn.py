import pandas as pd
import time
import sys
import torch
from torch.utils.data import DataLoader

from model.crnn import CRNN
from model.paragraph_reader import ParagraphReader
from model.simple_mdlstm import SimpleModelMDLSTM
from model.simple_model import SimpleModel
from utils.characters import index_char, blank_character, void_character, characters
from utils.data_utils import parse_args
from utils.image_helper import get_dataset, CustomDataSetSimple, CustomRawDataSet
from utils.tensor_helper import do_load_model, to_best_device

if torch.cuda.is_available():
    print("CUDA will be used")
else:
    print("CUDA won't be used")

args = parse_args()

data_path = args.data_path
models_rep = args.models_path

BATCH_SIZE = int(args.batch)
MAX_SENTENCE_LENGTH = int(args.sentence)


print(f"Loading dataset from {data_path}...")
ds = CustomRawDataSet(root_dir=data_path)
print(f"...dataset loaded")
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
model = to_best_device(CRNN())

do_load_model(models_rep, model, exit_on_error=True)

model.eval()
start = time.time()
results = []


def from_target_labels(target: torch.Tensor) -> str:
    """

    :param target: tensor of shape (n) with n being the length of the sequence
    and each element containing the index of one of the character
    :return: a trimmed string containing only relevant characters
    """
    return ''.join([characters[i] for i in target.cpu().numpy().astype(int)])


def get_selected_character(i: torch.Tensor):
    return torch.argmax(i).item()


def from_predicted_labels(predicted: torch.Tensor) -> str:
    """

    :param predicted: tensor of shape (L, X) with :
    - L being the length of the sequence
    - X being the size of the list of known characters
    and each element containing the index of one of the character
    :return: a trimmed string containing only relevant characters
    """
    as_np = predicted.detach()
    all_chars = [get_selected_character(i) for _, i in enumerate(as_np)]
    final = []
    current_char = None
    for char in all_chars:
        if not char == current_char:
            current_char = char
            if char == 0:
                pass
            else:
                final.append(char)
    return ''.join([str(characters[i]) for i in final])


for i, batch_data in enumerate(dataloader):
    data, labels = batch_data
    data = to_best_device(data)
    labels = to_best_device(labels)
    outputs = model(data)
    for j in range(len(outputs)):
        results.append({'target': from_target_labels(labels[j]), 'predicted': from_predicted_labels(outputs[j])})
end = time.time()
print(f"It took {end - start}")
df = pd.DataFrame({'target': [r['target'] for r in results], 'predicted': [r['predicted'] for r in results]})
print(df)