import pandas as pd
import time
import sys
import torch
from torch.utils.data import DataLoader

from model.paragraph_reader import ParagraphReader
from utils.characters import index_char, blank_character, void_character
from utils.image_helper import get_dataset
from utils.tensor_helper import do_load_model


if torch.cuda.is_available():
    print("CUDA will be used")
else:
    print("CUDA won't be used")

data_path = sys.argv[1] # '/data/train/one-line'
models_rep = sys.argv[2] # '/data/models'

BATCH_SIZE = 5
HEIGHT = 80
WIDTH = 80
MAX_SENTENCE_LENGTH = int(sys.argv[3])


print(f"Loading dataset from {data_path}...")
ds = get_dataset(data_path, width=WIDTH, height=HEIGHT, target_length=MAX_SENTENCE_LENGTH)
print(f"...dataset loaded")
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
model = ParagraphReader(height=HEIGHT, width=WIDTH)

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
    as_np = target.numpy().astype(int)
    all_chars = [index_char(i) for i in as_np]
    final = []
    current_char = None
    for char in all_chars:
        if not char == current_char:
            current_char = char
            if char == blank_character:
                pass
            if char == void_character:
                final.append(" ")
            else:
                final.append(char)
    return ''.join(final)


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
    all_chars = [index_char(get_selected_character(i)) for _, i in enumerate(as_np)]
    final = []
    current_char = None
    for char in all_chars:
        if not char == current_char:
            current_char = char
            if char == blank_character:
                pass
            if char == void_character:
                final.append(" ")
            else:
                final.append(char)
    return ''.join(final)


for i, batch_data in enumerate(dataloader):
    data, labels = batch_data
    outputs = model(data)
    for j in range(len(outputs)):
        results.append({'target': from_target_labels(labels[j]), 'predicted': from_predicted_labels(outputs[j])})
end = time.time()
print(f"It took {end - start}")
df = pd.DataFrame({'target': [r['target'] for r in results], 'predicted': [r['predicted'] for r in results]})
print(df)