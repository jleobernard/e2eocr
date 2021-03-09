import os

import matplotlib.pyplot as plt
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.crnn import CRNN
from utils.characters import get_sentence_length, characters, get_selected_character
from utils.data_utils import parse_args
from utils.image_helper import CustomRawDataSet
from utils.tensor_helper import to_best_device, do_load_model

args = parse_args()

data_path = args.data_path
models_rep = args.models_path
load_model = 'True' == args.load
NUM_EPOCHS = int(args.epoch)
BATCH_SIZE = int(args.batch)
MOMENTUM = 0.9
MAX_SENTENCE_LENGTH = int(args.sentence)
LEARNING_RATE = float(args.lr)
MAX_LR = float(args.max_lr)
features_multiplicity = int(args.feat_mul)


if torch.cuda.is_available():
    print("CUDA will be used")
else:
    print("CUDA won't be used")

def imshow(inp):
    inp = inp.numpy()[0]
    mean = 0.1307
    std = 0.3081
    inp = ((mean * inp) + std)
    plt.imshow(inp, cmap='gray')
    plt.show()

print(f"Loading dataset ...")
ds = CustomRawDataSet(root_dir=data_path)
#imshow(ds[5][0])
#exit()
print(f"...dataset loaded")
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
model = to_best_device(CRNN())
best_model = to_best_device(CRNN())


if load_model:
    if not do_load_model(models_rep, model):
        model.initialize_weights()
else:
    if not os.path.exists(models_rep):
        os.makedirs(models_rep)
    model.initialize_weights()

model.train()
loss = to_best_device(nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean"))
#optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', cooldown=0, verbose=True, patience=10)
optimizer = torch.optim.Adadelta(model.parameters())
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
#                                          max_lr=MAX_LR,
#                                          steps_per_epoch=len(ds),
#                                          epochs=NUM_EPOCHS,
#                                          anneal_strategy='linear')
start = time.time()
losses = []
min_loss = sys.maxsize
do_save = True


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

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, batch_data in enumerate(dataloader):
        data_cpu, labels_cpu = batch_data
        data = to_best_device(data_cpu)
        labels = to_best_device(labels_cpu)
        optimizer.zero_grad()
        outputs = model(data)
        for output in outputs:
            print(from_predicted_labels(output))
        # Because outputs is of dimension (batch_size, seq, nb_chars) we have to permute the dimensions to fit cttloss
        # expected inputs
        outputs = outputs.permute(1, 0, 2) # seq, batch_size, nb_chars = outputs.shape
        curr_loss = loss(nn.functional.log_softmax(outputs, 2), labels,
                         torch.tensor(outputs.shape[1] * [outputs.shape[0]], dtype=torch.long),
                         torch.tensor([get_sentence_length(label) for label in labels], dtype=torch.long))
        curr_loss.backward()
        optimizer.step()
        running_loss += curr_loss.item()
    #scheduler.step(round(running_loss * 1000))
    print(f'[{epoch}]Loss is {running_loss}')
    losses.append(running_loss)
    if running_loss < min_loss:
        do_save = True
        best_model.load_state_dict(model.state_dict())
        min_loss = running_loss
    else:
        if do_save:
            print(f'[{epoch}] Best loss so far is {min_loss} so we will save in best')
            torch.save(best_model.state_dict(), f"{models_rep}/best.pt")
        do_save = False
end = time.time()
print(f"It took {end - start}")
if do_save:
    print(f'[END] Best loss was {min_loss} so we will save in best')
    torch.save(best_model.state_dict(), f"{models_rep}/best.pt")
plt.plot(losses)
plt.show()
