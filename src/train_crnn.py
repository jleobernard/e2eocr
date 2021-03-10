import os

import matplotlib.pyplot as plt
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.crnn import CRNN
from utils.characters import get_sentence_length, from_target_labels, \
    from_predicted_labels
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
val_freq = int(args.val_freq)
val_patience = int(args.val_patience)


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


def should_stop_training(losses) -> bool:
    if len(losses) > val_patience:
        selected_values = losses[-val_patience:]
        idx_min = selected_values.index(min(selected_values))
        return idx_min < (val_patience / 2 - 1)
    else:
        return False

print(f"Loading dataset ...")
ds = CustomRawDataSet(root_dir=data_path)
len_train = int(len(ds) * 0.8)
train_set, val_set = torch.utils.data.random_split(ds, [len_train, len(ds) - len_train])
#imshow(train_set[5][0])
#exit()
print(f"...dataset loaded")
dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
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

val_losses = []

for epoch in range(NUM_EPOCHS + int(NUM_EPOCHS / val_freq)):
    running_loss = 0.0
    if epoch > 0 and epoch % (val_freq - 1) == 0:
        print(f"[{epoch}]Validating...")
        validation = True
        df = dataloader_val
    else:
        validation = False
        df = dataloader
    model.train(not validation)
    for i, batch_data in enumerate(df):
        data_cpu, labels_cpu = batch_data
        data = to_best_device(data_cpu)
        labels = to_best_device(labels_cpu)
        optimizer.zero_grad()
        outputs = model(data)
        if i == 0:
            print(f"[{epoch}]{from_target_labels(labels[0])} VS. {from_predicted_labels(outputs[0])}")
        # Because outputs is of dimension (batch_size, seq, nb_chars) we have to permute the dimensions to fit cttloss
        # expected inputs
        outputs = outputs.permute(1, 0, 2) # seq, batch_size, nb_chars = outputs.shape
        curr_loss = loss(nn.functional.log_softmax(outputs, 2), labels,
                         torch.tensor(outputs.shape[1] * [outputs.shape[0]], dtype=torch.long),
                         torch.tensor([get_sentence_length(label) for label in labels], dtype=torch.long))
        curr_loss.backward()
        optimizer.step()
        running_loss += curr_loss.item()
    if validation:
        print("...validation done")
        val_losses.append(running_loss)
        should_stop = should_stop_training(val_losses)
    else:
        print(f'[{epoch}]Loss is {running_loss}')
        losses.append(running_loss)
        should_stop = False
    if running_loss < min_loss:
        do_save = True
        best_model.load_state_dict(model.state_dict())
        min_loss = running_loss
    else:
        if do_save:
            print(f'[{epoch}] Best loss so far is {min_loss} so we will save in best')
            torch.save(best_model.state_dict(), f"{models_rep}/best.pt")
        do_save = False
    if should_stop:
        print(f"Early stopping at {epoch}")
        break
end = time.time()
print(f"It took {end - start}")
if do_save:
    print(f'[END] Best loss was {min_loss} so we will save in best')
    torch.save(best_model.state_dict(), f"{models_rep}/best.pt")
plt.plot(losses)
plt.plot(val_losses)
plt.show()
