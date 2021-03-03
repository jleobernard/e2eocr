import argparse
import os

import matplotlib.pyplot as plt
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.crnn import CRNN
from model.my_lstm import CustomLSTM
from model.simple_mdlstm import SimpleModelMDLSTM
from model.simple_model import SimpleModel
from utils.image_helper import CustomDataSetSimple, get_dataset
from utils.tensor_helper import to_best_device, do_load_model

parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('--data', dest='data_path',
                    help='Path to the folder containing training data', required=True)
parser.add_argument('--models', dest='models_path',
                    help='Path to the folder containing the models (load and save)', required=True)
parser.add_argument('--epoch', dest='epoch', default=10,
                    help='Path to the folder containing training data')
parser.add_argument('--batch', dest='batch', default=10,
                    help='Number of images per batch')
parser.add_argument('--height', dest='height', default=80,
                    help='Height of source images')
parser.add_argument('--width', dest='width', default=80,
                    help='Width of source images')
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
args = parser.parse_args()

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
HEIGHT = int(args.height)
WIDTH = int(args.width)


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
#ds = CustomDataSetSimple(nb_digit=MAX_SENTENCE_LENGTH, nb_samples=1000)
ds = get_dataset(data_path, width=WIDTH, height=HEIGHT, target_length=MAX_SENTENCE_LENGTH)
#imshow(ds[5][0])
#exit()
print(f"...dataset loaded")
dataloader = DataLoader(ds, batch_size=int(len(ds) / 3), shuffle=True)
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
loss = to_best_device(nn.CTCLoss(blank=0, zero_infinity=True, reduction="sum"))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                          max_lr=MAX_LR,
                                          steps_per_epoch=int(len(ds) / 3),
                                          epochs=NUM_EPOCHS,
                                          anneal_strategy='linear')
start = time.time()
losses = []
min_loss = sys.maxsize
do_save = True
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, batch_data in enumerate(dataloader):
        data_cpu, labels_cpu = batch_data
        data = to_best_device(data_cpu)
        labels = to_best_device(labels_cpu)
        optimizer.zero_grad()
        outputs = model(data)
        # Because outputs is of dimension (batch_size, seq, nb_chars) we have to permute the dimensions to fit cttloss
        # expected inputs
        outputs = outputs.permute(1, 0, 2) # seq, batch_size, nb_chars = outputs.shape
        curr_loss = loss(nn.functional.log_softmax(outputs, 2), labels.flatten(),
                         torch.tensor(outputs.shape[1] * [outputs.shape[0]], dtype=torch.long),
                         torch.tensor([len(label) for label in labels], dtype=torch.long))
        curr_loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += curr_loss.item()
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
