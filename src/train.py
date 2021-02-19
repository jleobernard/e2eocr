import os
from typing import Union

import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from model.paragraph_reader import ParagraphReader
from utils.data_utils import get_last_model_params
from utils.image_helper import get_dataset
from utils.characters import blank_id
import matplotlib.pyplot as plt

data_path = 'data/train/one-line'
models_rep = 'data/models'
load_model = True
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 5
HEIGHT = 80
WIDTH = 80
MOMENTUM = 0.9
MAX_SENTENCE_LENGTH = 10


def imshow(inp):
    inp = inp.numpy()[0]
    mean = 0.1307
    std = 0.3081
    inp = ((mean * inp) + std)
    plt.imshow(inp, cmap='gray')
    plt.show()


print(f"Loading dataset from {data_path}...")
ds = get_dataset(data_path, width=WIDTH, height=HEIGHT, target_length=MAX_SENTENCE_LENGTH)
#imshow(ds[8][0])
#exit()
print(f"...dataset loaded")
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
model = ParagraphReader(height=HEIGHT, width=WIDTH)

if load_model:
    last_model_file = get_last_model_params(models_rep)
    if last_model_file:
        print(f"Loading model parameters from {last_model_file}")
        model.load_state_dict(torch.load(last_model_file))

model.train()
loss = nn.CTCLoss(blank=blank_id)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
start = time.time()
losses =[]
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, batch_data in enumerate(dataloader):
        data, labels = batch_data
        optimizer.zero_grad()
        outputs = model(data)
        #print(f"Shape of ouputs before {outputs.shape}")
        outputs = outputs.permute(1, 0, 2)
        #print(f"Shape of ouputs after {outputs.shape}")
        bs = len(data)
        curr_loss = loss(outputs.log_softmax(2), labels, bs * [outputs.shape[0]], [len(label) for label in labels])
        curr_loss.backward()
        optimizer.step()
        running_loss += curr_loss.item()
    print(f'[{epoch}]Loss is {running_loss}')
    losses.append(running_loss)
end = time.time()
print(f"It took {end - start}")
save_path = f"data/models/{time.time()}.pt"
print(f"Saving to {save_path}")
torch.save(model.state_dict(), save_path)
print("Done")
plt.plot(losses)
plt.show()
