import time
import torch
import string
from torch import nn
from torch.utils.data import DataLoader

from model.paragraph_reader import ParagraphReader
from image_helper import get_dataset


def split(word):
    return [char for char in word]

data_path = 'data/00-alphabet'
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1
BATCH_SIZE = 10
HEIGHT = 64
WIDTH = 64
MOMENTUM = 0.9
CLASSES = split(string.ascii_letters)

print(f"Loading dataset from {data_path}...")
ds = get_dataset(data_path, width=WIDTH, height=HEIGHT)
print(f"...dataset loaded")
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
model = ParagraphReader(height=HEIGHT, width=WIDTH)
model.train()
loss = nn.CTCLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
start = time.time()
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, batch_data in enumerate(dataloader):
        data, labels = batch_data
        optimizer.zero_grad()
        outputs = model(data)
        curr_loss = loss(outputs, labels.view(outputs.shape))
        curr_loss.backward()
        optimizer.step()
        running_loss += curr_loss.item()
        if epoch % 10 == 0:
            print(f'[epoch - {epoch}]Loss is {running_loss}')
end = time.time()
print(f"It took {end - start}")