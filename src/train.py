import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from model.paragraph_reader import ParagraphReader
from utils.image_helper import get_dataset
from utils.characters import blank_id
import matplotlib.pyplot as plt

data_path = 'data/00-alphabet'
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1
BATCH_SIZE = 1
HEIGHT = 68
WIDTH = 68
MOMENTUM = 0.9
MAX_SENTENCE_LENGTH = 10

#torch.autograd.set_detect_anomaly(True)

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
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
model = ParagraphReader(height=HEIGHT, width=WIDTH)
model.train()
loss = nn.CTCLoss(blank=blank_id)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
start = time.time()
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, batch_data in enumerate(dataloader):
        data, labels = batch_data
        optimizer.zero_grad()
        outputs = model(data)
        print(f"Shape of ouputs before {outputs.shape}")
        outputs = outputs.permute(1, 0, 2)
        print(f"Shape of ouputs after {outputs.shape}")
        curr_loss = loss(outputs, labels.view(BATCH_SIZE, MAX_SENTENCE_LENGTH), BATCH_SIZE * [outputs.shape[0]], BATCH_SIZE * [MAX_SENTENCE_LENGTH])
        curr_loss.backward()
        optimizer.step()
        running_loss += curr_loss.item()
        if epoch % 10 == 0:
            print(f'[epoch - {epoch}]Loss is {running_loss}')
end = time.time()
print(f"It took {end - start}")