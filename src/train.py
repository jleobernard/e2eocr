import matplotlib.pyplot as plt
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from model.paragraph_reader import ParagraphReader
from utils.characters import blank_id, get_sentence_length
from utils.image_helper import get_dataset
from utils.tensor_helper import to_best_device, do_load_model

if torch.cuda.is_available():
    print("CUDA will be used")
else:
    print("CUDA won't be used")

data_path = sys.argv[1] # '/data/train/one-line'
models_rep = sys.argv[2] # '/data/models'
load_model = True
LEARNING_RATE = 0.0001
NUM_EPOCHS = int(sys.argv[3]) # 100
BATCH_SIZE = int(sys.argv[4]) # 100
HEIGHT = int(sys.argv[5]) # 80
WIDTH = int(sys.argv[6]) # 80
MOMENTUM = 0.9
MAX_SENTENCE_LENGTH = int(sys.argv[7]) # 10


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
    do_load_model(models_rep, model)

model.train()
loss = to_best_device(nn.CTCLoss(blank=blank_id, zero_infinity=True))
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
start = time.time()
losses =[]
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, batch_data in enumerate(dataloader):
        data, labels = batch_data
        data = to_best_device(data)
        labels = to_best_device(labels)
        optimizer.zero_grad()
        outputs = model(data)
        outputs = outputs.permute(1, 0, 2)
        bs = len(data)
        curr_loss = loss(outputs.log_softmax(2), labels, bs * [outputs.shape[0]], [get_sentence_length(label) for label in labels])
        curr_loss.backward()
        optimizer.step()
        running_loss += curr_loss.item()
    print(f'[{epoch}]Loss is {running_loss}')
    losses.append(running_loss)
end = time.time()
print(f"It took {end - start}")
save_path = f"{models_rep}/{time.time()}.pt"
print(f"Saving to {save_path}")
torch.save(model.state_dict(), save_path)
print("Done")
plt.plot(losses)
plt.show()
