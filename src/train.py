import matplotlib.pyplot as plt
import sys
import time
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader

from model.paragraph_reader import ParagraphReader
from utils.characters import blank_id, get_sentence_length
from utils.image_helper import get_dataset
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
parser.add_argument('--load', dest='load', default=False,
                    help='Load model if true')
args = parser.parse_args()

data_path = args.data_path
models_rep = args.models_path
load_model = 'True' == args.load
NUM_EPOCHS = int(args.epoch)
BATCH_SIZE = int(args.batch)
HEIGHT = int(args.height)
WIDTH = int(args.width) # 80
MOMENTUM = 0.9
MAX_SENTENCE_LENGTH = int(args.sentence) # 10
LEARNING_RATE = float(args.lr) # 0.00001

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


print(f"Loading dataset from {data_path}...")
ds = get_dataset(data_path, width=WIDTH, height=HEIGHT, target_length=MAX_SENTENCE_LENGTH)
#imshow(ds[8][0])
#exit()
print(f"...dataset loaded")
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
model = to_best_device(ParagraphReader(height=HEIGHT, width=WIDTH))

if load_model:
    do_load_model(models_rep, model)
else:
    model.initialize_weights()

model.train()
loss = to_best_device(nn.CTCLoss(blank=blank_id, zero_infinity=True))
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
start = time.time()
losses = []
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, batch_data in enumerate(dataloader):
        data, labels = batch_data
        data = to_best_device(data)
        labels = to_best_device(labels)
        optimizer.zero_grad()
        outputs = model(data)
        # Because outputs is of dimension (batch_size, seq, nb_chars) we have to permute the dimensions to fit cttloss
        # expected inputs
        outputs = outputs.permute(1, 0, 2)
        bs = len(data)
        curr_loss = loss(outputs.log_softmax(2), labels, bs * [outputs.shape[0]], [get_sentence_length(label) for label in labels])
        curr_loss.backward()
        optimizer.step()
        running_loss += curr_loss.item()
    print(f'[{epoch}]Loss is {running_loss}')
    if epoch % 20 == 19:
        path_to_epoch_file = f"{models_rep}/{time.time()}-{epoch}.pt"
        print(f'Saving epoch {epoch} in {path_to_epoch_file} with loss {running_loss}')
        torch.save(model.state_dict(), path_to_epoch_file)
    losses.append(running_loss)
end = time.time()
print(f"It took {end - start}")
save_path = f"{models_rep}/{time.time()}-final.pt"
print(f"Saving to {save_path}")
torch.save(model.state_dict(), save_path)
print("Done")
plt.plot(losses)
plt.show()
