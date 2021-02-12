import time
from torch.utils.data import DataLoader

from paragraph_reader import ParagraphReader
from image_helper import get_dataset

data_path = 'data/00-alphabet'
print(f"Loading dataset from {data_path}...")
ds = get_dataset(data_path, width=64, height=64)
print(f"...dataset loaded")
dataloader = DataLoader(ds, batch_size=10, shuffle=False)
model = ParagraphReader(height=64, width=64)
start = time.time()
for i, batch_data in enumerate(dataloader):
    data, labels = batch_data
    model.forward(data)
    print(f"Batch {i} done")
end = time.time()
print(f"It took {end - start}")