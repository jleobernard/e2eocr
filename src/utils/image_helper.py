import os
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from utils.characters import sentence_to_list


class MyPad(torch.nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.height, self.width = shape

    def forward(self, image):
        w, h = image.size
        hp = int((self.width - w) / 2)
        vp = int((self.height - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 255, 'constant')


class CustomDataSet(Dataset):
    def __init__(self, root_dir, transform, target_length):
        self.root_dir = root_dir
        self.transform = transform
        self.target_length = target_length
        self.csv = pd.read_csv(root_dir + '/dataset.csv')

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        entry = self.csv.iloc[idx]
        img_name = os.path.join(self.root_dir, entry.file)
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(sentence_to_list(entry.label, padding=self.target_length), dtype=torch.float32)


def get_dataset(path, width=50, height=50, target_length=100):
    """
    :param path: Path of the directory containing a csv file named dataset.csv
    containing all the pairs <clazz, short-filename> included in this dataset.
    :param width: Width of the target image
    :param height: Height of the target image
    :param target_length: Maximum length of target labels
    :return: CustomDataSet
    """
    transform = transforms.Compose([transforms.ToPILImage(),
                                    MyPad((height, width)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    return CustomDataSet(root_dir=path, transform=transform, target_length=target_length)
