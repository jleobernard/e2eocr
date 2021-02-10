import pandas as pd
import os
import torch
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from skimage import io, transform


class CustomDataSet(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
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

        return image, entry.clazz


def get_dataset(path, width=50, height=50):
    """
    :param path: Path of the directory containing a csv file named dataset.csv
    containing all the pairs <clazz, short-filename> included in this dataset.
    :param width: Width of the target image
    :param height: Height of the target image
    :return: CustomDataSet
    """
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((height, width)),
                                    transforms.CenterCrop((height, width)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    return CustomDataSet(root_dir=path, transform=transform)
