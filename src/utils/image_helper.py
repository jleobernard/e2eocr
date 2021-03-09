import os
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from utils.characters import sentence_to_list
from PIL import Image, ImageDraw
import utils.characters as characters
import random


class TextGenerator:

    def generate(self, out_dir: str, nb_images: int, nb_characters: int, height: int, width: int, nb_lines = 1):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        #with open(f"{out_dir}/dataset.csv", "w") as dataset_file:
        #dataset_file.write("label,file\n")
        labels = []
        for i in range(nb_images):
            label = self.__get_label__(nb_characters)
            labels.append(label)
            img = Image.new('RGB', (width, height), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text(self.__get_start_position__(0, nb_lines=nb_lines, nb_characters=nb_characters, dimensions=(width, height)), label, fill=(0, 0, 0), spacing=10)
            img.save(f'{out_dir}/{i}.png')
            #dataset_file.write(f"{label},{i}.png\n")
        pd.DataFrame({'label': labels, 'file': [f'{i}.png' for i, _ in enumerate(labels)]}).to_csv(f"{out_dir}/dataset.csv", index=False)

    def __get_label__(self, nb_characters: int) -> str:
        return ''.join(random.choices(characters.characters[2:4], k=nb_characters)) # We remove the first 2 characters
        #return ''.join(random.choices(characters.characters[2:], k=nb_characters)) # We remove the first 2 characters

    def __get_start_position__(self, param, nb_lines, nb_characters, dimensions):
        width, height = dimensions
        return random.randint(0, width / 2 - 10), random.randint(0, height - 10)


class MyPad(torch.nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.height, self.width = shape

    def forward(self, image):
        w, h = image.size
        hp = int((self.width - w) / 2)
        vp = int((self.height - h) / 2)
        padding = (hp, vp, self.width - w - hp, self.height - h - vp)
        return F.pad(image, padding, 255, 'constant')


class CustomDataSetSimple(Dataset):
    def __init__(self, nb_digit, nb_samples):
        self.samples = []
        self.labels = []
        for i in range(nb_samples):
            selected_digits = random.choices(range(10), k=nb_digit)
            self.labels.append(torch.tensor(selected_digits, dtype=torch.long))
            tensor_digits = [torch.tensor((dig * [1] + 10 * [0])[:10], dtype=torch.float) for dig in selected_digits]
            sample = torch.stack(tensor_digits).transpose(0, 1).unsqueeze(dim=0)
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class CustomRawDataSet(Dataset):
    def __init__(self, root_dir):
        file_list = os.listdir(root_dir)
        self.labels = [f[:-4] for f in file_list]
        file_list = [os.path.join(root_dir, f) for f in file_list if f.endswith('.jpg')]
        raw_images = [io.imread(f) for f in file_list]
        heights = [image.shape[0] for image in raw_images]
        widths = [image.shape[1] for image in raw_images]
        max_height = max(heights)
        max_width = max(widths)
        transform = transforms.Compose([transforms.ToPILImage(),
            MyPad((max_height, max_width + 64)),
            transforms.Grayscale(),
            transforms.ToTensor()])
        self.images = [transform(image) for image in raw_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image, torch.tensor(sentence_to_list(self.labels[idx], 200), dtype=torch.float32)


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


