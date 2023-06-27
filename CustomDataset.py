import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import transforms


class CustomDataset(Dataset):
    def __init__(self, path):
        self.images = []
        self.labels = []
        self.path = path
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path)
        label = self.labels[index]
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def getDatasetPath(self):
        return self.path
    
    def getImage(self):
        for file in os.listdir(self.path):
            self.images.append(self.path + '/' + file)
            self.labels.append(0)  # 0 means have fish

    def getSingleImage(self):
        self.images.append(self.path)
        self.labels.append(0)  # 0 means have fish
        return

    def initDataset(self):
        for file in os.listdir(self.path + '/hasFish'):
            self.images.append(self.path + '/hasFish/' + file)
            self.labels.append(1)

        for file in os.listdir(self.path + '/notHasFish'):
            self.images.append(self.path + '/notHasFish/' + file)
            self.labels.append(0)
        return
