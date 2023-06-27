import os

import cv2
import numpy as np
import torch
import tqdm
from torchvision import models
from model import initialize_model
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = initialize_model(model_name="resnet18", num_classes=2, feature_extract=False, use_pretrained=False)
state_dict = torch.load('runs/exp31/best.pt', map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640))
])

test_set = ImageFolder(root='../resnet-dataset/test', transform=transform)
test_loader = DataLoader(test_set, batch_size=16, shuffle=True)
class_names = test_set.classes
class_idx = test_set.class_to_idx
print(class_names)
print(class_idx)


classes = test_set.classes

counts = {}
for c in classes:
    counts[c] = test_set.targets.count(test_set.class_to_idx[c])
print(counts)

running_corrects = 0
has_corrects = 0
no_corrects = 0
criterion = nn.CrossEntropyLoss()
running_loss = 0.0

for input, label in tqdm.tqdm(test_loader, desc="Run on dataset: "):
    input = input.to(device)
    label = label.to(device)
    outputs = model(input)
    loss = criterion(outputs, label)
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == label.data)
    has_corrects += torch.sum((preds == label.data) * (label.data == 0))
    no_corrects += torch.sum((preds == label.data) * (label.data == 1))

running_loss += loss.item() * input.size(0)
acc = running_corrects.double() / len(test_loader.dataset)
has_acc = has_corrects.double() / counts['hasFish']
no_acc = no_corrects.double() / counts['notHasFish']
print("Average Acc:", acc)
print("Has Acc:", has_acc)
print("No Acc:", no_acc)
print("Val loss", running_loss)