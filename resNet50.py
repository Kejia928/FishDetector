from __future__ import division
from __future__ import print_function

from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from model import train_model, initialize_model
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Models to choose from [resnet18, resnet50]
model_name = "resnet50_Dropout"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train for
num_epochs = 500

# Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
feature_extract = False

# Initialize the model for this run
print("Initialize model .....")
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
print("Finish Iinitialize model .....")

# Print the model we just instantiated
# print(model_ft)

# Create training and validation dataloaders
train_dataset = CustomDataset('resnet-dataset/train')
test_dataset = CustomDataset('resnet-dataset/test')
train_dataset.initDataset()
test_dataset.initDataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# images = []
# labels = []
# for image, label in train_loader:
#     images.append(image)
#     labels.append(label)
#
# print(images[0].shape)
# print(labels[0].shape)

dataloaders_dict = {
    'train': train_loader,
    'val': test_loader
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9, weight_decay=1e-4)

# Set up the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                             is_inception=(model_name == "inception"))

