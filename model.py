import copy
import os.path
import re
import time
import torch
from matplotlib import pyplot as plt
from torchvision import models
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import tqdm

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm.tqdm(dataloaders[phase], desc="Run on dataset: "):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    path = 'ResNet/runs'
    if not os.path.exists(path):
        os.mkdir(path)
    all_exp = os.listdir(path)
    all_exp.sort()
    # num = 0
    if all_exp is None:
        path = path + '/exp' + str(0)
        os.mkdir(path)
    else:
        # print(all_exp)
        # num = int(re.findall(r'\d+', all_exp[-1])[-1])
        # print(num)
        path = path + '/exp' + str(len(all_exp))
        os.mkdir(path)
    print("The result save in ", path)
    torch.save(best_model_wts, path + '/best.pt')

    train_acc_history_cpu = [t.cpu().numpy() for t in train_acc_history]
    val_acc_history_cpu = [t.cpu().numpy() for t in val_acc_history]

    # plot diagram
    plt.plot(range(num_epochs), train_acc_history_cpu, "g*-", label='train_acc')
    plt.plot(range(num_epochs), val_acc_history_cpu, "b*-", label='val_acc')
    plt.xlabel('num_epoch')
    plt.title('Train and Val acc')
    plt.legend()
    plt.savefig(path + '/acc.png')
    plt.clf()

    plt.plot(range(num_epochs), train_loss_history, "k*-", label='train_loss')
    plt.plot(range(num_epochs), val_loss_history, "y*-", label='test_loss')
    plt.xlabel('num_epoch')
    plt.title('Train and Val loss')
    plt.legend()
    plt.savefig(path + '/loss.png')

    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, Dropout=0.5):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = None

    if model_name == "resnet18":
        """ Resnet18
        """
        if use_pretrained:
            model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model_ft = models.resnet18()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        """ Resnet50
        """
        # model_ft = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=use_pretrained)
        if use_pretrained:
            model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model_ft = models.resnet50()
        print(model_ft)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet18_Dropout":
        """ Resnet18 with add Dropout layer
        """
        if use_pretrained:
            model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model_ft = models.resnet18()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Dropout(Dropout),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == "resnet50_Dropout":
        """ Resnet50 with add Dropout layer
        """
        if use_pretrained:
            model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model_ft = models.resnet50()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Dropout(Dropout),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft