import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), stride=1, padding=1),  # 3x224x224 => 8x224x224
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 64, (3, 3), padding=1),  # 8x112x112 => 64x112x112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3, 3), padding=1),  # 64x56x56 => 128x56x56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x56x56 => 128x24x24
            #                                     nn.Conv2d(128,256,(3,3),padding=1),
            #                                     nn.ReLU(),
            #                                     nn.MaxPool2d(2,2), # 256x12x12
            nn.AdaptiveAvgPool2d(output_size=(6, 6)),
            nn.Flatten()
        )
        self.linear = nn.Sequential(nn.Linear(4608, 2048, bias=True),
                                    nn.Dropout(0.5),
                                    nn.ReLU(),
                                    nn.Linear(2048, 100, bias=True),
                                    nn.Dropout(0.5),
                                    nn.ReLU(),
                                    nn.Linear(100, 2, bias=True))

    #         num_ftrs = self.network.fc.in_features
    #         self.network.fc = nn.Linear(num_ftrs, 2)

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True

    def training_step(self, batch):
        image, label = batch
        out = self(image)
        loss = F.cross_entropy(out, label)
        return loss

    def forward(self, batch):
        out = self.network(batch)
        out = self.linear(out)
        return out

    def validation_step(self, batch):
        image, label = batch
        out = self(image)
        loss = F.cross_entropy(out, label)
        #         print("out:",type(out),'\n',out)
        _, y_pred = torch.max(out, dim=1)
        #         print('\nLabel:',type(label),'\n',label)
        label = label.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        print(label)
        print(y_pred)
        accuracy = accuracy_score(label, y_pred)
        precision = recall_score(label, y_pred, average='micro')
        print(confusion_matrix(label, y_pred))
        return {'val_loss': loss.detach(), 'val_accuracy': torch.Tensor([accuracy]),
                'precision': torch.Tensor([precision])}

    def validation_epoch_end(self, outputs):
        val_loss = [x['val_loss'] for x in outputs]
        val_loss_n = torch.stack(val_loss).mean()
        val_score = [x['val_accuracy'] for x in outputs]
        val_score_n = torch.stack(val_score).mean()
        precision = [x['precision'] for x in outputs]
        precision = torch.stack(precision).mean().item()
        return {'val_loss': val_loss_n, 'val_score': val_score_n, 'precision': precision}

    def epoch_end(self, epoch, result):
        print('Epoch {}: train_loss: {:.4f} val_loss: {:.4f} val_score: {:.4f} precision: {}'.format(epoch, result[
            'train_loss'], result['val_loss'], result['val_score'], result['precision']))

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        #         print('its a instance:',type(data))
        return [to_device(x, device) for x in data]
    #     print(type(data))
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class MainScript():
    def __init__(self):
        self.model = to_device(CNN(),get_default_device())
        self.model.load_state_dict(torch.load('mask_best_model.pth'))
        self.d = ImageFolder('RMDFDATA',transform=tt.ToTensor())

    def predict_image(self,img):
        # img is a numpy array overhere
        tfms = tt.Compose([
            tt.ToPILImage(),
            tt.Resize((230, 230)),
            tt.RandomHorizontalFlip(),

            tt.ToTensor(),
        ])
        img = tfms(img)
        # Convert to a batch of 1
        xb = to_device(img.unsqueeze(0), get_default_device())
        self.model.eval()
        # Get predictions from model
        yb = self.model(xb)
        # Pick index with highest probability
        # sm = torch.nn.Softmax()
        # yb = sm(yb)
        print(yb.detach())
        _, preds = torch.max(yb, dim=1)
        # Retrieve the class label
        return self.d.classes[preds[0].item()]
