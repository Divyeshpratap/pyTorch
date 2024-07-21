import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

class vgg_net(nn.Module):

    def __init__(self, model, in_channels, input_size, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = model
        self.conv_layers = self.create_layers(self.in_channels, self.model)
        self.network = nn.Sequential[*conv_layers]
        self.fcs = nn.Sequential(nn.Linear(512*7*7, 4096, bias = True),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(4096, 4096, bias = True),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(4096, self.num_classes, bias = True)
                                 )
         


    def forward(self, x):
        x = self.network(x)
        x = x.reshape(x.shape[0], -1)
        return self.fcs(x)

    def create_layers(self, in_channels, architecture):
        layers = []
        for layer in architecture:
            if type(layer) == int:
                layer = [nn.Conv2d(in_channels, layer, kernel_size = (3,3), padding = (1,1), stride = (1,1)),
                         nn.BatchNorm2d(x),
                         nn.ReLU()]
                layers += layer
                in_channels = layer
            elif typ(layer) == str:
                layer = [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))]
                layers += layer

        return layers


