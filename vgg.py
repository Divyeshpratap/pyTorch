import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

VGG_models = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}



class vgg_net(nn.Module):

    def __init__(self, model, in_channels, input_size, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = model
        self.conv_layers = self.create_layers(self.in_channels, self.model)
        self.network = nn.Sequential(*self.conv_layers)
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
        x = x.view(x.size(0), -1)
        return self.fcs(x)

    def create_layers(self, in_channels, architecture):
        layers = []
        for block in architecture:
            if type(block) == int:
                layer = [nn.Conv2d(in_channels, block, kernel_size = (3,3), padding = (1,1), stride = (1,1)),
                         nn.BatchNorm2d(block),
                         nn.ReLU()]
                layers += layer
                in_channels = block
            elif block == 'M':
                layer = [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))]
                layers += layer

        return layers


dropout = 0.2
in_channels = 3
batch_size = 16
num_classes= 10


model = vgg_net(model=VGG_models['VGG19'], in_channels=in_channels, input_size = 224, num_classes = num_classes)
model.to(device)
input_data = torch.rand((batch_size, in_channels, 224, 224)).to(device)
m = model(input_data)
print(m.shape)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor()
])


# train_dataset = datasets.CIFAR10(root ='./datasets', train=True, transform=transform, download=True)
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size)

# test_dataset = datasets.CIFAR10(root = './datasets', train=False, transform=transform, download=True)
# test_loader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size)

