import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torchvision
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_checkpoint(path, model, optimizer):
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.train()
    
def save_checkpoint(path, model, optimizer):
    print(f'saving checkpoint at location {path}')      
    checkpoint = {'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()}
    torch.save(checkpoint, path)

# class CNN(nn.Module):

#     def __init__(self, in_channels=1, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size = (3,3), stride=(1,1), padding=(1,1))
#         self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride= (1,1), padding=(1,1))
#         self.fc1 = nn.Linear(16*7*7, num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc1(x)

#         return x
    
# model = CNN(3, 10)

# x = torch.rand((100, 3, 28, 28))
# print(f'output shape is {model(x).shape}')

model = torchvision.models.vgg16(pretrained=True)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class classificationLayer(nn.Module):
 
    def __init__(self):
        super().__init__()
        self.sq1 = nn.Sequential(
            nn.Linear(512, 1024, bias = True),
            nn.Linear(1024, 256, bias = True),
            nn.Linear(256, 10,bias = True)
        )
    
    def forward(self, x):
        return self.sq1(x) 




in_channels = 1
num_classes = 10
learning_rate = 3e-4
batch_size = 32
num_epochs = 4
script_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint_path = os.path.join(script_dir, 'checkpoints/CNN/')
checkpoint_name = 'checkpoint_epoch.pth.tar'
load_model = False
save_frequency = 2

for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()   
model.classifier = classificationLayer()
model.to(device)

train_dataset = datasets.CIFAR10(root='/datasets', train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='/datasets', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr =learning_rate)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if load_model:
    load_checkpoint(os.path.join(checkpoint_path, checkpoint_name))

print(f'Starting training')
for epoch in range(num_epochs):
    print(f'Currently at epoch number {epoch}')
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        logits = model(data)
        loss = criterion(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        
@torch.no_grad()
def check_accuracy(loader, model):
    num_correct = 0
    num_steps = 0
    if loader.dataset.train:
        print(f'evaluating on train dataset')
    else:
        print(f'evaulating on test dataset')
    model.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        _, predictions = logits.max(1)
        num_correct += (predictions == y).sum()
        num_steps += y.shape[0]
    print(f'Accuracy during evaluation phase is {((num_correct/num_steps)*100):.2f} percent')
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)


