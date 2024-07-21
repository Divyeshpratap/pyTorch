import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available else 'cpu'

batch_size = 64
learning_rate = 3e-4
input_size = 28
sequence_length = 28
num_classes = 10
num_epochs = 2
num_layers = 4
hidden_size = 64

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
    
train_dataset = datasets.MNIST(root = '/datasets', train=True, transform = transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size)

test_dataset = datasets.MNIST(root = '/datasets', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers = num_layers, num_classes=num_classes)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
print(f'Starting with model training next')
for epoch in range(num_epochs):
    for x, y in train_loader:
        x, y = x.to(device).squeeze(1), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def check_accuracy(loader):
    if loader.dataset.train:
        print(f'Working on train dataset')
    else:
        print(f'Working on test dataset')
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, y in loader:
        x, y = x.to(device).squeeze(1), y.to(device)
        logits = model(x)
        _, predictions = logits.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

    acc = (num_correct/num_samples)*100

    print(f'Accuracy on dataset is {acc:.2f}')


check_accuracy(train_loader)
check_accuracy(test_loader)







