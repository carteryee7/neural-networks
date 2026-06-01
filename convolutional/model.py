import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, in_channel=1, output=10):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 13 * 13, output, bias=True)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        return torch.softmax(self.fc(x), dim=1)

torch.manual_seed(67)
model = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten(), nn.Linear(32 * 13 * 13, 10, bias=True), nn.Softmax())