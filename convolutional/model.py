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
        return self.fc(x)

torch.manual_seed(67)

# v1.0
#model = nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten(), nn.Linear(16 * 13 * 13, 10, bias=True))

"""
# v2.0
model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(32 * 13 * 13, 128),
    nn.ReLU(),
    nn.Dropout(0.25),       # Regularize the fully-connected layer
    nn.Linear(128, 10)
)
"""

# v3.0 (full CNN) tensor(0.0786, device='mps:0')

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),   # 28x28
    nn.ReLU(),
    nn.MaxPool2d(2, 2),               # 14x14
    nn.Conv2d(32, 64, 3, padding=1),  # 14x14
    nn.ReLU(),
    nn.MaxPool2d(2, 2),               # 7x7
    nn.Flatten(),                     # 64*7*7 = 3136
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(128, 10),
)