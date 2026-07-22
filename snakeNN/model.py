import torch
import torch.nn as nn

class SnakeNN(nn.Module):
    def __init__(self, h1=128, h2=64, in_features=14, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.out(x)

        return out

"""
cnn = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),   # 28x28
    nn.ReLU(),
    nn.MaxPool2d(2, 2),               # 14x14
    nn.Conv2d(32, 64, 3, padding=1),  # 14x14
    nn.ReLU(),
    nn.MaxPool2d(2, 2),               # 7x7
    nn.Flatten(),                     # 64*7*7 = 3136
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 4),
)
"""

cnn = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),   # 28x28
    nn.ReLU(),
    nn.MaxPool2d(2, 2),               # 14x14
    nn.Conv2d(32, 64, 3, padding=1),  # 14x14
    nn.ReLU(),
    nn.MaxPool2d(2, 2),               # 7x7
    nn.Flatten(),                     # 64*7*7 = 3136
    nn.Linear(64 * 2 * 2, 128),
    nn.ReLU(),
    nn.Linear(128, 4),
)

# can try to make cnn model from scratch