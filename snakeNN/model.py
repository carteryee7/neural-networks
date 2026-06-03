import torch
import torch.nn as nn

class SnakeNN(nn.Module):
    def __init__(self, in_features=10, h1=64, h2=32, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        out = self.out(x)

        return out