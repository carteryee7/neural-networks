import torch
import torch.nn as nn

class SnakeNN(nn.Module):
    def __init__(self, h1=64, h2=32, in_features=11, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.out(x)

        return out