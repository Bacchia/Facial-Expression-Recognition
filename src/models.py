import torch
import torch.nn as nn

class LinearBaselineModel(nn.Module):
    def __init__(self, input_dim=1*48*48, num_classes=7):
        super(LinearBaselineModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)