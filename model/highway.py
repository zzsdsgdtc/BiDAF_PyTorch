import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.size = input_size
        self.gate = nn.Linear(self.size, self.size)
        self.transform = nn.Linear(self.size, self.size)
        
    def forward(self, x):
        res = x
        for i in range(2):
            H = F.relu(self.transform(res))
            G = F.sigmoid(self.gate(res))
            res = G * H + (1 - G) * res
        return res