import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    """Feed-Forward Network for allocation weights."""
    def __init__(self, input_dim, hidden_dim=32):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        w_eps = self.out(x)
        return w_eps