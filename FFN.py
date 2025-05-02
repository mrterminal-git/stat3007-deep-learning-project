import torch
import torch.nn as nn # building blocks for neural networks
import torch.nn.functional as F # access to functions like ReLU, sigmoid, etc.

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    # IDEA: maybe try other relu variants such as LeakyReLU, ELU, etc.
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        w_eps = self.out(x) # equation (10), single scalar value for each input
        return w_eps
    
# equation (11) from project framework
def soft_normalize(weights):
    """
    Normalize allocation weights using L1 norm (sum of absolute values).
    weights: Tensor of shape [batch_size, 1]
    Returns: Normalized weights of shape [batch_size, 1]
    """
    l1_norm = torch.sum(torch.abs(weights), dim=0, keepdim=True) + 1e-8 # avoid division by zero
    normalized_weights = weights / l1_norm
    return normalized_weights