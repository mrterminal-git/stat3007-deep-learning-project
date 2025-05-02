import torch.nn as nn
from .cnn_transformer import CNNTransformer
from .ffn import FFN

class StatArbModel(nn.Module):
    """Full Statistical Arbitrage model."""
    def __init__(self, input_length, num_features, hidden_dim=32):
        super(StatArbModel, self).__init__()
        self.cnn_transformer = CNNTransformer(input_length, num_features)
        self.ffn = FFN(input_dim=8, hidden_dim=hidden_dim)

    def forward(self, x):
        x, attn_weights = self.cnn_transformer(x)
        w_eps = self.ffn(x)
        return w_eps, attn_weights