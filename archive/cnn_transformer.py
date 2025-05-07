import torch
import torch.nn as nn
from .transformer_layer import TransformerLayer

class CNNTransformer(nn.Module):
    """Combined CNN + Transformer model."""
    def __init__(self, input_length, num_features, num_filters=8, filter_size=2, num_heads=4):
        super(CNNTransformer, self).__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.num_features = num_features

        self.conv1 = nn.Conv1d(num_features, num_filters, filter_size, stride=1, padding=0)
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.skip_conv = nn.Conv1d(num_features, num_filters, kernel_size=1, stride=1, padding=0)
        self.transformer = TransformerLayer(input_dim=num_filters, num_heads=num_heads)

        L_after_conv1 = input_length - filter_size + 1
        self.L_after_conv2 = L_after_conv1 - filter_size + 1

    def forward(self, x):
        """Forward pass."""
        x_original = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x_skip = self.skip_conv(x_original)
        diff = x_skip.size(2) - x.size(2)
        if diff > 0:
            x_skip = x_skip[:, :, diff:]

        x = x + x_skip
        x = x.transpose(1, 2)
        output, attn_weights = self.transformer(x)
        return output, attn_weights