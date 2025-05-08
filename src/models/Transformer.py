import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        pool: str = 'last'
    ):
        """
        Transformer module for time-series features output by CNN.

        Args:
            d_model (int): Dimensionality of feature embeddings (num_filters from CNN).
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout probability in Transformer.
            pool (str): Pooling strategy: 'last' or 'mean'.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = pool

    def forward(self, cnn_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_features: Tensor of shape [batch_size, d_model, seq_len]
        Returns:
            Tensor of shape [batch_size, d_model]
        """
        # Permute to [seq_len, batch_size, d_model]
        x = cnn_features.permute(2, 0, 1)
        # Apply Transformer encoder
        tr_out = self.transformer(x)  # [seq_len, batch_size, d_model]
        # Pool over the sequence dimension
        if self.pool == 'last':
            output = tr_out[-1]         # Last time step [batch_size, d_model]
        elif self.pool == 'mean':
            output = tr_out.mean(dim=0) # Mean over seq [batch_size, d_model]
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")
        return output

    def get_parameters(self):
        return self.parameters()
