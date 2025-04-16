import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_length, num_features, num_classes=2):
        """
        Initialize the CNN model based on the equations in the paper.
        
        Args:
            input_length (int): Length of the input sequence (L in the equations)
            num_features (int): Number of features per time step
            num_classes (int): Number of output classes
        """
        super(CNN, self).__init__()
        
        self.D = 8  # Number of filters (D = 8 according to equation)
        self.D_size = 2  # Filter size (D_size = 2 according to equation)
        
        # First convolutional layer (Equation 3)
        # Input shape: [batch_size, num_features, input_length]
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=self.D,
            kernel_size=self.D_size,
            stride=1,
            padding=0
        )
        
        # Second convolutional layer (Equation 4)
        self.conv2 = nn.Conv1d(
            in_channels=self.D,
            out_channels=self.D,
            kernel_size=self.D_size,
            stride=1,
            padding=0
        )
        
        # Calculate output sizes after convolutions
        L_after_conv1 = input_length - self.D_size + 1
        L_after_conv2 = L_after_conv1 - self.D_size + 1
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.D * L_after_conv2, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_features, input_length]
                             Represents the x_i(t) vector in Equation 2
        
        Returns:
            torch.Tensor: Output predictions
        """
        # Store original input for skip connection (Equation 5)
        x_original = x
        
        # First convolutional layer (Equation 3)
        x = self.relu(self.conv1(x))
        
        # Second convolutional layer (Equation 4)
        x = self.relu(self.conv2(x))
        
        # Skip connection (Equation 5)
        # Need to adjust dimensions for skip connection
        # Cut or pad x_original to match x dimensions
        if x_original.shape[2] > x.shape[2]:
            # If original is longer, cut it
            diff = x_original.shape[2] - x.shape[2]
            x_skip = x_original[:, :, diff:]
        else:
            # If original is shorter, this would require padding
            # For simplicity, we'll just use the original shape
            x_skip = x_original
        
        # Apply the skip connection if dimensions match
        if x.shape == x_skip.shape:
            x = x + x_skip
        
        # Flatten and pass through fully connected layer
        x = self.flatten(x)
        x = self.fc(x)
        
        return x
    
    def get_parameters(self):
        return self.parameters()
