import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from models.CNN import *
from models.FFN import *
from models.BacktestSharpeEvaluator import *

class PortfolioOptimizer:
    """
    A class to train and evaluate a CNN+FFN model for portfolio optimization,
    generating weights to maximize the Sharpe ratio based on electricity price residuals.
    """
    def __init__(self, cnn_input_array: np.array, next_day_returns: pd.DataFrame, device=None, num_filters=8, filter_size=2, hidden_dim=32,
                 lr=0.001, num_epochs=1000, batch_size=32):
        """
        Initialize the portfolio optimizer with input data and hyperparameters.

        Args:
            cnn_input_array (np.ndarray): Input residuals with shape [samples, num_countries, window_size].
            next_day_returns (pd.DataFrame): Next-day returns with shape [samples, num_countries].
            device (torch.device, optional): Device for computation (cuda, mps, or cpu). Defaults to auto-detection.
            num_filters (int): Number of CNN filters.
            filter_size (int): Size of CNN convolutional filters.
            hidden_dim (int): Hidden dimension for CNN output and FFN layers.
            lr (float): Learning rate for the optimizer.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        # Set device (GPU, MPS, or CPU)
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        # Data dimensions
        self.num_countries = cnn_input_array.shape[1]  # Number of countries (31)
        self.window_size = cnn_input_array.shape[2]    # Window size for residuals (30)
        self.num_samples = cnn_input_array.shape[0]    # Number of samples (329, or the number of days)

        # Hyperparameters
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Convert input data to PyTorch tensors
        self.cnn_input_tensor = torch.FloatTensor(cnn_input_array).to(self.device)
        self.next_day_returns_tensor = torch.FloatTensor(next_day_returns.values).to(self.device)

        # Initialize models
        self.cnn_model = self._initialize_cnn()
        self.FFN_model = self._initialize_FFN()

        # Initialize optimizer
        self.optimizer = optim.Adam(
            list(self.cnn_model.parameters()) + list(self.FFN_model.parameters()),
            lr=self.lr
        )

        # Initialize evaluator for Sharpe ratio calculation
        self.evaluator = BacktestSharpeEvaluator()

    def _initialize_cnn(self):
        """
        Initialize the CNN model.

        Returns:
            CNN: Initialized CNN model on the specified device.
        """
        # Calculate output sizes after convolutions
        L_after_conv1 = self.window_size - self.filter_size + 1  # e.g., 30 - 2 + 1 = 29
        L_after_conv2 = L_after_conv1 - self.filter_size + 1    # e.g., 29 - 2 + 1 = 28

        cnn = CNN(
            input_length=self.window_size,
            num_features=self.num_countries,
            num_filters=self.num_filters,
            num_classes=self.hidden_dim,  # Output matches FFN input dimension
            filter_size=self.filter_size
        ).to(self.device)
        return cnn

    def _initialize_FFN(self):
        """
        Initialize the FFN model.

        Returns:
            FFN: Initialized FFN model on the specified device.
        """
        ffn = FFN(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.num_countries  # One weight per country
        ).to(self.device)
        return ffn

    def soft_normalize(self, weights):
        """
        Normalize weights using L1 norm (sum of absolute values = 1).

        Args:
            weights (torch.Tensor): Raw weights with shape [batch_size, num_countries].

        Returns:
            torch.Tensor: Normalized weights with shape [batch_size, num_countries].
        """
        l1_norm = torch.sum(torch.abs(weights), dim=1, keepdim=True) + 1e-8  # Avoid division by zero
        normalized_weights = weights / l1_norm
        return normalized_weights

    def sharpe_ratio_loss(self, returns, risk_free_rate=0.0):
        """
        Compute negative Sharpe ratio as loss for optimization.

        Args:
            returns (torch.Tensor): Portfolio returns with shape [batch_size].
            risk_free_rate (float): Risk-free rate (default: 0.0).

        Returns:
            torch.Tensor: Negative Sharpe ratio (scalar).
        """
        excess_returns = returns - risk_free_rate
        mean_excess = torch.mean(excess_returns)
        std_excess = torch.std(excess_returns, unbiased=False) + 1e-5  # Avoid division by zero
        sharpe_ratio = mean_excess / std_excess
        return -sharpe_ratio

    def train(self):
        """
        Train the CNN+FFN model to maximize the Sharpe ratio.

        Returns:
            list: Portfolio returns from the final evaluation.
        """
        for epoch in range(self.num_epochs):
            # Set models to training mode
            self.cnn_model.train()
            self.FFN_model.train()

            # Process data in batches
            for batch_idx in range(0, len(self.cnn_input_tensor), self.batch_size):
                # Extract batch
                batch_end = min(batch_idx + self.batch_size, len(self.cnn_input_tensor))
                batch_inputs = self.cnn_input_tensor[batch_idx:batch_end]  # Shape: [batch_size, num_countries, window_size]
                batch_returns = self.next_day_returns_tensor[batch_idx:batch_end]  # Shape: [batch_size, num_countries]

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass: CNN processes residuals
                cnn_output = self.cnn_model(batch_inputs)  # Shape: [batch_size, hidden_dim]

                # Forward pass: FFN generates weights
                weights = self.FFN_model(cnn_output)  # Shape: [batch_size, num_countries]

                # Normalize weights to sum to 1 (L1 norm)
                normalized_weights = self.soft_normalize(weights)  # Shape: [batch_size, num_countries]

                # Compute portfolio returns (dot product of weights and returns)
                portfolio_returns = torch.sum(normalized_weights * batch_returns, dim=1)  # Shape: [batch_size]

                # Compute loss (negative Sharpe ratio)
                loss = self.sharpe_ratio_loss(portfolio_returns)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            
            if epoch % 10 == 0:
                # Evaluate Sharpe ratio for every 10th epoch
                sharpe = self._evaluate_epoch()
                print(f"Epoch {epoch}, Sharpe Ratio: {sharpe:.4f}")

        # Perform final evaluation
        final_sharpe, portfolio_returns = self.evaluate_final()
        print(f"\nFinal Sharpe Ratio: {final_sharpe:.4f}")

        return portfolio_returns

    def _evaluate_epoch(self):
        """
        Evaluate the model for one epoch, computing the Sharpe ratio.

        Returns:
            float: Sharpe ratio for the epoch.
        """
        self.evaluator.reset()
        self.cnn_model.eval()
        self.FFN_model.eval()

        with torch.no_grad():
            for i in range(0, len(self.cnn_input_tensor), self.batch_size):
                batch_end = min(i + self.batch_size, len(self.cnn_input_tensor))
                batch_inputs = self.cnn_input_tensor[i:batch_end]
                batch_returns = self.next_day_returns_tensor[i:batch_end]

                # Forward pass
                cnn_output = self.cnn_model(batch_inputs)
                weights = self.FFN_model(cnn_output)
                normalized_weights = self.soft_normalize(weights)

                # Compute portfolio returns
                portfolio_returns = torch.sum(normalized_weights * batch_returns, dim=1)
                portfolio_returns_np = portfolio_returns.cpu().numpy()

                # Store returns in evaluator
                for return_val in portfolio_returns_np:
                    self.evaluator.add_return(return_val)

        return self.evaluator.calculate_sharpe()

    def evaluate_final(self):
        """
        Perform final evaluation on all samples.

        Returns:
            tuple: (final Sharpe ratio, list of portfolio returns).
        """
        self.evaluator.reset()
        self.cnn_model.eval()
        self.FFN_model.eval()

        with torch.no_grad():
            for i in range(len(self.cnn_input_tensor)):
                inputs = self.cnn_input_tensor[i:i+1]  # Shape: [1, num_countries, window_size]
                returns = self.next_day_returns_tensor[i:i+1]  # Shape: [1, num_countries]

                # Forward pass
                cnn_output = self.cnn_model(inputs)
                weights = self.FFN_model(cnn_output)
                normalized_weights = self.soft_normalize(weights)

                # Compute portfolio return
                portfolio_return = torch.sum(normalized_weights * returns, dim=1).item()
                self.evaluator.add_return(portfolio_return)

        final_sharpe = self.evaluator.calculate_sharpe()
        return final_sharpe, self.evaluator.portfolio_returns
