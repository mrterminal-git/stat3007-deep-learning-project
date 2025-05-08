from models.CointegrationResidualGenerator import CointegrationResidualGenerator
import pandas as pd

class Utility:
    @staticmethod
    def process_training_data(
        x_tr_data: list,
        returns: pd.DataFrame,
        cumulative_residual_window: int = 30
    ):
        """
        Processes training data to generate CNN inputs and next-day returns.

        Parameters:
        - x_tr_data (list): List of rolling windows (price matrices).
        - returns (pd.DataFrame): DataFrame of daily returns.
        - cumulative_residual_window (int): Window size for cumulative residuals (default=30).

        Returns:
        - x_tr_data_cumulative_residuals (list): List of CNN input arrays.
        - y_tr_data_next_day_returns (list): List of next-day returns arrays.
        """
        x_tr_data_cumulative_residuals = []
        y_tr_data_next_day_returns = []

        for current_price_matrix in x_tr_data:
            # Create an instance of the CointegrationResidualGenerator
            residual_generator = CointegrationResidualGenerator(current_price_matrix)

            # Compute residuals
            residual_generator.compute_all_asset_residuals()

            # Get residuals
            asset_residuals = residual_generator.get_asset_residuals()

            # Prepare CNN input
            if len(asset_residuals) < cumulative_residual_window:
                raise ValueError("The cumulative residual window size exceeds the available data.")
            cnn_input = residual_generator.prepare_cnn_input_from_residuals(window=cumulative_residual_window)

            # Reshape CNN input to match the expected shape
            cnn_input_array = cnn_input.transpose(0, 2, 1)  # [samples, features, window]

            # Get the start index of the first 30-day cumulative residuals in the returns DataFrame
            start_idx_in_returns = returns.index.get_loc(asset_residuals.index[0])
            num_samples = len(asset_residuals) - cumulative_residual_window + 1
            next_day_indices = [start_idx_in_returns + i + cumulative_residual_window for i in range(num_samples)]

            # Get the next-day returns for the corresponding indices
            next_day_returns = returns.iloc[next_day_indices]

            # Append the CNN input and next-day returns to their respective lists
            x_tr_data_cumulative_residuals.append(cnn_input_array)
            y_tr_data_next_day_returns.append(next_day_returns.values)

        return x_tr_data_cumulative_residuals, y_tr_data_next_day_returns