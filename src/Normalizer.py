import pandas as pd
import torch

class Normalizer():
    """
    A static class which allows normalization of dataframes and tensors.
    """
    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize each column of a DataFrame to zero mean and unit variance.
        
        Args:
            df (pd.DataFrame): Input DataFrame to normalize.
        
        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        return (df - df.mean()) / (df.std() + 1e-5)  # Add epsilon to avoid division by zero

    @staticmethod
    def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor to zero mean and unit variance along the specified dimensions.
        
        Args:
            tensor (torch.Tensor): Input tensor to normalize.
        
        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True) + 1e-5  # Add epsilon to avoid division by zero
        return (tensor - mean) / std