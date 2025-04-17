import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from linearmodels.asset_pricing import LinearFactorModel
from sklearn.linear_model import LinearRegression

# Placeholder for IPCA import
# from your_ipca_module import IPCA

class ResidualGenerator:
    def __init__(self, price_data: pd.DataFrame, factor_data: pd.DataFrame = None):
        """
        Initializes the residual generator.

        Parameters:
        - price_data: pd.DataFrame
            Time-aligned price data. Each column corresponds to an asset.
        - factor_data: pd.DataFrame (optional)
            Required for Fama-French or custom factor models. Index must align with price_data.
        """
        self.price_data = price_data
        self.factor_data = factor_data
        self.returns = self._compute_returns(price_data)
        self.residuals = {
            'fama_french': None,
            'pca': None,
            'ipca': None
        }

    def _compute_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Computes log returns."""
        return np.log(price_data / price_data.shift(1)).dropna()

    def compute_fama_french_residuals(self):
        """Fits Fama-French linear factor model and stores residuals."""
        if self.factor_data is None:
            raise ValueError("Factor data must be provided for Fama-French model.")

        residuals = pd.DataFrame(index=self.returns.index, columns=self.returns.columns)
        for asset in self.returns.columns:
            model = LinearRegression().fit(self.factor_data.loc[self.returns.index], self.returns[asset])
            predicted = model.predict(self.factor_data.loc[self.returns.index])
            residuals[asset] = self.returns[asset] - predicted
        self.residuals['fama_french'] = residuals

    def compute_pca_residuals(self, n_components: int = 3):
        """Computes PCA-based residuals."""
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(self.returns)
        loadings = pca.components_.T
        reconstructed = scores @ loadings.T
        residuals = self.returns - reconstructed
        self.residuals['pca'] = pd.DataFrame(residuals, index=self.returns.index, columns=self.returns.columns)

    def compute_ipca_residuals(self):
        """Computes IPCA residuals. Placeholder — implement with your IPCA module."""
        # Example (replace with actual implementation):
        # ipca = IPCA(...)
        # ipca.fit(...)
        # residuals = self.returns - ipca.predict(...)
        # self.residuals['ipca'] = pd.DataFrame(residuals, index=self.returns.index, columns=self.returns.columns)
        raise NotImplementedError("IPCA method not implemented. Please plug in your IPCA estimator.")

    def get_residuals(self, method: str) -> pd.DataFrame:
        """
        Returns residuals for the given method.

        Parameters:
        - method: str ('fama_french' | 'pca' | 'ipca')

        Returns:
        - pd.DataFrame
        """
        if method not in self.residuals:
            raise ValueError(f"Unknown method '{method}'.")
        if self.residuals[method] is None:
            raise ValueError(f"Residuals for method '{method}' not yet computed.")
        return self.residuals[method]
