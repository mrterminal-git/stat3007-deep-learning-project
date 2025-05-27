from PriceDataLoader import PriceDataLoader
from WeatherDataLoader import WeatherDataLoader
from models.CNN import *
from models.FFN import *
import torch
from models.CointegrationResidualGenerator import CointegrationResidualGenerator
from PortfolioOptimizer import PortfolioOptimizer
from Trainer import Trainer
from PortfolioOptimizer import PortfolioOptimizer
from DataPreparation import DataPreparation
import pandas as pd
from datetime import datetime, timedelta
import os

device = torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def generate_all_data_via_rolling_windows(
    price_parser,
    weather_parser,
    countries_list,
    weather_features,
    window_size=30,
    cointegration_period=250,
    stride=5,
    start_date="2015-01-01",
    end_date="2024-12-31"
):
    """
    Generates train/val/test data using a rolling cointegration window.

    Parameters:
        price_parser: PriceDataLoader object
        weather_parser: WeatherDataLoader object
        cointegration_period: int, window size for cointegration period (default=250)
        stride: int, how much to roll the window forward each time (default=5)
        start_date: str, beginning of date range for rolling windows
        end_date: str, end of date range for rolling windows

    Returns:
        Tuple of lists of train/val/test data and returns:
        (
            all_train_data, all_train_returns,
            all_val_data, all_val_returns,
            all_test_data, all_test_returns
        )
    """

    all_train_data, all_train_returns = [], []
    all_val_data, all_val_returns = [], []
    all_test_data, all_test_returns = [], []

    current_start = pd.to_datetime(start_date)
    final_end = pd.to_datetime(end_date)

    while current_start + timedelta(days=cointegration_period + 30 + 1) <= final_end:
        current_end = current_start + timedelta(days=cointegration_period)
        date_range = f"{current_start.date()},{current_end.date()}"
        print(f"----------------------Processing date range: {date_range}----------------------")

        # countries_list = list(set(price_parser.get_countries_with_complete_data(date_range)) &
        #                       set(weather_parser.get_country_list()))
        if not countries_list:
            current_start += timedelta(days=stride)
            continue

        # Generate price matrix
        price_matrix = price_parser.get_price_matrix(
            time_range=date_range,
            countries=countries_list,
            fill_method="ffill"
        )
        # print(f"Shape of price_matrix: {price_matrix.shape}")

        # Compute returns
        returns = price_matrix.pct_change().dropna()
        # print(f"Shape of returns: {returns.shape}")

        # Generate asset residuals using cointegration
        residual_generator = CointegrationResidualGenerator(price_matrix)
        residual_generator.compute_all_asset_residuals()
        asset_residuals = residual_generator.get_asset_residuals()

        # print(f"Number of total features (price + weather): {len(weather_features) * len(countries_list) + 1}")

        # Generate weather matrix
        weather_matrix = weather_parser.get_weather_matrix(
            time_range=date_range,
            countries=countries_list,
            fill_method="ffill",
            features=weather_features
        )
        # print(f"Shape of weather_matrix: {weather_matrix.shape}")


        dp = DataPreparation(
            price_residuals=asset_residuals,
            weather_data=weather_matrix,
            countries=countries_list,
            weather_features=weather_features
        )
        window_size = 30
        inner_stride = 1
        combined_data = dp.prepare_rolling_windows(window_size=window_size, stride=inner_stride) 
        # [num_samples, num_features, window_size]
        
        next_day_returns = dp.prepare_next_day_returns(returns=returns, window_size=window_size, stride=inner_stride)
        # [num_samples, num_countries]
        
        # Split the data in time-series
        (train_data, train_returns), (val_data, val_returns), (test_data, test_returns) = \
        dp.create_train_val_test_split(combined_data=combined_data, next_day_returns=next_day_returns)

        if train_data is not None and len(train_data) > 0:
            all_train_data.extend(train_data)
            all_train_returns.extend(train_returns)

        if val_data is not None and len(val_data) > 0:
            all_val_data.extend(val_data)
            all_val_returns.extend(val_returns)

        if test_data is not None and len(test_data) > 0:
            all_test_data.extend(test_data)
            all_test_returns.extend(test_returns)

        current_start += timedelta(days=stride)

    return (
        all_train_data, all_train_returns,
        all_val_data, all_val_returns,
        all_test_data, all_test_returns
    )


price_parser = PriceDataLoader(file_path="../data/european_wholesale_electricity_price_data_daily.csv")
weather_parser = WeatherDataLoader(file_path="../data/aggregated_weather.csv")

data_path = "data/all_data.pt"
force_regenerate = True

if os.path.exists(data_path) and not force_regenerate:
    # Load pre-saved data
    print("Loading pre-saved data...")
    checkpoint = torch.load(data_path)
    all_train_data = checkpoint["train_data"]
    all_train_returns = checkpoint["train_returns"]
    all_val_data = checkpoint["val_data"]
    all_val_returns = checkpoint["val_returns"]
    all_test_data = checkpoint["test_data"]
    all_test_returns = checkpoint["test_returns"]
else:
    # Generate data
    print("Generating data...")
    
    # Define the time range and valid countries
    start_date="2015-01-01"
    end_date="2024-12-31"
    time_range = f"{start_date},{end_date}"
    countries_list = list(set(price_parser.get_countries_with_complete_data(time_range)) & 
                     set(weather_parser.get_country_list()))
    
    # Define weather features
    weather_features = [
        'temperature_2m_mean', 'temperature_2m_min', 'temperature_2m_max',
        'precipitation_mean', 'precipitation_min', 'precipitation_max',
        'wind_speed_mean', 'wind_speed_min', 'wind_speed_max'
    ]

    # Define window size for rolling windows
    window_size = 30

    # Generate all data using rolling windows
    all_train_data, all_train_returns, all_val_data, all_val_returns, all_test_data, all_test_returns = generate_all_data_via_rolling_windows(
        price_parser,
        weather_parser,
        countries_list=countries_list,
        weather_features=weather_features,
        window_size=30,
        cointegration_period=365,
        stride=20,
        start_date=start_date,
        end_date=end_date
    )

    # Save the result
    os.makedirs("data", exist_ok=True)
    torch.save({
        "train_data": all_train_data,
        "train_returns": all_train_returns,
        "val_data": all_val_data,
        "val_returns": all_val_returns,
        "test_data": all_test_data,
        "test_returns": all_test_returns
    }, data_path)
 
# Convert to tensors
train_data_tensor = torch.FloatTensor(all_train_data).to(device)
train_returns_tensor = torch.FloatTensor(all_train_returns).to(device)
val_data_tensor = torch.FloatTensor(all_val_data).to(device)
val_returns_tensor = torch.FloatTensor(all_val_returns).to(device)
test_data_tensor = torch.FloatTensor(all_test_data).to(device)
test_returns_tensor = torch.FloatTensor(all_test_returns).to(device)

print(f"Train data shape: {train_data_tensor.shape}") # [samples, num_features, window size]
print(f"Train returns shape: {train_returns_tensor.shape}") # [samples, num_countries (returns)]


# We initialize the portfolio optimizer and set up our trainer, which allows grid search for finding the best hyperparameters.
portfolio_optimizer = PortfolioOptimizer(
    window_size=window_size,
    num_countries=len(countries_list),
    num_weather_features=len(weather_features),
    num_filters=8,  # Default value
    filter_size=3,
    hidden_dim=64,
    num_heads=4,
    use_transformer=True,
    device=device
)

trainer = Trainer(
    optimizer=portfolio_optimizer,
    train_data=train_data_tensor,
    train_returns=train_returns_tensor,
    val_data=val_data_tensor,
    val_returns=val_returns_tensor,
    lr=0.001,
    num_epochs=100,
    batch_size=32,
    patience=50,
    device=device,
)

# Can add more parameters to test
param_grid = {
    'num_filters': [8, 16, 32],
    'filter_size': [3, 5, 7],
    'hidden_dim': [64, 128]
}
best_params, best_score, best_returns = trainer.grid_search(param_grid, verbose=False)
print(f"Best parameters: {best_params} with Sharpe Ratio: {best_score:.4f}")

# Reinitialize with best parameters and train on train+val
portfolio_optimizer = PortfolioOptimizer(
    window_size=window_size,
    num_countries=len(countries_list),
    num_weather_features=len(weather_features),
    num_filters=best_params['num_filters'],
    filter_size=best_params['filter_size'],
    hidden_dim=best_params['hidden_dim'],
    num_heads=4,
    use_transformer=True,
    device=device
)
trainer = Trainer(
    optimizer=portfolio_optimizer,
    train_data=torch.cat([train_data_tensor, val_data_tensor]),  # Combine train and val
    train_returns=torch.cat([train_returns_tensor, val_returns_tensor]),
    val_data=val_data_tensor,  # Still use val for early stopping
    val_returns=val_returns_tensor,
    lr=0.001,
    num_epochs=100,
    batch_size=32,
    patience=50,
    device=device
)
final_sharpe, _ = trainer.train(verbose=True)