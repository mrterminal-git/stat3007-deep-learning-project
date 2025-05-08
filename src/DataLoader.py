import pandas as pd
from io import StringIO
from typing import Optional, List

class DataLoader:
    """
    Parses European wholesale electricity price data, allowing filtering
    by country and date range.
    """
    def __init__(self, file_path="../data/european_wholesale_electricity_price_data_daily.csv"):
        """
        Initializes the parser and loads the data.

        Args:
            file_path (str): The path to the CSV file.
        """
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        """Loads and preprocesses the data from the CSV file."""
        try:
            df = pd.read_csv(self.file_path)
            # Convert 'Date' column to datetime objects
            df['Date'] = pd.to_datetime(df['Date'])
            # Drop ISO3 Code column
            df.drop(columns={'ISO3 Code'}, inplace=True)
            # Rename price column for easier access
            df.rename(columns={'Price (EUR/MWhe)': 'Price'}, inplace=True)
            print(f"Data loaded successfully from {self.file_path}")
            
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return None
        except KeyError as e:
            print(f"Error: Expected column '{e}' not found in the CSV.")
            return None
        except Exception as e:
            print(f"Error loading or processing file: {e}")
            return None

    def get_data_by_country_and_range(self, time_range:str, country=None):
        """
        Filters the data for a specific country and time range.

        Args:
            country (str): The name of the country to filter by (e.g., 'Germany').
            time_range (str): A string representing the date range in the format
                              'YYYY-MM-DD,YYYY-MM-DD'.

        Returns:
            pandas.DataFrame: A DataFrame containing the filtered data,
                              or None if an error occurs or no data is found.
        """
        if self.data is None:
            print("Error: Data not loaded.")
            return None

        try:
            start_date_str, end_date_str = time_range.split(',')
            start_date = pd.to_datetime(start_date_str.strip())
            end_date = pd.to_datetime(end_date_str.strip())
        except ValueError:
            print("Error: Invalid time_range format. Please use 'YYYY-MM-DD,YYYY-MM-DD'.")
            return None
        except Exception as e:
             print(f"Error parsing time range: {e}")
             return None

        outputData = self.data.copy()
        # If a country is specified, filter the data by country, if not use all data
        if country is not None:
            outputData = self.data[self.data['Country'].str.lower() == country.lower()]

        # Filter by date range (inclusive)
        filtered_data = outputData[
            (outputData['Date'] >= start_date) & (outputData['Date'] <= end_date)
        ]

        if filtered_data.empty:
            print(f"Warning: No data found for country '{country}' within the range {time_range}.")
            return pd.DataFrame() # Return empty DataFrame

        return filtered_data.copy() # Return a copy to avoid SettingWithCopyWarning

    def get_all_data(self):
        """
        Returns the entire dataset.

        Returns:
            pandas.DataFrame: The entire dataset.
        """
        if self.data is None:
            print("Error: Data not loaded.")
            return None
        return self.data.copy()

    def get_country_list(self):
        """
        Returns a list of unique countries in the dataset.

        Returns:
            list: A list of unique country names.
        """
        if self.data is None:
            print("Error: Data not loaded.")
            return None
        return self.data['Country'].unique().tolist()
    
    def get_price_matrix(
        self,
        time_range: str,
        countries: List[str],
        fill_method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Returns a price matrix where:
        - Rows = dates
        - Columns = countries
        - Values = daily electricity prices

        Parameters:
        - time_range (str): e.g. "2021-05-10,2021-05-16"
        - countries (List[str]): list of country names to include
        - fill_method (Optional[str]): 'ffill', 'bfill', or None

        Returns:
        - pd.DataFrame: index=date, columns=country names, values=prices
        """
        start_date, end_date = time_range.split(",")

        # Filter the master data once
        df = self.data.copy()
        df = df[df["Country"].isin(countries)]
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

        # Pivot: index=date, columns=country, values=price
        price_matrix = df.pivot(index="Date", columns="Country", values="Price").sort_index()

        # Handle missing data
        if fill_method:
            price_matrix = price_matrix.fillna(method=fill_method)
        else:
            price_matrix = price_matrix.dropna()

        return price_matrix
    

    def get_price_matrix_rolling_window(
        self,
        one_window_days: int,
        window_stride_days: int,
        time_range: str,
        countries: List[str],
        fill_method: Optional[str] = None
    ) -> List[pd.DataFrame]:
        """
        Returns a set of price matrices where:
        - Rows = dates
        - Columns = countries
        - Values = daily electricity prices

        Parameters:
        - one_window_days (int): number of days in one window
        - window_stride_days (int): number of days to stride the window
        - time_range (str): e.g. "2021-05-10,2021-05-16"
        - countries (List[str]): list of country names to include
        - fill_method (Optional[str]): 'ffill', 'bfill', or None

        Returns:
        - List[pd.DataFrame]: A list of price matrices, each representing a rolling window.
        """
        start_date, end_date = time_range.split(",")

        # Filter the master data once
        df = self.data.copy()
        df = df[df["Country"].isin(countries)]
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

        # Pivot: index=date, columns=country, values=price
        price_matrix = df.pivot(index="Date", columns="Country", values="Price").sort_index()

        # Handle missing data
        if fill_method:
            price_matrix = price_matrix.fillna(method=fill_method)
        else:
            price_matrix = price_matrix.dropna()

        # Generate rolling windows
        rolling_windows = []
        for start_idx in range(0, len(price_matrix) - one_window_days + 1, window_stride_days):
            end_idx = start_idx + one_window_days
            if end_idx > len(price_matrix):  # Ensure we don't go past the date range
                break
            rolling_window = price_matrix.iloc[start_idx:end_idx]
            rolling_windows.append(rolling_window)

        return rolling_windows
    
    def get_next_day_returns(
        self,
        rolling_windows: List[pd.DataFrame],
        price_matrix: pd.DataFrame
    ) -> List[pd.Series]:
        """
        Finds the next-day return for the last day in each rolling window.

        Parameters:
        - rolling_windows (List[pd.DataFrame]): A list of price matrices, each representing a rolling window.
        - price_matrix (pd.DataFrame): A DataFrame of daily prices (index=date, columns=country names).

        Returns:
        - List[pd.Series]: A list of Series, where each Series contains the next-day return
        for all countries corresponding to the last date of each rolling window.
        """
        # Calculate returns for the entire price matrix
        returns = price_matrix.pct_change().dropna()

        next_day_returns = []
        for window in rolling_windows:
            # Get the last date in the current rolling window
            last_date = window.index[-1]

            # Find the next day's return
            if last_date in returns.index:
                next_day_idx = returns.index.get_loc(last_date) + 1
                if next_day_idx < len(returns):
                    # Retrieve the return for the immediate next day
                    next_day_return = returns.iloc[next_day_idx]
                    next_day_returns.append(next_day_return)
                else:
                    # If no next day exists, append None
                    next_day_returns.append(None)
            else:
                # If last_date is not in returns, append None
                next_day_returns.append(None)

        return next_day_returns

