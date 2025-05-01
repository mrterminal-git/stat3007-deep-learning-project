import pandas as pd
from io import StringIO

class DataLoader:
    """
    Parses European wholesale electricity price data, allowing filtering
    by country and date range.
    """
    def __init__(self, file_path="./data/european_wholesale_electricity_price_data_daily.csv"):
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

# Example usage

parser = DataLoader()
# Example 1: Get data for Germany in 2020
germany_2020 = parser.get_data_by_country_and_range("2020-01-01,2020-12-31")

print("\n--- Germany 2020 Data ---")
print(germany_2020.head(20))
print(f"Shape: {germany_2020.shape}")
# Example 2: Get data for France for a specific week
france_week = parser.get_data_by_country_and_range("2021-05-10,2021-05-16","France")
print("\n--- France May 2021 Week Data ---")
print(france_week)
print(f"Shape: {france_week.shape}")
# Example 3: Country not found
print("\n--- Country Not Found Example ---")
non_existent = parser.get_data_by_country_and_range("Atlantis", "2020-01-01,2020-12-31")
