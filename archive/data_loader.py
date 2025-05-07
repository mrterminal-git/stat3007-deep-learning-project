import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    Parses European wholesale electricity price data, allowing filtering by country and date range.
    """
    def __init__(self, file_path="./data/european_wholesale_electricity_price_data_daily.csv"):
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        """Loads and preprocesses the data from the CSV file."""
        try:
            df = pd.read_csv(self.file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.drop(columns={'ISO3 Code'}, inplace=True)
            df.rename(columns={'Price (EUR/MWhe)': 'Price'}, inplace=True)
            logging.info(f"Data loaded successfully from {self.file_path}")
            return df
        except FileNotFoundError:
            logging.error(f"File not found at {self.file_path}")
            return None
        except KeyError as e:
            logging.error(f"Expected column '{e}' not found in the CSV")
            return None
        except Exception as e:
            logging.error(f"Error loading or processing file: {e}")
            return None

    def get_data_by_country_and_range(self, time_range: str, country=None):
        """Filters data for a specific country and time range."""
        if self.data is None:
            logging.error("Data not loaded")
            return None

        try:
            start_date_str, end_date_str = time_range.split(',')
            start_date = pd.to_datetime(start_date_str.strip())
            end_date = pd.to_datetime(end_date_str.strip())
        except ValueError:
            logging.error("Invalid time_range format. Use 'YYYY-MM-DD,YYYY-MM-DD'")
            return None

        output_data = self.data.copy()
        if country is not None:
            output_data = output_data[output_data['Country'].str.lower() == country.lower()]

        filtered_data = output_data[
            (output_data['Date'] >= start_date) & (output_data['Date'] <= end_date)
        ]

        if filtered_data.empty:
            logging.warning(f"No data found for country '{country}' in range {time_range}")
            return pd.DataFrame()

        return filtered_data.copy()

    def get_data(self):
        """Returns the loaded data."""
        return self.data