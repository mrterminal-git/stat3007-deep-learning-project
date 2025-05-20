import pandas as pd
import os 
import numpy as np 

class data_agg:
    '''
    This class is used to aggregate data from multiple CSV files into a single DataFrame.
    We have all the weather data in multiple CSV files named as weather_data_<country_name>.csv
    What we want to do is to aggregate all the data into a single DataFrame
    We also need the go make the data from hourly to daily.
    This is going to be done by taking th mean of the data for each day.
    We will also make max and min of the data for each day.
    This will be both the wind speed (which i think we should combined u10 and v10), the temperature and precipitation.
    '''
    def __init__(self, data_dir: str = "weather_data_output", output_file: str = "aggregated_weather_data.csv"):
        self.data_dir = data_dir
        self.output_file = output_file
        self.weather_data = None # Will store the combined hourly data
        self.aggregated_data = None # Will store the final daily aggregated data
        # Corrected list of countries based on typical European context, adjust if needed
        self.countries = ['Austria', 'Belgium', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom', 'Bulgaria', 'Serbia', 'Croatia', 'Montenegro', 'North Macedonia', 'Ireland']

    def load_all_country_data(self):
        """
        Loads weather data for all specified countries and concatenates them.
        """
        all_data_frames = []
        for country in self.countries:
            file_name = f"weather_data_{country.replace(' ', '_')}.csv" # Handle spaces in country names if any
            file_path = os.path.join(self.data_dir, file_name)
            try:
                df = pd.read_csv(file_path)
                df['country'] = country # Add a column for the country name
                all_data_frames.append(df)
                print(f"Successfully loaded {file_path}")
            except FileNotFoundError:
                print(f"Warning: File not found for {country} at {file_path}")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

        if not all_data_frames:
            print("No data files were loaded. Please check the data_dir and country list.")
            self.weather_data = pd.DataFrame() # Initialize as empty DataFrame
            return
        
        self.weather_data = pd.concat(all_data_frames, ignore_index=True)
        print(f"Combined data shape: {self.weather_data.shape}")
        # Ensure 'valid_time' column is parsed as datetime
        if 'valid_time' in self.weather_data.columns:
            self.weather_data['valid_time'] = pd.to_datetime(self.weather_data['valid_time'])
        else:
            print("Warning: 'valid_time' column not found in the loaded data. Daily aggregation might fail.")
        return self.weather_data

    def _calculate_wind_speed(self, df):
        """
        Calculates wind speed from u and v components.
        Assumes columns 'u10' and 'v10' exist.
        """
        if 'u10' in df.columns and 'v10' in df.columns:
            df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
        else:
            print("Warning: Wind component columns (u10, v10) not found. Cannot calculate wind speed.")
            df['wind_speed'] = np.nan # Add an empty column if components are missing
        return df

    def aggregate_to_daily(self):
        """
        Aggregates the loaded hourly data to daily mean, min, and max values
        for temperature (t2m), precipitation (tp), and calculated wind speed.
        The final columns will be named according to the specified output format.
        """
        if self.weather_data is None or self.weather_data.empty:
            print("No weather data loaded. Please run load_all_country_data() first.")
            return None

        if 'valid_time' not in self.weather_data.columns or not pd.api.types.is_datetime64_any_dtype(self.weather_data['valid_time']):
            print("Error: 'valid_time' column is missing or not in datetime format. Cannot perform daily aggregation.")
            return None

        # Calculate wind speed first
        processed_data = self._calculate_wind_speed(self.weather_data.copy()) # Use a copy to avoid modifying original

        # Define aggregation functions using source column names
        aggregations = {
            't2m': ['mean', 'min', 'max'],          # Source column for temperature
            'tp': ['mean', 'min', 'max'],           # Source column for precipitation
            'wind_speed': ['mean', 'min', 'max']    # Calculated wind speed
        }
        
        # Check if columns exist before trying to aggregate them
        cols_to_aggregate = {
            col: funcs for col, funcs in aggregations.items() if col in processed_data.columns
        }
        if not cols_to_aggregate:
            print(f"Warning: None of the target source columns ({', '.join(aggregations.keys())}) found for aggregation.")
            self.aggregated_data = pd.DataFrame()
            return self.aggregated_data

        # Group by country and date (derived from 'valid_time')
        daily_aggregated = processed_data.groupby(['country', pd.Grouper(key='valid_time', freq='D')]).agg(cols_to_aggregate)

        # Flatten MultiIndex columns (e.g., t2m_mean, tp_min)
        daily_aggregated.columns = ['_'.join(col).strip() for col in daily_aggregated.columns.values]
        
        # Rename columns to the desired output format
        column_rename_map = {
            't2m_mean': 'temperature_2m_mean',
            't2m_min': 'temperature_2m_min',
            't2m_max': 'temperature_2m_max',
            'tp_mean': 'precipitation_mean',
            'tp_min': 'precipitation_min',
            'tp_max': 'precipitation_max',
            # wind_speed columns are already in the desired format (e.g., wind_speed_mean)
        }
        daily_aggregated = daily_aggregated.rename(columns=column_rename_map)
        
        daily_aggregated = daily_aggregated.reset_index() # Bring 'country' and 'valid_time' back as columns

        self.aggregated_data = daily_aggregated
        print(f"Daily aggregated data shape: {self.aggregated_data.shape}")
        if not self.aggregated_data.empty:
            print(f"Aggregated columns: {self.aggregated_data.columns.tolist()}")
        return self.aggregated_data

    def save_aggregated_data(self):
        """
        Saves the aggregated daily data to the specified output file.
        """
        if self.aggregated_data is None or self.aggregated_data.empty:
            print("No aggregated data to save. Please run aggregate_to_daily() first.")
            return

        try:
            self.aggregated_data.to_csv(self.output_file, index=False)
            print(f"Aggregated data saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving aggregated data: {e}")

    def process(self):
        """
        Runs the full pipeline: load, aggregate, and save.
        """
        self.load_all_country_data()
        if self.weather_data is not None and not self.weather_data.empty:
            self.aggregate_to_daily()
            if self.aggregated_data is not None and not self.aggregated_data.empty:
                self.save_aggregated_data()
        else:
            print("Processing stopped due to issues in loading data.")

# Example Usage (you can put this in a separate script or notebook cell)
if __name__ == '__main__':

    # Initialize and run the aggregator with the dummy data
    aggregator = data_agg(data_dir='weather_data_output', output_file="aggregated_weather.csv")
    aggregator.process()





