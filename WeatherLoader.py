import cdsapi
import zipfile
import os
import time # Optional: for adding delays if you encounter rate limiting

class WeatherLoader:
    """
    A class to load weather data from the Copernicus Climate Data Store (CDS).
    """

    def __init__(self, output_dir="weather_data_output"):
        """
        Initializes the WeatherLoader.
        :param output_dir: Directory to save the processed weather files.
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
        self.client = cdsapi.Client()

    def download_weather_data_for_country(self, country_name, longitude, latitude):
        """
        Downloads weather data for a given country, unpacks if it's a zip,
        and renames the CSV file.

        Note: The ERA5 dataset provides data on a grid. Requesting a single point
        via 'area' will typically give you the data for the grid cell(s) covering that point.
        The CSV output might contain headers indicating the specific lat/lon of the grid point.
        """
        # Using 'reanalysis-era5-single-levels-monthly-means' for monthly averaged data,
        # which is more manageable for long time series.
        # If daily or hourly data is needed, 'reanalysis-era5-single-levels' should be used,
        # and the request parameters (especially for date, time, day) would need adjustment.
        dataset = "reanalysis-era5-single-levels-monthly-means"

        # Define a temporary name for the downloaded file.
        # Based on the problem description, we expect a zip file.
        temp_download_filename = f"temp_{country_name}_download.zip"
        # Define the final path for the CSV file
        final_csv_filename = os.path.join(self.output_dir, f"weather_data_{country_name}.csv")

        # Check if the final CSV already exists to avoid re-downloading
        if os.path.exists(final_csv_filename):
            print(f"Data for {country_name} already exists: {final_csv_filename}. Skipping.")
            return

        print(f"Preparing to download weather data for {country_name} (Lon: {longitude}, Lat: {latitude})...")


        dataset = "reanalysis-era5-single-levels-timeseries"
        request = {
            "variable": [
                "2m_temperature",
                "total_precipitation",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind"
            ],
            "location": {"longitude": longitude, "latitude": latitude},
            "date": ["2015-01-01/2025-05-01"],
            "data_format": "csv"
        }
        # Filter out months for the year 2025 beyond May (as per original date "2025-05-01")
        # For monthly data, this means up to and including April 2025.
        # However, the API request will fetch all specified months for all specified years.
        # The data itself will only be available up to the present minus a few months.
        # For 2025, if current date is May 2025, data might only be available up to Feb or Mar 2025.
        # No specific filtering by day "01" needed for monthly means.

        try:
            print(f"Requesting data for {country_name} from CDS...")
            # The download() method can take a target file path
            self.client.retrieve(dataset, request, temp_download_filename)
            print(f"Download complete for {country_name}. File saved as {temp_download_filename}")

            # Unpack the zip file
            if not zipfile.is_zipfile(temp_download_filename):
                # This case should ideally not happen if 'csv_zip' format works as expected
                # or if the API always zips CSVs.
                print(f"Warning: Downloaded file for {country_name} ({temp_download_filename}) is not a zip file as expected.")
                # Attempt to treat it as a CSV directly
                if temp_download_filename.lower().endswith(".csv"):
                    os.rename(temp_download_filename, final_csv_filename)
                    print(f"Renamed {temp_download_filename} to {final_csv_filename} (assumed it was a direct CSV).")
                else: # If it's not a zip and not ending with .csv, unsure what to do.
                    print(f"Error: Downloaded file {temp_download_filename} is not a zip and not a .csv. Cannot process.")
                return # Exit this function for this country

            print(f"Unzipping {temp_download_filename}...")
            with zipfile.ZipFile(temp_download_filename, 'r') as zip_ref:
                # List files in zip to find the CSV
                csv_files_in_zip = [name for name in zip_ref.namelist() if name.lower().endswith('.csv')]

                if not csv_files_in_zip:
                    print(f"Error: No CSV file found in the zip archive {temp_download_filename} for {country_name}.")
                    return

                # Assume the first CSV found is the one we want
                extracted_csv_name_from_zip = csv_files_in_zip[0]
                # Extract the CSV to the output directory, it will have its original name from the zip
                zip_ref.extract(extracted_csv_name_from_zip, self.output_dir)
                # Path to the extracted CSV with its original name
                temp_extracted_csv_path = os.path.join(self.output_dir, extracted_csv_name_from_zip)

                print(f"Extracted '{extracted_csv_name_from_zip}' from zip for {country_name}.")

            # Rename the extracted CSV file
            os.rename(temp_extracted_csv_path, final_csv_filename)
            print(f"Successfully renamed '{temp_extracted_csv_path}' to '{final_csv_filename}'.")
        except zipfile.BadZipFile:
            print(f"Error: The downloaded file {temp_download_filename} for {country_name} was not a valid zip file.")
        except Exception as e:
            print(f"CDS API Error for {country_name}: {e}")
            print(f"Request details were: {request}")
            print(f"An unexpected error occurred while processing {country_name}: {e}")
        finally:
            # Clean up the temporary downloaded zip file
            if os.path.exists(temp_download_filename):
                os.remove(temp_download_filename)
                print(f"Removed temporary file: {temp_download_filename}")
            # Optional: Add a small delay to be courteous to the API
            # time.sleep(1)


# --- Main execution ---
country_coordinates = {
    # 'Austria': (13, 48),
    'Belgium': (5, 51),
    'Czechia': (16, 50),
    'Denmark': (12, 56),
    'Estonia': (26, 59),
    'Finland': (26, 64),
    'France': (2, 47),
    'Germany': (10, 51),
    'Greece': (21, 40),
    'Hungary': (20, 47),
    'Italy': (13, 43),
    'Latvia': (25, 57),
    'Lithuania': (24, 55),
    'Luxembourg': (6, 50),
    'Netherlands': (5, 52),
    'Norway': (12, 64),
    'Poland': (20, 52),
    'Portugal': (-8, 40),
    'Romania': (25, 46),
    'Slovakia': (19, 49),
    'Slovenia': (15, 46),
    'Spain': (-4, 40),
    'Sweden': (16, 62),
    'Switzerland': (8, 47),
    'United Kingdom': (-3, 54),
    'Bulgaria': (26, 43),
    'Serbia': (21, 44),
    'Croatia': (16, 45),
    'Montenegro': (19, 43),
    'North Macedonia': (22, 42),
    'Ireland': (-8, 53)
}

if __name__ == "__main__":
    # Create an instance of the WeatherLoader
    # This will create a folder named "weather_data_output" in the same directory as the script
    # to store the final CSV files.
    weather_loader = WeatherLoader()

    # Loop through the countries and download/process data
    for country_name_key, coords in country_coordinates.items():
        longitude_val, latitude_val = coords
        weather_loader.download_weather_data_for_country(country_name_key, longitude_val, latitude_val)
        time.sleep(1)
        print("-" * 30) # Separator for console output
        

    print("All weather data download and processing attempts are complete.")
    print(f"Final CSV files should be in the '{weather_loader.output_dir}' directory.")


