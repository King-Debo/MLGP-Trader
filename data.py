# Import the necessary libraries and modules
import numpy as np
import pandas as pd
import requests # For accessing the APIs
import bs4 # For web scraping
import sqlite3 # For connecting to the databases
import sklearn.preprocessing as skp # For data transformation

# Define the constants and variables
API_KEY = "YOUR_API_KEY" # The API key for accessing the APIs
API_URL = "https://www.alphavantage.co/query" # The API URL for accessing the APIs
DB_NAME = "market_data.db" # The name of the database file
DB_TABLE = "prices" # The name of the database table
TRAIN_RATIO = 0.8 # The ratio of the training data to the total data

# Define the functions for data acquisition and preprocessing
def get_data(assets, start_date, end_date, frequency):
    """
    Get the market data for the specified assets, time period, and frequency.
    The market data includes the prices and volumes of the assets.
    The market data can be obtained from the APIs, web scraping, or databases.
    The market data is returned as a pandas data frame.
    """
    # Initialize an empty data frame
    data = pd.DataFrame()
    # For each asset in the assets list
    for asset in assets:
        # Try to get the data from the APIs
        try:
            # Set the parameters for the API request
            params = {
                "function": "TIME_SERIES_" + frequency.upper(), # The function for the time series data
                "symbol": asset, # The symbol of the asset
                "apikey": API_KEY, # The API key
                "outputsize": "full", # The output size of the data
                "datatype": "json" # The data type of the data
            }
            # Send the API request and get the response
            response = requests.get(API_URL, params=params)
            # Convert the response to a JSON object
            response_json = response.json()
            # Get the key for the time series data
            time_series_key = "Time Series (" + frequency.capitalize() + ")"
            # Get the time series data as a dictionary
            time_series_data = response_json[time_series_key]
            # Convert the time series data to a data frame
            time_series_df = pd.DataFrame.from_dict(time_series_data, orient="index")
            # Rename the columns of the data frame
            time_series_df.columns = [asset + "_" + col[3:] for col in time_series_df.columns]
            # Join the data frame to the main data frame
            data = data.join(time_series_df, how="outer")
        # If there is an exception while getting the data from the APIs
        except Exception as e:
            # Print the exception message
            print(f"Exception while getting the data from the APIs for {asset}: {e}")
            # Try to get the data from web scraping
            try:
                # Set the URL for web scraping
                url = f"https://finance.yahoo.com/quote/{asset}/history?period1={start_date}&period2={end_date}&interval={frequency}&filter=history&frequency={frequency}"
                # Send the web scraping request and get the response
                response = requests.get(url)
                # Parse the response using BeautifulSoup
                soup = bs4.BeautifulSoup(response.text, "html.parser")
                # Find the table element that contains the market data
                table = soup.find("table", {"data-test": "historical-prices"})
                # Find all the table row elements that contain the market data
                rows = table.find_all("tr")
                # Initialize an empty list for the market data
                market_data = []
                # For each table row element in the rows list
                for row in rows:
                    # Find all the table data elements that contain the market data
                    cols = row.find_all("td")
                    # If there are six table data elements in the cols list
                    if len(cols) == 6:
                        # Get the date, open, high, low, close, and volume values from the cols list
                        date = cols[0].text
                        open = cols[1].text
                        high = cols[2].text
                        low = cols[3].text
                        close = cols[4].text
                        volume = cols[5].text
                        # Append the values to the market data list as a tuple
                        market_data.append((date, open, high, low, close, volume))
                # Convert the market data list to a data frame
                market_data_df = pd.DataFrame(market_data, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
                # Set the date column as the index of the data frame
                market_data_df.set_index("Date", inplace=True)
                # Rename the columns of the data frame
                market_data_df.columns = [asset + "_" + col for col in market_data_df.columns]
                # Join the data frame to the main data frame
                data = data.join(market_data_df, how="outer")
            # If there is an exception while getting the data from web scraping
            except Exception as e:
                # Print the exception message
                print(f"Exception while getting the data from web scraping for {asset}: {e}")
                # Try to get the data from the databases
                try:
                    # Connect to the database file
                    conn = sqlite3.connect(DB_NAME)
                    # Create a cursor object
                    cur = conn.cursor()
                    # Execute the SQL query to get the market data from the database table
                    cur.execute(f"SELECT * FROM {DB_TABLE} WHERE asset = '{asset}' AND date BETWEEN '{start_date}' AND '{end_date}'")
                    # Fetch all the rows from the cursor object
                    rows = cur.fetchall()
                    # Close the cursor object
                    cur.close()
                    # Close the connection
                    conn.close()
                    # Convert the rows to a data frame
                    database_data_df = pd.DataFrame(rows, columns=["Date", "Asset", "Open", "High", "Low", "Close", "Volume"])
                    # Set the date column as the index of the data frame
                    database_data_df.set_index("Date", inplace=True)
                    # Drop the asset column from the data frame
                    database_data_df.drop("Asset", axis=1, inplace=True)
                    # Rename the columns of the data frame
                    database_data_df.columns = [asset + "_" + col for col in database_data_df.columns]
                    # Join the data frame to the main data frame
                    data = data.join(database_data_df, how="outer")
                # If there is an exception while getting the data from the databases
                except Exception as e:
                    # Print the exception message
                    print(f"Exception while getting the data from the databases for {asset}: {e}")
                    # Raise an error
                    raise ValueError(f"Unable to get the data for {asset}")
    # Return the main data frame
    return data

def add_indicators(data, indicators):
    """
    Add the indicators to the market data.
    The indicators are calculated using the prices and volumes of the assets.
    The indicators are added as new columns to the market data.
    The market data is returned as a pandas data frame.
    """
    # For each asset in the assets list
    for asset in ASSETS:
        # For each indicator in the indicators list
        for indicator in indicators:
            # If the indicator is SMA (Simple Moving Average)
            if indicator == "SMA":
                # Calculate the SMA using the close price of the asset
                data[asset + "_SMA"] = data[asset + "_close"].rolling(window=20).mean()
            # If the indicator is EMA (Exponential Moving Average)
            elif indicator == "EMA":
                # Calculate the EMA using the close price of the asset
                data[asset + "_EMA"] = data[asset + "_close"].ewm(span=20).mean()
            # If the indicator is RSI (Relative Strength Index)
            elif indicator == "RSI":
                # Calculate the RSI using the close price of the asset
                delta = data[asset + "_close"].diff() # Get the difference between consecutive close prices
                up = delta.clip(lower=0) # Get the positive changes
                down = -delta.clip(upper=0) # Get the negative changes
                ema_up = up.ewm(com=13, adjust=False).mean() # Get the exponential moving average of the positive changes
                ema_down = down.ewm(com=13, adjust=False).mean() # Get the exponential moving average of the negative changes
                rs = ema_up / ema_down # Get the relative strength
                data[asset + "_RSI"] = 100 - (100 / (1 + rs)) # Get the relative strength index
            # If the indicator is MACD (Moving Average Convergence Divergence)
            elif indicator == "MACD":
                # Calculate the MACD using the close price of the asset
                ema_12 = data[asset + "_close"].ewm(span=12, adjust=False).mean() # Get the 12-period exponential moving average of the close price
                ema_26 = data[asset + "_close"].ewm(span=26, adjust=False).mean() # Get the 26-period exponential moving average of the close price
                macd = ema_12 - ema_26 # Get the MACD line
                signal = macd.ewm(span=9, adjust=False).mean() # Get the signal line
                data[asset + "_MACD"] = macd # Add the MACD line to the market data
                data[asset + "_SIGNAL"] = signal # Add the signal line to the market data
            # If the indicator is ADX (Average Directional Index)
            elif indicator == "ADX":
                # Calculate the ADX using the high, low, and close prices of the asset
                high = data[asset + "_high"] # Get the high price of the asset
                low = data[asset + "_low"] # Get the low price of the asset
                close = data[asset + "_close"] # Get the close price of the asset
                up = high - high.shift(1) # Get the difference between consecutive high prices
                down = low.shift(1) - low # Get the difference between consecutive low prices
                plus_dm = up.where((up > 0) & (up > down), 0) # Get the positive directional movement
                minus_dm = down.where((down > 0) & (down > up), 0) # Get the negative directional movement
                tr = pd.DataFrame({"high-low": high - low, "high-close": np.abs(high - close.shift(1)), "low-close": np.abs(low - close.shift(1))}) # Get the true range components
                tr = tr.max(axis=1) # Get the true range
                atr = tr.rolling(window=14).mean() # Get the average true range
                plus_di = 100 * plus_dm.rolling(window=14).mean() / atr # Get the positive directional indicator
                minus_di = 100 * minus_dm.rolling(window=14).mean() / atr # Get the negative directional indicator
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di) # Get the directional movement index
                adx = dx.rolling(window=14).mean() # Get the average directional index
                data[asset + "_ADX"] = adx # Add the ADX to the market data
            # If the indicator is invalid
            else:
                # Raise an error
                raise ValueError(f"Invalid indicator: {indicator}")
    # Return the market data with the indicators
    return data
