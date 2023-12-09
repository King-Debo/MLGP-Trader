# Import the necessary libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests # For accessing the real market APIs
import json # For parsing the real market data

# Define the constants and variables
REAL_MARKET_API_KEY = "YOUR_API_KEY" # The API key for accessing the real market APIs
REAL_MARKET_API_URL = "https://www.alphavantage.co/query" # The API URL for accessing the real market APIs
INITIAL_CAPITAL = 10000 # The initial capital for the trading simulation
COMMISSION = 0.01 # The commission fee for each trade

# Define the functions for simulating and evaluating the trading performance
def simulate_trading(data, signal, environment):
    """
    Simulate the trading performance using the market data and the trading signal in the specified environment.
    The market data includes the prices and volumes of the assets.
    The trading signal includes the buy, sell, or hold signals for each asset and time.
    The environment can be either virtual or real.
    The trading performance includes the profit curve, the number of trades, the win rate, the drawdown, and the benchmarks.
    The trading performance is returned as a dictionary object.
    """
    # Initialize an empty dictionary for the trading performance
    trading_performance = {}
    # Initialize the profit curve as a series object with the same index as the data
    profit_curve = pd.Series(0, index=data.index)
    # Initialize the number of trades as a scalar object
    number_of_trades = 0
    # Initialize the number of wins as a scalar object
    number_of_wins = 0
    # Initialize the drawdown as a scalar object
    drawdown = 0
    # Initialize the benchmarks as a dictionary object
    benchmarks = {}
    # Initialize the portfolio as a dictionary object with the keys as the assets and the values as the number of shares
    portfolio = {asset: 0 for asset in ASSETS}
    # Initialize the cash as a scalar object with the value as the initial capital
    cash = INITIAL_CAPITAL
    # For each row in the data and signal data frames
    for index, row in zip(data.index, signal.index):
        # If the index of the data and signal rows are the same
        if index == row:
            # Get the prices and volumes of the assets from the data row
            prices = data.loc[index, [asset + "_close" for asset in ASSETS]].values
            volumes = data.loc[index, [asset + "_volume" for asset in ASSETS]].values
            # Get the signals of the assets from the signal row
            signals = signal.loc[index, :].values
            # For each asset, price, volume, and signal in the lists
            for asset, price, volume, signal in zip(ASSETS, prices, volumes, signals):
                # If the signal is buy
                if signal == "buy":
                    # Calculate the number of shares to buy based on the cash and the price
                    shares = int(cash / price)
                    # If the number of shares is positive
                    if shares > 0:
                        # Update the portfolio with the number of shares
                        portfolio[asset] += shares
                        # Update the cash with the price and the commission fee
                        cash -= price * shares * (1 + COMMISSION)
                        # Update the number of trades
                        number_of_trades += 1
                # If the signal is sell
                elif signal == "sell":
                    # Get the number of shares to sell from the portfolio
                    shares = portfolio[asset]
                    # If the number of shares is positive
                    if shares > 0:
                        # Update the portfolio with the number of shares
                        portfolio[asset] -= shares
                        # Update the cash with the price and the commission fee
                        cash += price * shares * (1 - COMMISSION)
                        # Update the number of trades
                        number_of_trades += 1
                        # Calculate the profit from the sell trade
                        profit = price * shares - data.loc[index, asset + "_open"] * shares
                        # If the profit is positive
                        if profit > 0:
                            # Update the number of wins
                            number_of_wins += 1
                # If the signal is hold
                elif signal == "hold":
                    # Do nothing
                    pass
                # If the signal is invalid
                else:
                    # Raise an error
                    raise ValueError(f"Invalid signal: {signal}")
            # Calculate the total value of the portfolio and the cash
            total_value = cash + np.sum(prices * list(portfolio.values()))
            # Update the profit curve with the total value
            profit_curve[index] = total_value
            # Calculate the peak value of the profit curve
            peak_value = profit_curve[:index].max()
            # Calculate the current drawdown
            current_drawdown = (peak_value - total_value) / peak_value
            # If the current drawdown is greater than the previous drawdown
            if current_drawdown > drawdown:
                # Update the drawdown
                drawdown = current_drawdown
        # If the index of the data and signal rows are not the same
        else:
            # Raise an error
            raise ValueError(f"Index mismatch: {index} != {row}")
    # If the environment is real
    if environment == "real":
        # For each asset in the assets list
        for asset in ASSETS:
            # Try to get the real market data from the APIs
            try:
                # Set the parameters for the API request
                params = {
                    "function": "GLOBAL_QUOTE", # The function for the global quote data
                    "symbol": asset, # The symbol of the asset
                    "apikey": REAL_MARKET_API_KEY # The API key
                }
                # Send the API request and get the response
                response = requests.get(REAL_MARKET_API_URL, params=params)
                # Convert the response to a JSON object
                response_json = response.json()
                # Get the key for the global quote data
                global_quote_key = "Global Quote"
                # Get the global quote data as a dictionary
                global_quote_data = response_json[global_quote_key]
                # Get the price and volume of the asset from the global quote data
                price = float(global_quote_data["05. price"])
                volume = float(global_quote_data["06. volume"])
                # Update the prices and volumes lists with the real market data
                prices[ASSETS.index(asset)] = price
                volumes[ASSETS.index(asset)] = volume
            # If there is an exception while getting the real market data from the APIs
            except Exception as e:
                # Print the exception message
                print(f"Exception while getting the real market data from the APIs for {asset}: {e}")
                # Raise an error
                raise ValueError(f"Unable to get the real market data for {asset}")
        # For each asset, price, volume, and signal in the lists
        for asset, price, volume, signal in zip(ASSETS, prices, volumes, signals):
            # If the signal is buy
            if signal == "buy":
                # Calculate the number of shares to buy based on the cash and the price
                shares = int(cash / price)
                # If the number of shares is positive
                if shares > 0:
                    # Update the portfolio with the number of shares
                    portfolio[asset] += shares
                    # Update the cash with the price and the commission fee
                    cash -= price * shares * (1 + COMMISSION)
                    # Update the number of trades
                    number_of_trades += 1
            # If the signal is sell
            elif signal == "sell":
                # Get the number of shares to sell from the portfolio
                shares = portfolio[asset]
                # If the number of shares is positive
                if shares > 0:
                    # Update the portfolio with the number of shares
                    portfolio[asset] -= shares
                    # Update the cash with the price and the commission fee
                    cash += price * shares * (1 - COMMISSION)
                    # Update the number of trades
                    number_of_trades += 1
                    # Calculate the profit from the sell trade
                    profit = price * shares - data.loc[index, asset + "_open"] * shares
                    # If the profit is positive
                    if profit > 0:
                        # Update the number of wins
                        number_of_wins += 1
            # If the signal is hold
            elif signal == "hold":
                # Do nothing
                pass
            # If the signal is invalid
            else:
                # Raise an error
                raise ValueError(f"Invalid signal: {signal}")
        # Calculate the total value of the portfolio and the cash
        total_value = cash + np.sum(prices * list(portfolio.values()))
        # Update the profit curve with the total value
        profit_curve[index] = total_value
        # Calculate the peak value of the profit curve
        peak_value = profit_curve[:index].max()
        # Calculate the current drawdown
        current_drawdown = (peak_value - total_value) / peak_value
        # If the current drawdown is greater than the previous drawdown
        if current_drawdown > drawdown:
            # Update the drawdown
            drawdown = current_drawdown
    # Calculate the win rate
    win_rate = number_of_wins / number_of_trades
    # Calculate the annualized return
    annualized_return = profit_curve[-1] / INITIAL_CAPITAL ** (252 / len(profit_curve)) - 1
    # Calculate the annualized volatility
    annualized_volatility = profit_curve.pct_change().std() * np.sqrt(252)
    # Calculate the annualized Sharpe ratio
    annualized_sharpe_ratio = annualized_return / annualized_volatility
    # Update the trading performance dictionary with the annualized return, volatility, and Sharpe ratio
    trading_performance["annualized_return"] = annualized_return
    trading_performance["annualized_volatility"] = annualized_volatility
    trading_performance["annualized_sharpe_ratio"] = annualized_sharpe_ratio

    # Compare the trading performance with the benchmarks
    # For each asset in the assets list
    for asset in ASSETS:
        # Get the price of the asset from the data
        price = data[asset + "_close"]
        # Calculate the return of the asset
        return_asset = price.pct_change()
        # Calculate the cumulative return of the asset
        cumulative_return_asset = (1 + return_asset).cumprod()
        # Calculate the annualized return of the asset
        annualized_return_asset = cumulative_return_asset[-1] ** (252 / len(cumulative_return_asset)) - 1
        # Calculate the annualized volatility of the asset
        annualized_volatility_asset = return_asset.std() * np.sqrt(252)
        # Calculate the annualized Sharpe ratio of the asset
        annualized_sharpe_ratio_asset = annualized_return_asset / annualized_volatility_asset
        # Update the benchmarks dictionary with the asset metrics
        benchmarks[asset] = {
            "annualized_return": annualized_return_asset,
            "annualized_volatility": annualized_volatility_asset,
            "annualized_sharpe_ratio": annualized_sharpe_ratio_asset
        }

    # Plot the trading performance metrics using matplotlib
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    # Plot the profit curve on the first subplot
    axes[0, 0].plot(profit_curve, label="Profit curve")
    axes[0, 0].set_title("Profit curve")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Profit")
    axes[0, 0].legend()
    # Plot the cumulative returns of the assets and the portfolio on the second subplot
    axes[0, 1].plot(cumulative_return_asset, label="Asset")
    axes[0, 1].plot(profit_curve / INITIAL_CAPITAL, label="Portfolio")
    axes[0, 1].set_title("Cumulative returns")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Cumulative return")
    axes[0, 1].legend()
    # Plot the annualized returns of the assets and the portfolio on the third subplot
    axes[1, 0].bar(ASSETS + ["Portfolio"], [benchmarks[asset]["annualized_return"] for asset in ASSETS] + [annualized_return])
    axes[1, 0].set_title("Annualized returns")
    axes[1, 0].set_xlabel("Asset")
    axes[1, 0].set_ylabel("Annualized return")
    # Plot the annualized Sharpe ratios of the assets and the portfolio on the fourth subplot
    axes[1, 1].bar(ASSETS + ["Portfolio"], [benchmarks[asset]["annualized_sharpe_ratio"] for asset in ASSETS] + [annualized_sharpe_ratio])
    axes[1, 1].set_title("Annualized Sharpe ratios")
    axes[1, 1].set_xlabel("Asset")
    axes[1, 1].set_ylabel("Annualized Sharpe ratio")
    # Adjust the layout of the subplots
    fig.tight_layout()
    # Save the figure as a PNG file
    fig.savefig("trading_performance.png")

    # Create a summary table that compares the trading performance metrics of the portfolio and the benchmarks using pandas
    # Create a data frame with the trading performance metrics of the portfolio
    trading_performance_df = pd.DataFrame({
        "Portfolio": {
            "Profit": profit,
            "Number of trades": number_of_trades,
            "Win rate": win_rate,
            "Drawdown": drawdown,
            "Annualized return": annualized_return,
            "Annualized volatility": annualized_volatility,
            "Annualized Sharpe ratio": annualized_sharpe_ratio
        }
    })
    # For each asset in the assets list
    for asset in ASSETS:
        # Add a column with the trading performance metrics of the asset from the benchmarks dictionary
        trading_performance_df[asset] = {
            "Profit": np.nan,
            "Number of trades": np.nan,
            "Win rate": np.nan,
            "Drawdown": np.nan,
            "Annualized return": benchmarks[asset]["annualized_return"],
            "Annualized volatility": benchmarks[asset]["annualized_volatility"],
            "Annualized Sharpe ratio": benchmarks[asset]["annualized_sharpe_ratio"]
        }
    # Transpose the data frame
    trading_performance_df = trading_performance_df.T
    # Save the data frame as a CSV file
    trading_performance_df.to_csv("trading_performance.csv")

