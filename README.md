# MLGP-Trader

MLGP-Trader is a project that uses machine learning and genetic programming to generate trading signals for a portfolio of assets, and simulates and evaluates the trading performance in a virtual or real market environment.

## Objectives

The main objectives of this project are:

- To use machine learning techniques, such as neural networks, to learn the patterns and features of the market data, and to predict the future prices and movements of the assets.
- To use genetic programming techniques, such as symbolic regression, to evolve and optimize the trading rules and strategies based on the predictions and the fitness criteria.
- To generate trading signals, such as buy, sell, or hold, for each asset and time, based on the trading rules and strategies.
- To simulate and evaluate the trading performance using the market data and the trading signals in a virtual or real market environment, and to calculate and record the trading metrics, such as the profit, the number of trades, the win rate, the drawdown, and the benchmarks.

## Methods

The main methods of this project are:

- Data manipulation and visualization: The project uses various libraries and modules, such as numpy, pandas, and matplotlib, to manipulate and visualize the market data, which includes the prices and volumes of the assets.
- Machine learning: The project uses various libraries and modules, such as sklearn, tensorflow, and keras, to implement and train the neural network models, which are used to predict the future prices and movements of the assets.
- Genetic programming: The project uses various libraries and modules, such as deap, to implement and run the genetic programming algorithm, which is used to evolve and optimize the trading rules and strategies based on the predictions and the fitness criteria.
- Trading simulation and evaluation: The project uses various libraries and modules, such as requests and json, to simulate and evaluate the trading performance using the market data and the trading signals in a virtual or real market environment, and to calculate and record the trading metrics, such as the profit, the number of trades, the win rate, the drawdown, and the benchmarks.

## Requirements

The project requires the following libraries and modules to be installed:

- numpy
- pandas
- matplotlib
- sklearn
- tensorflow
- keras
- deap
- requests
- json

The project also requires the following constants and variables to be defined:

- ASSETS: A list of the symbols of the assets to be traded, such as ["AAPL", "MSFT", "GOOG", "AMZN", "FB"].
- REAL_MARKET_API_KEY: A string of the API key for accessing the real market APIs, such as "YOUR_API_KEY".
- REAL_MARKET_API_URL: A string of the API URL for accessing the real market APIs, such as "https://www.alphavantage.co/query".
- INITIAL_CAPITAL: A scalar of the initial capital for the trading simulation, such as 10000.
- COMMISSION: A scalar of the commission fee for each trade, such as 0.01.

## Usage

The project consists of five files, which are:

- main.py: The file that runs the whole project, and calls the functions from the other files.
- data.py: The file that contains the functions for manipulating and visualizing the market data, such as getting the data, cleaning the data, and plotting the data.
- model.py: The file that contains the functions for implementing and training the neural network models, such as creating the model, compiling the model, fitting the model, and evaluating the model.
- algorithm.py: The file that contains the functions for implementing and running the genetic programming algorithm, such as defining the primitives, creating the population, evaluating the individuals, selecting the parents, mating the parents, mutating the offspring, and selecting the survivors.
- simulation.py: The file that contains the functions for simulating and evaluating the trading performance, such as simulating the trading, calculating the metrics, comparing the benchmarks, and plotting the results.

To run the project, simply execute the main.py file, and follow the instructions on the console. The project will output the trading signals, the trading performance metrics, and the trading performance plots for the portfolio and the assets.

## Results

The project will output the following results:

- The trading signals, which are the buy, sell, or hold signals for each asset and time, based on the trading rules and strategies.
- The trading performance metrics, which are the profit, the number of trades, the win rate, the drawdown, the annualized return, the annualized volatility, and the annualized Sharpe ratio for the portfolio and the assets, and the comparison with the benchmarks.
- The trading performance plots, which are the profit curve, the cumulative returns, the annualized returns, and the annualized Sharpe ratios for the portfolio and the assets.

The project will also save the following files:

- trading_performance.png: A PNG file that contains the trading performance plots for the portfolio and the assets.
- trading_performance.csv: A CSV file that contains the trading performance metrics for the portfolio and the assets.

The results of the project may vary depending on the market data, the neural network models, the genetic programming algorithm, and the trading simulation and evaluation. The results are not guaranteed to be accurate, reliable, or profitable. The project is for educational and research purposes only, and is not intended to provide any financial advice or recommendations. The project is not responsible for any losses or damages that may arise from the use of the project. The user should exercise caution and discretion when using the project, and should do their own due diligence and research before making any trading decisions. The user should also abide by the terms and conditions of the real market APIs, and should not use the project for any illegal or unethical purposes. The user should acknowledge and accept the risks and limitations of the project, and should use the project at their own risk and responsibility. 😊
