# Import the necessary libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
import tensorflow as tf
from tensorflow import keras
import deap
from deap import base, creator, tools, algorithms
import data # The file that handles the data acquisition and preprocessing
import model # The file that defines and trains the neural network model
import algorithm # The file that implements the genetic algorithm
import simulation # The file that simulates and evaluates the trading performance

# Define the constants and variables
ASSETS = ["AAPL", "MSFT", "GOOG", "AMZN", "FB"] # The assets that are traded in the market
START_DATE = "2020-01-01" # The start date of the market data
END_DATE = "2020-12-31" # The end date of the market data
FREQUENCY = "daily" # The frequency of the market data
INDICATORS = ["SMA", "EMA", "RSI", "MACD", "ADX"] # The indicators that are calculated and added to the market data
INPUT_SIZE = len(ASSETS) * (len(INDICATORS) + 2) # The size of the input layer of the neural network model, based on the number of assets, indicators, and price and volume features
HIDDEN_SIZE = 10 # The size of the hidden layer of the neural network model
OUTPUT_SIZE = 3 # The size of the output layer of the neural network model, based on the buy, sell, or hold signals
ACTIVATION = "relu" # The activation function of the neural network model
LEARNING_RATE = 0.01 # The learning rate of the neural network model
LOSS_FUNCTION = "categorical_crossentropy" # The loss function of the neural network model
POPULATION_SIZE = 50 # The size of the population of the genetic algorithm
GENERATION_NUMBER = 100 # The number of generations of the genetic algorithm
MUTATION_RATE = 0.1 # The mutation rate of the genetic algorithm
CROSSOVER_RATE = 0.8 # The crossover rate of the genetic algorithm
SELECTION_METHOD = "tournament" # The selection method of the genetic algorithm
TOURNAMENT_SIZE = 3 # The tournament size of the genetic algorithm, if the selection method is tournament
SIMULATION_ENVIRONMENT = "virtual" # The market environment for the trading simulation, either virtual or real
BENCHMARKS = ["SPY", "random", "human"] # The benchmarks for the trading performance evaluation

# Acquire and preprocess the market data
market_data = data.get_data(ASSETS, START_DATE, END_DATE, FREQUENCY) # Get the market data from the data file
market_data = data.add_indicators(market_data, INDICATORS) # Add the indicators to the market data
market_data = data.clean_data(market_data) # Clean the market data
market_data = data.transform_data(market_data) # Transform the market data
train_data, test_data = data.split_data(market_data) # Split the market data into training and testing sets

# Define and train the neural network model
neural_network = model.create_model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, ACTIVATION, LEARNING_RATE, LOSS_FUNCTION) # Create the neural network model from the model file
model.train_model(neural_network, train_data) # Train the neural network model on the training data
model.evaluate_model(neural_network, test_data) # Evaluate the neural network model on the testing data

# Implement the genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Create the fitness class for the genetic algorithm
creator.create("Individual", list, fitness=creator.FitnessMax) # Create the individual class for the genetic algorithm
toolbox = base.Toolbox() # Create the toolbox for the genetic algorithm
toolbox.register("individual", algorithm.init_individual, creator.Individual, neural_network) # Register the individual initialization function from the algorithm file
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Register the population initialization function
toolbox.register("evaluate", algorithm.evaluate_individual, neural_network, train_data) # Register the individual evaluation function from the algorithm file
toolbox.register("mutate", algorithm.mutate_individual) # Register the individual mutation function from the algorithm file
toolbox.register("mate", algorithm.crossover_individual) # Register the individual crossover function from the algorithm file
if SELECTION_METHOD == "tournament": # If the selection method is tournament
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE) # Register the tournament selection function
elif SELECTION_METHOD == "roulette": # If the selection method is roulette
    toolbox.register("select", tools.selRoulette) # Register the roulette selection function
else: # If the selection method is invalid
    raise ValueError("Invalid selection method") # Raise an error
population = toolbox.population(n=POPULATION_SIZE) # Initialize the population
best_individual = None # Initialize the best individual
best_fitness = 0 # Initialize the best fitness
for generation in range(GENERATION_NUMBER): # For each generation
    offspring = algorithms.varAnd(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE) # Apply the crossover and mutation operators to the population
    fits = toolbox.map(toolbox.evaluate, offspring) # Evaluate the fitness of the offspring
    for fit, ind in zip(fits, offspring): # For each fitness and individual pair
        ind.fitness.values = fit # Assign the fitness value to the individual
    population = toolbox.select(offspring, k=len(population)) # Select the next generation population
    top_individual = tools.selBest(population, k=1)[0] # Select the top individual from the population
    top_fitness = top_individual.fitness.values[0] # Get the fitness value of the top individual
    if top_fitness > best_fitness: # If the top fitness is better than the best fitness
        best_individual = top_individual # Update the best individual
        best_fitness = top_fitness # Update the best fitness
    print(f"Generation {generation}: Best fitness = {best_fitness}") # Print the generation and the best fitness

# Simulate and evaluate the trading performance
best_model = algorithm.individual_to_model(best_individual, neural_network) # Convert the best individual to a neural network model using the function from the algorithm file
trading_signal = model.predict_model(best_model, test_data) # Predict the trading signal using the best model on the testing data
trading_performance = simulation.simulate_trading(test_data, trading_signal, SIMULATION_ENVIRONMENT) # Simulate the trading performance using the function from the simulation file
simulation.evaluate_trading(trading_performance, BENCHMARKS) # Evaluate the trading performance using the function from the simulation file

# Display the results and outputs
print(f"The best neural network model has the following parameters:") # Print the best neural network model parameters
print(f"Number of neurons in the hidden layer: {best_individual[0]}") # Print the number of neurons in the hidden layer
print(f"Activation function: {best_individual[1]}") # Print the activation function
print(f"Learning rate: {best_individual[2]}") # Print the learning rate
print(f"Loss function: {best_individual[3]}") # Print the loss function
print(f"The best neural network model has the following trading performance metrics:") # Print the best neural network model trading performance metrics
print(f"Total profit: {trading_performance['total_profit']}") # Print the total profit
print(f"Number of trades: {trading_performance['number_of_trades']}") # Print the number of trades
print(f"Win rate: {trading_performance['win_rate']}") # Print the win rate
print(f"Drawdown: {trading_performance['drawdown']}") # Print the drawdown

# Plot the trading performance of the best neural network model and the benchmarks
plt.figure(figsize=(10, 10)) # Create a figure with a specified size
plt.subplot(2, 2, 1) # Create a subplot in the first position
plt.plot(trading_performance['profit_curve'], label="Best model") # Plot the profit curve of the best model
for benchmark in BENCHMARKS: # For each benchmark
    plt.plot(trading_performance[benchmark]['profit_curve'], label=benchmark) # Plot the profit curve of the benchmark
plt.title("Profit curve") # Set the title of the subplot
plt.xlabel("Time") # Set the x-axis label of the subplot
plt.ylabel("Profit") # Set the y-axis label of the subplot
plt.legend() # Show the legend of the subplot
plt.subplot(2, 2, 2) # Create a subplot in the second position
plt.bar(["Best model"] + BENCHMARKS, [trading_performance['number_of_trades']] + [trading_performance[benchmark]['number_of_trades'] for benchmark in BENCHMARKS]) # Plot the number of trades of the best model and the benchmarks
plt.title("Number of trades") # Set the title of the subplot
plt.xlabel("Strategy") # Set the x-axis label of the subplot
plt.ylabel("Number of trades") # Set the y-axis label of the subplot
plt.subplot(2, 2, 3) # Create a subplot in the third position
plt.bar(["Best model"] + BENCHMARKS, [trading_performance['win_rate']] + [trading_performance[benchmark]['win_rate'] for benchmark in BENCHMARKS]) # Plot the win rate of the best model and the benchmarks
plt.title("Win rate") # Set the title of the subplot
plt.xlabel("Strategy") # Set the x-axis label of the subplot
plt.ylabel("Win rate") # Set the y-axis label of the subplot
plt.subplot(2, 2, 4) # Create a subplot in the fourth position
plt.bar(["Best model"] + BENCHMARKS, [trading_performance['drawdown']] + [trading_performance[benchmark]['drawdown'] for benchmark in BENCHMARKS]) # Plot the drawdown of the best model and the benchmarks
plt.title("Drawdown") # Set the title of the subplot
plt.xlabel("Strategy") # Set the x-axis label of the subplot
plt.ylabel("Drawdown") # Set the y-axis label of the subplot
plt.tight_layout() # Adjust the layout of the figure
plt.show() # Show the figure

# Create a summary table that compares the trading performance metrics of the best neural network model and the benchmarks
summary_table = pd.DataFrame(columns=["Strategy", "Total profit", "Number of trades", "Win rate", "Drawdown"]) # Create an empty data frame with the specified columns
summary_table = summary_table.append({"Strategy": "Best model", "Total profit": trading_performance['total_profit'], "Number of trades": trading_performance['number_of_trades'], "Win rate": trading_performance['win_rate'], "Drawdown": trading_performance['drawdown']}, ignore_index=True) # Append the trading performance metrics of the best model to the data frame
for benchmark in BENCHMARKS: # For each benchmark
    summary_table = summary_table.append({"Strategy": benchmark, "Total profit": trading_performance[benchmark]['total_profit'], "Number of trades": trading_performance[benchmark]['number_of_trades'], "Win rate": trading_performance[benchmark]['win_rate'], "Drawdown": trading_performance[benchmark]['drawdown']}, ignore_index=True) # Append the trading performance metrics of the benchmark to the data frame
print(summary_table) # Print the data frame
