# Import the necessary libraries and modules
import numpy as np
import pandas as pd
import deap
from deap import base, creator, tools, algorithms
import model # The file that defines and trains the neural network model

# Define the constants and variables
NEURONS = [5, 10, 15, 20, 25] # The possible values for the number of neurons in the hidden layer
ACTIVATIONS = ["relu", "sigmoid", "tanh", "linear"] # The possible values for the activation function
LEARNING_RATES = [0.01, 0.001, 0.0001] # The possible values for the learning rate
LOSS_FUNCTIONS = ["categorical_crossentropy", "mean_squared_error", "mean_absolute_error"] # The possible values for the loss function

# Define the functions for implementing the genetic algorithm
def init_individual(icls, model):
    """
    Initialize an individual for the genetic algorithm.
    An individual is a list of four parameters: the number of neurons, the activation function, the learning rate, and the loss function.
    The parameters are randomly selected from the possible values.
    The individual is returned as a list object.
    """
    # Initialize an empty list for the individual
    individual = []
    # Randomly select the number of neurons from the possible values
    neurons = np.random.choice(NEURONS)
    # Append the number of neurons to the individual
    individual.append(neurons)
    # Randomly select the activation function from the possible values
    activation = np.random.choice(ACTIVATIONS)
    # Append the activation function to the individual
    individual.append(activation)
    # Randomly select the learning rate from the possible values
    learning_rate = np.random.choice(LEARNING_RATES)
    # Append the learning rate to the individual
    individual.append(learning_rate)
    # Randomly select the loss function from the possible values
    loss_function = np.random.choice(LOSS_FUNCTIONS)
    # Append the loss function to the individual
    individual.append(loss_function)
    # Convert the individual to the specified class
    individual = icls(individual)
    # Return the individual
    return individual

def evaluate_individual(individual, model, train_data):
    """
    Evaluate the fitness of an individual for the genetic algorithm.
    The fitness is the accuracy of the neural network model with the individual parameters on the training data.
    The individual is converted to a neural network model using the function from the model file.
    The neural network model is trained and evaluated on the training data using the function from the model file.
    The fitness is returned as a tuple object.
    """
    # Convert the individual to a neural network model using the function from the model file
    neural_network = model.individual_to_model(individual, model)
    # Train and evaluate the neural network model on the training data using the function from the model file
    neural_network = model.train_model(neural_network, train_data)
    accuracy = model.evaluate_model(neural_network, train_data)
    # Return the fitness as a tuple
    return (accuracy,)

def mutate_individual(individual):
    """
    Mutate an individual for the genetic algorithm.
    The mutation is a random change of one of the individual parameters to a different value from the possible values.
    The individual is modified in place and returned as a list object.
    """
    # Randomly select the index of the parameter to mutate
    index = np.random.randint(0, len(individual))
    # If the index is 0, mutate the number of neurons
    if index == 0:
        # Randomly select a different value for the number of neurons from the possible values
        neurons = np.random.choice([n for n in NEURONS if n != individual[index]])
        # Assign the new value to the individual
        individual[index] = neurons
    # If the index is 1, mutate the activation function
    elif index == 1:
        # Randomly select a different value for the activation function from the possible values
        activation = np.random.choice([a for a in ACTIVATIONS if a != individual[index]])
        # Assign the new value to the individual
        individual[index] = activation
    # If the index is 2, mutate the learning rate
    elif index == 2:
        # Randomly select a different value for the learning rate from the possible values
        learning_rate = np.random.choice([l for l in LEARNING_RATES if l != individual[index]])
        # Assign the new value to the individual
        individual[index] = learning_rate
    # If the index is 3, mutate the loss function
    elif index == 3:
        # Randomly select a different value for the loss function from the possible values
        loss_function = np.random.choice([l for l in LOSS_FUNCTIONS if l != individual[index]])
        # Assign the new value to the individual
        individual[index] = loss_function
    # Return the mutated individual
    return individual

def crossover_individual(individual1, individual2):
    """
    Crossover two individuals for the genetic algorithm.
    The crossover is a random exchange of one or more of the individual parameters between the two individuals.
    The individuals are modified in place and returned as list objects.
    """
    # Randomly select the number of points for the crossover
    points = np.random.randint(1, len(individual1))
    # Randomly select the indices of the parameters to crossover
    indices = np.random.choice(range(len(individual1)), size=points, replace=False)
    # For each index in the indices list
    for index in indices:
        # Swap the values of the parameters between the two individuals
        individual1[index], individual2[index] = individual2[index], individual1[index]
    # Return the crossed individuals
    return individual1, individual2

def individual_to_model(individual, model):
    """
    Convert an individual to a neural network model.
    The individual is a list of four parameters: the number of neurons, the activation function, the learning rate, and the loss function.
    The neural network model is created and compiled using the function from the model file.
    The neural network model is returned as a keras model object.
    """
    # Get the individual parameters
    neurons = individual[0]
    activation = individual[1]
    learning_rate = individual[2]
    loss_function = individual[3]
    # Create and compile the neural network model using the function from the model file
    neural_network = model.create_model(INPUT_SIZE, neurons, OUTPUT_SIZE, activation, learning_rate, loss_function)
    # Return the neural network model
    return neural_network
