from __future__ import annotations
from edge import Edge
from layer import InputLayer, HiddenLayer, OutputLayer
from neuron import Neuron
from itertools import pairwise
from typing import TYPE_CHECKING
import random
from math_helper import gradient_descent
import pickle

if TYPE_CHECKING:
    from collections.abc import Callable

class NeuralNetwork:
    def __init__(self, layers_setup: list[int],
                 hidden_activation: Callable[[float], float],
                 output_activation: Callable[[float], float],
                 loss_func: Callable[[list[float], list[float]], float],
                 learning_rate: float):
        '''
        Args:
            layers_setup: Number of neurons in each layer,
            from left to right, including input and output layers.

            hidden_activation: Activation function used when iterating over hidden layers.
            output_activation: Activation function used in the output layer.
            loss_func: Desired loss function to optimize against.
        '''

        if len(layers_setup) < 3:
            print("Invalid layer setup, aborting graph generation")
            return
        
        # Generate input layer's neurons
        input_neurons = []
        for _ in range(layers_setup[0]):
            input_neurons.append(Neuron(None))
        
        # Generate hidden layers' neurons
        hidden_layers = []
        for neuron_count in layers_setup[1:-1]:
            neurons = []
            for _ in range(neuron_count):
                neurons.append(Neuron(hidden_activation))
            hidden_layers.append(neurons)

        # Generate output neurons
        output_neurons = []
        for _ in range(layers_setup[-1]):
            output_neuron = Neuron(output_activation)
            output_neurons.append(output_neuron)

        self._create_edges(input_neurons, hidden_layers, output_neurons)

        self.input_layer = InputLayer(input_neurons)
        self.hidden_layers = [HiddenLayer(x, learning_rate) for x in hidden_layers]
        self.output_layer = OutputLayer(output_neurons, learning_rate, loss_func)
        self.loss_func = loss_func
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def _create_edges(self, input_neurons: list[Neuron],
                     hidden_layers: list[list[Neuron]],
                     output_neurons: list[Neuron]):
        # Create edges from input layer to
        # second layer (first hidden layer)
        for input_neuron in input_neurons:
            for neuron in hidden_layers[0]:
                edge = Edge(input_neuron, neuron)
                neuron.add_edge(edge)

        # Create edges connecting all hidden layers
        for previous, current in pairwise(hidden_layers):
            for current_neuron in current:
                for previous_neuron in previous:
                  edge = Edge(previous_neuron, current_neuron)
                  current_neuron.add_edge(edge)
        
        # Create edges connecting output neurons to the
        # last hidden layer
        for output_neuron in output_neurons:
            for neuron in hidden_layers[-1]:
                edge = Edge(neuron, output_neuron)
                output_neuron.add_edge(edge)

    def stochastic_train(self, x_train: list[list[float]], y_train: list[list[float]], iterations: int):
        '''Train this neural network utilizing stochastic gradient descent.

        Args:
            x_train: training features.
            y_train: matching predictions.
            iterations: number of iterations over dataset.
        '''
        for _ in range(iterations):
            samples = list(zip(x_train, y_train))
            random.shuffle(samples)

            for x, y in samples:
                self.predict(x)
                self.backpropagate(y)

    def evaluate(self, x_test, y_test):
            samples = list(zip(x_test, y_test))

            total_error = 0
            for x, y in samples:
                predictions = self.predict(x)
                predictions = [p * 10 for p in predictions]
                y = [yi * 10 for yi in y]
                error = self.calculate_error(predictions, y)
                total_error += error

            return ((total_error / len(samples)))

    def save(self, path: str):
        '''Saves the current neural network to a pickle file'''
        with open(path, "wb") as model_pickle:
            pickle.dump(self, model_pickle)

    @staticmethod
    def load(path: str) -> NeuralNetwork:
        with open(path, "rb") as model_pickle:
            model = pickle.load(model_pickle)
        return model
    
    def predict(self, inputs: list[float]) -> list[float]:
        '''Activate all layers and return generated outputs.'''
        self.input_layer.load_inputs(inputs)
        for layer in self.hidden_layers:
            layer.activate()
        self.output_layer.activate()
        return self.output_layer.get_outputs()
    
    def calculate_error(self, predictions, expected):
        '''Calculates the error of the outputs.'''
        return self.loss_func(predictions, expected)
    
    def backpropagate(self, expected: list[float]):
        '''Update weights and biases to minimize error.

        Requires that all layers have been previously activated.
        '''
        gradients, current_backpropagation = self.output_layer.output_backpropagate(expected)
        for hidden_layer in reversed(self.hidden_layers):
            gradients, current_backpropagation = hidden_layer.backpropagate(gradients, current_backpropagation)
        
        # applying gradients to hidden layers
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                neuron.bias = gradient_descent(neuron.bias, gradients[neuron], hidden_layer.learning_rate)
                for edge in neuron.incoming_edges:
                    edge.weight = gradient_descent(edge.weight, gradients[edge], hidden_layer.learning_rate)

        # applying gradients to output layer
        for neuron in self.output_layer:
            neuron.bias = gradient_descent(neuron.bias, gradients[neuron], self.output_layer.learning_rate)
            for edge in neuron.incoming_edges:
                edge.weight = gradient_descent(edge.weight, gradients[edge], self.output_layer.learning_rate)

    def draw(self):
        print("Input layer")
        for neuron in self.input_layer:
            print(f"[{neuron.output}] ", end="")

        print("")
        print("")

        for id, layer in enumerate(self.hidden_layers):
            print(f"Hidden layer {id}")
            for neuron in layer:
                print(f"[{neuron.output}] ", end="")
            print("")
            print("")

        print("Output layer")
        for neuron in self.output_layer:
            print(f"[{neuron.output}] ", end="")
        print("")