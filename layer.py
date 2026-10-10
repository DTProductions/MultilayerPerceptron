from __future__ import annotations
from typing import TYPE_CHECKING
from math_helper import loss_partial_derivative

if TYPE_CHECKING:
    from neuron import Neuron
    from edge import Edge
    from collections.abc import Callable

class TrainableLayer():
    '''Represents a layer that can have its weights and biases updated,
    i.e hidden and output layers.
    '''

    def __init__(self, neurons: list[Neuron], learning_rate: float):
        self.neurons = neurons
        self.learning_rate = learning_rate

    def __iter__(self):
        return iter(self.neurons)
    
    def activate(self) -> None:
        '''Activates all neurons in this layer
        (requires neurons in previous layers to be activated).
        '''
        for neuron in self.neurons:
            neuron.activate()
    
    def get_outputs(self) -> list[float]:
        '''Returns the calculated outputs of this layer
        (Values are only meaningful if layer has already
        been activated).
        '''
        return [neuron.output for neuron in self.neurons]
    
class HiddenLayer(TrainableLayer):
    def backpropagate(self, previous_gradients: dict[Neuron | Edge, float],
                      previous_propagated: dict[Neuron, float]) -> tuple[dict[Edge | Neuron, float], dict[Neuron, float]]:
        gradients = previous_gradients
        propagated = {}

        for neuron in self.neurons:
            # Derivative of this neuron's output to the loss function.
            loss_derivative = previous_propagated[neuron]
            neuron._backpropagate_neuron(loss_derivative, gradients, propagated)
        return (gradients, propagated)
        
class InputLayer():
    '''Holds the input neurons.'''
    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    def __iter__(self):
        return iter(self.neurons)
    
    def load_inputs(self, inputs: list[float]) -> None:
        '''Loads an array of inputs into the input layer's neurons.'''
        if len(inputs) != len(self.neurons):
            print("Number of inputs does not match number of neurons")
            return
        for neuron, input in zip(self.neurons, inputs):
            neuron.output = input

class OutputLayer(TrainableLayer):
    def __init__(self, neurons: list[Neuron],
                 learning_rate: float,
                 loss_func: Callable[[list[float], list[float]], float]):
        super().__init__(neurons, learning_rate)
        self.loss_func = loss_func

    def output_backpropagate(self, expected: list[float]) -> tuple[dict[Edge | Neuron, float], dict[Neuron, float]]:
        '''Calculates the first backpropagated values based on this layer's loss function.

        Returns:
            tuple[dict[object, float], dict[object, float]]:
                A tuple of gradients and propagations, in the ordering **(gradients, propagations)**,
                where **gradients** is a dict mapping edges (for weights) and neurons (for biases)
                to their calculated gradient, while **propagations** maps neurons in the
                previous layer to the derivative of their output in relation to the loss function.
        '''
        gradients = {}
        propagated = {}
        # i represents ith neuron in the same ordering as get_outputs
        for i, neuron in enumerate(self.neurons):
            loss_derivative = loss_partial_derivative(self.loss_func, self.get_outputs(), expected, idx=i)
            neuron._backpropagate_neuron(loss_derivative, gradients, propagated)

        return (gradients, propagated)