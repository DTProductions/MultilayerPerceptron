import numpy as np
import copy
from collections.abc import Callable

def derivative(func: Callable[[float], float], x: float, h=0.0000001) -> float:
    '''
    Calculate derivative using the central difference formula.
    Args:
        func: function to apply the derivative to.
        x: point in which to calculate the derivative.
        h: precision.
    '''
    return ((func(x + h) - func(x - h)) / (2 * h))

def partial_derivative(func: Callable[[list[float]], float], x: list[float], idx: int, h=0.1) -> float:
    '''
    Calculate partial derivative using the central difference formula.
    Args:
        func: function to apply the partial derivative to.
        x: function inputs. Remain unchanged.
        idx: index of the changing value in the partial derivative.
        h: precision.
    '''
    x1 = copy.deepcopy(x)
    x2 = copy.deepcopy(x)

    x1[idx] = x1[idx] + h
    x2[idx] = x2[idx] - h

    return ((func(x1) - func(x2)) / 2 * h)

def loss_partial_derivative(func: Callable[[list[float], list[float]], float], x: list[float], expected: list[float], idx: int, h=0.001) -> float:
    '''Calculate partial derivative of a loss function using the central difference formula.

    Args:
        func: function to apply the partial derivative to.
        x: function inputs (Remains unchanged).
        expected: matching desired outputs (Remains unchanged).
        idx: index of the changing value in the partial derivative.
        h: precision.
    '''
    x1 = copy.deepcopy(x)
    x2 = copy.deepcopy(x)

    x1[idx] = x1[idx] + h
    x2[idx] = x2[idx] - h

    return ((func(x1, expected) - func(x2, expected)) / (2 * h))

def gradient_descent(x: float, derivative: float, learning_rate: float) -> float:
    '''
    Args:
        x: The starting point of descent.
        derivative: Derivative of a function in **x**.
        learning_rate: How far of a movement to take towards the descent direction.
    '''
    return (x - learning_rate * derivative)

def calculate_z(inputs: list[float], weights: list[float], bias: float) -> float:
        '''Calculates the Z function.
        
        The Z function is defined as the weighted sum of the inputs and weights plus the bias,
        used as input for activation functions in neural networks.

        Returns:
            weights[0] * inputs[0] + weights[1] * inputs[1] ... + weights[n] * inputs[n] + bias
        '''
        return (weighted_sum(inputs, weights) + bias)

def weighted_sum(inputs: list[float], weights: list[float]) -> float:
        '''Calculates the weighted sum of the inputs and weights.

        Returns:
            weight[0] * input[0] + weight[1] * input[1] ... + weight[n] * input[n]
        '''
        if len(inputs) != len(weights):
            return None
        
        sum = 0
        for input, weight in zip(inputs, weights):
            sum += input * weight
        
        return sum

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def mse(predictions: list[float], expected: list[float]) -> float:
    return (1 / len(predictions)) * sum([np.square(p - e) for p, e in zip(predictions, expected)])

def linear(x: float) -> float:
     '''Represents a linear function'''
     return x
