from __future__ import annotations
from typing import TYPE_CHECKING
from math_helper import calculate_z, derivative

if TYPE_CHECKING:
    from edge import Edge
    from collections.abc import Callable

class Neuron:
    def __init__(self, activation_func: Callable[[float], float]):
        self.bias = 0
        self.incoming_edges = []
        self.output = None
        self.activation_func = activation_func

    def add_edge(self, edge: Edge) -> None:
        if edge.target is self:
            self.incoming_edges.append(edge)

    def activate(self) -> None:
        '''Activate this neuron with the defined activation function.

        Updates the neuron's output.
        '''
        self.output = self.activation_func(self.z())

    def z(self) -> float:
        '''Calculate the Z function for this neuron's inputs, weights and bias.
        
        State remains unchanged.
        '''
        inputs = []
        weights = []
        for edge in self.incoming_edges:
            if edge.source.output != None:
                weights.append(edge.weight)
                inputs.append(edge.source.output)
            else:
                print("Previous layer's neurons not activated!")
        
        return calculate_z(inputs, weights, self.bias)

    def _backpropagate_neuron(self, loss_derivative: float,
                            gradients: dict[Neuron | Edge, float],
                            propagated: dict[Neuron, float]) -> None:
        '''Calculate the gradients and propagations of a single neuron given the
        derivative of a neuron's output in relation the loss function.

        Results are stored in the **gradients** and **propagated** dicts.
        '''
        activation_derivative = derivative(self.activation_func, self.z())
        # Z'(bias) is 1, hence why no other derivative needs to
        # be calculated in the chain rule.
        bias_derivative = loss_derivative * activation_derivative

        gradients[self] = bias_derivative

        for edge in self.incoming_edges:
            # Z'(w) = a(L-1), where a(L-1) is the previous layer neuron's output associated
            # with this specific weight.
            # This is an application of the power rule, given that a(L-1) is treated as
            # a constant in this partial derivative.
            weight_derivative = loss_derivative * activation_derivative * edge.source.output
            gradients[edge] = weight_derivative

            # Z'(a(L-1)) = w, because a(L-1) is the output matching this weight.
            edge_source_activation_derivative = loss_derivative * activation_derivative * edge.weight

            # Used to calculate the total derivative
            if edge.source in propagated:
                propagated[edge.source] += edge_source_activation_derivative
            else:
                propagated[edge.source] = edge_source_activation_derivative
