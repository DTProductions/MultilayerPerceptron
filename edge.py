from __future__ import annotations
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from neuron import Neuron

class Edge:
    def __init__(self, source: Neuron, target: Neuron):
        self.source = source
        self.target = target
        # The weights should be set such that when performing the activation
        # of the weighted sum of inputs in each layer the gradients don't vanish
        # but still go towards the deactivation of some layers and activation of others
        self.weight = random.uniform(-1, 1)
