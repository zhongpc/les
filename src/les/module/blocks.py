from typing import Callable, Union, Optional, Sequence, List

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ["build_mlp", "Dense"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable

def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[List[int]] = None,
    n_layers: int = 2,
    activation = F.silu,
    bias: bool = True,
):
    """
    Build multiple layer fully connected perceptron neural network.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        n_neurons = [0] * (n_layers + 1)
        n_neurons[0] = int(n_in)
        current = int(n_in)
        for i in range(1, n_layers):
            current = max(int(n_out), current // 2)
            n_neurons[i] = current
        n_neurons[n_layers] = int(n_out)
    else:
        # get list of number of nodes hidden layers
        if isinstance(n_hidden, int):
            n_neurons = [int(n_in)] + [int(n_hidden)] * (n_layers - 1) + [int(n_out)]
        else:
            n_neurons = [int(n_in)] + [int(h) for h in n_hidden] + [int(n_out)]

    # assign a Dense layer (with activation function) to each hidden layer
    layers = [
        Dense(n_neurons[i], n_neurons[i + 1], activation=activation, bias=bias)
        for i in range(n_layers - 1)
    ]

    # assign a Dense layer (without activation function) to the output layer
    layers.append(
        Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=bias)
    )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net

class Dense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = nn.Identity(),
    ):
        """
        Fully connected linear layer with an optional activation function and batch normalization.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If False, the layer will not have a bias term.
            activation (Callable or nn.Module): Activation function. Defaults to Identity.
        """
        super().__init__()
        # Dense layer
        self.linear = nn.Linear(in_features, out_features, bias)

        # Activation function
        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def forward(self, input: torch.Tensor):
        y = self.linear(input)
        y = self.activation(y)
        return y

