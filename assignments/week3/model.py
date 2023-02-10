import torch
from typing import Callable
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super().__init__() # The super fucnction lets us initialize values from the class we're inheriting from (nn.Module) fom insiede the MLP class.
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hidden_count = hidden_count
        self.activation = activation
        self.initializer = initializer

        # Define the layers of the network.
        self.layers = nn.ModuleList() #like a regular list, but takes modules as elements. This is a list of layers.
        self.layers.append(nn.Linear(input_size, hidden_size)) #append the first layer to the list of layers.
        
        for _ in range(hidden_count - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size)) #append the hidden layers to the list of layers.
        
        self.layers.append(nn.Linear(hidden_size, num_classes)) #append the output layer to the list of layers.

        # Initialize the weights of the network.
        for layer in self.layers:
            initializer(layer.weight) #initialize the weights of each layer.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """

        for layer in self.layers[:-1]:
            x = self.activation()(layer(x))

        return self.layers[-1](x)