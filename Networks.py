# RL challenge in the Platform Environment
# Code author: Sergi Andreu

import torch
import torch.nn as nn
  
### Neural Network ###
class TestNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, input_size, output_size):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, 8)
        self.input_layer_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(8, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        # Compute output layer
        out = self.output_layer(l1)
        return out


class BigNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, input_size, output_size):
        super().__init__()

        # Create input layer with ReLU activation
        self.layer1 = nn.Linear(input_size, 32)
        self.layer1_activation = nn.ReLU()

        # Create input layer with ReLU activation
        self.layer2 = nn.Linear(32, 8)
        self.layer2_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(8, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.layer1(x)
        l1 = self.layer1_activation(l1)

        # Compute second layer
        l2 = self.layer2(l1)
        l2 = self.layer2_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)

        return out

class BiggerNetwork(nn.Module):
    """ Create a feedforward neural network 
        with three layers"""
    def __init__(self, input_size, output_size):
        super().__init__()

        # Create input layer with ReLU activation
        self.layer1 = nn.Linear(input_size, 64)
        self.layer1_activation = nn.ReLU()

        self.layer2 = nn.Linear(64, 32)
        self.layer2_activation = nn.ReLU()

        self.layer3 = nn.Linear(32, 16)
        self.layer3_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(16, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.layer1(x)
        l1 = self.layer1_activation(l1)

        # Compute second layer
        l2 = self.layer2(l1)
        l2 = self.layer2_activation(l2)

        l3 = self.layer3(l2)
        l3 = self.layer3_activation(l3)

        # Compute output layer
        out = self.output_layer(l3)

        return out

class DuelingBigNetwork(nn.Module):
    """ Create a feedforward neural network
        with dueling (approximating value and advantage separately)"""
    def __init__(self, input_size, output_size, hidden_size_1=64, hidden_size_2=32):
        super().__init__()

        # Create input layer with ReLU activation
        self.layer1 = nn.Linear(input_size, hidden_size_1)
        self.layer1_activation = nn.ReLU()

        # Create input layer with ReLU activation
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer2_activation = nn.ReLU()

        # Create output layer
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size_2, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, 1)
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_size_2, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size)
        )


    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.layer1(x)
        l1 = self.layer1_activation(l1)

        # Compute second layer
        l2 = self.layer2(l1)
        l2 = self.layer2_activation(l2)

        values = self.value_layer(l2)
        advantage = self.advantage_layer(l2)

        out = values + (advantage - advantage.mean())

        return out

class ConvNetwork(nn.Module):

    """ Create a feedforward convolutional neural network """
    """ NOT WORKING FOR NOW (and not being used)"""
    def __init__(self, input_size, output_size):
        super().__init__()

        # Create input layer with ReLU activation
        self.layer1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2)
        self.layer1_activation = nn.ReLU()

        # Create input layer with ReLU activation
        self.layer2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2)
        self.layer2_activation = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Create output layer
        self.output_layer = nn.Linear(8, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.layer1(x)
        l1 = self.layer1_activation(l1)

        # Compute second layer
        l2 = self.layer2(l1)
        l2 = self.layer2_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)

        return out