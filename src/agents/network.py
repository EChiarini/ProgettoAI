import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Deep Q-Network (DQN) architecture.
    This MLP acts as a function approximator that maps a given state (flattened agent view)
    to the estimated Q-values for each possible action in the action space.
    """
class Network(nn.Module):
    def __init__(self, input_dim, action_size, seed=42):
        """
        Initializes parameters and builds the neural network layers.
        
        Args:
            input_dim (int): Dimension of the flattened observation space (view_size * view_size).
            action_size (int): Dimension of the action space (number of possible discrete actions).
            seed (int): Random seed for reproducibility of weight initialization.
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Hidden Layer 1: Fully connected layer with 256 neurons
        self.fc1 = nn.Linear(input_dim, 256)
        # Hidden Layer 2: Deeper representation layer to capture non-linear spatial relations
        self.fc2 = nn.Linear(256, 256)
        # Output Layer: Maps hidden features to Q-values for each action
        self.fc3 = nn.Linear(256, action_size)


    def forward(self, x):
        """
        Defines the forward pass of the network using ReLU activation functions.
        
        Args:
            x (torch.Tensor): The input state tensor.
        Returns:
            torch.Tensor: Predicted Q-values for each action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



    def get_description(self):
        """
        Utility to describe the network architecture.
        Useful for automated reporting and debugging.
        """
        info = []
        for name, layer in self.named_children():
            if isinstance(layer, nn.Linear):
                info.append(f"Layer ({name}): Input {layer.in_features} -> Neurons {layer.out_features}")
        return "\n".join(info)