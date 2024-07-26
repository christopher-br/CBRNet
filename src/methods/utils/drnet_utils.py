# LOAD MODULES
# Standard library
...

# Third party
import torch
import torch.nn as nn

class DRNetHeadLayer(nn.Module):
    """
    DRNet head layer.
    
    This class represents a head layer of a DRNet. In inherits from PyTorch's nn.Module class.
    It takes as input the size of the input layer, the size of the output layer, an indicator
    if it is the last layer, and an activation function. It returns a tensor.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        last_layer: bool = False,
        activation: nn.Module = nn.ELU(),
    ):
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified input size, output size, 
        whether it is the last layer, and the activation function.

        Parameters:
            input_size (int): The size of the input to the layer.
            output_size (int): The size of the output from the layer.
            last_layer (bool, optional): Whether this is the last layer in the network. Defaults to False.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ELU().
        """
        super(DRNetHeadLayer, self).__init__()
        self.last_layer = last_layer
        self.layer1 = nn.Linear(input_size + 1, output_size)
        self.layer2 = activation

    def forward(self, x):
        """
        Performs a forward pass through the layer using the input tensor `x`.

        This method takes as input a tensor `x` and performs a forward pass through the layer. 
        The method returns a tensor representing the output of the layer.
        The method splits the dose from x, and concatenates it to the output of the layer,
        to strengthen the effect of the dose variable in the network.

        Parameters:
            x (torch.Tensor): The input tensor. A tensor where each row is an observation and 
                            each column is a feature.

        Returns:
            torch.Tensor: The output of the layer. A tensor where each row is an observation and 
                        each column is a feature.
        """
        # Take d from x and save separately
        d = x[:, -1].reshape(-1, 1).detach()
        # Feed x through network
        x = self.layer1(x)
        # Activate
        x = self.layer2(x)
        # If not the last layer, concatenate t to x
        if self.last_layer == False:
            x = torch.cat((x, d), dim=1)

        return x