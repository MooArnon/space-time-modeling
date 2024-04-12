from typing import Union

import torch
import torch.nn as nn

from .__base import BaseDeep, BaseWrapper

##########
# Models #
##############################################################################
# Deep Neuron Network #
#######################

class DNN(BaseDeep):
    def __init__(
            self,
            input_dim: int, 
            hidden_layers: list[int], 
            output_dim: int,
            dropout: float = 0.0,
    ):
        super(DNN, self).__init__()
        
        # Initialize parameters
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Define the layers
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.ReLU())
        if dropout > 0:
            self.hidden_layers.append(nn.Dropout(dropout))
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_layers[i], hidden_layers[i+1])
            )
            self.hidden_layers.append(nn.ReLU())
            if dropout > 0:
                self.hidden_layers.append(nn.Dropout(dropout))
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(
            self, 
            x: Union[
                list[float], 
                torch.Tensor,
            ],  
    ) -> torch.Tensor:
        """Forward layer

        Parameters
        ----------
        x : Union[torch.Tensor]
            x as a feature data

        Returns
        -------
        torch.Tensor
            The label as a tensor
        """
        x = self.tensorize_data(x)
        
        # Forward pass through layers
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)
        out = self.sigmoid(out)
        
        return out
    ##########################################################################   

##############################################################################
# Long Short-Term Memory #
##########################

class LSTM(BaseDeep):
    def __init__(
            self,
            input_dim: int, 
            hidden_layers: list[int], 
            output_dim: int,
            dropout: float = 0.0
    ):
        super(LSTM, self).__init__()
        
        self.num_layers = len(hidden_layers)
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Define the LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(input_dim, hidden_layers[0]))
        for i in range(1, self.num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    hidden_layers[i-1], 
                    hidden_layers[i], 
                )
            )
            
        self.dropout = nn.Dropout(p=dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        
        # Activation function
        self.sigmoid = nn.Sigmoid()

    ##########################################################################
    
    def forward(self, x: Union[list[float], torch.Tensor]) -> torch.Tensor:
        # Check and convert x to tensor
        x = self.tensorize_data(x)
        
        # Forward pass through LSTM layers
        # Add batch dimension
        x = x.unsqueeze(0)  
        
        # Feed data to each layer
        for layer in self.lstm_layers:
            x, _ = layer(x)
            x = self.dropout(x)
        
        # Only take the output from the final LSTM layer
        out = x.squeeze(0)  # Remove batch dimension
        out = self.output_layer(out)
        out = self.sigmoid(out)
        
        return out

    ##########################################################################

############
# Wrappers #
##############################################################################
# Deep Neuron Network #
#######################

class DNNWrapper(BaseWrapper):
    name = "dnn_wrapper"
    def __init__(self, model: nn.Module=None, feature: list[str] = None):
        super(DNNWrapper, self).__init__(feature)
        if model:
            self.set_model(model)
    
    ##############
    # Properties #
    ##########################################################################
    # Model #
    #########
    
    def set_model(self, **kwargs) -> None:
        self.__model = DNN(**kwargs)
    
    ##########################################################################
    
    @property
    def model(self) -> nn.Module:
        return self.__model

    ###########
    # Methods #
    ##########################################################################
    
    def forward(self, x, detensor: bool = False):
        if detensor:
            return self.detensor(self.model(x))
        else:
            return self.model(x)
    
    ##########################################################################
    
##############################################################################
# Long Short-Term Memory #
##########################

class LSTMWrapper(BaseWrapper):
    name = "lstm_wrapper"
    def __init__(self, model: nn.Module=None, feature: list[str] = None):
        super(LSTMWrapper, self).__init__(feature)
        if model:
            self.set_model(model)

    ##############
    # Properties #
    ##########################################################################
    # Model #
    #########
    
    def set_model(self, **kwargs) -> None:
        self.__model = LSTM(**kwargs)
    
    ##########################################################################
    
    @property
    def model(self) -> nn.Module:
        return self.__model

    ###########
    # Methods #
    ##########################################################################
    
    def forward(self, x, detensor: bool = False):
        if detensor:
            return self.detensor(self.model(x))
        else:
            return self.model(x)
    
    ##########################################################################

##############################################################################