##########
# Import #
##############################################################################

from typing import Union

import numpy as np
from pandas.core.api import DataFrame as DataFrame
from pandas.core.api import Series as Series
import torch
import torch.nn as nn

##############################################################################

class BaseDeep(nn.Module):
    def tensorize_data(self, x: Union[np.ndarray, list]) -> torch.tensor:
        """Tensorize data

        Parameters
        ----------
        x : Union[np.ndarray, list]
            Feature

        Returns
        -------
        torch.tensor
            Tensor data type
        """
        # Check a convert x to the tensor
        if isinstance(x, np.ndarray):
            x = self.tensorize_array(x)
        elif isinstance(x, list):
            x = self.tensorize_list(x)
        return x
    
    ########
    # Util #
    ##########################################################################
    
    @staticmethod
    def tensorize_array(array: np.array) -> torch.tensor:
        """Create tensor object

        Parameters
        ----------
        array : np.array
            Input

        Returns
        -------
        torch.tensor
            Tensored data
        """
        return torch.tensor(array)
    
    ##########################################################################
    
    @staticmethod
    def tensorize_list(list_object: list) -> torch.tensor:
        """Create tensor object

        Parameters
        ----------
        array : list
            Input

        Returns
        -------
        torch.tensor
            Tensored data
        """
        return torch.tensor(list_object)
    
    ##########################################################################
    
    @staticmethod
    def detensor(tensor_object: torch.tensor) -> list:
        """Convert tensor to list 

        Parameters
        ----------
        tensor_object : torch.tensor
            Tensor object

        Returns
        -------
        list
            List of object extracted from tensor
        """
        out_numpy = tensor_object.detach().numpy()
        return out_numpy.tolist()
    
    ##########################################################################
    
##############################################################################

###########
# Wrapper #
##############################################################################

class BaseWrapper(nn.Module):
    def __init__(self, feature: list[str] = None):
        super(BaseWrapper, self).__init__()
        
        if feature:
            self.set_feature(feature)
    
    ##############
    # Properties #
    ##########################################################################
    # Feature #
    ###########
    
    def set_feature(self, feature: list[str]) -> None:
        self.__feature = feature
    
    ##########################################################################
    
    @property
    def feature(self) -> list[str]:
        return self.__feature
    
    #############
    # Utilities #
    ##########################################################################
    
    @staticmethod
    def detensor(tensor_object: torch.tensor) -> list:
        """Convert tensor to list 

        Parameters
        ----------
        tensor_object : torch.tensor
            Tensor object

        Returns
        -------
        list
            List of object extracted from tensor
        """
        out_numpy = tensor_object.detach().numpy()
        return out_numpy.tolist()
    
    ##########################################################################
    
##############################################################################
