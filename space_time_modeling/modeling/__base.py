##########
# Import #
##############################################################################

from datetime import datetime
from typing import Union

import numpy as np
from pandas.core.api import DataFrame as DataFrame
from pandas.core.api import Series as Series
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from ..utilities import read_df
from ..fe.__base import BaseFE

###########
# Classes #
##############################################################################

class BaseModel:
    """Base class for modeling
    """ 
    def __init__(
            self, 
            label_column: str = None, 
            feature_column: list[str] = None,
            result_path: str = None,
            test_size: float = 0.2,
    ) -> None:
        """Initiate BaseModel

        Parameters
        ----------
        df : Union[str, DataFrame]
            data frame can be 2 types,
            `str` as path of data frame
            `DataFrame` as data frame itself
        label_column : str
            String of target column
        feature_column : str
            String of feature column
        test_size : float = 0.2
            Proportion of test
        """
        # Set main attribute
        self.set_label_column(label_column)
        self.set_feature_column(feature_column)
        
        # Set preparing attribute
        self.set_test_size(test_size)
        
        # Path
        now = datetime.now()
        formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
        self.result_path = f"{result_path}_{formatted_datetime}"

    ##########################################################################
    
    def read_df(self, df: Union[str, DataFrame]):
        """set df attribute

        Parameters
        ----------
        df : Union[str, DataFrame]
            data frame can be 2 types,
            `str` as path of data frame
            `DataFrame` as data frame itself

        Raises
        ------
        ValueError
            if type is not string and DataFrame
        """
        # Check type of df 
        if isinstance(df, str):
            df = read_df(file_path = df)
        elif isinstance(df, DataFrame):
            df = df
        else:
            raise ValueError(f"{type(df)} is not supported")
        
        return df
    
    ##########################################################################
    
    @property
    def label_column(self) -> str:
        """ Label column """
        return self.__label_column
    
    ##########################################################################
    
    def set_label_column(self, label_column: str) -> None:
        """Set label_column attribute

        Parameters
        ----------
        label_column : str
            String of target column
        """
        self.__label_column = label_column
    
    ##########################################################################
    
    @property
    def feature_column(self) -> str:
        """ Feature column """
        return self.__feature_column
    
    ##########################################################################
    
    def set_feature_column(self, feature_column: str) -> None:
        """Set feature_column attribute

        Parameters
        ----------
        feature_column : str
            String of feature column
        """
        if self.label_column in feature_column:
            feature_column.remove(self.label_column)
        self.__feature_column = feature_column
        
    ##########################################################################
    # preparing #
    #############
    
    @property
    def test_size(self) -> float:
        """ Size of test set """
        return self.__test_size
    
    ##########################################################################
    
    def set_test_size(self, test_size: float) -> None:
        """Set size of test in ratio

        Parameters
        ----------
        test_size : float
            Size of test
        """
        self.__test_size = test_size
    
    ##########
    # Method #
    ##########################################################################
    # Feature prepare #
    ###################
    
    def prepare(
            self, 
            df: DataFrame,
    ) -> tuple[DataFrame, DataFrame, Series, Series]:
        """Get prepared data for machine learning

        Parameters
        ----------
        df : DataFrame
            FEd data frame

        Returns
        -------
        tuple[DataFrame, DataFrame, Series, Series]
            x_train, x_test, y_train, y_test
        """
        label, feature = self.split_feature_label(df)
        x_train, x_test, y_train, y_test = self.split_test_train(
            label, 
            feature,
        )
        return x_train, x_test, y_train, y_test
    
    ##########################################################################
    
    def split_feature_label(self, df: DataFrame) -> tuple[Series, DataFrame]:
        """Split feature and label from data frame

        Parameters
        ----------
        df : DataFrame
            Target data frame, contains only nesssary column

        Returns
        -------
        tuple[Series, DataFrame]
            tuple[label, feature]
        """
        if self.label_column in self.feature_column:
            self.feature_column.remove(self.label_column)
        
        # Split
        label = df[self.label_column]
        feature = df[self.feature_column]
        
        return label, feature
    
    ##########################################################################
    
    def split_test_train(
            self, 
            label: Series,
            feature: DataFrame,
    ) -> tuple[DataFrame, DataFrame, Series, Series]:
        """Split test and train, followed by the elf.test_size

        Parameters
        ----------
        label : Series
            Series of label
        feature : DataFrame
            Data frame of feature

        Returns
        -------
        tuple[DataFrame, DataFrame, Series, Series]
            x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(
            feature, 
            label, 
            test_size=self.test_size, 
            random_state=42,
        )
        
        return x_train, x_test, y_train, y_test
    
    ##########################################################################

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

