##########
# Import #
##############################################################################

from abc import abstractmethod
from typing import Union

import pandas as pd
from pandas.core.api import DataFrame as DataFrame

from space_time_modeling.utilities import read_df

###########
# Classes #
##############################################################################
# Base class #
##############

class BaseEDA:
    """
    The base class stored mutual methods, shared between inherited
    classes.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, str]
        receive either pd.DataFrame or path of file as string.
    """
    
    def __init__(self,  df: Union[pd.DataFrame, str]) -> None:
        """Initiate the base class

        Parameters
        ----------
        df : Union[pd.DataFrame, str]
            receive either pd.DataFrame or path of file as string.
        """
        # Set df
        self.set_df(df)
    
    ##############
    # Properties #
    ##########################################################################
    # DF #
    ######
    
    @property
    def df(self) -> pd.DataFrame:
        return self.__df
    
    ##########################################################################
    
    def set_df(self, df: Union[pd.DataFrame, str]) -> None:
        """Set data-frame that be used in the class

        Parameters
        ----------
        df : Union[pd.DataFrame, str]
            receive either pd.DataFrame or path of file as string.
        
        Raise
        -----
        ValueError
            When df is not both pd.DataFrame and path of string
        """
        # Check if path of df
        if isinstance(df, str):
            self.__df = read_df(df)
        
        # Check if pd.df
        elif isinstance(df, pd.DataFrame):
            self.__df = df
        
        # If not both, raise error
        else:
            raise ValueError(
                """Assign only path of data frame or data frame itself"""
            )
    
    ##########
    # Method #
    ##########################################################################
    
    @abstractmethod
    def get_eda(path: str) -> None:
        """Export the plot

        Parameters
        ----------
        path : str
            Path to store file.
        """
        pass
        
    ##########################################################################
    
##############################################################################
