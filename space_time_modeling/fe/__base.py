#--------#
# Import #
#----------------------------------------------------------------------------#

from abc import abstractmethod
import pandas
from typing import Union

from space_time_modeling.utilities import read_df

#---------#
# Classes #
#----------------------------------------------------------------------------#

class BaseFE:
    
    def __init__(self, df: Union[str, pandas.DataFrame]) -> None:
        self.set_df(df)
    
    #------------#
    # Properties #
    #------------------------------------------------------------------------#
    
    @property
    def df(self) -> pandas.DataFrame:
        """Data frame attribute"""
        return self.__df
    
    #------------------------------------------------------------------------#
    
    def set_df(self, df: Union[str, pandas.DataFrame]) -> None:
        """Set data frame attribute
        
        Parameters
        ----------
        df: Union[str, pandas.DataFrame]
            Path of df or df itself
        """
        if isinstance(df, str):
            self.__df = read_df(df)
            
        elif isinstance(df, pandas.DataFrame):
            self.__df = df
            
        else: 
            raise ValueError(
                """This instance takes only path of df and 
                    pandas.DataFrame itself.
                """
            )
        
    #--------#
    # Method #
    #------------------------------------------------------------------------#
    
    @staticmethod
    def add_label(
            df: pandas.DataFrame, 
            target_column: str,
    ) -> pandas.DataFrame:
        """Create label for the data
        attach the label 

        Parameters
        ----------
        df : pandas.DataFrame
            Target data-frame
        target_column: str
            Column to convert to label

        Returns
        -------
        pandas.DataFrame
            Returned data frame with label column
            `price_diff`: differences at row on target column
            `price_diff_shift`: price_diff but shift
            `signal`: sell if minus, buy if plus and still if neutral
        """
        # Diff between row of column
        df["price_diff"] = df[target_column].diff()
        
        # Shift 
        df["price_diff_shift"] = df["price_diff"].shift(-1)
        
        # Signal Label
        df.loc[df["price_diff_shift"] < 0, "signal" ] = 0
        df.loc[df["price_diff_shift"] > 0, "signal" ] = 1
        df.loc[df["price_diff_shift"] == 0, "signal" ] = 1
        
        return df
    
    #------------------------------------------------------------------------#
    
    @abstractmethod
    def transform_df(
            self, 
            fe_name_list: list[str], 
    ) -> pandas.DataFrame:
        """To create df based on FE config provided
        This method need to be override on the engine
        """
        pass
    
    #------------------------------------------------------------------------#
    
#----------------------------------------------------------------------------#
