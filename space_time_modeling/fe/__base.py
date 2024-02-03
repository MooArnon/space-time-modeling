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
    
    #--------#
    # Method #
    #------------------------------------------------------------------------#
    
    @staticmethod
    def read_df(df: Union[str, pandas.DataFrame]) -> None:
        """Set data frame attribute
        
        Parameters
        ----------
        df: Union[str, pandas.DataFrame]
            Path of df or df itself
        """
        if isinstance(df, str):
            return read_df(df)
            
        elif isinstance(df, pandas.DataFrame):
            return df
            
        else: 
            raise ValueError(
                """This instance takes only path of df and 
                    pandas.DataFrame itself.
                """
            )
    
    #------------------------------------------------------------------------#
    
    @staticmethod
    def add_label(
            df: pandas.DataFrame, 
            target_column: str,
    ) -> pandas.DataFrame:
        """Create label for the data
        attach the label , 0 = sell, 1 = buy

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
            df: Union[str, pandas.DataFrame],
            fe_name_list: list[str], 
            serialized: bool = False,
    ) -> pandas.DataFrame:
        """To create df based on FE config provided
        This method need to be override on the engine
        """
        pass

    #------------------------------------------------------------------------#
    
#----------------------------------------------------------------------------#
