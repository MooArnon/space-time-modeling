#--------#
# Import #
#----------------------------------------------------------------------------#

from abc import abstractmethod
import pandas
import pandas as pd
from typing import Union

from space_time_modeling.utilities import read_df

#---------#
# Classes #
#----------------------------------------------------------------------------#

class BaseFE:
    
    ##########
    # Wraper #
    ##########################################################################
    
    def process_dataframe_decorator(func):
        """
        Decorator to process the DataFrame by converting 
        the specified column to datetime format
        and sorting the DataFrame based on that column.

        Parameters
        ----------
        func : function
            The function to be decorated.

        Returns
        -------
        function
            Decorator function to process the DataFrame.
        """
        def wrapper(self, df, *args, **kwargs):
            df[self.control_column] = pd.to_datetime(df[self.control_column])
            df = df.sort_values(by=self.control_column, ascending=True)
            return func(self, df, *args, **kwargs)
        return wrapper
    
    #--------#
    # Method #
    ##########################################################################
    
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
    
    ##########################################################################
    
    @process_dataframe_decorator
    def add_label(
            self,
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
    
    ##########################################################################
    
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

    ##########################################################################
    
#----------------------------------------------------------------------------#
