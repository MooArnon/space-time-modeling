#--------#
# Import #
#----------------------------------------------------------------------------#

import pandas as pd
from typing import Union

from .__base import BaseFE

#-------#
# Class #
#----------------------------------------------------------------------------#

class ClassificationFE(BaseFE):
    """
    ClassificationFE
    ==========================
    Feature engineering for regression classification
    
    Attributes
    ----------
    target_column : str
    control_column : str
    label : str
    n_lag : int
    n_window : list[int]
    
    Methods
    -------
    transform_df(self, fe_name_list: list[str]) -> pd.DataFrame:
        Create transformed data frame  followed by name of fe
        approach in fe_name_list.
        `lag_df`
        `rolling_df`
        `percent_change_df`
        `rsi_df`
    """
    name = 'classification-feature-engineering-instance'
    def __init__(
            self, 
            df: Union[str, pd.DataFrame],
            control_column: str,
            target_column: str,
            label: str = None,
            n_lag: int = 15, 
            n_window: list[int] = [3, 9, 12, 15, 30]
    ) -> None:
        """Initiate `ClassificationFE` instance, inherited base class.
        
        Parameters
        ----------
        df : Union[str, pd.DataFrame]
            target transform
        control_column : str
            In regression, its name of time control
        target_column : str
            Name of target column
        label : str, optional
            Name of label column, by default None
        n_lag : int, optional
            Number of lag days that need to use, by default 15
        n_window : list[int], optional
            List of window that uses to construct the window statistics
            , by default [3, 9, 12, 15, 30]
        """
        super().__init__(df)
        
        # Attributes
        # Main attributes
        self.set_control_column(control_column)
        self.set_target_column(target_column)
        
        if label:
            self.set_label(label)
        
        # Function attributes
        self.set_n_lag(n_lag)
        self.set_n_window(n_window)
        
        
    #------------#
    # Properties #
    #------------------------------------------------------------------------#
    # Main #
    #------#
    
    @property
    def target_column(self) -> str:
        return self.__target_column
    
    #------------------------------------------------------------------------#
    
    def set_target_column(self, column: str) -> None:
        """Target column for the class

        Parameters
        ----------
        column : str
            Target column
        """
        self.__target_column = column
    
    #------------------------------------------------------------------------#
    
    @property
    def control_column(self) -> str:
        return self.__control_column
    
    #------------------------------------------------------------------------#
    
    def set_control_column(self, column: str) -> None:
        """Control column for the class

        Parameters
        ----------
        column : str
            Target column
        """
        self.__control_column = column
    
    #------------------------------------------------------------------------#
    
    @property
    def label(self) -> str:
        return self.__label
    
    #------------------------------------------------------------------------#
    
    def set_label(self, column: str) -> None:
        """Control column for the class

        Parameters
        ----------
        column : str
            Target column
        """
        self.__label = column
        
    #------------------------------------------------------------------------#
    # FE #
    #----#
    
    @property
    def n_lag(self) -> int:
        """int of number of lag that used in `lag_df`
        """
        return self.__n_lag
    
    #------------------------------------------------------------------------#
    
    def set_n_lag(self, n_lag: int) -> None:
        """Control column for the class

        Parameters
        ----------
        n_lag : int
            number of lag column
        """
        self.__n_lag = n_lag
        
    #------------------------------------------------------------------------#
    
    @property
    def n_window(self) -> list[int]:
        """List of int of number of window
        """
        return self.__n_window
    
    #------------------------------------------------------------------------#
    
    def set_n_window(self, n_window: list[int]) -> None:
        """Control column for the class

        Parameters
        ----------
        n_window : int
            list of window column
        """
        self.__n_window = n_window
    
    #---------------------#
    # Feature engineering #
    #------------------------------------------------------------------------#
    # Main #
    #------#
    
    def transform_df(
            self, 
            fe_name_list: list[str], 
    ) -> pd.DataFrame:
        """Transform df to listed fe
        Available fe names are `lag_df`, `rolling_df`, 
        `percent_change_df`, `rsi_df`,
        

        Parameters
        ----------
        fe_name_list : list[str]
            List of name of transformation

        Returns
        -------
        pd.DataFrame
            Returned the transformation
        """
        df = self.df
        
        df = self.delete_unused_columns(
            df = df,
            target_column = self.target_column,
            control_column = self.control_column,
            label = self.label
        )
        
        # Iterate over fe_name_list
        for fe_name in fe_name_list:
            
            # Get function for each fe name, using name
            fe_function = getattr(self, fe_name)
            
            # Check if the fe function is found
            if fe_function is None or not callable(fe_function):
                print(f"Warning: FE function '{fe_name}' not found.")
                continue
            
            # Execute fe function
            df = fe_function(df)
        
        df.dropna(inplace=True)
        
        return df
    
    #------------------------------------------------------------------------#
    # Element #
    #---------#
    
    def lag_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag feature by number of days

        Parameters
        ----------
        df : pd.DataFrame
            Target df

        Returns
        -------
        pd.DataFrame
            pandas data frame with column `lag_<n>_day`
        """
        # Iterate over lag_day
        for n in range(1, self.n_lag + 1):
            df[f"lag_{n}_day"] = df[self.target_column].shift(n)
        
        return df
    
    #------------------------------------------------------------------------#
    
    def rolling_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """calculate mean of target column

        Parameters
        ----------
        df : pd.DataFrame
            Target df

        Returns
        -------
        pd.DataFrame
            pd.DataFrame
            pandas data frame with column `mean_<n>_day` and
            `std_<n>_day`
        """
        # Iterate over list of window applied at instance
        for window in self.n_window:
            
            # Calculate mean
            df[f'mean_{window}_day'] = df[self.target_column]\
                .rolling(window=window)\
                    .mean()
            
            # Calculate std
            df[f'std_{window}_day'] = df[self.target_column]\
                .rolling(window=window)\
                    .std()
        
        return df
    
    #------------------------------------------------------------------------#
    
    def percent_change_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percent changed from target column

        Parameters
        ----------
        df : pd.DataFrame
            Target df

        Returns
        -------
        pd.DataFrame
            pandas data frame with column `percentage_change`
        """
        df['percentage_change'] = df[self.target_column].pct_change()
        return df
    
    #------------------------------------------------------------------------#
    
    def rsi_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rsi from target column

        Parameters
        ----------
        df : pd.DataFrame
            Target df

        Returns
        -------
        pd.DataFrame
            pd.DataFrame
            pandas data frame with column `rsi_<n>`
        """
        # Iterate over window
        for window in self.n_window: 
            
            df = self.calculate_rsi(
                df = df,
                window = window,
                target_column = self.target_column
            )
            
        return df
        
    #------------------------------------------------------------------------#
    
    @staticmethod
    def calculate_rsi(
            df: pd.DataFrame, 
            window: int,
            target_column: str
    ) -> pd.DataFrame:
        """Calculate rsi from target column, window and target_column

        Parameters
        ----------
        df : pd.DataFrame
            Target df

        Returns
        -------
        pd.DataFrame
            pd.DataFrame
            pandas data frame with column `rsi_<n>`
        """
        # Calculate price changes
        delta = df[target_column].diff(1)

        # Define gain and loss
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and average loss over the specified window
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        # Calculate relative strength (RS)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        # Add RSI column to the DataFrame
        df[f'rsi_{window}'] = rsi

        return df
        
    #-----------#
    # Utilities #
    #------------------------------------------------------------------------#
    
    @staticmethod
    def delete_unused_columns(
            df: pd.DataFrame,
            target_column: str,
            control_column: str,
            label: str = None
    ) -> pd.DataFrame:
        """Keep only used columns

        Parameters
        ----------
        df : pd.DataFrame
            Target df
        target_column : str
            Target column
        control_column : str
            Control column
        label: str = None
            Label column

        Returns
        -------
        pd.DataFrame
            Result df, it keeps only the target_column and 
            control_column
        """
        
        if label:
            df = df[[target_column, control_column, label]]
        
        elif label is None:
            df = df[[target_column, control_column]]
        
        return df
    
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
