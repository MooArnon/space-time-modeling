#--------#
# Import #
#----------------------------------------------------------------------------#

import os
import pandas as pd
from typing import Union

from .__base import BaseFE
from ..utilities import serialize_instance

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
        `date_hour_df`
    """
    name = 'fe'
    def __init__(
            self,
            control_column: str,
            target_column: str,
            label: str = None,
            fe_name_list: list[str] = [
                "lag_df",
                "rolling_df",
                "percent_change_df",
                "rsi_df",
                "date_hour_df"
            ], 
            n_lag: int = 15, 
            n_window: list[int] = [3, 9, 12, 15, 30],
            name: str = None,
    ) -> None:
        """Initiate `ClassificationFE` instance, inherited base class.
        
        Parameters
        ----------
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
        name: str
            Name of instance
            , by default None
        """
        
        # Attributes
        # Main attributes
        self.set_control_column(control_column)
        self.set_target_column(target_column)
        
        if label:
            self.set_label(label)
        
        # Function attributes
        self.set_fe_name_list(fe_name_list)
        self.set_n_lag(n_lag)
        self.set_n_window(n_window)
        
        # Set name of instance
        self.set_name(name)
        
    #------------#
    # Properties #
    #------------------------------------------------------------------------#
    # Main #
    #------#
    
    @property
    def name(self) -> str:
        return self.__name
    
    #------------------------------------------------------------------------#
    
    def set_name(self, name: str = None) -> None:
        """Set name of instance
        
        Parameters
        ----------
        name: str
            If not none, just fe
        """
        # Set base name
        __name = "fe_" if name is None else name
        
        # Iterate over fe_name_list
        for fe_name in self.fe_name_list:
            if fe_name == "lag_df":
                __name += f"{self.n_lag}lag_"
            elif fe_name == "rolling_df":
                __name += f"{'-'.join([str(x) for x in self.n_window])}rolling_"
            elif fe_name == "percent_change_df":
                __name += "percent-change_"
            elif fe_name == "rsi_df":
                __name += f"{'-'.join([str(x) for x in self.n_window])}rsi"
            
        self.__name = __name
    
    #------------------------------------------------------------------------#
    
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
    def fe_name_list(self) -> list[str]:
        """ list of feature engineering component"""
        return self.__fe_name_list
    
    #------------------------------------------------------------------------#
    
    def set_fe_name_list(self, fe_name_list: list[str]):
        """Set fe_name_list

        Parameters
        ----------
        fe_name_list : _type_
            List of fe

        Returns
        -------
        list[str]
            List of fe
        """
        self.__fe_name_list = fe_name_list
    
    #------------------------------------------------------------------------#
    
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
            df: Union[str, pd.DataFrame],
            serialized: bool = False,
    ) -> pd.DataFrame:
        """Transform df to listed fe
        Available fe names are `lag_df`, `rolling_df`, 
        `percent_change_df`, `rsi_df`,
        
        Parameters
        ----------
        df: Union[str, pd.DataFrame]
            Target data frame
        serialized : bool
            If not None, export the fe instance to working dir

        Returns
        -------
        pd.DataFrame
            Returned the transformation
        """
        df = self.read_df(df)
        
        df[self.control_column] = pd.to_datetime(df[self.control_column])
        
        df = df.sort_values(by=self.control_column)
        
        df = self.delete_unused_columns(
            df = df,
            target_column = self.target_column,
            control_column = self.control_column,
            label = self.label
        )
        
        # Iterate over fe_name_list
        for fe_name in self.fe_name_list:
            
            # Get function for each fe name, using name
            fe_function = getattr(self, fe_name)
            
            # Check if the fe function is found
            if fe_function is None or not callable(fe_function):
                print(f"Warning: FE function '{fe_name}' not found.")
                continue
            
            # Execute fe function
            df = fe_function(df)
        
        df.dropna(inplace=True)
        
        df.drop(
            columns=[self.target_column, self.control_column], 
            inplace=True,
        )
        
        # Serialize fe if the path is specified
        if serialized:
            serialize_instance(
                self, 
                self.name
            )
        
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
    
    #------------------------------------------------------------------------#
    
    def date_hour_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract `week_of_month`, `date_of_month`, `hour`

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        target_column : str
            _description_
        control_column : str
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        df['datetime_column'] = pd.to_datetime(df[self.control_column])

        # Extract week of the month
        df['week_of_month'] = (df['datetime_column'].dt.day - 1) // 7 + 1

        # Extract date of the month
        df['date_of_month'] = df['datetime_column'].dt.day

        # Extract day of the week (0 for Monday, 1 for Tuesday, ..., 6 for Sunday)
        df['day_of_week'] = df['datetime_column'].dt.dayofweek

        # Extract hour
        df['hour'] = df['datetime_column'].dt.hour
        
        df.drop(columns="datetime_column", inplace=True)
        
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
        columns = list(df.columns)
        
        if label in columns:
            df = df[[target_column, control_column, label]]
        
        elif label not in columns:
            df = df[[target_column, control_column]]
        
        return df
    
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
