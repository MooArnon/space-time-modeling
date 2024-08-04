#--------#
# Import #
#----------------------------------------------------------------------------#

import os
import pandas as pd
from typing import Union

from sklearn.feature_selection import mutual_info_classif

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
            ununsed_feature: list[str] = None
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
        ununsed_feature: list[str]
            Name of unused feature, which will be dropped after processing.
            , by default None
        """
        
        # Attributes
        # Main attributes
        self.set_control_column(control_column)
        self.set_target_column(target_column)
        self.set_unused_feature(ununsed_feature)
        
        if label:
            self.set_label(label)
        
        # Function attributes
        self.set_fe_name_list(fe_name_list)
        self.set_n_lag(n_lag)
        self.set_n_window(n_window)
        
        # Set name of instance
        self.set_name(name)
        
    ##############
    # Properties #
    ##########################################################################
    # Main #
    ########
    
    @property
    def name(self) -> str:
        return self.__name
    
    ##########################################################################
    
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
    
    ##########################################################################
    
    @property
    def target_column(self) -> str:
        return self.__target_column
    
    ##########################################################################
    
    def set_target_column(self, column: str) -> None:
        """Target column for the class

        Parameters
        ----------
        column : str
            Target column
        """
        self.__target_column = column
    
    ##########################################################################
    
    @property
    def control_column(self) -> str:
        return self.__control_column
    
    ##########################################################################
    
    def set_control_column(self, column: str) -> None:
        """Control column for the class

        Parameters
        ----------
        column : str
            Target column
        """
        self.__control_column = column
    
    ##########################################################################
    
    @property
    def label(self) -> str:
        return self.__label
    
    ##########################################################################
    
    def set_label(self, column: str) -> None:
        """Control column for the class

        Parameters
        ----------
        column : str
            Target column
        """
        self.__label = column
        
    ##########################################################################
    # FE #
    ######
    
    @property
    def fe_name_list(self) -> list[str]:
        """ list of feature engineering component"""
        return self.__fe_name_list
    
    ##########################################################################
    
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
    
    ##########################################################################
    
    @property
    def n_lag(self) -> int:
        """int of number of lag that used in `lag_df`
        """
        return self.__n_lag
    
    ##########################################################################
    
    def set_n_lag(self, n_lag: int) -> None:
        """Control column for the class

        Parameters
        ----------
        n_lag : int
            number of lag column
        """
        self.__n_lag = n_lag
        
    ##########################################################################
    
    @property
    def n_window(self) -> list[int]:
        """List of int of number of window
        """
        return self.__n_window
    
    ##########################################################################
    
    def set_n_window(self, n_window: list[int]) -> None:
        """Control column for the class

        Parameters
        ----------
        n_window : int
            list of window column
        """
        self.__n_window = n_window
    
    ##########################################################################
    
    @property
    def unused_feature(self) -> list[int]:
        """List of unused feature
        """
        return self.__unused_feature
    
    ##########################################################################
    
    def set_unused_feature(self, unused_feature: list[str]) -> None:
        """Control column for the class

        Parameters
        ----------
        unused_feature : list[str]
            list of window column
        """
        self.__unused_feature = unused_feature
    
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
    
    #######################
    # Feature engineering #
    ##########################################################################
    # Main #
    ########
    
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
        
        # Drop unwanted column
        if self.unused_feature:
            df.drop(columns=self.unused_feature, inplace=True)
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
    
    ##########################################################################
    # Element #
    ###########
    
    @process_dataframe_decorator
    def lag_df(self, df: pd.DataFrame, n_lag: int) -> pd.DataFrame:
        """Create lag feature by number of days

        Parameters
        ----------
        df : pd.DataFrame
            Target df
        n_lag: int
            Number of lag, performing n_lag - 1 for each iteration
        
        Returns
        -------
        pd.DataFrame
            pandas data frame with column `lag_<n>_day`
        """
        # Iterate over lag_day
        for n in range(1, n_lag + 1):
            df[f"lag_{n}_day"] = df[self.target_column].shift(n)
        
        return df
    
    ##########################################################################
    
    @process_dataframe_decorator
    def rolling_df(
            self, 
            df: pd.DataFrame, 
            n_window: list[int],
    ) -> pd.DataFrame:
        """calculate mean of target column

        Parameters
        ----------
        df : pd.DataFrame
            Target df
        n_window: list[int]
            List of window
            
        Returns
        -------
        pd.DataFrame
            pd.DataFrame
            pandas data frame with column `mean_<n>_day` and
            `std_<n>_day`
        """
        # Iterate over list of window applied at instance
        for window in n_window:
            
            # Calculate mean
            df[f'mean_{window}_day'] = df[self.target_column]\
                .rolling(window=window)\
                    .mean()
            
            # Calculate std
            df[f'std_{window}_day'] = df[self.target_column]\
                .rolling(window=window)\
                    .std()
        
        return df
    
    ##########################################################################
    
    @process_dataframe_decorator
    def bollinger_bands(
            self, 
            df:pd.DataFrame,
            n_window: list[int] = None,   
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands

        Parameters
        ----------
        df : pd.DataFrame
            Target df
        n_window : list[int], optional
            List of window, by default None

        Returns
        -------
        pd.DataFrame
            pandas data frame with column `bollinger_bands`
        """
        if not n_window:
            n_window = self.n_window
        for window in n_window:
            
            # Moving Averages
            df['SMA_10'] = df['price'].rolling(window=10).mean()
            df['EMA_10'] = df['price'].ewm(span=10, adjust=False).mean()
            
            # Bollinger Bands
            df[f'upper_band_{window}'] = df['SMA_10'] \
                + 2 * df['price'].rolling(window=10).std()
            df[f'lower_band_{window}'] = df['SMA_10'] \
                - 2 * df['price'].rolling(window=10).std()

        return df
    
    ##########################################################################
    
    @process_dataframe_decorator
    def percent_change_df(
            self,
            df: pd.DataFrame, 
            n_window: list[int] = None,
    ) -> pd.DataFrame:
        """Calculate percent changed from target column

        Parameters
        ----------
        df : pd.DataFrame
            Target df
        n_window: list[int]
            List of window
            
        Returns
        -------
        pd.DataFrame
            pandas data frame with column `percentage_change`
        """
        if not n_window:
            n_window = self.n_window
        for window in n_window:
            df[f'percentage_change_{window}'] = df[self.target_column]\
                .pct_change(periods=window)
        return df
    
    ##########################################################################
    
    @process_dataframe_decorator
    def rsi_df(
            self, 
            df: pd.DataFrame, 
            n_window: list[int] = None,
    ) -> pd.DataFrame:
        """Calculate rsi from target column

        Parameters
        ----------
        df : pd.DataFrame
            Target df
        n_window: list[int]
            List of window

        Returns
        -------
        pd.DataFrame
            pd.DataFrame
            pandas data frame with column `rsi_<n>`
        """
        if not n_window:
            n_window = self.n_window
        
        # Iterate over window
        for window in n_window: 
            
            df[f"rsi_{window}"] = self.calculate_rsi(
                df = df,
                price_column = self.target_column,
                window = window,
            )
            
        return df
        
    ##########################################################################
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, price_column: str, window: int):
        # Calculate price difference
        delta = df[price_column].diff()  
        up, down = delta.copy(), delta.copy()
        
        # Set negative differences to 0 (upward movement)
        up[up < 0] = 0  
        
        # Set positive differences to 0 (downward movement)
        down[down > 0] = 0  
        ema_up = up.ewm(
            alpha=1/window, 
            min_periods=window - 1, 
            adjust=False,
        ).mean()
        ema_down = down.abs().ewm(
            alpha=1/window, 
            min_periods=window - 1, 
            adjust=False,
        ).mean()
        
        # Calculate Relative Strength
        rs = ema_up / ema_down  
        
        # Calculate RSI
        rsi = 100 - 100 / (1 + rs) 

        return rsi
    
    ##########################################################################
    
    @process_dataframe_decorator
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
        
    ##########################################################################
    
    @process_dataframe_decorator
    def ma(self, df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
        """Moving average of target_column

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Output data frame with ma_< period int > as a column
        """
        for period in periods:
            df[f'ma_{period}'] = df[self.target_column]\
                .rolling(window=period).mean()

        return df
    
    ##########################################################################
    
    @process_dataframe_decorator
    def ema(self, df: pd.DataFrame, n_window: list[int] = None) -> pd.DataFrame:
        """Exponential moving average of target_column

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        n_window: list[int]
            List of window size

        Returns
        -------
        pd.DataFrame
            Output data frame with ma_< period int > as a column
        """
        if not n_window:
            n_window = self.n_window
        for window in n_window:
            df[f'ema_{window}'] = df[self.target_column]\
                .ewm(span=window, adjust=False).mean()

        return df

    ##########################################################################
    
    @process_dataframe_decorator
    def percent_diff_ema(
            self, 
            df: pd.DataFrame, 
            n_window: list[int] = None,
    ) -> pd.DataFrame:
        if not n_window:
            n_window = self.n_window
        for window in n_window:
            df[f'percentage_change_ema_{window}'] = \
                (df['price'] - df[f'ema_{window}']) \
                    / df[f'ema_{window}']
        return df

    #############
    # Utilities #
    ##########################################################################
    
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
    
    ##########################################################################
    
    def mutual_info(self, df: pd.DataFrame, label: str = None) -> pd.DataFrame:
        
        if not label:
            label = self.label
        
        column = df.columns.to_list()
        
        column.remove(label)
        
        X = df[column]
        y = df[self.label]
        mutual_info = mutual_info_classif(X, y)

        mutual_info_df = pd.DataFrame(
            {'feature': column, 'mutual_information': mutual_info}
        )
        mutual_info_df.sort_values(
            by='mutual_information', 
            ascending=False, 
            inplace=True,
        )
        
        return mutual_info_df

    ##########################################################################

##############################################################################
