##########
# Import #
##############################################################################

import os
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pandas.core.api import DataFrame as DataFrame
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


from .__base import BaseEDA

##############################################################################
# Regression EDA class #
########################

class TimeSeriesEDA(BaseEDA):
    """The class for explore the simple regression metric

    Inherit
    -------
    BaseEDA : object
        To access the universal eda function
    """
    def __init__(
            self, 
            df: Union[DataFrame, str],
            control_column: str,
            target_column: str,
    ) -> None:
        super().__init__(df)
        
        # Set up attribute
        self.set_control_column(control_column)
        self.set_target_column(target_column)
    
    ##############
    # Properties #
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
    
    ##########
    # Method #
    ##########################################################################

    def get_eda(self, plot_name_list: list[str], store_at: str) -> None:
        """Create eda followed by plot_name_list

        Parameters
        ----------
        plot_name_list : list[str]
            List of name of plot
        store_at : str
            Directory to store result
        """
        # Execute plot
        plot_dict = self.plot(plot_name_list) 
        
        # Save image to the dir
        # Iterate over all plot
        for name in plot_dict:
            
            # Select plot from its name
            plot = plot_dict[name]
            
            # Check if plot is type matplotlib.axes
            if isinstance(plot, matplotlib.axes._axes.Axes):
                fig = plot.get_figure()
            
            # Check if plot is sns
            elif isinstance(plot, sns.PairGrid):
                fig = plot.fig
            
            # If fig itself, save and pass the loop
            elif isinstance(plot, matplotlib.figure.Figure):
                plot.savefig(
                    os.path.join(store_at, f"{name}.png")
                )
                continue
            
            fig.savefig(
                os.path.join(store_at, f"{name}.png")
            )
    
    ##########################################################################
    
    def plot(self, plot_name_list: list[str]) -> dict:
        """Create each plot

        Parameters
        ----------
        plot_name_list : list[str]
            List of plot name

        Returns
        -------
        dict
            result as dict with format {"name_of_plot": "plot"}
        """
        plot_dict = {}
        
        # Iterate over plot_name_list
        for plot_name in plot_name_list:
            
            # Get function for each plot name, using name
            plot_function = getattr(self, plot_name)
            
            # Check if the plot function is found
            if plot_function is None or not callable(plot_function):
                print(f"Warning: Plot function '{plot_name}' not found.")
                continue
            
            # Execute plot function
            plot = plot_function(
                self.df,
                self.control_column,
                self.target_column,
            )
            
            # Add title
            plt.title(plot_name)
            
            # Store plot with plot name
            plot_dict[plot_name] = plot
            
        return plot_dict
    
    ##########################################################################
    # Plots #
    #########
    
    @staticmethod
    def data_date_trend(
            df: pd.DataFrame, 
            control_column: str, 
            target_column: str, 
    ) -> plt.axes:
        """Plot the control_column against target_column

        Parameters
        ----------
        df : pd.DataFrame
            Data-frame
        control_column : str
            dependence column
        target_column : str
            independence column

        Returns
        -------
        plt.axes
            plot
        """
        
        # PLot data
        plot = sns.lineplot(
            data = df,
            x = control_column,
            y = target_column,
        )
        
        # Create parameter to filter x column
        records = df.shape[0]
        date_interval = round(records/100)
        
        # Filter x column
        plot.xaxis.set_major_locator(
            mdates.MonthLocator(interval=date_interval)
        )
        
        return plot
        
    
    ##########################################################################
    
    @staticmethod
    def pair_plot(
            df: pd.DataFrame, 
            control_column: str, 
            target_column: str, 
    ) -> plt.axes:
        """Generate pair-plot of each column in data frame

        Parameters
        ----------
        df : pd.DataFrame
            Source data frame
        control_column : str
            mock
        target_column : str
            mock

        Returns
        -------
        plt.axes
        """
        plot = sns.pairplot(df)
        
        return plot
    
    ##########################################################################
    
    @staticmethod
    def acf_plot(
            df: pd.DataFrame, 
            control_column: str, 
            target_column: str, 
    ) -> plt.axes:
        """Create autocorrelation plot

        Parameters
        ----------
        df : pd.DataFrame
            Source data frame
        control_column : str
            mock
        target_column : str
            mock
        
        Returns
        -------
        plt.axes
        """
        plot = plot_acf(
            df[target_column], 
            lags=40, 
            title='Autocorrelation Function (ACF)'
        )
        return plot

    ##########################################################################
    
    @staticmethod
    def pacf_plot(
            df: pd.DataFrame, 
            control_column: str, 
            target_column: str, 
    ) -> plt.axes:
        """Create autocorrelation plot

        Parameters
        ----------
        df : pd.DataFrame
            Source data frame
        control_column : str
            mock
        target_column : str
            mock
        
        Returns
        -------
        plt.axes
        """
        plot = plot_pacf(
            df[target_column], 
            lags=40, 
            title='Partial Autocorrelation Function (PACF)'
        )
        
        return plot
    
    ##########################################################################
    
    @staticmethod
    def rolling_statistics(
            df: pd.DataFrame, 
            control_column: str, 
            target_column: str, 
    ) -> plt.axes:
        """Create rolling stats

        Parameters
        ----------
        df : pd.DataFrame
            Source data frame
        control_column : str
            mock
        target_column : str
            mock
        
        Returns
        -------
        plt.axes
        """
        # Calculate mean and std
        roll_mean = df[target_column].rolling(window=30).mean()
        roll_std = df[target_column].rolling(window=30).std()
        
        # Create plot
        plt.plot(df[target_column], label='Original')
        plt.plot(roll_mean, label='Rolling Mean')
        plt.plot(roll_std, label='Rolling Std')
        
        plt.legend(loc='upper left')
        plt.title('Rolling Mean and Standard Deviation')
        
        return plt.gcf()
    
    ##########################################################################
    
    @staticmethod
    def correlation_plot(
            df: pd.DataFrame, 
            control_column: str, 
            target_column: str, 
    ) -> plt.axes:
        """Create rolling stats

        Parameters
        ----------
        df : pd.DataFrame
            Source data frame
        control_column : str
            mock
        target_column : str
            mock
        
        Returns
        -------
        plt.axes
        """
        # Exclude none numeric column
        non_numeric_columns = df.select_dtypes(
            exclude=['float64', 'int64']    
        ).columns
        
        # Drop none numeric
        df_numeric = df.drop(columns=non_numeric_columns)
        
        # Get correlation
        df_corr = df_numeric.corr()
        
        # Plot
        plot = sns.heatmap(df_corr)
        
        return plot
    
    ##########################################################################

##############################################################################
