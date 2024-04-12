##########
# Import #
##############################################################################

import pandas
from typing import Union

from space_time_modeling.eda.__engine import TimeSeriesEDA
from ..utilities import create_dir_w_timestamp

############
# Function #
##############################################################################

def eda(
        df: Union[str, pandas.DataFrame],
        store_at: str,
        plot: list[str] = None,
        plot_attribute: dict = None,
) -> None:
    """Create the EDA of `df` and store result at `store_at` 
    This function wii call the eda instance.
    
    Parameters
    ==========
    df : Union[str, pandas.DataFrame]
        The target data frame ,can be either path of csv, excel and 
        pandas data frame
    store_at : str
        Path to store the result. It will came with timestamp
        <store_at>_YYYYMMMDD_HHMMSS file format.
    
    Raise
    =====
    ValueError
        if plot was assign but not for plot_attribute.
        This raise handel error at TimeSeriesEDA
    
    Additional Parameters
    =====================
    TimeSeriesEDA engine
    --------------------
    plot_attribute : dict
        The attribute for plot stored as dictionary and pass **kwargs
        to the instance.
        {
            control_column: "Name of column as X variable",
            target_column: "Name of column as Y variable",
        }
    """
    # Init path
    # Create directory to store
    dir_name = create_dir_w_timestamp(store_at)
    
    # If plot is indicated
    if plot:
        
        if plot_attribute is None:
            raise ValueError(
                """Please assign the plot_attribute
                see the attribute of plot at help.
                """
            )
        
        # Initialize class
        eda_plot = TimeSeriesEDA(df = df, **plot_attribute)
        
        # Get plot
        eda_plot.get_eda(
            plot_name_list=plot,
            store_at=dir_name
        )
    
##############################################################################
