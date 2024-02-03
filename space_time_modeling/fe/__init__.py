#--------#
# Import #
#----------------------------------------------------------------------------#

import pandas
from typing import Union

from .__base import BaseFE
from .classification_fe import ClassificationFE


#----------#
# Function #
#----------------------------------------------------------------------------#

def fe_engine(
        engine: str,
        control_column: str,
        target_column: str,
        **kwargs,
) -> BaseFE:
    """Create instance of feature engineering

    Parameters
    ==========
    engine : str
        Name of engine, now is 
        `classification`
    control_column : str
        Stamp column for df, x
    target_column : str
        Value column for df, y

    Returns
    =======
    BaseFE
        Instance of BaseFE
    
    Raise
    =====
    ValueError
        If engine is not exists
    
    Additional Parameters
    =====================
    classification: ClassificationFE
    ---------------------------------
    Feature for classifier, `lag_df`, `rolling_df`,  
    `percent_change_df`, `rsi_df`  
    label: str = None
        Label of data, `signal`
    n_lag: int = 15, 
        Number of lag, it will iterate over int to create lag column.
        Used by `lag_df` method
    n_window: list[int] = [3, 9, 12, 15, 30]
        List of integer, create n column.
        Used by `rolling_df`, `rsi_df`
    
    Example
    =======
    classification
    --------------
    ```python
    # Import
    from from space_time_modeling.fe import engine
    
    # Initiate engine
    fe = engine(
        df = df_path,
        control_column = control_column,
        target_column = target_column,
        label = label,
        engine = "classification",
    )
    
    # Add label
    df = fe.add_label(
        df = fe.df, 
        target_column = "Open",
    )
    
    # Transform data frame with specified fe functions
    df = fe.transform_df(
        [
            "lag_df",
            "rolling_df",
            "percent_change_df",
            "rsi_df",
        ]
    )
    ```
    """
    # Check engine
    if engine == "classification":
        fe = ClassificationFE(
            control_column = control_column,
            target_column = target_column,
            **kwargs,
        )
    else: 
        raise ValueError(f"{engine} is not supported")
    
    return fe

#----------------------------------------------------------------------------#
