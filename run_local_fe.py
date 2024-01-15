#--------#
# Import #
#----------------------------------------------------------------------------#

import os

from space_time_modeling.fe import engine

#-----#
# Use #
#----------------------------------------------------------------------------#
if __name__ == "__main__":
    
    #-----------#
    # Attribute #
    #------------------------------------------------------------------------#
    
    df_path = os.path.join("tests", "df_test.csv")
    control_column = "Date"
    target_column = "Open"
    
    #---------#
    # Get eda #
    #------------------------------------------------------------------------#
    
    # Initiate engine
    fe = engine(
        df = df_path,
        control_column = control_column,
        target_column = target_column,
        engine = "classification"
    )
    
    # label data
    df = fe.label(
        df = fe.df, 
        target_column = "Open",
    )
    
    df = fe.transform_df(
        [
            "lag_df",
            "rolling_df",
            "percent_change_df",
            "rsi_df",
        ]
    )
    
    print(df.columns)
    
    
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
