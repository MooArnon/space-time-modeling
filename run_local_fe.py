#--------#
# Import #
#----------------------------------------------------------------------------#

import os
import pandas as pd

from space_time_modeling.fe import fe_engine

#-----#
# Use #
#----------------------------------------------------------------------------#
if __name__ == "__main__":
    
    #-----------#
    # Attribute #
    #------------------------------------------------------------------------#
    
    df_path = os.path.join("tests", "df_test.csv")
    df = pd.read_csv(df_path)
    
    control_column = "Date"
    target_column = "Open"
    label = "signal"
    
    #---------#
    # Get eda #
    #------------------------------------------------------------------------#
    
    # Initiate engine
    fe = fe_engine(
        control_column = control_column,
        target_column = target_column,
        label = label,
        engine = "classification",
        fe_name_list=[
            "lag_df",
            "rolling_df",
            "percent_change_df",
            "rsi_df",
        ],
    )
    
    # label data
    df = fe.add_label(
        df = df, 
        target_column = "Open",
    )
    
    # Get transform
    df = fe.transform_df(
        df = df,
        serialized = True
    )
    

    print(df.columns)
    
    df.to_csv("preprocessed.csv")
    
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
