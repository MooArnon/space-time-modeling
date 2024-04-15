##########
# Import #
##############################################################################

import os
import pandas as pd

from space_time_modeling.fe import ClassificationFE 

#######
# Use #
##############################################################################
if __name__ == "__main__":
    
    #############
    # Attribute #
    ##########################################################################
    
    df_path = os.path.join("local", "BTC-Hourly.csv")
    df = pd.read_csv(df_path)
    
    control_column = "date"
    target_column = "open"
    label = "signal"
    
    df = df[[target_column, control_column]]
    
    ###########
    # Get eda #
    ##########################################################################

    # Initiate engine
    fe = ClassificationFE(
        control_column = control_column,
        target_column = target_column,
    )
    
    df_label = fe.add_label(
        df,
        target_column
    )
    print(df_label.head(4))
    
    df_ma = fe.ma(df, [3, 7, 25, 99])
    print(df_ma.head(4))
    
    df_ema = fe.ema(df, [3, 7, 25, 99])
    print(df_ema.head(4))
    
    df_date = fe.date_hour_df(df)
    print(df_date.head(4))
    
    df_date = fe.percent_change_df(df, [1, 2, 3, 4, 5, 10])
    print(df_date.head(4))
    
    df_date = fe.rsi_df(df, [3, 7, 25, 99])
    print(df_date.head(10))
    
    ##########################################################################

##############################################################################
