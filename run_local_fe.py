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
    
    df_path = os.path.join("local", "btc-all.csv")
    df = pd.read_csv(df_path)
    
    control_column = "scraped_timestamp"
    target_column = "price"
    label = "signal"
    
    df = df[[target_column, control_column]]
    
    # statics 
    label_column = "signal"
    control_column = "scraped_timestamp"
    target_column = "price"
    
    # Feature col 
    feature_column = [
        "percent_change_df",
        "rsi_df",
        "date_hour_df",
        "ema",
        "percent_diff_ema",
        "bollinger_bands",
    ]
    
    ###########
    # Get eda #
    ##########################################################################

    df_path = os.path.join("local", "btc-all.csv")
    
    # Preprocess data
    df = pd.read_csv(df_path)
    df.dropna(inplace=True)
    df = df[[target_column, control_column]]
    
    n_window = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 25, 75, 99]
    ununsed_feature = [f"ema_{win}" for win in n_window]
    
    fe = ClassificationFE(
        control_column = control_column,
        target_column = target_column,
        label = label_column,
        fe_name_list = feature_column,
        n_window = n_window,
        ununsed_feature = ununsed_feature
    )
    
    df_label = fe.add_label(
        df,
        target_column
    )
    
    df_train = fe.transform_df(
        df_label
    )
    
    df_train.to_csv("btc-all-fe.csv")
    
    ##########################################################################

##############################################################################
