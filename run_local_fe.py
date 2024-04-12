##########
# Import #
##############################################################################

import os
import pandas as pd
import pickle

from space_time_modeling.fe import fe_engine 

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
    
    ###########
    # Get eda #
    ##########################################################################

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
            "date_hour_df",
        ],
    )
    
    # label data
    df = fe.add_label(
        df = df, 
        target_column = "open",
    )
    
    # Get transform
    df = fe.transform_df(
        df = df,
        serialized = True
    )
    

    print(df.columns)
    
    df.to_csv("local/preprocessed.csv")

    """
    pickle_file_path = os.path.join(
        "fe_15lag_3-9-12-15-30rolling_percent-change_3-9-12-15-30rsi",
        "fe_15lag_3-9-12-15-30rolling_percent-change_3-9-12-15-30rsi_20240212_195034.pkl"
    )
    with open(pickle_file_path, 'rb') as f:
        # Load the object stored in the pickle file
        preprocessor = pickle.load(f)
    
    df = preprocessor.transform_df(
        df
    )
    
    df.to_csv("preprocessed.csv")
    """
    
    ##########################################################################

##############################################################################
