#--------#
# Import #
#----------------------------------------------------------------------------#

import os

from space_time_modeling.modeling import modeling_engine

#-----#
# Use #
#----------------------------------------------------------------------------#
if __name__ == "__main__":
    
    #-----------#
    # Attribute #
    #------------------------------------------------------------------------#
    
    df_path = os.path.join("result", "preprocessed.csv")
    label_column = "signal"
    feature_column = [
        'signal', 'lag_1_day', 'lag_2_day',
        'lag_3_day', 'lag_4_day', 'lag_5_day', 'lag_6_day', 'lag_7_day',
        'lag_8_day', 'lag_9_day', 'lag_10_day', 'lag_11_day', 'lag_12_day',
        'lag_13_day', 'lag_14_day', 'lag_15_day', 'mean_3_day', 'std_3_day',
        'mean_9_day', 'std_9_day', 'mean_12_day', 'std_12_day', 'mean_15_day',
        'std_15_day', 'mean_30_day', 'std_30_day', 'percentage_change', 'rsi_3',
        'rsi_9', 'rsi_12', 'rsi_15', 'rsi_30'
    ]
    label = "signal"
    
    #---------#
    # Get model #
    #------------------------------------------------------------------------#
    
    modeling = modeling_engine(
        engine = "classification",
        label_column = label_column,
        feature_column = feature_column,
        result_path = os.path.join("result"),
    )
    
    modeling.modeling(df = os.path.join("tests", "preprocessed.csv"),)
    
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
