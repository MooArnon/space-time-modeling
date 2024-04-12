##########
# Import #
##############################################################################

import pickle
import os

import pandas as pd
import torch.nn as nn

from space_time_modeling.modeling import modeling_engine
from space_time_modeling.modeling import DeepClassificationModel
from space_time_modeling.utilities import load_model_dnn

#######
# Use #
##############################################################################
if __name__ == "__main__":
    
    #############
    # Attribute #
    ##########################################################################
    
    pickle_preprocess_path = os.path.join(
        "fe_15lag_3-9-12-15-30rolling_percent-change_3-9-12-15-30rsi",
        "fe_15lag_3-9-12-15-30rolling_percent-change_3-9-12-15-30rsi_20240412_121420.pkl"
    )
    
    df_path = os.path.join("local", "BTC-Hourly.csv")
    
    label_column = "signal"
    feature_column = [ "signal",
        'lag_1_day', 'lag_2_day',
        'lag_3_day',  'lag_4_day', 'lag_5_day', 'lag_6_day', 'lag_7_day',
        'lag_8_day',  'lag_9_day', 'lag_10_day', 'lag_11_day', 'lag_12_day',
        'lag_13_day', 'lag_14_day', 'lag_15_day', 'mean_3_day', 'std_3_day',
        'mean_9_day', 'std_9_day', 'mean_12_day', 'std_12_day', 'mean_15_day',
        'std_15_day', 'mean_30_day', 'std_30_day', 'percentage_change', 'rsi_3',
        'rsi_9',      'rsi_12', 'rsi_15', 'rsi_30'
    ]
    
    ###################
    # Preprocess data #
    ##########################################################################
    
    df = pd.read_csv(df_path)
    
    with open(pickle_preprocess_path, 'rb') as f:
        
        # Load the object stored in the pickle file
        preprocessor = pickle.load(f)
    
    df = preprocessor.add_label(
        df = df, 
        target_column = "open",
    )
    
    df = preprocessor.transform_df(df)
    
    #############
    # Get model #
    ##########################################################################
    
    modeling: DeepClassificationModel = modeling_engine(
        engine = "deep_classification",
        label_column = label_column,
        feature_column = feature_column,
        result_path = os.path.join("test_dnn"),
        mode ='random_search',
        n_iter = 2,
        lstm_params_dict = {
            'lr': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'epochs':[15, 20],
            'criterion':[nn.BCELoss(), nn.HuberLoss()],
            'module__hidden_layers': [
                [8, 16, 8],
                [16, 32, 16],
                [8, 16, 16, 8],
                [16, 32, 32, 16],
                [8, 16, 32, 16, 8],
            ],
            'module__dropout': [0.1, 0.15, 0.2, 0.25]
        }
    )
    
    modeling.modeling(
        df = df,
        model_name_list = ['lstm', 'dnn'],
        batch_size = 32,
    )
    
    ##########################################################################
    """
    model = load_model_dnn("test_dnn_20240408_233107/lstm/lstm.pth")
    print(model)
    print(df.head(5))
    last_row = df.iloc[-1].to_list()
    print(last_row)
    pred = model(last_row)
    print(pred)
    """
    ##########################################################################

##############################################################################
