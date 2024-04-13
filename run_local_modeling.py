##########
# Import #
##############################################################################

import pickle
import os

import numpy as np
import pandas as pd

from space_time_modeling.modeling import modeling_engine
from space_time_modeling.modeling import ClassificationModel, ClassifierWrapper 
from space_time_modeling.utilities import load_instance

#######
# Use #
##############################################################################
if __name__ == "__main__":
    
    #############
    # Attribute #
    ##########################################################################
    
    pickle_preprocess_path = os.path.join(
        "fe_15lag_3-9-12-15-30rolling_percent-change_3-9-12-15-30rsi",
        "fe_15lag_3-9-12-15-30rolling_percent-change_3-9-12-15-30rsi_20240412_191021.pkl"
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
    label = "signal"
    
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
    """
    modeling: ClassificationModel = modeling_engine(
        engine = "classification",
        label_column = label_column,
        feature_column = feature_column,
        result_path = os.path.join("classifier"),
        n_iter = 1,
    )
    
    modeling.modeling(df = df)
    """
    ##############
    # Test model #
    ##########################################################################
    
    model_types = [
        "catboost", 
        "knn", 
        "logistic_regression", 
        "random_forest", 
        "xgboost"
    ]
    for model_type in model_types:
        path = os.path.join(
            "classifier_20240413_160045",
            model_type,
            f"{model_type}.pkl"
        )
        
        model_wrapper: ClassifierWrapper = load_instance(path)
        last_row = df[model_wrapper.feature].iloc[-1].to_list()
        try:
            if model_type == 'catboost':
                pred = model_wrapper(last_row)
            else: 
                pred = model_wrapper([last_row])
        except ValueError:
            raise ValueError(f"Error at {model_type}")
        
        print(pred)
    
    ##########################################################################
    
##############################################################################
