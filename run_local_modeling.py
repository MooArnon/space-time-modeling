##########
# Import #
##############################################################################

import os
import json

import pandas as pd
import requests

from space_time_modeling.fe import ClassificationFE 
from space_time_modeling.modeling import modeling_engine
from space_time_modeling.modeling import ClassificationModel 
from space_time_modeling.modeling.__classification_wrapper import ClassifierWrapper
from space_time_modeling.utilities import load_instance

#########
# Train #
##############################################################################

def train_model() -> None:
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
    ]
    
    n_window = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 25, 75, 99]
    ununsed_feature = [f"ema_{win}" for win in n_window]
    
    df_path = os.path.join("local", "btc-all.csv")
    
    # Preprocess data
    df = pd.read_csv(df_path)
    df.dropna(inplace=True)
    df = df[[target_column, control_column]]
    
    fe = ClassificationFE(
        control_column = control_column,
        target_column = target_column,
        label = label_column,
        fe_name_list = feature_column,
        n_window = n_window,
        ununsed_feature = ununsed_feature,
    )
    
    df_label = fe.add_label(
        df,
        target_column
    )
    
    df_train = fe.transform_df(
        df_label
    )
    
    # return df.columns
    # Train model
    modeling: ClassificationModel = modeling_engine(
        engine = "classification",
        label_column = label_column,
        feature_column = feature_column,
        result_path = os.path.join("fine-tune-lightgbm"),
        test_size = int(56),
        n_iter = 10,
        push_to_s3 = True,
        aws_s3_bucket = 'space-time-model',
        aws_s3_prefix = 'classifier/btc',
    )
    
    modeling.modeling(
        df = df_train, 
        preprocessing_pipeline=fe,
        model_name_list=['lightgbm'],
        feature_rank = 15,
    )
    
########
# Test #
##############################################################################

def test_model(path: str, type: str) -> None:
    model_path = os.path.join(
        path,
        type,
        f"{type}.pkl",
    )
    
    data_path = os.path.join(
        "local",
        "sample-test.csv",
    )
    
    # Load model
    model: ClassifierWrapper = load_instance(model_path)
    
    print(model.version)
    print(model.name)
    
    df = pd.read_csv(data_path)
    pred = model(df)
    
    print(pred)
    print('\n')

##############################################################################

def eval_model(path: str, type: str) -> None:
    model_path = os.path.join(
        path,
        type,
        f"{type}.pkl",
    )
    
    data_path = os.path.join(
        "local",
        "sample-test.csv",
    )
    
    # Load model
    model: ClassifierWrapper = load_instance(model_path)
    
    print(model.version)
    print(model.name)
    
    df = pd.read_csv(data_path)
    
    pred = model.evaluate(
        x_test=df,
    )
    
    print(pred)
    print('\n')

#######
# Use #
##############################################################################

if __name__ == "__main__":
    
    train_model()
    """
    model_type_list = ["catboost", "knn", "logistic_regression", "random_forest", "xgboost"]
    result_path =  "test-mutual-feature_20240803_191220"
    
    for model_type in model_type_list:
        test_model(result_path, model_type)
    
    """
    """
    
    result_path =  "feat-wrap-eval_20240901_123301"
    
    test_model(result_path, 'xgboost')
    
    eval_model(
        "feat-wrap-eval_20240901_123301", 
        'xgboost'
    )
    """
    ##########################################################################

##############################################################################