##########
# Import #
##############################################################################

import os

import pandas as pd

from space_time_modeling.fe import ClassificationFE 
from space_time_modeling.modeling import modeling_engine
from space_time_modeling.modeling import DeepClassificationModel 
from space_time_modeling.modeling.deep_learning_model import DeepWrapper
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
        "macd",
        "roc",
        "bollinger_bands",
        "volatility",
        "moving_average_crossover",
        "stochastic_oscillator",
    ]
    
    n_window = [1, 3, 5, 7, 10, 15]
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
        df = df_label,
    )
    
    # return df.columns
    # Train model
    modeling: DeepClassificationModel = modeling_engine(
        engine = "deep_classification",
        label_column = label_column,
        feature_column = feature_column,
        result_path = os.path.join("short-term-high_feature"),
        test_size = int(24),
        epoch_per_trial = 35,
        max_trials = 40,
        early_stop_min_delta = 0.001,
        early_stop_patience = 5,
        push_to_s3 = True,
        override_model_name_dict = {
            "dnn": "dnn-short",
            "lstm": "lstm-short",
            "gru": "gru-short",
            "cnn": "cnn-short",
        },
        aws_s3_bucket = 'space-time-model',
        aws_s3_prefix = 'classifier/btc',
    )
    
    modeling.modeling(
        df = df_train, 
        preprocessing_pipeline=fe,
        model_name_list=['dnn'],
        feature_rank = 71,
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
    model: DeepWrapper = load_instance(model_path)
    
    print(model.name)
    print(model.feature)
    
    data_df = pd.read_csv(data_path)
    print(data_df.head(4))
    pred = model(data_df)
    
    print(pred)
    print('\n')


#######
# Use #
##############################################################################

if __name__ == "__main__":
    # train_model()
    
    result_path =  "prod-test"
    test_model(result_path, 'dnn-short')
    # test_model(result_path, 'lstm-short')
    # test_model(result_path, 'gru-short')
    # test_model(result_path, 'dnn-short')
    
    ##########################################################################

##############################################################################