#--------#
# Import #
#----------------------------------------------------------------------------#

import random
import os

from space_time_modeling.modeling import modeling_engine
from space_time_modeling.modeling import ClassificationModel

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
        'lag_1_day', 'lag_2_day',
        'lag_3_day',  'lag_4_day', 'lag_5_day', 'lag_6_day', 'lag_7_day',
        'lag_8_day',  'lag_9_day', 'lag_10_day', 'lag_11_day', 'lag_12_day',
        'lag_13_day', 'lag_14_day', 'lag_15_day', 'mean_3_day', 'std_3_day',
        'mean_9_day', 'std_9_day', 'mean_12_day', 'std_12_day', 'mean_15_day',
        'std_15_day', 'mean_30_day', 'std_30_day', 'percentage_change', 'rsi_3',
        'rsi_9',      'rsi_12', 'rsi_15', 'rsi_30'
    ]
    label = "signal"
    
    #-----------#
    # Get model #
    #------------------------------------------------------------------------#
    
    modeling: ClassificationModel = modeling_engine(
        engine = "classification",
        label_column = label_column,
        feature_column = feature_column,
        result_path = os.path.join("test_catboost"),
        n_iter = 1,
    )
    
    modeling.modeling(
        df = os.path.join("tests", "preprocessed.csv"),
    )
    
    #------------#
    # Test model #
    #------------------------------------------------------------------------#
    
    """
    model = ClassificationModel()
    model = model.load_catboost("test_catboost_20240224_225157/catboost_best_model.bin")

    # Generate a list of 10 random float numbers
    random_floats = [random.random() for _ in range(len(feature_column))]
    
    pred = model.predict_proba(random_floats)
    
    print(model.classes_)
    """
    #------------------------------------------------------------------------#
    
#----------------------------------------------------------------------------#
