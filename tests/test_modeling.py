#--------#
# Import #
#----------------------------------------------------------------------------#

import os
import shutil

import pytest

from space_time_modeling.modeling import modeling_engine
from space_time_modeling.modeling import (
    ClassificationModel,
)

#---------#
# Classes #
#----------------------------------------------------------------------------#

class TestEngineFE:
    
    # Init attribute
    control_column = "Date"
    target_column = "Open"
    label = "signal"
    
    #------------------------------------------------------------------------#
    # Classification fe #
    #-------------------# 
    
    @pytest.fixture
    def engine(self):
        df_path = os.path.join("tests", "preprocessed.csv")
        
        label_column = "signal"
        feature_column = [
            'signal', 'lag_1_day', 'lag_2_day',
            'lag_3_day', 'lag_4_day', 'lag_5_day', 'lag_6_day', 'lag_7_day',
            'lag_8_day', 'lag_9_day', 'lag_10_day', 'lag_11_day', 
            'lag_12_day', 'lag_13_day', 'lag_14_day', 'lag_15_day', 
            'mean_3_day', 'std_3_day', 'mean_9_day', 'std_9_day', 
            'mean_12_day', 'std_12_day', 'mean_15_day', 'std_15_day', 
            'mean_30_day', 'std_30_day', 'percentage_change', 'rsi_3',
            'rsi_9', 'rsi_12', 'rsi_15', 'rsi_30'
        ]
        
        model_engine:ClassificationModel = modeling_engine(
            "classification",
            df = df_path,
            label_column = label_column,
            feature_column = feature_column,
            result_path = os.path.join("result_tests"),
            n_iter = 5,
        )
        
        return model_engine
    
    #------------------------------------------------------------------------#
    
    def test_result(self, engine: ClassificationModel):
        """ Modeling properly works """
        
        try:
            engine.modeling(
                model_name_list = ["xgboost"]
            )
            is_pass = True
            
        except:
            is_pass = False
        
        assert is_pass
        
    #------------------------------------------------------------------------#
    
    def readable_model(self, engine: ClassificationModel):
        
        try:
            engine.load_xgboost(
                os.path.join("result_tests", "xgboost.xgb")
            )
            is_pass = True
        
        except:
            is_pass = False
        
        assert is_pass
    
    #------------------------------------------------------------------------#
    
    def test_delete_result(self):
        shutil.rmtree("result_tests")
    
    #------------------------------------------------------------------------#
    
    
    
    #------------------------------------------------------------------------#
    
#----------------------------------------------------------------------------#
