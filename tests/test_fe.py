##########
# Import #
##############################################################################

import os
import pandas
import shutil

import pytest

from space_time_modeling.fe import fe_engine
from space_time_modeling.fe.classification_fe import (
    ClassificationFE,
)

pandas.options.mode.chained_assignment = None

#########
# Tests #
##############################################################################
# Engine #
##########

class TestEngineFE:
    
    # Init attribute
    df_path = os.path.join("tests", "df_test.csv")
    df = pandas.read_csv(os.path.join("tests", "df_test.csv"))
    control_column = "Date"
    target_column = "Open"
    label = "signal"
    
    ##########################################################################
    # Classification fe #
    ##################### 
    
    @pytest.fixture
    def engine(self):
        return fe_engine(
            control_column = self.control_column,
            target_column = self.target_column,
            label = self.label,
            engine = "classification",
        )
    
    ##########################################################################
    
    def test_result_type(self, engine: ClassificationFE, ):
        
        df = engine.transform_df(
            df = self.df,
            serialized=False,
        )
        
        assert type(df) == pandas.DataFrame
        assert df.shape != (0, 0)
        
    ##########################################################################
    
    def test_result(self, engine: ClassificationFE, ):
    
        engine.set_n_lag = 3
        engine.set_n_window = [3, 4, 5]
        
        df = engine.transform_df(
            df = self.df,
            serialized=False,
        )
        
        assert type(df) == pandas.DataFrame
        assert df.shape != (0, 0)
        
        

##############################################################################
# Classification Fe #
#####################

class TestClassificationFE:
    df_path = os.path.join("tests", "df_test.csv")
    df = pandas.read_csv(os.path.join("tests", "df_test.csv"))
    
    @pytest.fixture
    def engine(self):
        return ClassificationFE(
            control_column = "Date",
            target_column = "Open",
        )
    
    #######################
    # Feature engineering #
    ##########################################################################
    # Return type #
    ###############
    
    def test_lag_df_type(self, engine: ClassificationFE, ):
        """ Need to return pandas data frame"""
        
        df = engine.lag_df(self.df)
        
        assert type(df) == pandas.DataFrame
    
    ##########################################################################
    
    def test_rolling_df_type(self, engine: ClassificationFE, ):
        """ Need to return pandas data frame"""
        
        df = engine.rolling_df(self.df)
        
        assert type(df) == pandas.DataFrame
    
    ##########################################################################
    
    def test_del_df_type(self, engine: ClassificationFE, ):
        """ Need to return pandas data frame"""
        
        df = engine.delete_unused_columns(
            df = self.df,
            target_column = "Open",
            control_column = "Date",
        )
        
        assert type(df) == pandas.DataFrame
        
    ##########################################################################
    # Return result #
    #################
    
    def test_del_df_column(self, engine: ClassificationFE, ):
        """ Need to return pandas data frame"""
        
        df = engine.delete_unused_columns(
            df = self.df,
            target_column = "Open",
            control_column = "Date",
        )
        
        assert df.shape[1] == 2
    
    ##########################################################################
    
    def test_lag_df_column(self, engine: ClassificationFE, ):
        """ Shape and column must be the same """
        
        lag_columns = [
            "lag_1_day", 
            "lag_2_day", 
            "lag_3_day", 
            "lag_4_day", 
            "lag_5_day"
        ]
        
        engine.set_n_lag(5)
        
        df = engine.delete_unused_columns(
            df = self.df,
            target_column = "Open",
            control_column = "Date",
        )
        
        df = engine.lag_df(df)
        
        df_column = list(df.columns)
        
        # Test shape
        assert df.shape[1] == 7
        
        # Test column name
        for lag in lag_columns:
            assert lag in df_column
            df_column.remove(lag)
        
        # Must be only 2 element in list, 
        # generated column fit the parameter
        assert len(df_column) == 2
    
    ##########################################################################
    
    def test_rolling_df_column(self, engine: ClassificationFE, ):
        "Test the rolling method"
        rolling_columns = [
            "mean_3_day",
            "mean_6_day",
            "mean_12_day",
            "std_3_day",
            "std_6_day",
            "std_12_day",
        ]
        
        df = engine.delete_unused_columns(
            df = self.df,
            target_column = "Open",
            control_column = "Date",
        )
        
        engine.set_n_window([3, 6, 12])
        
        df = engine.rolling_df(df)
        
        df_column = list(df.columns)
        
        # Test shape
        assert df.shape[1] == 8
        
        # Test column name
        for lag in rolling_columns:
            assert lag in df_column
            df_column.remove(lag)
        
        # Must be only 2 element in list, 
        # generated column fit the parameter
        assert len(df_column) == 2
        
    ##########################################################################
    
    def test_rsi_df_column(
        self,
        engine: ClassificationFE,
    ):
        "Test the rolling method"
        rolling_columns = [
            "rsi_3",
            "rsi_6",
            "rsi_11",
        ]
        
        df = engine.delete_unused_columns(
            df = self.df,
            target_column = "Open",
            control_column = "Date",
        )
        
        engine.set_n_window([3, 6, 11])
        
        df = engine.rsi_df(df)
        
        df_column = list(df.columns)
        
        # Test shape
        assert df.shape[1] == 5
        
        # Test column name
        for lag in rolling_columns:
            assert lag in df_column
            df_column.remove(lag)
        
        # Must be only 2 element in list, 
        # generated column fit the parameter
        assert len(df_column) == 2
    
    ##########################################################################
    
##############################################################################
