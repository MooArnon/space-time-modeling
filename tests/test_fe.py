#--------#
# Import #
#----------------------------------------------------------------------------#

import os
import pandas

import pytest

from space_time_modeling.fe.classification_fe import ClassificationFE

#-------#
# Tests #
#----------------------------------------------------------------------------#

class TestClassificationFE:
    
    @pytest.fixture
    def engine(self):
        df_path = os.path.join("tests", "df_test.csv")
        return ClassificationFE(
            df = df_path,
            control_column = "Date",
            target_column = "Open",
        )
    
    #---------------------#
    # Feature engineering #
    #------------------------------------------------------------------------#
    # Return type #
    #-------------#
    
    def test_lag_df_type(self, engine: ClassificationFE):
        """ Need to return pandas data frame"""
        
        df = engine.lag_df(engine.df)
        
        assert type(df) == pandas.DataFrame
    
    #------------------------------------------------------------------------#
    
    def test_rolling_df_type(self, engine: ClassificationFE):
        """ Need to return pandas data frame"""
        
        df = engine.rolling_df(engine.df)
        
        assert type(df) == pandas.DataFrame
    
    #------------------------------------------------------------------------#
    
    def test_del_df_type(self, engine: ClassificationFE):
        """ Need to return pandas data frame"""
        
        df = engine.delete_unused_columns(
            df = engine.df,
            target_column = "Open",
            control_column = "Date",
        )
        
        assert type(df) == pandas.DataFrame
        
    #------------------------------------------------------------------------#
    # Return result #
    #---------------#
    
    def test_del_df_column(self, engine: ClassificationFE):
        """ Need to return pandas data frame"""
        
        df = engine.delete_unused_columns(
            df = engine.df,
            target_column = "Open",
            control_column = "Date",
        )
        
        assert df.shape[1] == 2
    
    #------------------------------------------------------------------------#
    
    def test_lag_df_column(self, engine: ClassificationFE):
        """ Need to return pandas data frame"""
        
        df = engine.delete_unused_columns(
            df = engine.df,
            target_column = "Open",
            control_column = "Date",
        )
        
        engine.set_n_lag(5)
        
        df = engine.lag_df(
            df
        )
        
        print(df.columns)
        
        assert df.shape[1] == 7
    
    #------------------------------------------------------------------------#
    
#----------------------------------------------------------------------------#
