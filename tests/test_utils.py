##########
# Import #
##############################################################################

import os
import shutil

import pandas
import pytest

from space_time_modeling.utilities import (
    read_df, 
    serialize_instance, 
    load_instance
)

###########
# Statics #
##############################################################################

class MockUpInstance:
    """Use only for test the instance importation and exportation."""
    return_1 = 1
    return_a = 'a'
    name = "mockup_instance"
    
    def plus(a: int, b: int) -> int:
        return a+b

#########
# Tests #
##############################################################################

@pytest.fixture
def path_2_df() -> os.path:
    """Path to df.csv at tests directory"""
    return os.path.join("tests", "df_test.csv")
    
class TestUtils:
    
    def test_read_df_type(self, path_2_df: os.path):
        """ Need to return pandas data frame """
        
        df = read_df(path_2_df)
        
        assert type(df) == pandas.DataFrame
        
    ##########################################################################
    
    def test_serialize(self):
        """ Export the instance """
        try:
            serialize_instance(
                MockUpInstance, 
                os.path.join("test_mockup")
            )
            x = True
            
        except:
            x = False
            pass
        
        assert x is True
        
    ##########################################################################
    
    def test_load_instance(self):
        for filename in os.listdir("test_mockup"):
            if filename.startswith("mockup_instance"):
                mockup_name = filename
        
        obj:MockUpInstance = load_instance(
            os.path.join("test_mockup",mockup_name)
        )
        
        assert obj.return_1 == 1
        assert obj.return_a == 'a'
        assert obj.plus(1, 2) == 3
    
    ##########################################################################
    
    def test_clear_mockup(self):
        shutil.rmtree("test_mockup")

##############################################################################
