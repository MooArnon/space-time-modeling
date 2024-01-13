#--------#
# Import #
#----------------------------------------------------------------------------#

import os
import pandas

from space_time_modeling.utilities import read_df

#-------#
# Tests #
#----------------------------------------------------------------------------#

class TestUtils:
    
    def test_red_df_type(self):
        """ Need to return pandas data frame"""
        
        df = read_df(
            os.path.join("tests", "df_test.csv")
        )
        
        assert type(df) == pandas.DataFrame
        
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
