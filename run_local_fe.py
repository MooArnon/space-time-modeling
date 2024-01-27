#--------#
# Import #
#----------------------------------------------------------------------------#

import os

from space_time_modeling.fe import fe_engine

#-----#
# Use #
#----------------------------------------------------------------------------#
if __name__ == "__main__":
    
    #-----------#
    # Attribute #
    #------------------------------------------------------------------------#
    
    df_path = os.path.join("tests", "df_test.csv")
    control_column = "Date"
    target_column = "Open"
    label = "signal"
    
    #---------#
    # Get eda #
    #------------------------------------------------------------------------#
    
    # Initiate engine
    fe = fe_engine(
        df = df_path,
        control_column = control_column,
        target_column = target_column,
        label = label,
        engine = "classification",
    )
    
    # label data
    df = fe.add_label(
        df = fe.df, 
        target_column = "Open",
    )
    
    
    
    print(df.columns)
    
    df.to_csv("result/preprocessed.csv")
    
    
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
