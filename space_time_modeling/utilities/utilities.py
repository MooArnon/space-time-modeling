#--------#
# Import #
#---------------------------------------------------------------------#

import os

import pandas as pd
import datetime

#-----------#
# functions #
#---------------------------------------------------------------------#

def read_df(file_path: str) -> pd.DataFrame:
    """Read source file and convert to pandas data frame

    Parameters
    ----------
    file_path : str
        Path of tabular file

    Returns
    -------
    pd.DataFrame
        Data frame result
    
    Raise
    -----
    ValueError
        when file type is not appropriate
    """
    # Split file name to get the file type
    file_type = file_path.split(sep=".")[-1]
    
    # Check if file is csv
    if file_type == "csv":
        df = pd.read_csv(file_path)
        
    # Check if file is excel
    elif file_type == "xlsx":
        df = pd.read_excel(file_path)
    
    # If not all, raise value error
    else:
        raise ValueError(
            f"""
                File type {file_type} has not support yet, 
                contact the developer
            """
        )
    
    return df

#----------------------------------------------------------------------------#

def create_dir_w_timestamp(dir_name: str) -> str:
    """Create directory with timestamp tag

    Parameters
    ----------
    dir_name : str
        Name of directory
        
    Returns
    -------
    str
        name of directory
    """
    # Create timestamp
    time_stamp = datetime.datetime.now()
    time_stamp_format = time_stamp.strftime("%Y%m%d_%H%M%S")
    
    # Combine name of dir and time tag
    dir_name = dir_name + "_" + str(time_stamp_format)
    
    # Create directory
    os.mkdir(dir_name)
    
    return dir_name

#----------------------------------------------------------------------------#
