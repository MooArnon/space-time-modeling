##########
# Import #
##############################################################################

import os

import pandas as pd
import pickle

import datetime

#############
# functions #
##############################################################################

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

##############################################################################

def now_formatted_timestamp() -> str:
    """Get formatted timestamp

    Returns
    -------
    str
        Formatted timestamp
    """
    # Create timestamp
    time_stamp = datetime.datetime.now()
    return time_stamp.strftime("%Y%m%d_%H%M%S")

##############################################################################

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
    time_stamp_format = now_formatted_timestamp()
    
    # Combine name of dir and time tag
    dir_name = dir_name + "_" + str(time_stamp_format)
    
    # Create directory
    os.mkdir(dir_name)
    
    return dir_name

##############################################################################

def serialize_instance(
        instance: object, 
        path: str, 
        add_time: bool = True    
) -> None:
    """Serialize and export instance  

    Parameters
    ----------
    instance : object
        Target exporting instance
    path : str
        Path to export
    add_time: bool = True    
        If true, add timestamp as suffix
    """
    time_stamp_format = now_formatted_timestamp()
    
    if not os.path.exists(path):
        os.makedirs(path)
    if add_time:
        path = os.path.join(path, f"{instance.name}_{time_stamp_format}.pkl")
    else:
        path = os.path.join(path, f"{instance.name}.pkl")
    
    with open(path, "wb") as f:
        pickle.dump(instance, f)
    
##############################################################################

def load_instance(path: str) -> object:
    """Load instance

    Parameters
    ----------
    path : str
        Part to .pkl file

    Returns
    -------
    object
        Loaded instance
    """
    with open(path, "rb") as f:
        return pickle.load(f)
    
##############################################################################
