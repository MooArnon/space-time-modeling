##########
# Import #
##############################################################################

import os

import boto3
from botocore.exceptions import NoCredentialsError
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
) -> str:
    """Serialize and export instance  

    Parameters
    ----------
    instance : object
        Target exporting instance
    path : str
        Path to export
    add_time: bool = True    
        If true, add timestamp as suffix
    
    Return 
    ------
    str
        Relative path of file
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
    
    return path
    
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

def clear_s3_prefix(bucket, prefix):
    """Delete all files under a specific prefix in an S3 bucket, pass if no files found."""
    s3 = boto3.client('s3')
    
    try:
        # List all the objects under the given prefix
        objects_to_delete = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if 'Contents' in objects_to_delete:
            # Create a list of object identifiers (Key, VersionId) for deletion
            delete_keys = [{'Key': obj['Key']} for obj in objects_to_delete['Contents']]

            # Perform the delete operation
            s3.delete_objects(Bucket=bucket, Delete={'Objects': delete_keys})
            print(f"All files under prefix {prefix} have been deleted from {bucket}")
        else:
            print(f"No files found under prefix {prefix} in bucket {bucket}. Continuing with upload.")
            pass  # Just pass if no files are found
        return True

    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    
##############################################################################

def upload_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket.

    Parameter
    ---------
    file_name: str
        File to upload
    bucket: str
        Bucket to upload to
    object_name: str
        S3 object name. If not specified then file_name is used
    
    Return
    ------
    bool
        True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client('s3')

    try:
        # Upload the file
        s3_client.upload_file(file_name, bucket, object_name)
        print(f"File {file_name} uploaded successfully to {bucket}/{object_name}")
        return True
    except FileNotFoundError:
        print(f"The file {file_name} was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

##############################################################################

def clear_and_push_to_s3(file_name, bucket, prefix):
    """Clear a specific prefix and then upload a file to S3.

    Parameter
    ---------
    file_name: 
        The local file to upload
    bucket: 
        The name of the S3 bucket
    prefix: 
        The prefix (folder path) to clear and then upload the file to
    """
    # Step 1: Clear the target prefix
    cleared = clear_s3_prefix(bucket, prefix)
    
    # Step 2: Upload the new file to the cleared prefix
    if cleared:
        object_name = prefix + file_name.split('/')[-1]  
        upload_to_s3(file_name, bucket, object_name)

##############################################################################
