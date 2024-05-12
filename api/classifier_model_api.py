##########
# Import #
##############################################################################

import os
import joblib
from typing import Union

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import logging

from space_time_modeling.modeling import ClassifierWrapper
from space_time_modeling.utilities import load_instance

###########
# Statics #
##############################################################################

app = Flask(__name__)

# Load the pre-trained machine learning model
path = joblib.load('model.pkl')

# Add logger object
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
)

#######
# API #
##############################################################################

@app.route('/predict', methods=['POST'])
def predict(data: Union[dict, pd.DataFrame], model_type: str):
    """Predict fuction

    Parameters
    ----------
    data : Union[dict, pd.DataFrame]
        If dict, convert to dataframe
    model_type : str
        Assigned for the differeces data prepare logic

    Raises
    ------
    SystemError
        If any error occure
    """
    
    # Load data into dataframe
    if isinstance(data, dict):
        df = pd.DataFrame(data = data)
    
    # Load model warpper
    model_wrapper: ClassifierWrapper = load_instance(path)
    
    # Ensure that the latest row was feed to model
    last_row = df[model_wrapper.feature].iloc[-1].to_list()
    
    # Try and if error raise SystemError
    try:
        
        # Spacial logic for catboost
        if model_type == 'catboost':
            pred = model_wrapper(last_row)
        else: 
            pred = model_wrapper([last_row])
        logger.info(f"Predicted: {pred}")
    except SystemError:
        logger.error(f"Error at {model_type}")
        raise SystemError(f"Error at {model_type}")
    
    print(pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
