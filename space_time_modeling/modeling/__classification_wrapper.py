##########
# Import #
##############################################################################

from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from .__base import BaseWrapper
from .custom_metric import custom_metric

###########
# Wrapper #
##############################################################################

class ClassifierWrapper(BaseWrapper):
    """Wrapper for classification

    Load model into wrapper with an useful method and properties.
    This class was implemented to serve the feature store feature.
    
    Initiate
    --------
    ```python
    from space_time_modeling.utilities import load_instance
    model_wrapper: ClassifierWrapper = load_instance(path)
    ```
    
    Call
    ----
    Call function activates at instance call. Will call predict.  
    Example, `pred = model_wrapper(data_list)`
    x: list
        List of data
    clean: bool = True
        If true, clean a data
    
    Notes
    -----
    The feeding data would different by the type of model.
    `catboost`: receive `list[float]`
    others model: receive `list[list[float]]`
    """
    def __init__(
        self, 
        model: object, 
        name: str, 
        feature: list[str] = None,
        preprocessing_pipeline: object = None,
    ) -> None:
        super(ClassifierWrapper, self).__init__(feature, preprocessing_pipeline)
        self.set_model(model)
        self.set_name(name)
        self.set_version()

    ##############
    # Properties #
    ##########################################################################
    # Model #
    #########
    
    def set_model(self, model: object) -> None:
        self.__model = model
    
    ##########################################################################
    
    @property
    def model(self) -> object:
        return self.__model

    ##########################################################################
    
    def set_name(self, name: str) -> None:
        self.__name = name 
    
    ##########################################################################
    
    @property
    def name(self) -> str:
        return self.__name
    
    ##########################################################################
    
    def set_version(self, name: str = None) -> None:
        now = datetime.now()
        self.__version = f"{now.strftime('%Y%m%d.%H%M%S')}"
        if name:
            self.__version = f"{self.__version}.{name}"
    
    ##########################################################################
    
    @property
    def version(self) -> str:
        return self.__version
    
    ###########
    # Methods #
    ##########################################################################
    
    def __call__(self, x: Union[list, pd.DataFrame], clean: bool = True): 
        x = self.preprocessing_pipeline.transform_df(x)[self.feature].iloc[[-1]]
        pred = self.model.predict(x)
        if clean:
            pred = self.extract_value(pred)
            pred = int(pred)
        return pred
    
    ##########################################################################
    
    @staticmethod
    def extract_value(nested_list: Union[list, np.ndarray]):
        if isinstance(nested_list, np.ndarray):
            nested_list = nested_list.tolist()
        while isinstance(nested_list, list):
            nested_list = nested_list[-1]
        return nested_list

    ##########################################################################
    
    def evaluate(self, x_test: pd.DataFrame) -> dict:
        """Evaluate the model using the stored evaluation metric.

        Parameters
        ----------
        x_test : pd.DataFrame
            Test data features.

        Returns
        -------
        dict
            A dictionary containing evaluation metrics.
        """
        # Add label at data
        label = self.preprocessing_pipeline.add_label(
            x_test,
            self.preprocessing_pipeline.target_column
        )[self.preprocessing_pipeline.label]
        
        # Preprocess the test data
        x_test_processed = self.preprocessing_pipeline.transform_df(x_test)\
            [self.feature]
        
        # Predict on the test data
        y_pred = self.model.predict(x_test_processed)
        label = label[-y_pred.shape[0]:]

        # Evaluate using the custom metric if provided
        evaluation_result = custom_metric(label, y_pred)
        
        return evaluation_result

##############################################################################
