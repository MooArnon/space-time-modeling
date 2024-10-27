##########
# Import #
##############################################################################

from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from .__base import BaseWrapper

#################
# Custom metric #
##############################################################################

def custom_metric(y_true, y_pred, alpha=0.5, beta1=1):
    
    # Convert predictions to class labels (0 or 1) based on a threshold
    y_pred_labels = (y_pred > 0.5).astype(float)

    # Calculate precision and recall for buy signals
    true_positives = np.sum(y_true * y_pred_labels)
    predicted_positives = np.sum(y_pred_labels)
    actual_positives = np.sum(y_true)

    precision_buy = true_positives \
        / (predicted_positives + np.finfo(float).eps)
    recall_buy = true_positives \
        / (actual_positives + np.finfo(float).eps)

    f1_buy = 2 * (precision_buy * recall_buy) \
        / (precision_buy + recall_buy + np.finfo(float).eps)

    # Calculate precision and recall for sell signals
    true_negatives = np.sum((1 - y_true) * (1 - y_pred_labels))
    predicted_negatives = np.sum(1 - y_pred_labels)
    actual_negatives = np.sum(1 - y_true)

    precision_sell = true_negatives \
        / (predicted_negatives + np.finfo(float).eps)
    recall_sell = true_negatives \
        / (actual_negatives + np.finfo(float).eps)

    f1_sell = 2 * (
        precision_sell * recall_sell) \
            / (precision_sell + recall_sell + np.finfo(float).eps
    )

    # PRB (Precision-Recall Balance)
    prb = alpha * f1_buy + (1 - alpha) * f1_sell

    # Composite Financial Performance Index (CFPI)
    cfpi = beta1 * prb

    return cfpi

##############################################################################

def profit_factor_metric(
        y_true: pd.DataFrame, 
        y_pred: list, 
        price_data: pd.DataFrame=None, 
        weights: dict=None,
) -> float:
    """Proceed financial evaluation

    Parameters
    ----------
    y_true : pd.dataframe
        Dataframe of true labels
    y_pred : list
        List of predicted labels
    price_data : pd.dataframe, optional
        Dataframe of price, by default None
    weights : dict, optional
        Weight for each metrics, by default None
        if None, using 
        weights = {
            'pf': 0.25,         
            'sr': 0.25,         
            'win_rate': 0.5,   
        }

    Returns
    -------
    float
        Weighted combine metrics

    Raises
    ------
    ValueError
        if no price data assigned
    """
    if price_data is None:
        raise ValueError(
            "price_data is required for financial metrics calculation."
        )
    
    # Set default weights if none provided
    if weights is None:
        weights = {
            'pf': 0.25,         
            'sr': 0.25,         
            'win_rate': 0.5,   
        }
    
    # Align price_data with y_true indices
    sliced_price_data = price_data.loc[y_true.index]
    
    # Calculate gains and losses
    short_gains = ((y_pred == 0) & (y_true == 0)) \
        * abs(sliced_price_data.shift(-1) - sliced_price_data)
    long_gains = ((y_pred == 1) & (y_true == 1)) \
        * (sliced_price_data.shift(-1) - sliced_price_data)
    short_losses = ((y_pred == 0) & (y_true == 1)) \
        * abs(sliced_price_data.shift(-1) - sliced_price_data)
    long_losses = ((y_pred == 1) & (y_true == 0)) \
        * abs(sliced_price_data.shift(-1) - sliced_price_data)

    # Fill NaN from shifting prices
    short_gains, long_gains = short_gains.fillna(0), long_gains.fillna(0)
    short_losses, long_losses = short_losses.fillna(0), long_losses.fillna(0)
    
    # Aggregate gains and losses
    total_gains = short_gains.sum() + long_gains.sum()
    total_losses = short_losses.sum() + long_losses.sum()
    
    # Profit Factor (PF)
    profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')
    
    # Sharpe Ratio (SR)
    returns = pd.concat([short_gains, long_gains]) - pd.concat([short_losses, long_losses])
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = mean_return / std_return if std_return != 0 else 0
    
    # Win Rate
    win_trades = (short_gains > 0).sum() + (long_gains > 0).sum()
    total_trades = len(y_pred)
    win_rate = win_trades / total_trades if total_trades != 0 else 0

    # Combine metrics into a single score using weighted sum
    combined_score = (
        weights['pf'] * profit_factor +
        weights['sr'] * sharpe_ratio +
        weights['win_rate'] * win_rate 
    )

    return combined_score

##############################################################################

def total_pnl(y_true, y_pred, price_data=None):
    if price_data is None:
        raise ValueError("price_data is required for P&L calculation.")
    
    # Align price_data with y_true indices
    sliced_price_data = price_data.loc[y_true.index]
    
    # Initialize total P&L
    total_pnl_value = 0

    # Calculate gains and losses for SHORT and LONG positions
    short_gains = ((y_pred == 0) & (y_true == 0)) * abs(sliced_price_data.shift(-1) - sliced_price_data)
    long_gains = ((y_pred == 1) & (y_true == 1)) * (sliced_price_data.shift(-1) - sliced_price_data)
    short_losses = ((y_pred == 0) & (y_true == 1)) * abs(sliced_price_data.shift(-1) - sliced_price_data)
    long_losses = ((y_pred == 1) & (y_true == 0)) * abs(sliced_price_data.shift(-1) - sliced_price_data)

    # Fill NaN from shifting prices
    short_gains, long_gains = short_gains.fillna(0), long_gains.fillna(0)
    short_losses, long_losses = short_losses.fillna(0), long_losses.fillna(0)

    # Aggregate gains and losses
    total_gains = short_gains.sum() + long_gains.sum()
    total_losses = short_losses.sum() + long_losses.sum()
    
    # Calculate total P&L
    total_pnl_value = total_gains - total_losses

    return total_pnl_value

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
