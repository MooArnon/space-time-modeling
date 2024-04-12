##########
# Import #
##############################################################################

from typing import Union

from pandas.core.api import DataFrame as DataFrame

from .__base import BaseModel
from .__classification import ClassificationModel
from .__deep_classification import DeepClassificationModel

############
# Function #
##############################################################################

def modeling_engine(
        engine: str,
        label_column: str,
        feature_column: list[str],
        result_path: str,
        **kwargs,
) -> BaseModel:
    """Create instance of modeling

    Parameters
    ==========
    engine : str
        Name of engine,
        `classification`
    df : Union[str, DataFrame]
        Target FEd data frame
    label_column : str
        NAme of label column
    feature_column : list[str]
        List of feature column
    result_path : str
        Path to result

    Returns
    =======
    BaseModel
        Model

    Raises
    ======
    ValueError
        If engine is not exists 
        
    Additional Parameters
    =====================
    test_size: float = 0.2
        Size of test data
    n_iter: int = 30
        Number of searched
    cv: int = 5
        Number of cross validation
    
    deep_classification: DeepClassificationModel
    --------------------------------------------
    mode: str 
        mode of search
        ,by default 'random_search'
    
    Search default parameters
    ------------------------
    `xgboost`: {
        'learning_rate': uniform(0.001, 0.9),
        'n_estimators': randint(10, 1000),
        'max_depth': randint(3, 60),
        'subsample': uniform(0.1, 0.9),
        'colsample_bytree': uniform(0.1, 0.9),
        'gamma': uniform(0, 0.9)
    }
    `dnn`: {
        'lr': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'epochs':[5, 10, 15, 30, 60, 100],
        'criterion':[nn.BCELoss(), nn.HuberLoss()],
        'module__hidden_layers': [
            [8],
            [16],
            [32],
            [64],
            [8, 16, 8],
            [16, 32, 16],
            [8, 16, 16, 8],
            [16, 32, 32, 16],
            [8, 16, 32, 16, 8],
        ],
        'module__dropout': [0.1, 0.15, 0.2, 0.25]
    }
    `lstm`:
    """
    if engine == 'classification':
        modeling = ClassificationModel(
            label_column,
            feature_column,
            result_path,
            **kwargs,
        )
    elif engine == 'deep_classification':
        modeling = DeepClassificationModel(
            label_column,
            feature_column,
            result_path,
            **kwargs
        )
    
    else: 
        raise ValueError(f"{engine} is not supported")
    
    return modeling

##############################################################################
