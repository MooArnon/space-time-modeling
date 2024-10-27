##########
# Import #
##############################################################################

from pandas.core.api import DataFrame as DataFrame

from .__base import BaseModel

# Handel the import error from seperate 2 reqs.
from .__classification import ClassificationModel
try:
    from .__classification import ClassificationModel
except ImportError:
    ClassificationModel = None

try:
    from .__deep_classification import DeepClassificationModel
except ImportError:
    DeepClassificationModel = None


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
    max_trials: int = 10
        Number of trial at Gaussian search
    executions_per_trial: int = 1
        Number of execution per epoch, to reduce SD of accuracy
    epoch_per_trial = 20
        Number of epoch per trial
    early_stop_min_delta=0.0001
        The threshold to check at search stop
    early_stop_patience=20
        Number of minimum threshold that would end a trial
    early_stop_verbose=1
        Early stopping
    """
    print(ClassificationModel)
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

if __name__ == "main":
    pass
