#--------#
# Import #
#----------------------------------------------------------------------------#

import pandas
from typing import Union

from .classification_fe import ClassificationFE


#----------#
# Function #
#----------------------------------------------------------------------------#

def engine(
        df: Union[str, pandas.DataFrame],
        control_column: str,
        target_column: str,
        engine: str,
        **kwargs,
) -> ClassificationFE:
    
    # Check engine
    if engine == "classification":
        fe = ClassificationFE(
            df = df,
            control_column = control_column,
            target_column = target_column,
            **kwargs,
        )
    
    return fe