##########
# Import #
##############################################################################

from datetime import datetime
import os
from typing import Union

from pandas.core.api import DataFrame as DataFrame
from pandas.core.api import Series as Series
from sklearn.model_selection import train_test_split

from ..utilities import read_df
from space_time_modeling.fe import ClassificationFE 

###########
# Classes #
##############################################################################

class BaseModel:
    """Base class for modeling
    """ 
    def __init__(
            self, 
            label_column: str = None, 
            feature_column: list[str] = None,
            result_path: str = None,
            test_size: float = 0.2,
            push_to_s3: bool = False,
            aws_key: str = os.getenv("AWS_KEY"),
            aws_secret: str = os.getenv("AWS_SECRET"),
            aws_s3_bucket: str = None,
            aws_s3_prefix: str = None,
    ) -> None:
        """Initiate BaseModel

        Parameters
        ----------
        df : Union[str, DataFrame]
            data frame can be 2 types,
            `str` as path of data frame
            `DataFrame` as data frame itself
        label_column : str
            String of target column
        feature_column : str
            String of feature column
        test_size : float = 0.2
            Proportion of test
        """
        # Set main attribute
        self.set_label_column(label_column)
        self.set_feature_column(feature_column)
        
        # Set preparing attribute
        self.set_test_size(test_size)
        
        # Path
        now = datetime.now()
        formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
        self.result_path = f"{result_path}_{formatted_datetime}"
        
        self.push_to_s3 = push_to_s3
        self.aws_key = aws_key
        self.aws_secret = aws_secret
        self.aws_s3_bucket = aws_s3_bucket
        self.aws_s3_prefix = aws_s3_prefix

    ##########################################################################
    
    def read_df(self, df: Union[str, DataFrame]):
        """set df attribute

        Parameters
        ----------
        df : Union[str, DataFrame]
            data frame can be 2 types,
            `str` as path of data frame
            `DataFrame` as data frame itself

        Raises
        ------
        ValueError
            if type is not string and DataFrame
        """
        # Check type of df 
        if isinstance(df, str):
            df = read_df(file_path = df)
        elif isinstance(df, DataFrame):
            df = df
        else:
            raise ValueError(f"{type(df)} is not supported")
        
        return df
    
    ##########################################################################
    
    @property
    def label_column(self) -> str:
        """ Label column """
        return self.__label_column
    
    ##########################################################################
    
    def set_label_column(self, label_column: str) -> None:
        """Set label_column attribute

        Parameters
        ----------
        label_column : str
            String of target column
        """
        self.__label_column = label_column
    
    ##########################################################################
    
    @property
    def feature_column(self) -> list[str]:
        """ Feature column """
        return self.__feature_column
    
    ##########################################################################
    
    def set_feature_column(self, feature_column: list[str]) -> None:
        """Set feature_column attribute

        Parameters
        ----------
        feature_column : list[str]
            list[str] of feature column
        """
        if self.label_column in feature_column:
            feature_column.remove(self.label_column)
        self.__feature_column = feature_column
        
    ##########################################################################
    # preparing #
    #############
    
    @property
    def test_size(self) -> float:
        """ Size of test set """
        return self.__test_size
    
    ##########################################################################
    
    def set_test_size(self, test_size: float) -> None:
        """Set size of test in ratio

        Parameters
        ----------
        test_size : float
            Size of test
        """
        self.__test_size = test_size
    
    ##########
    # Method #
    ##########################################################################
    # Feature prepare #
    ###################
    
    def prepare(
            self, 
            df: DataFrame,
    ) -> tuple[DataFrame, DataFrame, Series, Series]:
        """Get prepared data for machine learning

        Parameters
        ----------
        df : DataFrame
            FEd data frame

        Returns
        -------
        tuple[DataFrame, DataFrame, Series, Series]
            x_train, x_test, y_train, y_test
        """
        label, feature = self.split_feature_label(df)
        x_train, x_test, y_train, y_test = self.split_test_train(
            label, 
            feature,
        )
        return x_train, x_test, y_train, y_test
    
    ##########################################################################
    
    def split_feature_label(self, df: DataFrame) -> tuple[Series, DataFrame]:
        """Split feature and label from data frame

        Parameters
        ----------
        df : DataFrame
            Target data frame, contains only nesssary column

        Returns
        -------
        tuple[Series, DataFrame]
            tuple[label, feature]
        """
        if self.label_column in self.feature_column:
            self.feature_column.remove(self.label_column)
            
        # Split
        label = df[self.label_column]
        feature = df.drop(columns=self.label_column)
        
        return label, feature
    
    ##########################################################################
    
    def split_test_train(
            self, 
            label: Series,
            feature: DataFrame,
    ) -> tuple[DataFrame, DataFrame, Series, Series]:
        """Split test and train, followed by the elf.test_size

        Parameters
        ----------
        label : Series
            Series of label
        feature : DataFrame
            Data frame of feature

        Returns
        -------
        tuple[DataFrame, DataFrame, Series, Series]
            x_train, x_test, y_train, y_test
        """
        if isinstance(self.test_size, float):
            test_size = int(len(feature) * self.test_size)
        elif isinstance(self.test_size, int):
            test_size = self.test_size
        train_size = len(feature) - test_size

        x_train = feature.iloc[:train_size]
        x_test = feature.iloc[train_size:]
        y_train = label.iloc[:train_size]
        y_test = label.iloc[train_size:]
        
        return x_train, x_test, y_train, y_test
    
    ##########################################################################

##############################################################################

class BaseWrapper():
    def __init__(
        self, 
        feature: list[str] = None, 
        preprocessing_pipeline: object = None
    ) -> None:
        super(BaseWrapper, self).__init__()
        
        if feature:
            self.set_feature(feature)
        if preprocessing_pipeline:
            self.set_preprocessing_pipeline(preprocessing_pipeline)
    
    ##############
    # Properties #
    ##########################################################################
    # Feature #
    ###########
    
    def set_feature(self, feature: list[str]) -> None:
        self.__feature = feature
    
    ##########################################################################
    
    @property
    def feature(self) -> list[str]:
        return self.__feature
    
    ##########################################################################
    
    def set_preprocessing_pipeline(self, preprocessing_pipeline: object) -> None:
        self.__preprocessing_pipeline = preprocessing_pipeline
    
    ##########################################################################
    
    @property
    def preprocessing_pipeline(self) -> ClassificationFE:
        return self.__preprocessing_pipeline
    
    ##########################################################################
    
##############################################################################
