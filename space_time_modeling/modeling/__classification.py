#--------#
# Import #
#----------------------------------------------------------------------------#

import os
from typing import Union

from pandas.core.api import DataFrame, Series
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import uniform, randint
import xgboost as xgb

from .__base import BaseModel

#---------#
# Classes #
#----------------------------------------------------------------------------#

class ClassificationModel(BaseModel):
    name = 'modeling-instance'
    def __init__(
            self, 
            label_column: str, 
            feature_column: list[str], 
            result_path: str,
            test_size: float = 0.2,
            n_iter: int = 30,
            cv: int = 5,
            xgboost_params_dict: dict = None,
    ) -> None:
        super().__init__(
            label_column, 
            feature_column, 
            result_path, 
            test_size,
        )
        
        # Set attribute for tuning
        self.set_n_iter(n_iter)
        self.set_cv(cv)
        
        # Set attribute for model
        ## XGBoost
        self.set_xgboost_params_dict(xgboost_params_dict)
        
        os.mkdir(result_path)
        
    #------------#
    # Properties #
    #------------------------------------------------------------------------#
    # Tuning #
    #--------#
    
    @property
    def n_iter(self) -> int:
        """ Number of search """
        return self.__n_iter
    
    #------------------------------------------------------------------------#
    
    def set_n_iter(self, n_iter: int) -> None:
        """Number of search

        Parameters
        ----------
        n_iter : int
            Number of search
        """
        self.__n_iter = n_iter
    
    #------------------------------------------------------------------------#
    
    @property
    def cv(self) -> int:
        """ Cross validation at search """
        return self.__cv
    
    #------------------------------------------------------------------------#
    
    def set_cv(self, cv: int) -> None:
        """Number of search

        Parameters
        ----------
        n_iter : int
            Number of search
        """
        self.__cv = cv
    
    #------------------------------------------------------------------------#
    # Model #
    #-------#
    
    @property
    def xgboost_params_dict(self):
        """ parameter of XGBoost model for random search """
        return self.__xgboost_params_dict

    #------------------------------------------------------------------------#
    
    def set_xgboost_params_dict(
            self, 
            xgboost_params_dict: dict = None
    ) -> None:
        """Set parameter of XGBoost model for random search

        Parameters
        ----------
        xgboost_params_dict : dict, optional
            Parameters, by default None
        """
        if xgboost_params_dict is None:
            xgboost_params_dict = {
                'learning_rate': uniform(0.001, 0.9),
                'n_estimators': randint(10, 1000),
                'max_depth': randint(3, 60),
                'subsample': uniform(0.1, 0.9),
                'colsample_bytree': uniform(0.1, 0.9),
                'gamma': uniform(0, 0.9)
            }
        self.__xgboost_params_dict = xgboost_params_dict
            
    #----------#
    # Modeling #
    #------------------------------------------------------------------------#
    
    def modeling(
            self,
            df : Union[DataFrame, str],
            model_name_list: list[str] = ['xgboost'], 
    ) -> None:
        """Tran and save model in model_name_list

        Parameters
        ----------
        df : Union[DataFrame, str]
            Could be either path to data frame or data frame itself.
        model_name_list : list[str]
            List of model name
            `xgboost`
        """
        x_train, x_test, y_train, y_test = self.prepare(
            self.read_df(df)
        )
        
        # Iterate over model_name_list
        for model_name in model_name_list:
            
            # Get train function for each model, using name
            train_function = getattr(self, model_name)
            
            # Check if the training function is found
            if train_function is None or not callable(train_function):
                print(f"Warning: Model '{model_name}' not found.")
                continue
            
            # Execute training function
            train_function(
                x_train, 
                x_test,
                y_train,
                y_test,
            )
        
    #------------------------------------------------------------------------#
    # Model #
    #---------#
    # XGBoost #
    #---------#
    
    def xgboost(
            self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
    ) -> xgb.XGBClassifier:
        
        # Print line separate
        print("\n", "-"*72)
        print("Tuning XGBoost")
        
        # Init model
        model = xgb.XGBClassifier(objective='binary:logistic',random_state=42)
        
        # Get random search
        tuned_model = self.random_search(
            model,
            self.xgboost_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        
        # Save model
        self.save_xgboost(tuned_model)
        
        return tuned_model
    #------------------------------------------------------------------------#
    
    def save_xgboost(self, model: xgb.XGBClassifier) -> None:
        """Save XGBoost model

        Parameters
        ----------
        model : xgb.XGBClassifier
            model
        """
        path = os.path.join(self.result_path, 'xgboost.xgb')
        
        model.save_model(path)
        
    #------------------------------------------------------------------------#
    
    @staticmethod
    def load_xgboost(model_path: str) -> xgb.XGBModel:
        """Load xgboost model

        Parameters
        ----------
        model_path : str
            path of `.xbg` model

        Returns
        -------
        xgb.XGBModel
            xgboost model
        """
        # Initiate xbg instance
        model = xgb.Booster()
        model.load_model(model_path)
        
        return model
        
    #-----------#
    # Utilities #
    #------------------------------------------------------------------------#
    
    def random_search(
            self,
            model: any, 
            param_dict: dict,
            x_train: Series,
            x_test: Series,
            y_train: DataFrame,
            y_test: DataFrame,
    ) -> any:
        """Fine tune by random search over the params dict

        Parameters
        ----------
        model : any
            Input model
        param_dist : dict
            Dictionary of parameters
        x_train : Series
            Train feature
        x_test : Series
            Test feature
        y_train : DataFrame
            Train label
        y_test : DataFrame
            Test label

        Returns
        -------
        any
            Fine tuned model
        """
        random_search = RandomizedSearchCV(
            model, 
            param_distributions=param_dict, 
            n_iter=self.n_iter, 
            cv=self.cv, 
            random_state=42,
            verbose = 2,
        )
        # Fit the model to the training data
        random_search.fit(x_train, y_train)

        # Print the best hyperparameters
        print("Best Hyperparameters:", random_search.best_params_)
        
        best_model = random_search.best_estimator_

        # Make predictions on the test set
        y_pred = best_model.predict(x_test)

        # Evaluate the model
        accuracy = classification_report(y_test, y_pred)
        print("Classification Report:\n", accuracy)
        
        return best_model
    
    #------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
