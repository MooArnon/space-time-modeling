##########
# Import #
##############################################################################

import os
import pickle
from typing import Union

from catboost import CatBoostClassifier, Pool
import pandas as pd
from pandas.core.api import DataFrame, Series
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import uniform, randint
import xgboost as xgb

from .__base import BaseModel

###########
# Classes #
##############################################################################

class ClassificationModel(BaseModel):
    name = 'modeling-instance'
    def __init__(
            self, 
            label_column: str = None, 
            feature_column: list[str] = None, 
            result_path: str = None,
            test_size: float = 0.2,
            n_iter: int = 15,
            cv: int = 5,
            xgboost_params_dict: dict = None,
            catboost_params_dict: dict = None,
            svc_params_dict: dict = None,
            random_forest_params_dict: dict = None,
            logistic_regression_params_dict: dict = None,
            knn_params_dict: dict = None,
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
        self.set_xgboost_params_dict(xgboost_params_dict)
        self.set_catboost_params_dict(catboost_params_dict)
        self.set_svc_params_dict(svc_params_dict)
        self.set_random_forest_params_dict(random_forest_params_dict)
        self.set_logistic_regression_params_dict(logistic_regression_params_dict)
        self.set_knn_params_dict(knn_params_dict)
        
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        
    ##############
    # Properties #
    ##########################################################################
    # Tuning #
    ##########
    
    @property
    def n_iter(self) -> int:
        """ Number of search """
        return self.__n_iter
    
    ##########################################################################
    
    def set_n_iter(self, n_iter: int) -> None:
        """Number of search

        Parameters
        ----------
        n_iter : int
            Number of search
        """
        self.__n_iter = n_iter
    
    ##########################################################################
    
    @property
    def cv(self) -> int:
        """ Cross validation at search """
        return self.__cv
    
    ##########################################################################
    
    def set_cv(self, cv: int) -> None:
        """Number of search

        Parameters
        ----------
        n_iter : int
            Number of search
        """
        self.__cv = cv
    
    ##########################################################################
    # Model #
    ###########
    # XGBoost #
    ###########
    
    @property
    def xgboost_params_dict(self):
        """ parameter of XGBoost model for random search """
        return self.__xgboost_params_dict

    ##########################################################################
    
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
                'max_depth': [2, 6, 8, 12, 16, 24, 30, 40, 50],
                'subsample': uniform(0.1, 0.9),
                'colsample_bytree': uniform(0.1, 0.9),
                'gamma': uniform(0, 0.9)
            }
        self.__xgboost_params_dict = xgboost_params_dict
    
    ##########################################################################
    # CatBoost #
    ############
    
    @property
    def catboost_params_dict(self):
        """ parameter of CatBoost model for random search """
        return self.__catboost_params_dict

    ##########################################################################
    
    def set_catboost_params_dict(
            self, 
            catboost_params_dict: dict = None
    ) -> None:
        """Set parameter of XGBoost model for random search

        Parameters
        ----------
        catboost_params_dict : dict, optional
            Parameters, by default None
        """
        if catboost_params_dict is None:
            catboost_params_dict = {
                'iterations': randint(10, 500),
                'learning_rate': uniform(0.001, 0.9),
                'depth': randint(2, 16),
            }
        self.__catboost_params_dict = catboost_params_dict
        
    ##########################################################################
    # svc #
    #######
    
    @property
    def svc_params_dict(self):
        """ parameter of CatBoost model for random search """
        return self.__svc_params_dict

    ##########################################################################
    
    def set_svc_params_dict(
            self, 
            svc_params_dict: dict = None
    ) -> None:
        """Set parameter of XGBoost model for random search

        Parameters
        ----------
        catboost_params_dict : dict, optional
            Parameters, by default None
        """
        if svc_params_dict is None:
            svc_params_dict = {
                'C': [0.01, 0.1, 1, 10, 50, 100],
                'gamma': [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
            }
        self.__svc_params_dict = svc_params_dict
        
    ##########################################################################
    # Random Forest #
    #################
    
    @property
    def random_forest_params_dict(self):
        """ parameter of CatBoost model for random search """
        return self.__random_forest_params_dict

    ##########################################################################
    
    def set_random_forest_params_dict(
            self, 
            random_forest_params_dict: dict = None
    ) -> None:
        """Set parameter of Random forest model for random search

        Parameters
        ----------
        random_forest_params_dict : dict, optional
            Parameters, by default None
        """
        if random_forest_params_dict is None:
            random_forest_params_dict = {
                'n_estimators': randint(10, 1000),  
                'max_features': ['log2', 'sqrt'],
                'max_depth': randint(10, 100), 
                'min_samples_split': randint(2, 100),  
                'min_samples_leaf': randint(1, 12), 
                'bootstrap': [True, False]  
            }
        self.__random_forest_params_dict = random_forest_params_dict
    
    ##########################################################################
    # Random Forest #
    #################
    
    @property
    def logistic_regression_params_dict(self):
        """ parameter of Logistic regression model for random search """
        return self.__logistic_regression_params_dict
    
    ##########################################################################
    
    def set_logistic_regression_params_dict(
            self, 
            logistic_regression_params_dict: dict = None
    ) -> None:
        """Set parameter of Random forest model for random search

        Parameters
        ----------
        random_forest_params_dict : dict, optional
            Parameters, by default None
        """
        if logistic_regression_params_dict is None:
            logistic_regression_params_dict = {
                'C': uniform(loc=0, scale=4),
                'penalty': ['l1', 'l2'] 
            }
        self.__logistic_regression_params_dict = logistic_regression_params_dict
        
    ##########################################################################
    # KNN #
    #######
    
    @property
    def knn_params_dict(self):
        """ parameter of Logistic regression model for random search """
        return self.__knn_params_dict
    
    ##########################################################################
    
    def set_knn_params_dict(
            self, 
            knn_params_dict: dict = None
    ) -> None:
        """Set parameter of Random forest model for random search

        Parameters
        ----------
        random_forest_params_dict : dict, optional
            Parameters, by default None
        """
        if knn_params_dict is None:
            knn_params_dict = {
                'n_neighbors': randint(1, 50),
                'weights': ['uniform', 'distance'],  
                'p': [1, 2] 
            }
        self.__knn_params_dict = knn_params_dict
    
    ############
    # Modeling #
    ##########################################################################
    
    def modeling(
            self,
            df : Union[DataFrame, str],
            model_name_list: list[str] = [
                'xgboost', 
                'catboost', 
                # 'svc',
                'random_forest',
                'logistic_regression',
                'knn',
            ], 
    ) -> None:
        """Tran and save model in model_name_list

        Parameters
        ----------
        df : Union[DataFrame, str]
            Could be either path to data frame or data frame itself.
        model_name_list : list[str]
            List of model name
            `xgboost`, `catboost`, `svc`, 
            `random_forest`, `logistic_regression`, `knn`,
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
        
    ##########################################################################
    # Model #
    ###########
    # XGBoost #
    ###########
    
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
    ##########################################################################
    
    def save_xgboost(self, model: xgb.XGBClassifier) -> None:
        """Save XGBoost model

        Parameters
        ----------
        model : xgb.XGBClassifier
            model
        """
        path = os.path.join(self.result_path, 'xgboost.xgb')
        
        model.save_model(path)
        
    ##########################################################################
    
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
    
    ##########################################################################
    # Cat Boost #
    #############
    
    def catboost(
            self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
    ) -> CatBoostClassifier:
        """Cat boosting model

        Parameters
        ----------
        x_train : DataFrame
            x train as pandas data-frame
        x_test : DataFrame
            x test as pandas data-frame
        y_train : Series
            y train as pandas data-series
        y_test : Series
            y test as pandas data-series

        Returns
        -------
        CatBoostClassifier
            Catboost model
        """
        # Create dataset objects for CatBoost
        # train_pool = Pool(data=x_train, label=y_train)
        # test_pool = Pool(data=x_test, label=y_test)
        
        # Print line separate
        print("\n", "-"*72)
        print("Tuning CatBoost")
        
        model = CatBoostClassifier(
            devices="0",
            loss_function = "Logloss",
            eval_metric = "AUC",
            verbose = False
        )
        
        # Get random search
        tuned_model = self.random_search(
            model,
            self.catboost_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        
        self.save_catboost(tuned_model)
        
    ##########################################################################
    
    def save_catboost(self, model: CatBoostClassifier) -> None:
        path = os.path.join(self.result_path, 'catboost.bin')
        model.save_model(path)
    
    ##########################################################################
    
    @staticmethod
    def load_catboost(model_path: str) -> CatBoostClassifier:
        """Load cat boost model"""
        model = CatBoostClassifier()
        model.load_model(model_path)
        
        return model
    
    ##########################################################################
    # SVC #
    #######
    
    def svc(self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
    ) -> SVC:
        """SVC model

        Parameters
        ----------
        x_train : DataFrame
            x train as pandas data-frame
        x_test : DataFrame
            x test as pandas data-frame
        y_train : Series
            y train as pandas data-series
        y_test : Series
            y test as pandas data-series
        
        Returns
        -------
        SVC
            SVC model
        """
        print("\n", "-"*72)
        print("Tuning SVC")
        model = SVC(verbose=True)
        
        # Get random search
        tuned_model = self.random_search(
            model,
            self.svc_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        self.save_svc(tuned_model)
        
    ##########################################################################
    
    def save_svc(self, model: CatBoostClassifier) -> None:
        # Save the model
        path = os.path.join(self.result_path, 'svc.pkl')
        with open(f'{path}', 'wb') as f:
            pickle.dump(model, f)
        
        model.save_model(path)
    
    ##########################################################################
    
    @staticmethod
    def load_svc(model_path: str) -> CatBoostClassifier:
        """Load cat boost model"""
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    ##########################################################################
    # Random forest #
    #################
    
    def random_forest(self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
    ) -> RandomForestClassifier:
        """Random forest model

        Parameters
        ----------
        x_train : DataFrame
            x train as pandas data-frame
        x_test : DataFrame
            x test as pandas data-frame
        y_train : Series
            y train as pandas data-series
        y_test : Series
            y test as pandas data-series
        
        Returns
        -------
        RandomForestClassifier
            Random Forest Classifier model
        """
        print("\n", "-"*72)
        print("Tuning Random forest")
        model = RandomForestClassifier()
        
        # Get random search
        tuned_model = self.random_search(
            model,
            self.random_forest_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        self.save_random_forest(tuned_model)
        
    ##########################################################################
    
    def save_random_forest(self, model: CatBoostClassifier) -> None:
        # Save the model
        path = os.path.join(self.result_path, 'random_forest.pkl')
        with open(f'{path}', 'wb') as f:
            pickle.dump(model, f)
    
    ##########################################################################
    
    @staticmethod
    def load_random_forest(model_path: str) -> CatBoostClassifier:
        """Load random forest boost model"""
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    ##########################################################################
    # Logistic regression #
    #######################
    
    def logistic_regression(self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
    ) -> LogisticRegression:
        """Logistic regression model

        Parameters
        ----------
        x_train : DataFrame
            x train as pandas data-frame
        x_test : DataFrame
            x test as pandas data-frame
        y_train : Series
            y train as pandas data-series
        y_test : Series
            y test as pandas data-series
        
        Returns
        -------
        LogisticRegression
            Logistic regression model
        """
        print("\n", "-"*72)
        print("Tuning Logistic regression")
        model = LogisticRegression(solver='saga', max_iter=10000)
        
        # Get random search
        tuned_model = self.random_search(
            model,
            self.logistic_regression_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        self.save_logistic_regression(tuned_model)
        
    ##########################################################################
    
    def save_logistic_regression(self, model: CatBoostClassifier) -> None:
        # Save the model
        path = os.path.join(self.result_path, 'logistic_regression.pkl')
        with open(f'{path}', 'wb') as f:
            pickle.dump(model, f)
    
    ##########################################################################
    
    @staticmethod
    def load_logistic_regression(model_path: str) -> CatBoostClassifier:
        """Load random forest boost model"""
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    ##########################################################################
    # Logistic regression #
    #######################
    
    def knn(self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
    ) -> LogisticRegression:
        """KNN model

        Parameters
        ----------
        x_train : DataFrame
            x train as pandas data-frame
        x_test : DataFrame
            x test as pandas data-frame
        y_train : Series
            y train as pandas data-series
        y_test : Series
            y test as pandas data-series
        
        Returns
        -------
        LogisticRegression
            Random Forest Classifier model
        """
        print("\n", "-"*72)
        print("Tuning Logistic regression")
        model = KNeighborsClassifier()
        
        # Get random search
        tuned_model = self.random_search(
            model,
            self.knn_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        self.save_knn(tuned_model)
        
    ##########################################################################
    
    def save_knn(self, model: CatBoostClassifier) -> None:
        # Save the model
        path = os.path.join(self.result_path, 'knn.pkl')
        with open(f'{path}', 'wb') as f:
            pickle.dump(model, f)
    
    ##########################################################################
    
    @staticmethod
    def load_knn(model_path: str) -> CatBoostClassifier:
        """Load random forest boost model"""
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
        
    #############
    # Utilities #
    ##########################################################################
    
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
        print(param_dict)
        random_search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_dict, 
            n_iter=self.n_iter, 
            cv=self.cv, 
            random_state=42,
            verbose = 10,
            scoring="f1",
        )
        # Fit the model to the training data
        random_search.fit(x_train, y_train)

        # Print the best hyperparameters
        print("Best Hyperparameters:", random_search.best_params_)
        
        best_model = random_search.best_estimator_

        # Make predictions on the test set
        y_pred = best_model.predict(x_test)

        # Evaluate the model
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Convert the classification report to a DataFrame
        df_classification_report = pd.DataFrame(report).transpose()

        # Save the classification report DataFrame to a CSV file
        df_classification_report.to_csv(
            os.path.join(self.result_path, f'{type(model)}.csv'), 
            index=True,
        )
        
        return best_model
    
    ##########################################################################

##############################################################################
