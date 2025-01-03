##########
# Import #
##############################################################################

import json
import os
from typing import Union

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from pandas.core.api import DataFrame, Series
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, make_scorer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import uniform, randint
import xgboost as xgb

from .__base import BaseModel
from .__classification_wrapper import ClassifierWrapper, custom_metric, profit_factor_metric, total_pnl
from ..utilities.utilities import serialize_instance, clear_and_push_to_s3

###########
# Classes #
##############################################################################
# Classifier #
##############

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
            lightgbm_params_dict: dict = None,
            mutual_feature: bool = True,
            push_to_s3: bool = False,
            aws_s3_bucket: str = None,
            aws_s3_prefix: str = None,
    ) -> None:
        super().__init__(
            label_column=label_column, 
            feature_column=feature_column, 
            result_path=result_path, 
            test_size=test_size,
            push_to_s3=push_to_s3,
            aws_s3_bucket=aws_s3_bucket,
            aws_s3_prefix=aws_s3_prefix,
        )
        
        # Set attribute for tuning
        self.set_mutual_feature(mutual_feature)
        self.set_n_iter(n_iter)
        self.set_cv(cv)
        
        # Set attribute for model
        self.set_xgboost_params_dict(xgboost_params_dict)
        self.set_catboost_params_dict(catboost_params_dict)
        self.set_svc_params_dict(svc_params_dict)
        self.set_random_forest_params_dict(random_forest_params_dict)
        self.set_logistic_regression_params_dict(logistic_regression_params_dict)
        self.set_knn_params_dict(knn_params_dict)
        self.set_lightgbm_params_dict(lightgbm_params_dict)

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
    
    @property
    def mutual_feature(self) -> bool:
        """ If true, rank important feature """
        return self.__mutual_feature
    
    ##########################################################################
    
    def set_mutual_feature(self, mutual_feature: bool) -> None:
        """set mutual_feature

        Parameters
        ----------
        mutual_feature : bool
            If true, rank important feature
        """
        self.__mutual_feature = mutual_feature
    
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
                'learning_rate': uniform(0.01, 0.5),  
                'n_estimators': randint(50, 300),  
                'max_depth': randint(3, 10),  
                'subsample': uniform(0.5, 1.0),  
                'colsample_bytree': uniform(0.5, 1.0),  
                'gamma': uniform(0, 5)  
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
    # Logistic regression #
    #######################
    
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
    
    ##########################################################################
    # LightGBM #
    ############
    
    @property
    def lightgbm_params_dict(self):
        """ parameter of XGBoost model for random search """
        return self.__lightgbm_params_dict

    ##########################################################################
    
    def set_lightgbm_params_dict(
            self, 
            lightgbm_params_dict: dict = None
    ) -> None:
        """Set parameter of XGBoost model for random search

        Parameters
        ----------
        lightgbm_params_dict : dict, optional
            Parameters, by default None
        """
        if lightgbm_params_dict is None:
            lightgbm_params_dict = {
                'n_estimators': randint(10, 1000),
                'num_leaves': randint(20, 150),
                'max_depth': randint(3, 20),
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_samples': randint(10, 100),
                'boosting_type': ['gbdt', 'dart', 'goss'],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.1, 0.5, 1],
                'max_bin': randint(10, 255),
            }
        self.__lightgbm_params_dict = lightgbm_params_dict
    
    ############
    # Modeling #
    ##########################################################################
    
    def modeling(
            self,
            df : Union[DataFrame, str],
            preprocessing_pipeline: object,
            model_name_list: list[str] = [
                'xgboost', 
                'catboost', 
                # 'svc',
                'random_forest',
                'logistic_regression',
                'knn',
            ], 
            feature_rank: int = 15,
            weights: dict = None,
            drop_target_column: bool = True,
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
        feature_rank: int
            Integer of top feature
        """
        self.price_data = df[preprocessing_pipeline.target_column]
        print(df.head())
        
        # Check if inportant feature is apply
        # Set up new feature
        if self.mutual_feature:
            feature = preprocessing_pipeline.mutual_info(
                df = df
            ).head(feature_rank)['feature'].to_list()
            
            self.set_feature_column(feature)
            
            feature_column = self.feature_column
            feature_column.append(self.label_column)
            
            if drop_target_column:
                feature_column.remove(preprocessing_pipeline.target_column)

        x_train, x_test, y_train, y_test = self.prepare(
            self.read_df(df)
        )
        
        x_train = x_train[preprocessing_pipeline.features]
        x_test = x_test[preprocessing_pipeline.features]
    
        # Iterate over model_name_list
        for model_name in model_name_list:
            
            # Get train function for each model, using name
            train_function = getattr(self, model_name)
            
            # Check if the training function is found
            if train_function is None or not callable(train_function):
                print(f"Warning: Model '{model_name}' not found.")
                continue
            
            # Execute training function
            tuned_model, df_classification_report = train_function(
                x_train, 
                x_test,
                y_train,
                y_test,
                weights,
            )
            
            # Wrap model
            wrapped_model = ClassifierWrapper(
                model = tuned_model, 
                name = model_name,
                feature = self.feature_column,
                preprocessing_pipeline = preprocessing_pipeline,
            )
            
            # Save model
            path = os.path.join(self.result_path, model_name)
            file_path = serialize_instance(
                instance = wrapped_model,
                path = path,
                add_time = False,
            )
            
            if self.push_to_s3:
                clear_and_push_to_s3(
                    file_path,
                    self.aws_s3_bucket,
                    f"{self.aws_s3_prefix}/{model_name}/",
                )
            
            # Save metrics
            df_classification_report.to_csv(
                os.path.join(
                    path,
                    "metrics.csv"
                )
            )
            
            best_model_param = os.path.join(path,"best_model_param.json")
            with open(best_model_param, 'w') as json_file:
                json.dump(self.best_model_params_serializable, json_file, indent=4)
        
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
            weights: dict,
    ) -> xgb.XGBClassifier:
        
        # Print line separate
        print("\n", "-"*72)
        print("Tuning XGBoost")
        
        # Init model
        model = xgb.XGBClassifier(objective='binary:logistic',random_state=42)
        
        # Get random search
        tuned_model, df_classification_report = self.random_search(
            model,
            self.xgboost_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
            weights,
        )
        
        return tuned_model, df_classification_report

    ##########################################################################
    # Cat Boost #
    #############
    
    def catboost(
            self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
            weights: dict,
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
        tuned_model, df_classification_report  = self.random_search(
            model,
            self.catboost_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
            weights,
        )

        return tuned_model, df_classification_report
        
    ##########################################################################
    # SVC #
    #######
    
    def svc(self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
            weights: dict,
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
        tuned_model, df_classification_report = self.random_search(
            model,
            self.svc_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
            weights,
        )
        
        return tuned_model, df_classification_report

    ##########################################################################
    # Random forest #
    #################
    
    def random_forest(self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
            weights: dict,
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
        tuned_model, df_classification_report = self.random_search(
            model,
            self.random_forest_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
            weights,
        )
        
        return tuned_model, df_classification_report

    ##########################################################################
    # Logistic regression #
    #######################
    
    def logistic_regression(self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
            weights: dict,
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
        tuned_model, df_classification_report = self.random_search(
            model,
            self.logistic_regression_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
            weights,
        )
        return tuned_model, df_classification_report
    
    ##########################################################################
    # Logistic regression #
    #######################
    
    def knn(self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
            weights: dict,
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
        tuned_model, df_classification_report = self.random_search(
            model,
            self.knn_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
            weights,
        )
        
        return tuned_model, df_classification_report

    ##########################################################################
    # LightGBM #
    ############
    
    def lightgbm(self,
            x_train: DataFrame, 
            x_test: DataFrame, 
            y_train: Series, 
            y_test: Series, 
            weights: dict,
    ) -> lgb.LGBMClassifier:
        """LightGBM model

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
        lgb
            LightFBM Classifier model
        """
        print("\n", "-"*72)
        print("Tuning LightGBM")
        model = lgb.LGBMClassifier()
        
        # Get random search
        tuned_model, df_classification_report = self.random_search(
            model,
            self.lightgbm_params_dict,
            x_train,
            x_test,
            y_train,
            y_test,
            weights,
        )
        
        return tuned_model, df_classification_report
    
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
            weights: dict,
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
        custom_scorer = make_scorer(
            profit_factor_metric, 
            greater_is_better=True,
            price_data=self.price_data,
            weights=weights
        )
        
        tscv = TimeSeriesSplit(n_splits=self.cv)
        
        print(param_dict)
        random_search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_dict, 
            n_iter=self.n_iter, 
            cv=tscv, 
            verbose=10, 
            scoring=custom_scorer,
            random_state=None,
        )
        # Fit the model to the training data
        random_search.fit(x_train, y_train)

        # Print the best hyperparameters
        print("Best Hyperparameters:", random_search.best_params_)
        
        best_model = random_search.best_estimator_
        model_params = best_model.get_params()
        self.best_model_params_serializable = {
            key: str(value) for key, value in model_params.items()
        }

        # Make predictions on the test set
        y_pred = best_model.predict(x_test)

        # Evaluate the model
        report = classification_report(y_test, y_pred, output_dict=True)
        combined_metrics = profit_factor_metric(y_test, y_pred, self.price_data)
        
        # Convert the classification report to a DataFrame
        df_classification_report = pd.DataFrame(report).transpose()
        df_classification_report['Custom metrics'] = combined_metrics
        df_classification_report['Custom metrics'] = combined_metrics
        
        # Add PnL for LONG and SHORT positions to the report
        pnl = total_pnl(y_test, y_pred, self.price_data)
        df_classification_report['LONG PnL'] = pnl['LONG PnL']
        df_classification_report['SHORT PnL'] = pnl['SHORT PnL']

        # Combine metrics (e.g., profit factor, Sharpe ratio, etc.)
        df_classification_report['PnL'] = pnl['LONG PnL'] + pnl['SHORT PnL']

        return best_model, df_classification_report, 
    
    ##########################################################################

##############################################################################