##########
# Import #
##############################################################################

import os
from typing import Union

import numpy as np
import pandas as pd
from pandas.core.api import DataFrame
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import BayesianOptimization, Objective
import numpy as np

from .__base import BaseModel
from .deep_learning_model import (
    DeepWrapper, 
    build_lstm_model, 
    build_gru_model,
    build_dnn_model,
    build_cnn_model,
)
from ..utilities.utilities import serialize_instance, clear_and_push_to_s3
from .__classification_wrapper import total_pnl

###########
# Classes #
##############################################################################
# Classifier #
##############

class DeepClassificationModel(BaseModel):
    name = 'modeling-instance'
    def __init__(
            self, 
            label_column: str = None, 
            feature_column: list[str] = None, 
            result_path: str = None,
            test_size: float = 0.2,
            mutual_feature: bool = False,
            max_trials: int = 10,
            executions_per_trial: int = 1,
            epoch_per_trial = 20,
            early_stop_min_delta=0.0001,
            early_stop_patience=20,
            early_stop_verbose=1,
            push_to_s3: bool = True,
            aws_s3_bucket: str = None,
            aws_s3_prefix: str = None,
            override_model_name_dict: str = None
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
        self.set_mutual_feature(mutual_feature)
        
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.epoch_per_trial = epoch_per_trial
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_patience = early_stop_patience
        self.early_stop_verbose = early_stop_verbose
        
        self.override_model_name_dict = override_model_name_dict
        
    ##############
    # Properties #
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
    
    @property
    def lstm_param(self) -> dict:
        """Parameter for random search

        Returns
        -------
        dict
        """
        return self.__lstm_param
    
    ##########################################################################
    
    def set_lstm_param(self, lstm_param: dict = None) -> None:
        """Set param dict for random search

        Parameters
        ----------
        lstm_param : dict, optional
            if none set 
            param_dist = {
                'lstm_units': [50, 100, 150],          
                'dense_units': [64, 128, 256],         
                'dropout_rate': [0.2, 0.3, 0.4],       
                'learning_rate': [0.001, 0.01, 0.1],   
                'batch_size': [16, 32, 64],            
                'epochs': [10, 20, 30]                 
            }
            , by default None
        """
        if not lstm_param:
            lstm_param  = {
                'build_fn__units': [50, 100, 150],   
                'build_fn__learning_rate': [0.001, 0.01, 0.1],           
                'epochs': [10, 20, 30]                 
            }
        self.__lstm_param = lstm_param
    
    ##########################################################################
    # Model #
    #########
    
    @property
    def cnn_learning_rate(self) -> float:
        """Learning rate used at CNN model

        Returns
        -------
        float
            learning_rate
        """
        return self.__cnn_learning_rate
    
    ##########################################################################
    
    def set_cnn_learning_rate(self, learning_rate: float) -> None:
        """Set learning rate used at CNN model

        Parameters
        ----------
        learning_rate : float
            Float of earning rate
        """
        self.__cnn_learning_rate = learning_rate
    
    ############
    # Training #
    ##########################################################################
    
    def modeling(
            self,
            df : Union[DataFrame, str],
            preprocessing_pipeline: object,
            model_name_list: list[str] = [
                'cnn',
            ], 
            feature_rank: int = 30,
    ) -> None:
        """Tran and save model in model_name_list

        Parameters
        ----------
        df : Union[DataFrame, str]
            Could be either path to data frame or data frame itself.
        model_name_list : list[str]
            List of model name `cnn`
        feature_rank: int
            Integer of top feature
        """
        self.price_data = df[preprocessing_pipeline.target_column]
        
        # Check if inportant feature is apply
        # Set up new feature
        if self.mutual_feature:
            feature = preprocessing_pipeline.mutual_info(
                df = df
            ).head(feature_rank)['feature'].to_list()
            
            self.set_feature_column(feature)
            
            feature_column = self.feature_column
            feature_column.append(self.label_column)
            df = df[feature_column]
        
        x_train, x_test, y_train, y_test = self.prepare(
            self.read_df(df)
        )
        
        print(f"features: {x_train.columns}")
        print(f"x_train's shape: {x_train.shape}")
        print(f"x_test's shape: {x_test.shape}")
        
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
            )
            
            # Separate model
            if self.override_model_name_dict:
                    model_name = self.override_model_name_dict[model_name]

            # Wrap model
            wrapped_model = DeepWrapper(
                model = tuned_model, 
                name = model_name,
                feature = x_train.columns.to_list(),
                preprocessing_pipeline=preprocessing_pipeline
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
            pd.DataFrame(data=df_classification_report).to_csv(
                os.path.join(
                    path,
                    "metrics.csv"
                )
            )
    
    ##########################################################################
    
    def train(
            self,
            model_builder,
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
            input_shape: tuple
        ) -> None:
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=self.early_stop_min_delta,
            patience=self.early_stop_patience,
            verbose=self.early_stop_verbose,
            restore_best_weights=True
        )
        
        # Instantiate the tuner
        tuner = BayesianOptimization(
            lambda hp: model_builder(hp, input_shape),
            objective=Objective("val_custom_metric", "max"),
            max_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,
            directory='deep-log',
            project_name=f"{model_builder.__name__}_{self.result_path}"
        )
        
        tuner.search(
            x_train, 
            y_train, 
            epochs=self.epoch_per_trial, 
            validation_data=(x_test, y_test),
            callbacks=[early_stopping],
        )

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"Best hyperparameters: {best_hyperparameters.values}")
        
        test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
        print(f"Test Accuracy: {test_accuracy}")
        
        # Predict the class labels for the test set
        y_pred = best_model.predict(x_test)
        y_pred = np.round(y_pred).astype(int) 
        
        # Create report
        # Evaluate the model
        report = classification_report(y_test, y_pred, output_dict=True)

        # Add PnL for LONG and SHORT positions to the report
        y_pred = np.concatenate(y_pred)
        pnl = total_pnl(y_test, y_pred, self.price_data)
        report['LONG PnL'] = pnl['LONG PnL']
        report['SHORT PnL'] = pnl['SHORT PnL']

        # Combine metrics (e.g., profit factor, Sharpe ratio, etc.)
        report['PnL'] = pnl['LONG PnL'] + pnl['SHORT PnL']
        
        return best_model, report

    ##########################################################################
    
    def lstm(
            self,
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
    ) -> None:
        
        input_shape = (x_train.shape[1], 1)
        
        return self.train(
            build_lstm_model,
            x_train, 
            x_test, 
            y_train, 
            y_test, 
            input_shape,
        )
    
    ##########################################################################
    
    def gru(
            self,
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
    ) -> None:
        
        input_shape = (x_train.shape[1], 1)
        
        return self.train(
            build_gru_model, 
            x_train, 
            x_test, 
            y_train, 
            y_test, 
            input_shape,
        )
    
    ##########################################################################
    
    def dnn(
            self,
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
    ) -> None:
        
        input_shape = (x_train.shape[1],)
        
        return self.train(
            build_dnn_model, 
            x_train, 
            x_test, 
            y_train, 
            y_test, 
            input_shape,
        )
    
    ##########################################################################
    
    def cnn(
            self,
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
    ) -> None:
        
        input_shape = (x_train.shape[1],)
        
        return self.train(
            build_cnn_model, 
            x_train, 
            x_test, 
            y_train, 
            y_test, 
            input_shape,
        )
    
    ##########################################################################
    
##############################################################################
