##########
# Import #
##############################################################################

import os
import random
from itertools import product
from typing import Union
from tqdm import tqdm


import numpy as np
import pandas as pd
from pandas.core.api import DataFrame
import json
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .__base import BaseModel
from .deep_learning_model import (
    DNNWrapper,
    LSTMWrapper
)
from ..utilities.utilities import serialize_instance

###########
# Statics #
##############################################################################

###########
# Classes #
##############################################################################

class DeepClassificationModel(BaseModel):
    name = 'modeling-instance'
    def __init__(
            self, 
            label_column: str = None, 
            feature_column: list[str] = None, 
            result_path: str = None,
            n_iter: int = 5,
            test_size: float = 0.2,
            mode: str = 'random_search',
            dnn_params_dict: dict = None,
            lstm_params_dict: dict = None,
            focus_metric: str = 'accuracy',
    ) -> None:
        super().__init__(
            label_column, 
            feature_column, 
            result_path, 
            test_size,
        )
        
        # Set param dict
        self.set_dnn_params_dict(
            mode = mode, 
            n_iter = n_iter, 
            dnn_params_dict = dnn_params_dict, 
        )
        self.set_lstm_params_dict(
            mode = mode, 
            n_iter = n_iter, 
            lstm_params_dict = lstm_params_dict,
        )
        
        
        self.set_focus_metric(focus_metric)
    
    ##############
    # Properties #
    ##########################################################################
    # DNN #
    #######
    
    @property
    def focus_metric(self) -> str:
        """Get focus metric at random search

        Returns
        -------
        focus_metric : str
            Focus metric
        """
        return self.__focus_metric
    
    ##########################################################################
    
    def set_focus_metric(self, focus_metric: str) -> None:
        """Set focus metric at random search

        Parameters
        ----------
        focus_metric : str
            Focus metric
        """
        self.__focus_metric = focus_metric
        
    ##########################################################################
    
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
    def dnn_params_dict(self):
        """ Parameter of DNN for random search """
        return self.__dnn_params_dict
    
    ##########################################################################
    
    def set_dnn_params_dict(
            self, 
            mode: str,
            n_iter: int,
            dnn_params_dict: dict = None,
    ) -> None:
        """Set parameter of DNN for random search

        Parameters
        ----------
        mode: str , optional
            Mode of search
        n_iter:
            number of iteration
        dnn_params_dict : dict, optional
            Parameters, by default None
        """
        if dnn_params_dict is None:
            dnn_params_dict = {
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
            
        if mode =='random_search':
            self.__dnn_params_dict = self.generate_model_param_dict(
                params_dict = dnn_params_dict,
                n_iter = n_iter
            )
            
    ##########################################################################
        
    @property
    def lstm_params_dict(self):
        """ Parameter of LSTM for random search """
        return self.__lstm_params_dict

    ##########################################################################
    
    def set_lstm_params_dict(
            self, 
            mode: str,
            n_iter: int,
            lstm_params_dict: dict = None,
    ) -> None:
        """Set parameter of DNN for random search

        Parameters
        ----------
        mode: str , optional
            Mode of search
        n_iter:
            number of iteration
        lstm_params_dict : dict, optional
            Parameters, by default None
        """
        if lstm_params_dict is None:
            lstm_params_dict = {
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
            
        if mode =='random_search':
            self.__lstm_params_dict = self.generate_model_param_dict(
                params_dict = lstm_params_dict,
                n_iter = n_iter
            )
    
    ###############
    # Main method #
    ##########################################################################
    
    def modeling(
            self,
            df : Union[DataFrame, str],
            model_name_list: list[str] = ['dnn', 'lstm'],
            batch_size: int = 32,
    ) -> None:
        """Modeling method

        Parameters
        ----------
        df : Union[DataFrame, str]
            Data frame object consisted of feature and label columns
        model_name_list : list[str], optional
            List of model, now `dnn`
            , by default ['dnn', 'lstm']
        batch_size: int, optional
            Size of each batch
            , by default 32
        """
        # Split feature and label
        y, x = self.split_feature_label(self.read_df(df))
        
        # Split train and test data
        x_train, x_test, y_train, y_test = self.split_test_train(
            label = y,
            feature = x
        )

        # Add data to custom dataset object
        dataset_train = CustomDataset(x_train, y_train)
        dataset_test = CustomDataset(x_test, y_test)
        
        # Create DataLoader
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        
        # Iterate over model_name_list
        for model_name in model_name_list:
            print("\n")
            
            # Get train function for each model, using name
            train_function = getattr(self, model_name)
            
            # Check if the training function is found
            if train_function is None or not callable(train_function):
                print(f"Warning: Model '{model_name}' not found.")
                continue
            
            # Execute training function
            best_model = train_function(
                num_feature = dataset_train.num_features,
                dataloader_train = dataloader_train,
                dataloader_test = dataloader_test,
            )
            
            metrics =self.test_deep(
                best_model['model_wrapper'], 
                dataloader_test, 
                metric_format='dataframe',
            )
            
            model_structure = self.generate_model_structure(
                best_model['model_wrapper'].model
            )
            
            model_structure["lr"] = best_model["params_dict"]["lr"]
            model_structure["epochs"] = best_model["params_dict"]["epochs"]
            model_structure["criterion"] = str(
                best_model["params_dict"]["criterion"]
            )
            
            # Save model
            ## Create path
            model_path = os.path.join(self.result_path, model_name)
            self.model_path = model_path
            
            # Sev attribute
            self.save_nn_wrapper(best_model['model_wrapper'], model_name)
            self.save_object(metrics, 'best_metrics')
            self.save_object(model_structure, 'best_model_structure')

    ###########
    # Element #
    ##########################################################################
    # DNN: Deep Neuron Network #
    ############################
    
    def dnn(self,
            num_feature: int,
            dataloader_train: DataLoader,
            dataloader_test: DataLoader,
    ) -> None:
        """dnn model

        Parameters
        ----------
        num_feature: int
            Number of features
        dataloader_train: DataLoader
            Dataloader object for training set
        dataloader_test: DataLoader
            Dataloader object for testing set
        """
        print("\nTuning Deep Neuron Network")
        print("--------------------------\n")
        
        wrapper = DNNWrapper(feature=self.feature_column)
        
        # Get random search
        best_model = self.random_search(
            num_feature = num_feature,
            wrapper = wrapper,
            params_dict_list = self.dnn_params_dict,
            dataloader_train = dataloader_train,
            dataloader_test = dataloader_test,
            focus_metric = self.focus_metric,
        )
        
        return best_model
    
    ##########################################################################
    # LSTM: Long Short-Term Memory #
    ################################
    
    def lstm(self,
            num_feature: int,
            dataloader_train: DataLoader,
            dataloader_test: DataLoader,
    ) -> None:
        """dnn model

        Parameters
        ----------
        num_feature: int
            Number of features
        dataloader_train: DataLoader
            Dataloader object for training set
        dataloader_test: DataLoader
            Dataloader object for testing set
        """
        print("\nTuning Long Short-Term Model")
        print("----------------------------\n")
        
        wrapper = LSTMWrapper(feature=self.feature_column)
        
        # Get random search
        best_model = self.random_search(
            num_feature = num_feature,
            wrapper = wrapper,
            params_dict_list = self.lstm_params_dict,
            dataloader_train = dataloader_train,
            dataloader_test = dataloader_test,
            focus_metric = self.focus_metric,
        )
        
        return best_model
    
    #####################
    # Training Function #
    ##########################################################################
    
    def random_search(
            self,
            num_feature: int,
            wrapper: torch.nn.Module,
            params_dict_list: list[dict],
            dataloader_train: DataLoader,
            dataloader_test: DataLoader,
            focus_metric: str,
    ) -> dict:
        """Random search method

        Parameters
        ----------
        num_feature : int
            Number of features
        wrapper : torch.nn.Module
            Wrapper object for model
        params_dict_list : list[dict]
            List of parameter dictionary
        dataloader_train: DataLoader
            Dataloader object for training set
        dataloader_test: DataLoader
            Dataloader object for testing set
        focus_metric : str
            Metric that need to focus

        Returns
        -------
        dict
            {
                params_dict: "parameter in param dict",
                module: "parameter in param dict but in module__ prefix",
                metrics: "Evaluation score",
                loss: "Final loss",
                model: "Best model
            }
        """
        model_accuracy_list = []
        for params_dict in params_dict_list:
            model_accuracy = {}
            
            print(f"Tuning parameters: {params_dict}")
            
            wrapper.set_model(
                input_dim = num_feature, 
                output_dim = 1,
                **params_dict["module"],
            )
            
            # Optimizer and criterion
            optimizer = torch.optim.Adam(
                wrapper.model.parameters(), 
                lr = params_dict['lr'],
            )
            criterion = params_dict["criterion"]

            metrics = self.train_deep(
                epochs = params_dict['epochs'], 
                dataloader_train = dataloader_train,
                dataloader_test = dataloader_test,
                optimizer = optimizer,
                model = wrapper.model,
                criterion = criterion,
            )
            
            model_accuracy["params_dict"] = params_dict
            model_accuracy["metrics"] = metrics
            model_accuracy['model_wrapper'] = wrapper
            
            model_accuracy_list.append(model_accuracy)
            
        # Choose best model
        best_model = self.choose_best_model(focus_metric, model_accuracy_list)
        return best_model

    ##########################################################################
    
    def train_deep(
            self,
            epochs: int, 
            dataloader_train: DataLoader,
            dataloader_test: DataLoader,
            optimizer: torch.optim,
            model: torch.nn.Module,
            criterion: nn
    ) -> dict:
        """Training function

        Parameters
        ----------
        epochs : int
            Epochs
        dataloader_train : DataLoader
            Train data loader object
        dataloader_test : DataLoader
            Test data loader object
        optimizer : torch.optim
            Optimizer
        model : torch.nn.Module
            Model
        criterion : nn
            Loss function

        Returns
        -------
        dict
            Metrics
        """
        # Iterate over epochs
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            
            # Iterate over data loader
            # Batching
            for i, (inputs, labels) in enumerate(dataloader_train):
                
                # Zero gradian
                optimizer.zero_grad()
                
                # Feed data and back propagation
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Add loss for print
                running_loss += loss.item()
            
            # Evaluate every 20 epochs
            if (epoch%10 == 0) and (epoch!=0):
                
                # Test model
                metrics = self.test_deep(model, dataloader_test)
                
                # Decorate terminal
                print(f'{epoch}/{epochs}')
                print(f'Loss: {running_loss}')
                print(f'Accuracy test set: {metrics["accuracy"]}')

        # Test model
        metrics = self.test_deep(model, dataloader_test)
        
        # Decorate terminal
        print("Finished")
        print(f'Loss: {running_loss}', end=" | ")
        print(f'Accuracy test set: {metrics["accuracy"]}')
        
        metrics = self.test_deep(model, dataloader_test)
        metrics["loss"] = running_loss
        
        return metrics
        
    ##########################################################################
        
    def test_deep(
            self, 
            model: torch.nn.Module, 
            dataloader_test: DataLoader,
            metric_format: str = 'dict'
    ) -> Union[dict, DataFrame]:
        """Test deep learning model

        Parameters
        ----------
        model : torch.nn.Module
            Model
        dataloader_test : DataLoader
            Data loader object, test set
        metric_format : str, optional
            Format of return metrics,
            `dict` return dictionary
            `dataframe` return pandas data-frame object
            by default 'dict'

        Returns
        -------
        Union[dict, DataFrame]
            Returned metrics
        
        raise
        -----
        ValueError
            if the metric_format is not in ['dict', 'dataframe']
        """
        # Statics
        model.eval()
        all_labels = []
        all_predictions = []

        
        # Set gradient to 0
        # Iterate over dataloader
        with torch.no_grad():
            for inputs, labels in dataloader_test:
                
                # Predict model
                # Re-dimension using squeeze
                outputs = model(inputs).squeeze()
                
                # Monitoring data
                labels_numpy = labels.detach().numpy()
                rounded_outputs_numpy = torch.round(outputs).detach().numpy()
                
                # Collect labels and predictions for entire dataset
                all_labels.extend(labels_numpy)
                all_predictions.extend(rounded_outputs_numpy)
        
        report = classification_report(
            all_labels, 
            all_predictions, 
            output_dict=True,
            zero_division=1,
        )
        
        if metric_format == 'dict':
            return report

        elif metric_format == 'dataframe':
            return pd.DataFrame(report).transpose()
        
        else:
            raise ValueError("metric_format is in ['dict', 'dataframe']")
        
    ##########################################################################
    # Deep learning util #
    ######################
    
    def save_nn_wrapper(self, model_wrapper: object, model_type: str) -> None:
        """Save Deep learning model wrapper

        Parameters
        ----------
        model : nn.Module
            Model
        model_type: str
            Type of model
        """
        path = os.path.join(self.model_path)
        serialize_instance(
            instance = model_wrapper, 
            path = path, 
            add_time = False
        )
    
    ##########################################################################
    
    def save_object(
            self, 
            metics: Union[dict, DataFrame], 
            prefix: str,
    ) -> None:
        """Save Deep learning model

        Parameters
        ----------
        metics : dict
            Model
        prefix: str
            Prefix before file extension, or file name itself
        """
        
        if isinstance(metics, dict):
            path = os.path.join(self.model_path, f'{prefix}.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metics, f, ensure_ascii=False, indent=4)
        
        if isinstance(metics, DataFrame):
            path = os.path.join(self.model_path, f'{prefix}.csv')
            metics.to_csv(path)
    
    ##########################################################################
    
    @staticmethod
    def load_dnn(model_path: str) -> nn.Module:
        """Load Deep learning model
        
        Parameters
        ----------
        model_path: str
            path of model
        """
        
        return torch.load(f'{model_path}')
        
    
    #############
    # Utilities #
    ##########################################################################
    
    def generate_model_param_dict(
            self, 
            params_dict: dict, 
            n_iter: int,
    ) -> dict:
        """Generate param dict

        Parameters
        ----------
        params_dict : dict
            param dict
        n_iter : int
            Number of iteration

        Returns
        -------
        dict
            Final dict
        """
        param_combinations = list(product(*params_dict.values()))

        # Randomly select 3 combinations
        random_combinations = random.sample(param_combinations, n_iter)

        # Combine keys with values to form dictionaries
        hyperparameter_sets = [
            dict(zip(params_dict.keys(), values)) \
                for values in random_combinations
        ]
        
        return self.rearrange_module_dict(hyperparameter_sets)
        
    ##########################################################################
    
    @staticmethod
    def rearrange_module_dict(param_dict_list: list[dict]) -> dict:
        """Restructure module__

        Parameters
        ----------
        param_dict_list : dict
            List of parameter dictionary

        Returns
        -------
        dict
            Result that expanded dictionary
        """
        new_param_dict = []
        
        # Iterate over list of dictionary
        for param_dict in param_dict_list:
            module_hyperparameters = {}

            # Iterate over key in param_dict
            for key, value  in param_dict.items():
                
                # Check if prefix is module__
                if key.startswith('module__'):
                    
                    # Replace the module__ with black
                    # Store information in module_hyperparameters dictionary
                    module_key = key.replace('module__', '')
                    module_hyperparameters[module_key] = value

            # Wrap all parameter and remove the one with module__ prefix
            hyperparameters = {
                key: value \
                    for key, value in param_dict.items() \
                        if not key.startswith('module__')
            }
            
            # Insert module's key to the original
            hyperparameters['module'] = module_hyperparameters
            new_param_dict.append(hyperparameters)
            
        return new_param_dict

    ##########################################################################
    
    @staticmethod
    def choose_best_model(
            focus_metric: str, 
            model_accuracy_list: list[dict]
    ) -> dict:
        """Compare and save model

        Parameters
        ----------
        focus_metric : str
            Metic that need to focus
        model_accuracy_list : list[dict]
            Result of model as an dictionary
        """
        # Initialize variables to store 
        # the maximum metric and corresponding dictionary
        max_focus_metric_score = float('-inf')
        best_model_info = None

        # Iterate over each dictionary in the list
        for item in model_accuracy_list:
            focus_metric_score = item['metrics'][focus_metric]
            
            # Check if the current maximum metric 
            # is higher than the maximum seen so far
            if focus_metric_score > max_focus_metric_score:
                max_focus_metric_score = focus_metric_score
                best_model_info = item
        
        return best_model_info
    
    ##########################################################################
    
    @staticmethod
    def generate_model_structure(model: nn.Module) -> dict:
        """Generate model structure

        Parameters
        ----------
        model : nn.Module
            Model object

        Returns
        -------
        dict
            Model information
        """
        model_info = {
            "name": type(model).__name__,
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "layers": []
        }

        # Iterate over each layer in the model
        for name, layer in model.named_children():
            layer_info = {
                "name": name,
                "type": layer.__class__.__name__,
                "params": []
            }
            
            # Iterate over each parameter in the layer
            for pname, pvalue in layer.named_parameters():
                param_info = {
                    "name": pname,
                    "shape": list(pvalue.shape)
                }
                layer_info["params"].append(param_info)

            model_info["layers"].append(layer_info)

        return model_info
    ##########################################################################

##############################################################################

class CustomDataset(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num_features = x.shape[1]
    
    ##########################################################################
    
    def __len__(self):
        return len(self.x)
    
    ##########################################################################
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.x.iloc[idx].values, dtype=torch.float32), 
            torch.tensor(self.y.iloc[idx], dtype=torch.float32)
        )
    
    ##########################################################################

##############################################################################