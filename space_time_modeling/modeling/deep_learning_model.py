##########
# Import #
##############################################################################

from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, 
    Dropout, 
    LSTM,
    GRU,
    Input,
)

from .__base import BaseWrapper

###########
# Wrapper #
##############################################################################

class DeepWrapper(BaseWrapper):
    name = "deep_wrapper"
    def __init__(
            self,
            model: Model, 
            name: str, 
            feature: list[str], 
            preprocessing_pipeline: object,
    ):
        super(DeepWrapper, self).__init__(feature, preprocessing_pipeline)
        self.set_model(model)
        self.set_name(name)
        self.set_version()

    ##############
    # Properties #
    #########################################################################
    
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
    #########################################################################
    
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

###############
# Build model #
##############################################################################
# LSTM #
########

def build_lstm_model(hp, input_shape):
    model = Sequential()
    
    model.add(Input(shape=input_shape))
    
    # Adding LSTM layers with tuned number of units
    for i in range(hp.Int('num_layers', 1, 3)):
        if i == 0:
            # The first LSTM layer needs the input_shape parameter
            model.add(
                LSTM(
                    units=hp.Int(
                        'units_' + str(i), 
                        min_value=16, 
                        max_value=20, 
                        step=2
                    ),
                    return_sequences=True \
                        if i < hp.Int('num_layers', 1, 3) - 1 \
                            else False
                )
            )
        else:
            # Subsequent LSTM layers don't need the input_shape parameter
            model.add(
                LSTM(
                    units=hp.Int(
                        'units_' + str(i), 
                        min_value=16, 
                        max_value=20, 
                        step=2
                    ),
                    return_sequences=True \
                        if i < hp.Int('num_layers', 1, 3) - 1 \
                            else False
                )
            )
    
    # Adding Dropout with tuned rate
    model.add(
        Dropout(
            rate=hp.Float(
                'dropout_rate', 
                min_value=0.1, 
                max_value=0.5, 
                step=0.1
            )
        )
    )
    
    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model with tuned learning rate
    model.compile(
        optimizer=Adam(
            hp.Float(
                'learning_rate', 
                min_value=1e-4, 
                max_value=10, 
                sampling='log'
            )
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

##############################################################################
# GRU #
#######

def build_gru_model(hp, input_shape):
    model = Sequential()
    
    # Adding Input layer
    model.add(Input(shape=input_shape))
    
    # Adding GRU layers with tuned number of units
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(
            GRU(
                units=hp.Int(
                    'units_' + str(i), 
                    min_value=16, 
                    max_value=20, 
                    step=2
                ),
                return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 \
                    else False
            )
        )
    
    # Adding Dropout with tuned rate
    model.add(
        Dropout(
            rate=hp.Float(
                'dropout_rate', 
                min_value=0.1, 
                max_value=0.5, 
                step=0.1
            )
        )
    )
    
    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model with tuned learning rate
    model.compile(
        optimizer=Adam(
            hp.Float(
                'learning_rate', 
                min_value=1e-4, 
                max_value=10, 
                sampling='log'
            )
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

##############################################################################
# DNN #
#######

def build_dnn_model(hp, input_shape):
    model = Sequential()
    
    # Adding Input layer
    model.add(Input(shape=input_shape))
    
    # Adding hidden Dense layers with tuned number of units 
    # and activation functions
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(
            Dense(
                units=hp.Int(
                    'units_' + str(i), 
                    min_value=32, 
                    max_value=512, 
                    step=32
                ),
                activation=hp.Choice(
                    'activation_' + str(i), 
                    ['relu', 'tanh', 'sigmoid'],
                )
            )
        )
        # Optional Dropout layer to prevent overfitting
        if hp.Boolean('dropout_' + str(i)):
            model.add(
                Dropout(
                    rate=hp.Float(
                        'dropout_rate_' + str(i), 
                        min_value=0.1, 
                        max_value=0.5, 
                        step=0.1
                    )
                )
            )
    
    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model with tuned learning rate
    model.compile(
        optimizer=Adam(
            hp.Float(
                'learning_rate', 
                min_value=1e-4, 
                max_value=10, 
                sampling='log'
            )
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

##############################################################################
