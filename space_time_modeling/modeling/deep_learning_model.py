##########
# Import #
##############################################################################

from datetime import datetime
from typing import Union
import math

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
    Conv1D,
    MaxPooling1D, 
    Flatten,
    Reshape,
)

from .__base import BaseWrapper
from .custom_metric import CustomMetric

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
    for i in range(hp.Int('num_layers', 1, 10)):
        if i == 0:
            # The first LSTM layer needs the input_shape parameter
            model.add(
                LSTM(
                    units=hp.Int(
                        'units_' + str(i), 
                        min_value=16, 
                        max_value=1024, 
                        step=16
                    ),
                    return_sequences=True \
                        if i < hp.Int('num_layers', 1, 10) - 1 \
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
                        max_value=1024, 
                        step=16
                    ),
                    return_sequences=True \
                        if i < hp.Int('num_layers', 1, 10) - 1 \
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
                min_value=1e-9, 
                max_value=1, 
                sampling='log'
            )
        ),
        loss='binary_crossentropy',
        metrics=[CustomMetric()]
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
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(
            GRU(
                units=hp.Int(
                    'units_' + str(i), 
                    min_value=16, 
                    max_value=512, 
                    step=16
                ),
                return_sequences=True if i < hp.Int('num_layers', 1, 5) - 1 \
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
                step=0.05
            )
        )
    )
    
    # Adding a fully connected layer with tuned number of units
    model.add(
        Dense(
            units=hp.Int(
                'fc_units', 
                min_value=16, 
                max_value=512, 
                step=16
            ), 
            activation='relu'
        )
    )
    
    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model with tuned learning rate
    model.compile(
        optimizer=Adam(
            hp.Float(
                'learning_rate', 
                min_value=1e-9, 
                max_value=1, 
                sampling='log'
            )
        ),
        loss='binary_crossentropy',
        metrics=[CustomMetric()]
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
    for i in range(hp.Int('num_layers', 1, 16)):
        model.add(
            Dense(
                units=hp.Int(
                    'units_' + str(i), 
                    min_value=16, 
                    max_value=2048, 
                    step=32
                ),
                activation=hp.Choice(
                    'activation_' + str(i), 
                    ['relu'],
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
                        max_value=0.7, 
                        step=0.05
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
                min_value=1e-9, 
                max_value=1, 
                sampling='log'
            )
        ),
        loss='binary_crossentropy',
        metrics=[CustomMetric()]
    )
    
    return model

##############################################################################

def build_cnn_model(hp, input_shape):
    model = Sequential()
    
    model.add(Reshape((input_shape[0], 1), input_shape=input_shape))
    
    for i in range(hp.Int('num_conv_layers', 1, 10)):
        model.add(
            Conv1D(
                filters=hp.Int('filters_' + str(i), min_value=32, max_value=256, step=32),
                kernel_size=hp.Choice('kernel_size_' + str(i), values=[3, 5, 7]),
                activation=hp.Choice('conv_activation_' + str(i), values=['relu', 'tanh']),
                input_shape=input_shape if i == 0 else None,
                padding='same'
            )
        )
        model.add(MaxPooling1D(pool_size=hp.Choice('pool_size_' + str(i), values=[2, 3])))

        if hp.Boolean('conv_dropout_' + str(i)):
            model.add(Dropout(rate=hp.Float('conv_dropout_rate_' + str(i), min_value=0.1, max_value=0.5, step=0.1)))
    
    # Flatten the output of the conv layers to connect to Dense layers
    model.add(Flatten())
    
    # Adding Fully Connected (Dense) layers
    # Number of dense layer
    for i in range(hp.Int('num_dense_layers', 1, 5)):  
        model.add(
            Dense(
                units=hp.Int(
                    'dense_units_' + str(i), 
                    min_value=32, 
                    max_value=512, 
                    step=32
                ),
                activation=hp.Choice(
                    'dense_activation_' + str(i), 
                    values=['relu', 'tanh']
                )
            )
        )
        
        # Optional Dropout to prevent overfitting in Dense layers
        if hp.Boolean('dense_dropout_' + str(i)):
            model.add(
                Dropout(
                    rate=hp.Float(
                        'dense_dropout_rate_' + str(i), 
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
                min_value=1e-9, 
                max_value=1, 
                sampling='log'
            )
        ),
        loss='binary_crossentropy',
        metrics=[CustomMetric()]
    )
    
    return model

##############################################################################
