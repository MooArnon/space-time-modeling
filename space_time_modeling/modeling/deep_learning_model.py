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
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score

from .__base import BaseWrapper

##############################################################################
# Custom metric #
#################

class CustomMetric(tf.keras.metrics.Metric):
    def __init__(
        self, 
        name="custom_metric", 
        alpha=0.5, 
        beta1=1, 
        beta2=0.4, 
        beta3=0.2, 
        **kwargs,
    ) -> None:
        super(CustomMetric, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.cfpi_sum = self.add_weight(name="cfpi_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class labels (0 or 1) based on a threshold
        y_pred_labels = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is also float32

        # Calculate precision and recall using TensorFlow operations
        true_positives = tf.reduce_sum(y_true * y_pred_labels)
        predicted_positives = tf.reduce_sum(y_pred_labels)
        actual_positives = tf.reduce_sum(y_true)

        precision_buy = true_positives / (predicted_positives + K.epsilon())
        recall_buy = true_positives / (actual_positives + K.epsilon())

        # Calculate F1-score for buy signals
        f1_buy = 2 * (precision_buy * recall_buy) / (precision_buy + recall_buy + K.epsilon())

        # Similarly, calculate for sell signals
        true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred_labels))
        predicted_negatives = tf.reduce_sum(1 - y_pred_labels)
        actual_negatives = tf.reduce_sum(1 - y_true)

        precision_sell = true_negatives / (predicted_negatives + K.epsilon())
        recall_sell = true_negatives / (actual_negatives + K.epsilon())

        f1_sell = 2 * (precision_sell * recall_sell) / (precision_sell + recall_sell + K.epsilon())

        # PRB (Precision-Recall Balance)
        prb = self.alpha * f1_buy + (1 - self.alpha) * f1_sell

        # Composite Financial Performance Index (CFPI) - 
        # since we don't have FIS and SRAP, it’s just PRB
        cfpi = self.beta1 * prb

        # Update the sum of cfpi and the count for averaging
        self.cfpi_sum.assign_add(cfpi)
        self.count.assign_add(1.0)

    def result(self):
        return self.cfpi_sum / self.count

    def reset_state(self):
        self.cfpi_sum.assign(0.0)
        self.count.assign(0.0)

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
    for i in range(hp.Int('num_layers', 1, 4)):
        if i == 0:
            # The first LSTM layer needs the input_shape parameter
            model.add(
                LSTM(
                    units=hp.Int(
                        'units_' + str(i), 
                        min_value=16, 
                        max_value=256, 
                        step=32
                    ),
                    return_sequences=True \
                        if i < hp.Int('num_layers', 1, 6) - 1 \
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
                        max_value=256, 
                        step=32
                    ),
                    return_sequences=True \
                        if i < hp.Int('num_layers', 1, 6) - 1 \
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
                min_value=1e-6, 
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
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(
            GRU(
                units=hp.Int(
                    'units_' + str(i), 
                    min_value=32, 
                    max_value=256, 
                    step=32
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
                max_value=256, 
                step=32
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
                min_value=1e-6, 
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
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(
            Dense(
                units=hp.Int(
                    'units_' + str(i), 
                    min_value=64, 
                    max_value=1024, 
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
                        max_value=0.5, 
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
                min_value=1e-6, 
                max_value=0.1, 
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
    
    for i in range(hp.Int('num_conv_layers', 1, 6)):
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
    for i in range(hp.Int('num_dense_layers', 1, 3)):  
        model.add(
            Dense(
                units=hp.Int(
                    'dense_units_' + str(i), 
                    min_value=32, 
                    max_value=256, 
                    step=32
                ),
                activation=hp.Choice(
                    'dense_activation_' + str(i), 
                    values=['relu']
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
                min_value=1e-6, 
                max_value=1, 
                sampling='log'
            )
        ),
        loss='binary_crossentropy',
        metrics=[CustomMetric()]
    )
    
    return model

##############################################################################
