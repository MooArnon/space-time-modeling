##########
# Import #
##############################################################################

from keras_tuner import HyperModel, RandomSearch
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score

###########
# Classes #
#################
# Deep learning #
##############################################################################

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

        # Composite Financial Performance Index (CFPI) - since we don't have FIS and SRAP, it’s just PRB
        cfpi = self.beta1 * prb

        # Update the sum of cfpi and the count for averaging
        self.cfpi_sum.assign_add(cfpi)
        self.count.assign_add(1.0)

    def result(self):
        return self.cfpi_sum / self.count

    def reset_state(self):
        self.cfpi_sum.assign(0.0)
        self.count.assign(0.0)
        
##############################################################################

# Implementing the composite metric function
def composite_metric(
        y_true, 
        y_pred, 
        prices, 
        alpha=0.5, 
        beta1=0.4, 
        beta2=0.4, 
        beta3=0.2
) -> tuple[float, float, float]:
    precision_buy = precision_score(y_true, y_pred, pos_label=1)
    recall_buy = recall_score(y_true, y_pred, pos_label=1)
    precision_sell = precision_score(y_true, y_pred, pos_label=0)
    recall_sell = recall_score(y_true, y_pred, pos_label=0)
    
    f1_buy = 2 * (precision_buy * recall_buy) / (precision_buy + recall_buy)
    f1_sell = 2 * (precision_sell * recall_sell) / (precision_sell + recall_sell)
    
    prb = alpha * f1_buy + (1 - alpha) * f1_sell
    
    profit_buy, profit_sell, loss_buy, loss_sell = calculate_profit_loss(
        y_true, 
        y_pred, 
        prices,
    )
    
    fis = (profit_buy + profit_sell) / (loss_buy + loss_sell)
    
    returns = np.array([profit_buy - loss_buy, profit_sell - loss_sell])
    srap = np.mean(returns) / np.std(returns)
    
    cfpi = beta1 * prb + beta2 * fis + beta3 * srap
    
    return cfpi, prb, fis, srap

##############################################################################

# Implementing the profit/loss function
def calculate_profit_loss(y_true, y_pred, prices):
    profit_buy = np.sum((y_pred == 1) & (y_true == 1) * prices)
    profit_sell = np.sum((y_pred == 0) & (y_true == 0) * prices)
    loss_buy = np.sum((y_pred == 1) & (y_true == 0) * prices)
    loss_sell = np.sum((y_pred == 0) & (y_true == 1) * prices)
    return profit_buy, profit_sell, loss_buy, loss_sell

################
# Scikit learn #
##############################################################################

def custom_metric(y_true, y_pred, alpha=0.5, beta1=1):
    
    # Convert predictions to class labels (0 or 1) based on a threshold
    y_pred_labels = (y_pred > 0.5).astype(float)

    # Calculate precision and recall for buy signals
    true_positives = np.sum(y_true * y_pred_labels)
    predicted_positives = np.sum(y_pred_labels)
    actual_positives = np.sum(y_true)

    precision_buy = true_positives \
        / (predicted_positives + np.finfo(float).eps)
    recall_buy = true_positives \
        / (actual_positives + np.finfo(float).eps)

    f1_buy = 2 * (precision_buy * recall_buy) \
        / (precision_buy + recall_buy + np.finfo(float).eps)

    # Calculate precision and recall for sell signals
    true_negatives = np.sum((1 - y_true) * (1 - y_pred_labels))
    predicted_negatives = np.sum(1 - y_pred_labels)
    actual_negatives = np.sum(1 - y_true)

    precision_sell = true_negatives \
        / (predicted_negatives + np.finfo(float).eps)
    recall_sell = true_negatives \
        / (actual_negatives + np.finfo(float).eps)

    f1_sell = 2 * (
        precision_sell * recall_sell) \
            / (precision_sell + recall_sell + np.finfo(float).eps
    )

    # PRB (Precision-Recall Balance)
    prb = alpha * f1_buy + (1 - alpha) * f1_sell

    # Composite Financial Performance Index (CFPI)
    cfpi = beta1 * prb

    return cfpi

##############################################################################
