"""Neural network components for solar production forecasting."""

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Concatenate, Input, Bidirectional
import tensorflow.keras.backend as K

def build_hourly_branch(hourly_input, config):
    """
    Build the hourly data processing branch.
    
    Args:
        hourly_input: Input tensor for hourly data
        config: Model configuration dictionary
        
    Returns:
        Output tensor from the hourly branch
    """
    # Convolutional layers for feature extraction
    conv1 = Conv1D(
        filters=config['cnn_filters'],
        kernel_size=config['cnn_kernel_size'],
        activation='relu',
        padding='same',
        name='hourly_conv1'
    )(hourly_input)
    
    conv2 = Conv1D(
        filters=config['cnn_filters'],
        kernel_size=config['cnn_kernel_size'],
        activation='relu',
        padding='same',
        name='hourly_conv2'
    )(conv1)
    
    # Max pooling to reduce dimensionality
    pool = MaxPooling1D(pool_size=2, name='hourly_pool')(conv2)
    
    # Dropout for regularization
    dropout1 = Dropout(config['dropout_rate'], name='hourly_dropout1')(pool)
    
    # LSTM layers for temporal dependencies
    lstm1 = Bidirectional(
        LSTM(config['hourly_lstm_units'], return_sequences=True, name='hourly_lstm1'),
        name='hourly_bidirectional'
    )(dropout1)
    
    lstm2 = LSTM(config['hourly_lstm_units'], name='hourly_lstm2')(lstm1)
    
    # Final dropout
    hourly_output = Dropout(config['dropout_rate'], name='hourly_dropout2')(lstm2)
    
    return hourly_output


def build_daily_branch(daily_input, config):
    """
    Build the daily data processing branch.
    
    Args:
        daily_input: Input tensor for daily data
        config: Model configuration dictionary
        
    Returns:
        Output tensor from the daily branch
    """
    # LSTM layers for temporal dependencies
    lstm1 = LSTM(
        config['daily_lstm_units'],
        return_sequences=True,
        name='daily_lstm1'
    )(daily_input)
    
    # Dropout for regularization
    dropout = Dropout(config['dropout_rate'], name='daily_dropout')(lstm1)
    
    # Second LSTM layer
    lstm2 = LSTM(config['daily_lstm_units'], name='daily_lstm2')(dropout)
    
    return lstm2


def build_dense_layers(merged_inputs, config):
    """
    Build the dense layers for the final prediction.
    
    Args:
        merged_inputs: Merged inputs from all branches
        config: Model configuration dictionary
        
    Returns:
        Output tensor from dense layers
    """
    # First dense layer
    dense1 = Dense(
        config['dense_units'],
        activation='relu',
        name='dense1'
    )(merged_inputs)
    
    # Dropout for regularization
    dropout = Dropout(config['dropout_rate'], name='dense_dropout')(dense1)
    
    # Second dense layer
    dense2 = Dense(
        config['dense_units'] // 2,
        activation='relu',
        name='dense2'
    )(dropout)
    
    # Output layer
    output = Dense(1, activation='linear', name='output')(dense2)
    
    return output