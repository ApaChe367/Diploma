"""Ensemble model for solar production forecasting."""

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Concatenate, Lambda, Add
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import os
import joblib

from physics_model import PhysicsLayer, physics_consistency_loss
from neural_components import build_hourly_branch, build_daily_branch, build_dense_layers

def build_lstm_model(hourly_features, daily_features, config):
    """Build a pure LSTM model for the ensemble."""
    # Hourly input
    hourly_input = Input(shape=(config['seq_length_hourly'], hourly_features), name='hourly_input')
    
    # Daily input
    daily_input = Input(shape=(config['seq_length_daily'], daily_features), name='daily_input')
    
    # Process hourly data with LSTM
    hourly_lstm = Bidirectional(LSTM(config['hourly_lstm_units'], return_sequences=True))(hourly_input)
    hourly_lstm = Dropout(config['dropout_rate'])(hourly_lstm)
    hourly_lstm = LSTM(config['hourly_lstm_units'] // 2)(hourly_lstm)
    
    # Process daily data with LSTM
    daily_lstm = LSTM(config['daily_lstm_units'], return_sequences=True)(daily_input)
    daily_lstm = Dropout(config['dropout_rate'])(daily_lstm)
    daily_lstm = LSTM(config['daily_lstm_units'] // 2)(daily_lstm)
    
    # Merge branches
    merged = Concatenate()([hourly_lstm, daily_lstm])
    
    # Dense layers for output
    dense = Dense(64, activation='relu')(merged)
    dense = Dropout(config['dropout_rate'])(dense)
    
    # Output multiple steps ahead (forecast_horizon)
    outputs = []
    for i in range(config['forecast_horizon']):
        output_i = Dense(1, name=f'output_{i}')(dense)
        outputs.append(output_i)
    
    # Create model
    model = Model(inputs=[hourly_input, daily_input], outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_cnn_lstm_model(hourly_features, daily_features, config):
    """Build a CNN-LSTM model for the ensemble."""
    # Hourly input
    hourly_input = Input(shape=(config['seq_length_hourly'], hourly_features), name='hourly_input')
    
    # Daily input
    daily_input = Input(shape=(config['seq_length_daily'], daily_features), name='daily_input')
    
    # Process hourly data with CNN
    hourly_conv = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(hourly_input)
    hourly_conv = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(hourly_conv)
    hourly_pool = MaxPooling1D(pool_size=2)(hourly_conv)
    hourly_conv = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(hourly_pool)
    hourly_pool = MaxPooling1D(pool_size=2)(hourly_conv)
    
    # Process pooled features with LSTM
    hourly_lstm = LSTM(config['hourly_lstm_units'])(hourly_pool)
    
    # Process daily data with LSTM
    daily_lstm = LSTM(config['daily_lstm_units'])(daily_input)
    
    # Merge branches
    merged = Concatenate()([hourly_lstm, daily_lstm])
    
    # Dense layers for output
    dense = Dense(128, activation='relu')(merged)
    dense = Dropout(config['dropout_rate'])(dense)
    dense = Dense(64, activation='relu')(dense)
    
    # Output multiple steps ahead (forecast_horizon)
    outputs = []
    for i in range(config['forecast_horizon']):
        output_i = Dense(1, name=f'output_{i}')(dense)
        outputs.append(output_i)
    
    # Create model
    model = Model(inputs=[hourly_input, daily_input], outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_attention_model(hourly_features, daily_features, config):
    """Build an attention-based model for the ensemble."""
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
    
    # Hourly input
    hourly_input = Input(shape=(config['seq_length_hourly'], hourly_features), name='hourly_input')
    
    # Daily input
    daily_input = Input(shape=(config['seq_length_daily'], daily_features), name='daily_input')
    
    # Self-attention for hourly data
    hourly_attention = MultiHeadAttention(
        num_heads=4, key_dim=hourly_features
    )(hourly_input, hourly_input)
    hourly_attention = LayerNormalization(epsilon=1e-6)(hourly_input + hourly_attention)
    
    # Process with LSTM
    hourly_lstm = LSTM(config['hourly_lstm_units'])(hourly_attention)
    
    # Process daily data with LSTM
    daily_lstm = LSTM(config['daily_lstm_units'])(daily_input)
    
    # Merge branches
    merged = Concatenate()([hourly_lstm, daily_lstm])
    
    # Dense layers for output
    dense = Dense(64, activation='relu')(merged)
    dense = Dropout(config['dropout_rate'])(dense)
    
    # Output multiple steps ahead (forecast_horizon)
    outputs = []
    for i in range(config['forecast_horizon']):
        output_i = Dense(1, name=f'output_{i}')(dense)
        outputs.append(output_i)
    
    # Create model
    model = Model(inputs=[hourly_input, daily_input], outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_weather_specialized_model(hourly_features, daily_features, config):
    """Build a model specialized for weather pattern recognition."""
    # Hourly input
    hourly_input = Input(shape=(config['seq_length_hourly'], hourly_features), name='hourly_input')
    
    # Daily input
    daily_input = Input(shape=(config['seq_length_daily'], daily_features), name='daily_input')
    
    # Weather feature extraction with CNN
    hourly_conv = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(hourly_input)
    hourly_conv = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(hourly_conv)
    hourly_pool = MaxPooling1D(pool_size=2)(hourly_conv)
    
    # LSTM for temporal patterns
    hourly_lstm = LSTM(config['hourly_lstm_units'], return_sequences=True)(hourly_pool)
    hourly_lstm = LSTM(config['hourly_lstm_units'] // 2)(hourly_lstm)
    
    # Process daily data
    daily_lstm = LSTM(config['daily_lstm_units'])(daily_input)
    
    # Merge branches
    merged = Concatenate()([hourly_lstm, daily_lstm])
    
    # Dense layers for output with weather specialization
    dense = Dense(128, activation='relu')(merged)
    dense = Dropout(config['dropout_rate'])(dense)
    dense = Dense(64, activation='relu')(dense)
    
    # Add weather specialization (more units for handling complex patterns)
    dense = Dense(32, activation='relu')(dense)
    
    # Output multiple steps ahead (forecast_horizon)
    outputs = []
    for i in range(config['forecast_horizon']):
        output_i = Dense(1, name=f'output_{i}')(dense)
        outputs.append(output_i)
    
    # Create model
    model = Model(inputs=[hourly_input, daily_input], outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_physics_hybrid_model(hourly_features, daily_features, config, physics_params):
    """Build a physics-informed hybrid model for the ensemble."""
    # Hourly input
    hourly_input = Input(shape=(config['seq_length_hourly'], hourly_features), name='hourly_input')
    
    # Daily input
    daily_input = Input(shape=(config['seq_length_daily'], daily_features), name='daily_input')
    
    # Process hourly data
    hourly_branch = build_hourly_branch(hourly_input, config)
    
    # Process daily data
    daily_branch = build_daily_branch(daily_input, config)
    
    # Merge branches
    merged = Concatenate(name='merge')([hourly_branch, daily_branch])
    
    # Dense network for neural component
    neural_dense = Dense(64, activation='relu')(merged)
    neural_dense = Dropout(config['dropout_rate'])(neural_dense)
    
    # Physics layer for physical estimates
    # Extract features needed for physics (radiation, temp, zenith)
    physics_features = Lambda(
        lambda x: x[:, -1, :3],  # Last timestep, first 3 features
        name='physics_extractor'
    )(hourly_input)
    
    physics_layer = PhysicsLayer(
        panel_efficiency=physics_params['panel_efficiency'],
        panel_area=physics_params['panel_area'],
        temp_coeff=physics_params['temp_coeff'],
        name='physics_layer'
    )
    
    physics_base = physics_layer(physics_features)
    
    # Output multiple steps ahead
    outputs = []
    for i in range(config['forecast_horizon']):
        # Neural output for this step
        neural_output = Dense(1, name=f'neural_output_{i}')(neural_dense)
        
        # Physics decay with time (physics becomes less reliable as we predict further ahead)
        physics_weight = 0.7 * (0.9 ** i)  # Exponential decay of physics weight
        
        # Combine neural and physics for this timestep
        combined_output = Lambda(
            lambda x: x[0] * physics_weight + x[1] * (1 - physics_weight),
            name=f'output_{i}'
        )([physics_base, neural_output])
        
        outputs.append(combined_output)
    
    # Create model
    model = Model(inputs=[hourly_input, daily_input], outputs=outputs)
    
    # Compile model with custom loss
    loss = physics_consistency_loss(physics_weight=physics_params['consistency_weight'])
    
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['mae']
    )
    
    return model

def create_ensemble(model_configs, hourly_features, daily_features, weights=None):
    """
    Create an ensemble of models for solar production forecasting.
    
    Args:
        model_configs: Dictionary with model configurations
        hourly_features: Number of hourly features
        daily_features: Number of daily features
        weights: Optional weights for ensemble models
        
    Returns:
        Dictionary with ensemble models
    """
    models = {}
    
    # Build each model type
    if 'physics_hybrid' in model_configs['model_types']:
        models['physics_hybrid'] = build_physics_hybrid_model(
            hourly_features, daily_features,
            model_configs, model_configs['physics_params']
        )
    
    if 'lstm_only' in model_configs['model_types']:
        models['lstm_only'] = build_lstm_model(
            hourly_features, daily_features, model_configs
        )
    
    if 'cnn_lstm' in model_configs['model_types']:
        models['cnn_lstm'] = build_cnn_lstm_model(
            hourly_features, daily_features, model_configs
        )
    
    if 'attention' in model_configs['model_types']:
        models['attention'] = build_attention_model(
            hourly_features, daily_features, model_configs
        )
    
    if 'weather_specialized' in model_configs['model_types']:
        models['weather_specialized'] = build_weather_specialized_model(
            hourly_features, daily_features, model_configs
        )
    
    return models

def ensemble_predict(models, hourly_input, daily_input, weights=None):
    """
    Make predictions using ensemble of models.
    
    Args:
        models: Dictionary of trained models
        hourly_input: Hourly input data
        daily_input: Daily input data
        weights: Optional weights for ensemble models
        
    Returns:
        Array with ensemble predictions
    """
    if weights is None:
        # Equal weights if not specified
        weights = {model_name: 1.0 / len(models) for model_name in models}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {name: weight / total_weight for name, weight in weights.items()}
    
    # Make predictions with each model
    all_predictions = {}
    for model_name, model in models.items():
        model_pred = model.predict([hourly_input, daily_input])
        
        # Handle case where model returns a list of outputs (one per timestep)
        if isinstance(model_pred, list):
            # Concatenate predictions into a single array of shape [samples, timesteps]
            model_pred = np.concatenate([pred.reshape(-1, 1) for pred in model_pred], axis=1)
        
        all_predictions[model_name] = model_pred
    
    # Combine predictions with weights
    ensemble_pred = np.zeros_like(next(iter(all_predictions.values())))
    for model_name, pred in all_predictions.items():
        ensemble_pred += weights[model_name] * pred
    
    return ensemble_pred

def optimize_ensemble_weights(models, X_hourly_val, X_daily_val, y_val, 
                              initial_weights=None, n_iterations=100):
    """
    Optimize ensemble weights using validation data.
    
    Args:
        models: Dictionary of trained models
        X_hourly_val: Hourly validation data
        X_daily_val: Daily validation data
        y_val: Validation targets
        initial_weights: Initial weights (optional)
        n_iterations: Number of optimization iterations
        
    Returns:
        Optimized weights dictionary
    """
    from scipy.optimize import minimize
    import numpy as np
    
    # Get model predictions on validation set
    model_predictions = {}
    for model_name, model in models.items():
        preds = model.predict([X_hourly_val, X_daily_val])
        
        # Handle case where model returns a list of outputs
        if isinstance(preds, list):
            # Convert to array of shape [samples, timesteps]
            preds = np.concatenate([p.reshape(-1, 1) for p in preds], axis=1)
        
        model_predictions[model_name] = preds
    
    # Define objective function (MSE)
    def objective(weights_array):
        # Convert weights array to dictionary
        weights_dict = {name: weight for name, weight in zip(models.keys(), weights_array)}
        
        # Normalize weights to sum to 1
        total = sum(weights_dict.values())
        weights_dict = {name: weight / total for name, weight in weights_dict.items()}
        
        # Compute weighted ensemble prediction
        ensemble_pred = np.zeros_like(next(iter(model_predictions.values())))
        for model_name, pred in model_predictions.items():
            ensemble_pred += weights_dict[model_name] * pred
        
        # Compute MSE
        mse = np.mean((ensemble_pred - y_val) ** 2)
        return mse
    
    # Initial weights
    if initial_weights is None:
        initial_weights = np.ones(len(models)) / len(models)
    else:
        initial_weights = np.array([initial_weights[name] for name in models.keys()])
    
    # Constraints: weights sum to 1 and are non-negative
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 1.0) for _ in range(len(models))]
    
    # Optimize weights
    result = minimize(
        objective, initial_weights, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': n_iterations}
    )
    
    # Convert optimized weights to dictionary
    optimized_weights = {name: weight for name, weight in zip(models.keys(), result.x)}
    
    return optimized_weights

def save_ensemble(models, weights, output_dir):
    """
    Save ensemble models and weights.
    
    Args:
        models: Dictionary of trained models
        weights: Ensemble weights dictionary
        output_dir: Output directory
    """
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save weights
    with open(os.path.join(output_dir, 'ensemble_weights.json'), 'w') as f:
        json.dump(weights, f, indent=4)
    
    # Save each model
    for model_name, model in models.items():
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir)
    
    print(f"Ensemble saved to {output_dir}")

def load_ensemble(output_dir):
    """
    Load ensemble models and weights.
    
    Args:
        output_dir: Directory with saved ensemble
        
    Returns:
        models, weights
    """
    import os
    import json
    
    # Load weights
    with open(os.path.join(output_dir, 'ensemble_weights.json'), 'r') as f:
        weights = json.load(f)
    
    # Load models
    models = {}
    for model_name in weights.keys():
        model_dir = os.path.join(output_dir, model_name)
        
        # Handle custom objects for physics model
        if model_name == 'physics_hybrid':
            models[model_name] = tf.keras.models.load_model(
                model_dir,
                custom_objects={
                    'loss': physics_consistency_loss(),
                    'PhysicsLayer': PhysicsLayer
                }
            )
        else:
            models[model_name] = tf.keras.models.load_model(model_dir)
    
    return models, weights