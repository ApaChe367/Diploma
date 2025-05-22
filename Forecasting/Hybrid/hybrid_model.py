"""Hybrid model combining physics-informed and neural components."""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda, Add
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from physics_model import PhysicsLayer, physics_consistency_loss, prepare_physics_inputs
from neural_components import build_hourly_branch, build_daily_branch, build_dense_layers

def build_physics_hybrid_model(hourly_features, daily_features, config, physics_params):
    """
    Build the complete hybrid model with physics-informed components.
    
    Args:
        hourly_features: Number of hourly features
        daily_features: Number of daily features
        config: Model configuration dictionary
        physics_params: Physics parameters dictionary
        
    Returns:
        Compiled Keras model
    """
    # Define input layers
    hourly_input = Input(
        shape=(config['seq_length_hourly'], hourly_features),
        name='hourly_input'
    )
    
    daily_input = Input(
        shape=(config['seq_length_daily'], daily_features),
        name='daily_input'
    )
    
    # Build hourly branch
    hourly_output = build_hourly_branch(hourly_input, config)
    
    # Build daily branch
    daily_output = build_daily_branch(daily_input, config)
    
    # Merge branches
    merged = Concatenate(name='merge')([hourly_output, daily_output])
    
    # Build dense layers for neural component
    neural_output = build_dense_layers(merged, config)
    
    # Physics-informed component
    # Extract features needed for physics calculation (radiation, temp, zenith)
    physics_features = Lambda(
        lambda x: x[:, -1, :3],  # Take last timestep, first 3 features
        name='physics_extractor'
    )(hourly_input)
    
    # Apply physics model
    physics_layer = PhysicsLayer(
        panel_efficiency=physics_params['panel_efficiency'],
        panel_area=physics_params['panel_area'],
        temp_coeff=physics_params['temp_coeff'],
        name='physics_layer'
    )
    
    physics_output = physics_layer(physics_features)
    
    # Learnable weight parameter for combining physics and neural outputs
    alpha = K.variable(0.5, name='physics_weight')
    
    # Combine physics and neural outputs with learnable weight
    combined_output = Lambda(
        lambda inputs: inputs[0] * alpha + inputs[1] * (1 - alpha),
        name='weighted_output'
    )([physics_output, neural_output])
    
    # Define the model
    model = Model(
        inputs=[hourly_input, daily_input],
        outputs=combined_output,
        name='physics_hybrid_solar_model'
    )
    
    # Compile the model with custom loss
    optimizer = Adam(learning_rate=config['learning_rate'])
    loss = physics_consistency_loss(physics_weight=physics_params['consistency_weight'])
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
    
    return model


def create_ensemble_model(models, weights=None):
    """
    Create an ensemble model from multiple base models.
    
    Args:
        models: List of trained models
        weights: List of weights for each model
        
    Returns:
        Ensemble model
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # Ensure weights sum to 1
    weights = [w / sum(weights) for w in weights]
    
    # Get input layers from the first model
    inputs = models[0].inputs
    
    # Get outputs from all models
    outputs = [model(inputs) for model in models]
    
    # Weighted average of outputs
    ensemble_output = outputs[0] * weights[0]
    for i in range(1, len(outputs)):
        ensemble_output = Add()([ensemble_output, outputs[i] * weights[i]])
    
    # Create ensemble model
    ensemble_model = Model(inputs=inputs, outputs=ensemble_output)
    
    # Compile model
    ensemble_model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return ensemble_model