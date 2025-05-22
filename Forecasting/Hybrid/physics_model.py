"""Physics-informed components for solar production forecasting."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Layer
import tensorflow.keras.backend as K

class PhysicsLayer(Layer):
    """Custom Keras layer implementing physics-based solar production calculation."""
    
    def __init__(self, panel_efficiency=0.15, panel_area=1.6, temp_coeff=-0.0044, **kwargs):
        """
        Initialize the physics layer with solar panel parameters.
        
        Args:
            panel_efficiency: Solar panel efficiency (between 0 and 1)
            panel_area: Solar panel area in m²
            temp_coeff: Temperature coefficient for efficiency (typically negative)
        """
        super(PhysicsLayer, self).__init__(**kwargs)
        self.panel_efficiency = panel_efficiency
        self.panel_area = panel_area
        self.temp_coeff = temp_coeff
    
    def build(self, input_shape):
        """Build the layer."""
        super(PhysicsLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass through the physics layer.
        
        Args:
            inputs: Tensor with shape [..., features]
                where features are: [radiation, temperature, zenith_angle, ...]
                
        Returns:
            Tensor with physics-based energy production estimate
        """
        # Extract relevant features
        # Assuming inputs has dimensions [..., features]
        # and the feature indices are consistent
        solar_radiation = inputs[..., 0]  # Horizontal solar radiation
        temperature = inputs[..., 1]      # Ambient temperature
        zenith = inputs[..., 2]           # Solar zenith angle
        
        # Convert zenith to radians and calculate cosine factor
        zenith_rad = tf.clip_by_value(zenith * np.pi / 180.0, 0, np.pi/2)
        cos_factor = tf.cos(zenith_rad)
        
        # Temperature correction factor
        temp_factor = 1.0 + self.temp_coeff * (temperature - 25.0)
        
        # Basic physical model calculation
        physics_output = (
            solar_radiation * 
            cos_factor * 
            self.panel_efficiency * 
            self.panel_area * 
            temp_factor
        )
        
        # Ensure non-negative output
        physics_output = tf.maximum(physics_output, 0.0)
        
        # Reshape to match expected output shape
        return tf.expand_dims(physics_output, -1)
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return input_shape[:-1] + (1,)
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super(PhysicsLayer, self).get_config()
        config.update({
            'panel_efficiency': self.panel_efficiency,
            'panel_area': self.panel_area,
            'temp_coeff': self.temp_coeff,
        })
        return config


def physics_consistency_loss(physics_weight=0.1):
    """
    Custom loss function that encourages consistency with physical laws.
    
    Args:
        physics_weight: Weight of physics consistency term
        
    Returns:
        Loss function that can be used in model compilation
    """
    def loss(y_true, y_pred):
        # Standard mean squared error
        mse = K.mean(K.square(y_true - y_pred))
        
        # Physics consistency terms could include:
        # 1. Non-negative energy production
        non_negative_constraint = K.mean(K.relu(-y_pred))
        
        # 2. Energy follows solar pattern
        # This is a simplified constraint based on common knowledge
        # that energy production should be close to zero at night
        
        # Full physics-based loss
        return mse + physics_weight * non_negative_constraint
    
    return loss


def prepare_physics_inputs(X_hourly, feature_indices):
    """
    Prepare inputs for the physics model from the full feature set.
    
    Args:
        X_hourly: Full hourly features
        feature_indices: Dictionary mapping feature names to indices
        
    Returns:
        Tensor with only the features needed for physics calculations
    """
    # Extract only the features needed for physics calculations
    required_features = ['SolRad_Hor', 'Air Temp', 'zenith']
    physics_indices = [feature_indices.get(feature, 0) for feature in required_features]
    
    # Select the features
    physics_inputs = tf.gather(X_hourly, physics_indices, axis=-1)
    
    return physics_inputs