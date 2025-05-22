"""Configuration for the solar production forecasting system."""

# Data paths
DATA_PATH = "C:/Users/kigeorgiou/Desktop/Forcasting/Results/forecast_data.csv"
OUTPUT_DIR = "C:/Users/kigeorgiou/Desktop/Forcasting/Results/Ensemble_Forecast"

# Physics parameters
PHYSICS_PARAMS = {
    'panel_efficiency': 0.146,
    'panel_area': 1.642,
    'temp_coeff': -0.0044,
    'consistency_weight': 0.1,
}

# Model parameters
MODEL_PARAMS = {
    'seq_length_hourly': 48,     # Using longer history for multi-day forecasting
    'seq_length_daily': 14,      # Two weeks of daily data
    'forecast_horizon': 72,      # 3 days ahead (72 hours)
    'hourly_lstm_units': 64,
    'daily_lstm_units': 32,
    'cnn_filters': 64,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
}

# Training parameters
TRAINING_PARAMS = {
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 15,
}

# OpenWeather API parameters
WEATHER_API = {
    'api_key': "7273588818d8b2bb8597ee797baf4935",  # Replace with your actual API key
    'latitude': 37.98983,
    'longitude': 23.74328,
    'units': 'metric',
    'forecast_days': 5,          # OpenWeather provides 5-day forecasts
}

# Ensemble configuration
ENSEMBLE_CONFIG = {
    'num_models': 5,             # Number of models in ensemble
    'model_types': [
        'physics_hybrid',        # Physics-informed model
        'lstm_only',             # Pure LSTM model
        'cnn_lstm',              # CNN-LSTM hybrid
        'attention',             # Attention-based model
        'weather_specialized'    # Model specialized for weather patterns
    ],
    'weights': [0.3, 0.2, 0.2, 0.15, 0.15]  # Initial weights (will be optimized)
}