"""Feature engineering for solar production forecasting."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def engineer_features(df, include_physics=True):
    """
    Create features for the model.
    
    Args:
        df: DataFrame with preprocessed data
        include_physics: Whether to include physics-based features
        
    Returns:
        DataFrame with engineered features
    """
    # Copy dataframe to avoid modifying the original
    data = df.copy()
    
    # Time-based features
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    data['day_sin'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
    data['day_cos'] = np.cos(2 * np.pi * data.index.dayofyear / 365)
    data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    data['weekend'] = data.index.dayofweek >= 5
    
    # Lag features for time series
    for lag in range(1, 25):
        data[f'E_ac_lag_{lag}'] = data['E_ac'].shift(lag)
        
    # Moving averages
    data['E_ac_ma_6h'] = data['E_ac'].rolling(window=6).mean()
    data['E_ac_ma_24h'] = data['E_ac'].rolling(window=24).mean()
    
    # Physics-based features
    if include_physics:
        # Temperature effect on panel efficiency
        if 'temperature_factor' not in data.columns and 'Air Temp' in data.columns:
            data['temperature_factor'] = 1 + (-0.0044) * (data['Air Temp'] - 25)
        
        # Clear sky ratio (if we have both actual and theoretical radiation)
        if 'SolRad_Hor' in data.columns:
            # Theoretical max based on zenith angle
            if 'zenith' in data.columns:
                zenith_rad = np.radians(data['zenith'])
                # Calculate theoretical clear sky radiation (simplified model)
                solar_constant = 1361  # W/m²
                # Avoid division by zero or negative values
                zenith_rad_clipped = np.clip(zenith_rad, 0, np.pi/2 - 0.01)
                theoretical_max = solar_constant * np.cos(zenith_rad_clipped)
                # Calculate clear sky ratio
                data['clear_sky_ratio'] = data['SolRad_Hor'] / theoretical_max
                # Clean up invalid values
                data['clear_sky_ratio'] = data['clear_sky_ratio'].replace([np.inf, -np.inf], np.nan)
                data['clear_sky_ratio'] = data['clear_sky_ratio'].fillna(0)
                # Cap at reasonable values
                data['clear_sky_ratio'] = np.clip(data['clear_sky_ratio'], 0, 1)
    
    # Drop rows with NaN values (from lagging operations)
    data = data.dropna()
    
    return data


def prepare_hourly_daily_sequences(df, daily_df, seq_length_hourly=24, seq_length_daily=7, forecast_horizon=1):
    """
    Prepare sequences for the model with both hourly and daily resolution.
    
    Args:
        df: DataFrame with hourly data
        daily_df: DataFrame with daily data
        seq_length_hourly: Length of hourly sequences
        seq_length_daily: Length of daily sequences
        forecast_horizon: Number of steps ahead to predict
        
    Returns:
        Tuple of input data and scalers
    """
    # Define hourly features
    hourly_features = [
        'E_ac', 'ac_power_output', 'Air Temp', 'SolRad_Hor', 'SolRad_Dif',
        'WS_10m', 'zenith', 'azimuth', 'hour_sin', 'hour_cos',
        'temperature_factor', 'clear_sky_ratio'
    ] + [f'E_ac_lag_{lag}' for lag in range(1, 25)]
    
    # Keep only available features
    hourly_features = [f for f in hourly_features if f in df.columns]
    
    # Define daily features
    daily_features = [
        'E_ac', 'ac_power_output', 'Air Temp', 'SolRad_Hor', 'SolRad_Dif',
        'WS_10m', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_weekend',
    ]
    
    # Keep only available features
    daily_features = [f for f in daily_features if f in daily_df.columns]
    
    # Prepare hourly inputs
    X_hourly = df[hourly_features].values
    
    # Prepare target (energy production)
    y = df['E_ac'].shift(-forecast_horizon).values[:-forecast_horizon]
    
    # Match X_hourly length with y
    X_hourly = X_hourly[:-forecast_horizon]
    
    # Scale the hourly features
    hourly_scaler = MinMaxScaler()
    X_hourly_scaled = hourly_scaler.fit_transform(X_hourly)
    
    # Get daily dates corresponding to hourly data
    hourly_dates = df.index[:-forecast_horizon]
    daily_dates = pd.DatetimeIndex([date.date() for date in hourly_dates])
    
    # Get daily data for each hourly timestamp
    daily_data = []
    for date in daily_dates:
        try:
            # Get the daily features for this date
            daily_row = daily_df.loc[date.strftime('%Y-%m-%d')][daily_features].values
            daily_data.append(daily_row)
        except KeyError:
            # If date not in daily_df, use zeros
            daily_data.append(np.zeros(len(daily_features)))
    
    X_daily = np.array(daily_data)
    
    # Scale the daily features
    daily_scaler = MinMaxScaler()
    X_daily_scaled = daily_scaler.fit_transform(X_daily)
    
    # Scale the target
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_hourly_seq, X_daily_seq, y_seq = create_sequences(
        X_hourly_scaled, X_daily_scaled, y_scaled, 
        seq_length_hourly, seq_length_daily
    )
    
    return (X_hourly_seq, X_daily_seq, y_seq, 
            hourly_scaler, daily_scaler, y_scaler, 
            len(hourly_features), len(daily_features))


def create_sequences(X_hourly, X_daily, y, seq_length_hourly, seq_length_daily):
    """
    Create sequences for the model.
    
    Args:
        X_hourly: Hourly features
        X_daily: Daily features
        y: Target values
        seq_length_hourly: Length of hourly sequences
        seq_length_daily: Length of daily sequences
        
    Returns:
        X_hourly_seq, X_daily_seq, y_seq
    """
    X_hourly_seq = []
    X_daily_seq = []
    y_seq = []
    
    for i in range(len(X_hourly) - seq_length_hourly):
        # Get the end index for this sequence
        end_idx = i + seq_length_hourly
        
        # Find the daily indices for this period
        # Map hourly indices to daily indices
        hourly_seq = X_hourly[i:end_idx]
        
        # For daily data, we need the surrounding days
        # Get the day index for the last hour in the sequence
        day_end_idx = i + seq_length_hourly
        day_start_idx = max(0, day_end_idx - seq_length_daily)
        daily_seq = X_daily[day_start_idx:day_end_idx]
        
        # Pad if needed
        if len(daily_seq) < seq_length_daily:
            pad_width = seq_length_daily - len(daily_seq)
            daily_seq = np.pad(daily_seq, ((pad_width, 0), (0, 0)), 'constant')
        
        X_hourly_seq.append(hourly_seq)
        X_daily_seq.append(daily_seq)
        y_seq.append(y[end_idx])
    
    return np.array(X_hourly_seq), np.array(X_daily_seq), np.array(y_seq)


def split_sequences(X_hourly_seq, X_daily_seq, y_seq, train_size=0.7, val_size=0.15):
    """
    Split sequences into training, validation, and test sets.
    
    Args:
        X_hourly_seq: Hourly sequence data
        X_daily_seq: Daily sequence data
        y_seq: Target sequence data
        train_size: Fraction of data for training
        val_size: Fraction of data for validation
        
    Returns:
        Training, validation, and test data
    """
    # Calculate split indices
    n = len(X_hourly_seq)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # Split hourly sequences
    X_hourly_train = X_hourly_seq[:train_end]
    X_hourly_val = X_hourly_seq[train_end:val_end]
    X_hourly_test = X_hourly_seq[val_end:]
    
    # Split daily sequences
    X_daily_train = X_daily_seq[:train_end]
    X_daily_val = X_daily_seq[train_end:val_end]
    X_daily_test = X_daily_seq[val_end:]
    
    # Split target values
    y_train = y_seq[:train_end]
    y_val = y_seq[train_end:val_end]
    y_test = y_seq[val_end:]
    
    print(f"Sequence split: Train={len(X_hourly_train)}, Val={len(X_hourly_val)}, Test={len(X_hourly_test)}")
    
    return (X_hourly_train, X_daily_train, y_train,
            X_hourly_val, X_daily_val, y_val,
            X_hourly_test, X_daily_test, y_test)