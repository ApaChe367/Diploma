"""Utility functions for solar production forecasting."""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_callbacks(output_dir, patience=10):
    """
    Create callbacks for model training.
    
    Args:
        output_dir: Directory to save model checkpoints
        patience: Patience for early stopping
        
    Returns:
        List of callbacks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    
    return callbacks


def train_model(model, X_hourly_train, X_daily_train, y_train, 
                X_hourly_val, X_daily_val, y_val, 
                batch_size=32, epochs=100, output_dir='./output'):
    """
    Train the model.
    
    Args:
        model: Model to train
        X_hourly_train: Hourly training data
        X_daily_train: Daily training data
        y_train: Training targets
        X_hourly_val: Hourly validation data
        X_daily_val: Daily validation data
        y_val: Validation targets
        batch_size: Batch size
        epochs: Number of epochs
        output_dir: Directory to save model checkpoints
        
    Returns:
        Training history
    """
    # Create callbacks
    callbacks = create_callbacks(output_dir)
    
    # Train the model
    history = model.fit(
        [X_hourly_train, X_daily_train],
        y_train,
        validation_data=([X_hourly_val, X_daily_val], y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_hourly_test, X_daily_test, y_test, y_scaler=None):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_hourly_test: Hourly test data
        X_daily_test: Daily test data
        y_test: Test targets
        y_scaler: Scaler for target values
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict([X_hourly_test, X_daily_test])
    
    # Inverse scale if scaler is provided
    if y_scaler is not None:
        y_pred = y_scaler.inverse_transform(y_pred)
        y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_test_orig = y_test
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_orig, y_pred)
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred)
    
    # Calculate MAPE only for non-zero targets
    non_zero_mask = y_test_orig > 1.0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_test_orig[non_zero_mask] - y_pred.flatten()[non_zero_mask]) / 
                              y_test_orig[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    # Return metrics
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'y_pred': y_pred.flatten(),
        'y_test': y_test_orig
    }


def save_artifacts(model, hourly_scaler, daily_scaler, y_scaler, output_dir):
    """
    Save model and scalers.
    
    Args:
        model: Trained model
        hourly_scaler: Scaler for hourly features
        daily_scaler: Scaler for daily features
        y_scaler: Scaler for targets
        output_dir: Directory to save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model.save(os.path.join(output_dir, 'model.h5'))
    
    # Save scalers
    joblib.dump(hourly_scaler, os.path.join(output_dir, 'hourly_scaler.pkl'))
    joblib.dump(daily_scaler, os.path.join(output_dir, 'daily_scaler.pkl'))
    joblib.dump(y_scaler, os.path.join(output_dir, 'y_scaler.pkl'))
    
    print(f"Model and scalers saved to {output_dir}")


def load_artifacts(output_dir):
    """
    Load model and scalers.
    
    Args:
        output_dir: Directory with saved artifacts
        
    Returns:
        model, hourly_scaler, daily_scaler, y_scaler
    """
    # Load model
    model = tf.keras.models.load_model(
        os.path.join(output_dir, 'model.h5'),
        custom_objects={
            'loss': physics_consistency_loss(),
            'PhysicsLayer': PhysicsLayer
        }
    )
    
    # Load scalers
    hourly_scaler = joblib.load(os.path.join(output_dir, 'hourly_scaler.pkl'))
    daily_scaler = joblib.load(os.path.join(output_dir, 'daily_scaler.pkl'))
    y_scaler = joblib.load(os.path.join(output_dir, 'y_scaler.pkl'))
    
    return model, hourly_scaler, daily_scaler, y_scaler


def predict_future(model, last_hourly_data, last_daily_data, 
                   hourly_scaler, daily_scaler, y_scaler,
                   seq_length_hourly, seq_length_daily,
                   future_steps=24):
    """
    Predict future energy production.
    
    Args:
        model: Trained model
        last_hourly_data: Recent hourly data
        last_daily_data: Recent daily data
        hourly_scaler: Scaler for hourly features
        daily_scaler: Scaler for daily features
        y_scaler: Scaler for targets
        seq_length_hourly: Length of hourly sequences
        seq_length_daily: Length of daily sequences
        future_steps: Number of steps to predict
        
    Returns:
        DataFrame with predictions
    """
    # Create copies to avoid modifying originals
    hourly_data = last_hourly_data.copy()
    daily_data = last_daily_data.copy()
    
    # Initialize results
    results = []
    last_date = hourly_data.index[-1]
    
    for step in range(future_steps):
        # Create input sequences
        hourly_features = hourly_data.iloc[-seq_length_hourly:].values
        hourly_scaled = hourly_scaler.transform(hourly_features)
        hourly_seq = hourly_scaled.reshape(1, seq_length_hourly, -1)
        
        # Get corresponding daily data
        daily_dates = [d.date() for d in hourly_data.index[-seq_length_daily:]]
        daily_features = []
        for date in daily_dates:
            try:
                daily_row = daily_data.loc[date.strftime('%Y-%m-%d')].values
                daily_features.append(daily_row)
            except:
                # Use the last available daily data
                daily_features.append(daily_data.iloc[-1].values)
        
        daily_features = np.array(daily_features)
        daily_scaled = daily_scaler.transform(daily_features)
        daily_seq = daily_scaled.reshape(1, seq_length_daily, -1)
        
        # Make prediction
        pred_scaled = model.predict([hourly_seq, daily_seq])
        pred = y_scaler.inverse_transform(pred_scaled)[0][0]
        
        # Next timestamp
        next_date = last_date + pd.Timedelta(hours=step+1)
        
        # Store result
        results.append({
            'timestamp': next_date,
            'predicted_energy': pred
        })
        
        # Update hourly data with the new prediction
        # This is a simplified approach - in a real system, you would
        # use actual weather forecasts for the future hours
        new_row = hourly_data.iloc[-1].copy()
        new_row['E_ac'] = pred
        new_row.name = next_date
        hourly_data = pd.concat([hourly_data, pd.DataFrame([new_row])])
        
        # Update daily data if needed
        if next_date.date() not in daily_data.index:
            # Create new daily row from recent hourly data
            new_daily_row = hourly_data.resample('D').agg({
                'E_ac': 'sum',
                'ac_power_output': 'mean',
                'Air Temp': 'mean',
                'SolRad_Hor': 'sum',
                'SolRad_Dif': 'sum',
                'WS_10m': 'mean',
                'zenith': 'mean',
                'azimuth': 'mean',
                'temperature_factor': 'mean',
            }).iloc[-1]
            
            new_daily_row.name = next_date.date()
            daily_data = pd.concat([daily_data, pd.DataFrame([new_daily_row])])
    
    # Convert results to DataFrame
    return pd.DataFrame(results)