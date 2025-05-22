"""Main script for multi-day solar production forecasting system."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time
from pathlib import Path

# Import local modules
from config import (
    DATA_PATH, OUTPUT_DIR, PHYSICS_PARAMS, MODEL_PARAMS, 
    TRAINING_PARAMS, WEATHER_API, ENSEMBLE_CONFIG
)
from data_processing import (
    load_and_preprocess_data, split_data, create_daily_aggregates
)
from feature_engineering import (
    engineer_features, prepare_hourly_daily_sequences, split_sequences
)
from weather_integration import (
    fetch_weather_forecast, process_weather_forecast, interpolate_hourly_forecast,
    create_radiation_estimate, merge_weather_with_production_data
)
from ensemble_model import (
    create_ensemble, train_model, optimize_ensemble_weights,
    ensemble_predict, save_ensemble, load_ensemble
)
from utils import (
    create_callbacks, evaluate_model, save_artifacts
)
from visualization import (
    plot_training_history, plot_predictions, create_dashboard
)

def main():
    """Main execution function for multi-day solar production forecasting."""
    # Start timing
    start_time = time.time()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("\n" + "="*70)
    print("MULTI-DAY SOLAR PRODUCTION FORECASTING SYSTEM")
    print("="*70)
    
    # 1. Load and preprocess historical data
    print("\n1. Loading and preprocessing historical data...")
    df = load_and_preprocess_data(DATA_PATH)
    print(f"Data loaded: {len(df)} records")
    
    # 2. Fetch weather forecast data
    print("\n2. Fetching weather forecast data...")
    forecast_json = fetch_weather_forecast(
        WEATHER_API['api_key'],
        WEATHER_API['latitude'],
        WEATHER_API['longitude'],
        WEATHER_API['units']
    )
    
    if forecast_json:
        # Process forecast data
        forecast_df = process_weather_forecast(forecast_json)
        print(f"Weather forecast obtained: {len(forecast_df)} timesteps")
        
        # Interpolate to hourly data
        hourly_forecast = interpolate_hourly_forecast(forecast_df)
        print(f"Interpolated to hourly data: {len(hourly_forecast)} hours")
        
        # Add solar radiation estimates
        forecast_with_radiation = create_radiation_estimate(
            hourly_forecast,
            WEATHER_API['latitude'],
            WEATHER_API['longitude']
        )
        print("Added solar radiation estimates to forecast")
        
        # Merge with historical data
        combined_df = merge_weather_with_production_data(df, forecast_with_radiation)
        print(f"Combined data created: {len(combined_df)} records")
    else:
        print("Error fetching weather data. Using historical data only.")
        combined_df = df
    
    # 3. Create daily aggregates
    print("\n3. Creating daily aggregates...")
    daily_df = create_daily_aggregates(combined_df)
    print(f"Daily data created: {len(daily_df)} records")
    
    # 4. Feature engineering
    print("\n4. Performing feature engineering...")
    df_engineered = engineer_features(combined_df, include_physics=True)
    print(f"Features engineered: {df_engineered.shape[1]} features")
    
    # Split into historical and forecast portions
    historical_mask = df_engineered['is_forecast'] == 0 if 'is_forecast' in df_engineered.columns else slice(None)
    historical_df = df_engineered.loc[historical_mask].copy()
    
    # 5. Prepare data sequences for historical data
    print("\n5. Preparing data sequences...")
    (X_hourly_seq, X_daily_seq, y_seq, 
     hourly_scaler, daily_scaler, y_scaler, 
     num_hourly_features, num_daily_features) = prepare_hourly_daily_sequences(
         historical_df, daily_df.loc[historical_df.index.date], 
         seq_length_hourly=MODEL_PARAMS['seq_length_hourly'],
         seq_length_daily=MODEL_PARAMS['seq_length_daily'],
         forecast_horizon=MODEL_PARAMS['forecast_horizon']
    )
    print(f"Sequences created: {len(X_hourly_seq)} samples")
    
    # 6. Split data
    print("\n6. Splitting data...")
    (X_hourly_train, X_daily_train, y_train,
     X_hourly_val, X_daily_val, y_val,
     X_hourly_test, X_daily_test, y_test) = split_sequences(
         X_hourly_seq, X_daily_seq, y_seq
    )
    
    # 7. Create ensemble models
    print("\n7. Creating ensemble models...")
    # Add physics params to ensemble config
    ENSEMBLE_CONFIG['physics_params'] = PHYSICS_PARAMS
    
    # Create models
    ensemble_models = create_ensemble(
        ENSEMBLE_CONFIG, 
        num_hourly_features, 
        num_daily_features
    )
    
    print(f"Created {len(ensemble_models)} models for ensemble:")
    for model_name in ensemble_models:
        print(f"  - {model_name}")
    
    # 8. Train each model
    print("\n8. Training ensemble models...")
    
    for model_name, model in ensemble_models.items():
        print(f"\nTraining {model_name} model...")
        
        # Create model-specific output directory
        model_output_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Create callbacks
        callbacks = create_callbacks(
            output_dir=model_output_dir,
            patience=TRAINING_PARAMS['early_stopping_patience']
        )
        
        # Train the model
        history = model.fit(
            [X_hourly_train, X_daily_train],
            [y_train] * MODEL_PARAMS['forecast_horizon'],  # Replicate for multi-output model
            validation_data=([X_hourly_val, X_daily_val], [y_val] * MODEL_PARAMS['forecast_horizon']),
            batch_size=TRAINING_PARAMS['batch_size'],
            epochs=TRAINING_PARAMS['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        plot_training_history(history, model_output_dir)
        
        print(f"{model_name} model trained.")
    
    # 9. Optimize ensemble weights
    print("\n9. Optimizing ensemble weights...")
    
    # Start with equal weights
    initial_weights = {model_name: 1.0 / len(ensemble_models) for model_name in ensemble_models}
    
    # Optimize weights
    optimized_weights = optimize_ensemble_weights(
        ensemble_models,
        X_hourly_val, X_daily_val, y_val,
        initial_weights
    )
    
    print("Optimized ensemble weights:")
    for model_name, weight in optimized_weights.items():
        print(f"  - {model_name}: {weight:.4f}")
    
    # 10. Evaluate ensemble on test data
    print("\n10. Evaluating ensemble on test data...")
    
    # Make ensemble predictions
    ensemble_predictions = ensemble_predict(
        ensemble_models,
        X_hourly_test, X_daily_test,
        optimized_weights
    )
    
    # Inverse transform predictions
    if y_scaler:
        y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        ensemble_preds_orig = y_scaler.inverse_transform(ensemble_predictions)
    else:
        y_test_orig = y_test
        ensemble_preds_orig = ensemble_predictions
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_orig, ensemble_preds_orig)
    mse = mean_squared_error(y_test_orig, ensemble_preds_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, ensemble_preds_orig)
    
    print(f"Ensemble Performance:")
    print(f"  MAE: {mae:.2f} Wh")
    print(f"  RMSE: {rmse:.2f} Wh")
    print(f"  R²: {r2:.4f}")
    
    # 11. Generate future predictions
    print("\n11. Generating 72-hour forecast...")
    
    # Get the latest data for prediction
    latest_hourly = df_engineered.iloc[-MODEL_PARAMS['seq_length_hourly']:]
    latest_daily = daily_df.iloc[-MODEL_PARAMS['seq_length_daily']:]
    
    # Scale input data
    hourly_scaled = hourly_scaler.transform(latest_hourly.values)
    hourly_input = hourly_scaled.reshape(1, MODEL_PARAMS['seq_length_hourly'], -1)
    
    daily_scaled = daily_scaler.transform(latest_daily.values)
    daily_input = daily_scaled.reshape(1, MODEL_PARAMS['seq_length_daily'], -1)
    
    # Generate predictions with ensemble
    future_predictions_scaled = ensemble_predict(
        ensemble_models,
        hourly_input, daily_input,
        optimized_weights
    )
    
    # Inverse transform predictions
    future_predictions = y_scaler.inverse_transform(future_predictions_scaled)
    
    # Create forecast timestamps
    last_timestamp = df.index[-1]
    forecast_timestamps = [last_timestamp + pd.Timedelta(hours=i+1) for i in range(MODEL_PARAMS['forecast_horizon'])]
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'timestamp': forecast_timestamps,
        'predicted_production': future_predictions.flatten()
    })
    
    print(f"72-hour forecast generated")
    
    # Save forecast to CSV
    forecast_path = os.path.join(OUTPUT_DIR, '72hour_forecast.csv')
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Forecast saved to {forecast_path}")
    
    # 12. Save ensemble
    print("\n12. Saving ensemble models and weights...")
    save_ensemble(ensemble_models, optimized_weights, os.path.join(OUTPUT_DIR, 'ensemble'))
    
    # 13. Create visualizations
    print("\n13. Creating visualizations...")
    create_dashboard(
        history=None,  # No single history for ensemble
        evaluation_results={
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test_orig,
            'y_pred': ensemble_preds_orig.flatten()
        },
        future_predictions=forecast_df,
        output_dir=os.path.join(OUTPUT_DIR, 'visualizations')
    )
    
    # Print execution time
    execution_time = time.time() - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nExecution completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    print("\n" + "="*70)
    print("MULTI-DAY SOLAR PRODUCTION FORECASTING SYSTEM COMPLETE")
    print("="*70)
    
    return ensemble_models, optimized_weights, forecast_df

if __name__ == "__main__":
    main()