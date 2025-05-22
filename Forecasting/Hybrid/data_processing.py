"""Data loading and preprocessing for solar production forecasting."""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime, timedelta


def check_data_path(data_path):
    """Check if the data path exists and is valid."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not data_path.endswith('.csv'):
        raise ValueError("Only CSV data files are supported")
    return True


def load_and_preprocess_data(data_path):
    """
    Load and preprocess solar production data.
    
    Args:
        data_path: Path to the CSV data file
        
    Returns:
        DataFrame with preprocessed data
    """
    # Check data path
    check_data_path(data_path)
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Get the datetime column name (should be first column)
    datetime_col_name = df.columns[0]
    print(f"Datetime column name: {datetime_col_name}")
    
    # Convert datetime strings to datetime objects
    try:
        # Step 1: Convert the datetime strings to datetime objects
        datetime_series = pd.to_datetime(df[datetime_col_name], errors='coerce')
        
        # Step 2: Remove timezone info if present (to avoid DatetimeIndex issues)
        if datetime_series.dt.tz is not None:
            datetime_series = datetime_series.dt.tz_localize(None)
        
        # Step 3: Set as index
        df_copy = df.copy()
        df_copy.index = datetime_series
        df_copy = df_copy.drop(columns=[datetime_col_name])  # Remove the original datetime column
        df = df_copy
        
        print("Datetime conversion successful!")
    except Exception as e:
        print(f"Automatic conversion failed: {e}")
        print("Using manual approach...")
        
        # Manual conversion - create datetime index from scratch
        # Assuming hourly data starting from first timestamp
        start_time = df.iloc[0, 0]
        # Remove timezone info from string if present
        if '+' in start_time:
            start_time = start_time.split('+')[0]
        elif 'T' in start_time and len(start_time.split('T')[1]) > 8:
            # Handle other timezone formats
            start_time = start_time[:19]  # Keep only YYYY-MM-DD HH:MM:SS
        
        # Create datetime range
        start_dt = pd.to_datetime(start_time)
        n_hours = len(df)
        new_index = pd.date_range(start=start_dt, periods=n_hours, freq='H')
        
        # Set new index and remove datetime column
        df.index = new_index
        df = df.drop(columns=[datetime_col_name])
    
    # Add time-based features
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek
    
    # Handle missing values
    print("Handling missing values...")
    
    # Wind speed is completely missing - use a reasonable default
    if 'WS_10m' in df.columns and df['WS_10m'].isna().all():
        print("Setting default wind speed (3 m/s)")
        df['WS_10m'] = 3.0
    
    # Mismatch calculation
    if 'mismatch' in df.columns and df['mismatch'].isna().all():
        print("Calculating mismatch values")
        df['mismatch'] = df['ac_power_output'] / 1000 - df['Load (kW)']
    
    # Fill remaining missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df


def fetch_weather_forecast(lat, lon, api_key):
    """
    Fetch weather forecast data from OpenWeatherMap API.
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: OpenWeatherMap API key
        
    Returns:
        JSON response with weather forecast data
    """
    # Example API call to OpenWeatherMap 5-day/3-hour forecast
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching forecast data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


def process_weather_forecast(forecast_json):
    """
    Process raw forecast JSON into a pandas DataFrame.
    
    Args:
        forecast_json: JSON response from weather API
        
    Returns:
        DataFrame with processed forecast data
    """
    if not forecast_json or 'list' not in forecast_json:
        return None
        
    forecast_data = []
    
    # Extract relevant features from forecast
    for item in forecast_json['list']:
        timestamp = datetime.fromtimestamp(item['dt'])
        
        forecast_data.append({
            'datetime': timestamp,
            'forecast_temp': item['main']['temp'],
            'forecast_humidity': item['main']['humidity'],
            'forecast_pressure': item['main']['pressure'],
            'forecast_clouds': item['clouds']['all'],  # Cloud cover percentage
            'forecast_wind_speed': item['wind']['speed'],
            'forecast_wind_direction': item['wind'].get('deg', 0),
            'forecast_precipitation': item.get('rain', {}).get('3h', 0),  # 3-hour precipitation
            'forecast_weather_code': item['weather'][0]['id'],
            'forecast_weather_main': item['weather'][0]['main'],
        })
    
    # Convert to DataFrame
    df_forecast = pd.DataFrame(forecast_data)
    df_forecast.set_index('datetime', inplace=True)
    
    return df_forecast


def split_data(df, train_size=0.7, val_size=0.15):
    """
    Split the data into training, validation, and test sets.
    
    Args:
        df: DataFrame with preprocessed data
        train_size: Fraction of data for training
        val_size: Fraction of data for validation
        
    Returns:
        train_df, val_df, test_df
    """
    # Sort by time
    df = df.sort_index()
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # Split data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


def create_daily_aggregates(df):
    """
    Create daily aggregated data from hourly data.
    
    Args:
        df: DataFrame with hourly data
        
    Returns:
        DataFrame with daily aggregated data
    """
    # Create daily aggregated data
    daily_df = df.resample('D').agg({
        'E_ac': 'sum',                 # Total daily energy production
        'ac_power_output': 'mean',     # Average power output
        'Air Temp': 'mean',            # Average temperature
        'SolRad_Hor': 'sum',           # Total horizontal solar radiation
        'SolRad_Dif': 'sum',           # Total diffuse solar radiation
        'WS_10m': 'mean',              # Average wind speed
        'zenith': 'mean',              # Average solar zenith angle
        'azimuth': 'mean',             # Average solar azimuth angle
        'temperature_factor': 'mean',  # Average temperature factor
    })
    
    # Add time-based features
    daily_df['day_of_week'] = daily_df.index.dayofweek
    daily_df['day_of_year'] = daily_df.index.dayofyear
    daily_df['month'] = daily_df.index.month
    daily_df['is_weekend'] = (daily_df.index.dayofweek >= 5).astype(int)
    
    return daily_df