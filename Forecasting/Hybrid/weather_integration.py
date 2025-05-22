"""Weather data integration with OpenWeather API."""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_weather_forecast(api_key, lat, lon, units='metric'):
    """
    Fetch 5-day weather forecast from OpenWeather API.
    
    Args:
        api_key: OpenWeather API key
        lat: Latitude
        lon: Longitude
        units: Units (metric or imperial)
        
    Returns:
        JSON response or None if request failed
    """
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units={units}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching forecast: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception during API request: {str(e)}")
        return None

def process_weather_forecast(forecast_json):
    """
    Process raw forecast JSON into a pandas DataFrame.
    
    Args:
        forecast_json: JSON response from OpenWeather API
        
    Returns:
        DataFrame with processed forecast data
    """
    if not forecast_json or 'list' not in forecast_json:
        print("Invalid forecast data received")
        return None
    
    forecast_data = []
    
    # Extract relevant features from forecast
    for item in forecast_json['list']:
        timestamp = datetime.fromtimestamp(item['dt'])
        
        # Extract all needed weather parameters
        data_point = {
            'datetime': timestamp,
            'temp': item['main']['temp'],
            'feels_like': item['main']['feels_like'],
            'pressure': item['main']['pressure'],
            'humidity': item['main']['humidity'],
            'clouds': item['clouds']['all'],
            'wind_speed': item['wind']['speed'],
            'wind_deg': item['wind'].get('deg', 0),
            'weather_id': item['weather'][0]['id'],
            'weather_main': item['weather'][0]['main'],
            'weather_description': item['weather'][0]['description'],
        }
        
        # Extract precipitation if available
        if 'rain' in item and '3h' in item['rain']:
            data_point['rain_3h'] = item['rain']['3h']
        else:
            data_point['rain_3h'] = 0
            
        if 'snow' in item and '3h' in item['snow']:
            data_point['snow_3h'] = item['snow']['3h']
        else:
            data_point['snow_3h'] = 0
        
        # Calculate approximate solar radiation based on weather
        # This is a simplification since OpenWeather doesn't provide direct solar radiation data
        if 'clouds' in item:
            cloud_cover = item['clouds']['all']  # 0-100%
            # Simple estimate: lower cloud cover = higher radiation
            clear_sky_factor = 1 - (cloud_cover / 100)
            
            # Apply weather condition factor
            weather_id = item['weather'][0]['id']
            # Thunderstorm (200-299), Drizzle (300-399), Rain (500-599), Snow (600-699)
            if 200 <= weather_id < 300:
                weather_factor = 0.2  # Thunderstorms block most radiation
            elif 300 <= weather_id < 400:
                weather_factor = 0.5  # Drizzle blocks some radiation
            elif 500 <= weather_id < 600:
                weather_factor = 0.3  # Rain blocks significant radiation
            elif 600 <= weather_id < 700:
                weather_factor = 0.4  # Snow reflects some radiation
            elif weather_id == 800:
                weather_factor = 1.0  # Clear sky
            elif 801 <= weather_id < 900:
                weather_factor = 0.7  # Partly cloudy
            else:
                weather_factor = 0.6  # Default
            
            # Approximate solar radiation (unitless factor 0-1)
            data_point['solar_factor'] = clear_sky_factor * weather_factor
        else:
            data_point['solar_factor'] = 0.5  # Default if no cloud data
        
        forecast_data.append(data_point)
    
    # Convert to DataFrame
    df_forecast = pd.DataFrame(forecast_data)
    df_forecast.set_index('datetime', inplace=True)
    
    return df_forecast

def interpolate_hourly_forecast(forecast_df):
    """
    Interpolate 3-hourly forecast data to hourly data.
    
    Args:
        forecast_df: DataFrame with 3-hourly forecast data
        
    Returns:
        DataFrame with hourly forecast data
    """
    # Resample to hourly and interpolate
    hourly_df = forecast_df.resample('H').asfreq()
    
    # Linear interpolation for most numeric columns
    for col in forecast_df.columns:
        if col not in ['weather_id', 'weather_main', 'weather_description']:
            hourly_df[col] = hourly_df[col].interpolate(method='linear')
    
    # Forward fill for categorical columns
    for col in ['weather_id', 'weather_main', 'weather_description']:
        if col in hourly_df.columns:
            hourly_df[col] = hourly_df[col].fillna(method='ffill')
    
    return hourly_df

def create_radiation_estimate(forecast_df, lat, lon):
    """
    Create solar radiation estimate based on weather data and location.
    
    Args:
        forecast_df: DataFrame with hourly forecast data
        lat: Latitude
        lon: Longitude
        
    Returns:
        DataFrame with added radiation estimate columns
    """
    import pvlib
    
    # Enhanced forecast with better radiation estimates
    df = forecast_df.copy()
    
    # Create a time series for the forecast period
    times = pd.DatetimeIndex(df.index)
    
    # Create a pvlib location object
    location = pvlib.location.Location(latitude=lat, longitude=lon)
    
    # Calculate solar position
    solar_position = location.get_solarposition(times=times)
    df['zenith'] = solar_position['zenith']
    df['azimuth'] = solar_position['azimuth']
    
    # Calculate clear-sky radiation
    clearsky = location.get_clearsky(times)
    df['clearsky_ghi'] = clearsky['ghi']  # Global horizontal irradiance
    df['clearsky_dni'] = clearsky['dni']  # Direct normal irradiance
    df['clearsky_dhi'] = clearsky['dhi']  # Diffuse horizontal irradiance
    
    # Adjust clear-sky radiation based on weather conditions
    df['SolRad_Hor_est'] = df['clearsky_ghi'] * df['solar_factor']
    df['SolRad_Dif_est'] = df['clearsky_dhi'] * df['solar_factor']
    
    # Night hours should have zero radiation
    night_mask = df['zenith'] >= 90
    df.loc[night_mask, ['SolRad_Hor_est', 'SolRad_Dif_est']] = 0
    
    return df

def merge_weather_with_production_data(production_df, weather_df):
    """
    Merge production data with weather forecast data.
    
    Args:
        production_df: DataFrame with historical production data
        weather_df: DataFrame with weather forecast data
        
    Returns:
        DataFrame with merged historical and forecast data
    """
    # Keep only columns needed for forecasting from weather data
    forecast_features = [
        'temp', 'humidity', 'clouds', 'wind_speed', 
        'SolRad_Hor_est', 'SolRad_Dif_est', 'zenith', 'azimuth'
    ]
    
    weather_subset = weather_df[forecast_features].copy()
    
    # Rename columns to match production data
    weather_subset.rename(columns={
        'temp': 'Air Temp',
        'humidity': 'RH',
        'clouds': 'cloud_cover',
        'wind_speed': 'WS_10m',
        'SolRad_Hor_est': 'SolRad_Hor',
        'SolRad_Dif_est': 'SolRad_Dif'
    }, inplace=True)
    
    # Add is_forecast flag
    weather_subset['is_forecast'] = 1
    
    # Add is_forecast flag to historical data
    production_df_copy = production_df.copy()
    production_df_copy['is_forecast'] = 0
    
    # Combine historical data with forecast
    combined_df = pd.concat([production_df_copy, weather_subset], axis=0)
    
    # For forecast period, set production target (E_ac) to NaN
    # (these are what we want to predict)
    if 'E_ac' in combined_df.columns:
        forecast_mask = combined_df['is_forecast'] == 1
        combined_df.loc[forecast_mask, 'E_ac'] = np.nan
    
    return combined_df