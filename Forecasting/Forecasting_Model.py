import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# -------------------------------------------------
# Data Fetching Module
# -------------------------------------------------

def fetch_nasa_power_data(latitude, longitude, start_date, end_date, parameters):
    """
    Fetches data from the NASA POWER API.

    Parameters:
    - latitude (float): Latitude of the location.
    - longitude (float): Longitude of the location.
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - parameters (list): List of parameters to request.

    Returns:
    - df_nasa (DataFrame): DataFrame containing the NASA data.
    """
    # Build the API request URL
    api_url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters={','.join(parameters)}&"
        f"community=RE&"  # Renewable Energy community
        f"longitude={longitude}&"
        f"latitude={latitude}&"
        f"start={start_date.replace('-', '')}&"
        f"end={end_date.replace('-', '')}&"
        f"timeStandard=UTC&"
        f"format=JSON"
    )

    # Make the API request
    response = requests.get(api_url)
    data = response.json()

    # Check for errors
    if 'properties' not in data or 'parameter' not in data['properties']:
        raise ValueError("Error fetching data from NASA POWER API.")

    # Extract the data into a DataFrame
    parameters_data = data['properties']['parameter']
    df_nasa = pd.DataFrame(parameters_data)

    # Transpose and reset index
    df_nasa = df_nasa.transpose().reset_index()
    df_nasa.rename(columns={'index': 'datetime'}, inplace=True)

    # Convert datetime strings to datetime objects
    df_nasa['datetime'] = pd.to_datetime(df_nasa['datetime'])

    # Set 'datetime' as index and localize to UTC
    df_nasa.set_index('datetime', inplace=True)
    df_nasa.index = df_nasa.index.tz_localize('UTC')

    # Convert to local time zone
    df_nasa.index = df_nasa.index.tz_convert('Europe/Athens')

    return df_nasa

# -------------------------------------------------
# Data Preprocessing Module
# -------------------------------------------------

def preprocess_local_data(file_path):
    """
    Preprocesses the local energy data.

    Parameters:
    - file_path (str): Path to the CSV file containing energy data.

    Returns:
    - df_local (DataFrame): Preprocessed local data.
    """
    # Load your data
    df_local = pd.read_csv(file_path)

    # Convert 'hour' to datetime
    start_year = 2023  # Replace with the correct year
    df_local['datetime'] = pd.to_datetime(df_local['hour'] - 1, unit='h', origin=pd.Timestamp(f'{start_year}-01-01 00:00:00'))

    # Set 'datetime' as index
    df_local.set_index('datetime', inplace=True)

    # Localize to local time zone
    df_local.index = df_local.index.tz_localize('Europe/Athens')

    return df_local

def align_data(df_local, df_nasa):
    """
    Aligns and merges local data with NASA data.

    Parameters:
    - df_local (DataFrame): Local energy data.
    - df_nasa (DataFrame): NASA POWER data.

    Returns:
    - df_combined (DataFrame): Combined DataFrame.
    """
    # Merge data on datetime index
    df_combined = df_local.merge(df_nasa, left_index=True, right_index=True, how='inner')

    # Handle missing values
    df_combined.fillna(method='ffill', inplace=True)
    df_combined.fillna(method='bfill', inplace=True)

    return df_combined

# -------------------------------------------------
# Feature Engineering Module
# -------------------------------------------------

def create_features(df):
    """
    Creates additional features for the model.

    Parameters:
    - df (DataFrame): Combined DataFrame.

    Returns:
    - df (DataFrame): DataFrame with new features.
    """
    # Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['month'] = df.index.month

    # Lag features for energy consumption and production
    for lag in range(1, 25):
        df[f'lag_consumption_{lag}'] = df['Load (kW)'].shift(lag)
        df[f'lag_production_{lag}'] = df['SolRad_Hor'].shift(lag)  # Assuming SolRad_Hor relates to production

    # Drop rows with NaN values introduced by lagging
    df.dropna(inplace=True)

    return df

# -------------------------------------------------
# Modeling Module
# -------------------------------------------------

def build_cnn_lstm_model(input_shape):
    """
    Builds the CNN-LSTM model.

    Parameters:
    - input_shape (tuple): Shape of the input data.

    Returns:
    - model (Sequential): Compiled CNN-LSTM model.
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model

def train_model(model, X_train, y_train):
    """
    Trains the model.

    Parameters:
    - model (Sequential): The model to train.
    - X_train (ndarray): Training features.
    - y_train (ndarray): Training targets.

    Returns:
    - history: Training history.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    return history

def evaluate_model(model, X_test, y_test, scaler_y):
    """
    Evaluates the model.

    Parameters:
    - model (Sequential): Trained model.
    - X_test (ndarray): Test features.
    - y_test (ndarray): Test targets.
    - scaler_y (Scaler): Scaler used for the target variable.

    Returns:
    - mae (float): Mean Absolute Error.
    - rmse (float): Root Mean Squared Error.
    """
    y_pred = model.predict(X_test)

    # Inverse transform to get actual values
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    return mae, rmse, y_test_inv, y_pred_inv

# -------------------------------------------------
# Main Execution
# -------------------------------------------------

def main():
    # Define your location and time range
    latitude = 37.98983  # Replace with your latitude
    longitude = 23.74328  # Replace with your longitude
    start_date = '2018-01-01'  # Start date in YYYY-MM-DD format
    end_date = '2022-12-31'    # End date in YYYY-MM-DD format

    # Define the parameters you want to retrieve
    parameters = [
        'ALLSKY_SFC_SW_DWN',  # Solar irradiance
        'T2M',                # Temperature at 2 meters
        'WS2M',               # Wind speed at 2 meters
        'RH2M',               # Relative humidity at 2 meters
        'PRECTOTCORR',        # Precipitation
        # Add more parameters as needed
    ]

    # Fetch NASA POWER data
    df_nasa = fetch_nasa_power_data(latitude, longitude, start_date, end_date, parameters)

    # Preprocess local data
    file_path = r'C:\Users\Lospsy\Desktop\Thesis\DATA_kkav.csv'
    df_local = preprocess_local_data(file_path)

    # Align and merge data
    df_combined = align_data(df_local, df_nasa)

    # Create features
    df_features = create_features(df_combined)

    # Define features and target
    features = [
        'ALLSKY_SFC_SW_DWN',
        'T2M',
        'WS2M',
        'RH2M',
        'hour',
        'day_of_week',
        # Include lag features
        # 'lag_consumption_1', 'lag_consumption_2', ..., 'lag_production_1', etc.
    ]

    # Add lag features to the list of features
    lag_consumption_features = [f'lag_consumption_{lag}' for lag in range(1, 25)]
    lag_production_features = [f'lag_production_{lag}' for lag in range(1, 25)]
    features.extend(lag_consumption_features)
    features.extend(lag_production_features)

    target = 'Load (kW)'  # Assuming 'Load (kW)' is the target variable for energy consumption

    # Prepare data for modeling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(df_features[features])
    y = scaler_y.fit_transform(df_features[[target]])

    # Create sequences
    def create_sequences(X, y, time_steps=24):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    time_steps = 24  # Use past 24 hours to predict the next hour
    X_seq, y_seq = create_sequences(X, y, time_steps)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_model(input_shape)
    model.summary()

    # Train model
    history = train_model(model, X_train, y_train)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    # Evaluate model
    mae, rmse, y_test_inv, y_pred_inv = evaluate_model(model, X_test, y_test, scaler_y)
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')

    # Plot predictions vs actual
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_inv, label='Actual')
    plt.plot(y_pred_inv, label='Predicted')
    plt.legend()
    plt.show()

    # Save the model
    model.save('energy_forecast_model.h5')

    # Load the model (optional)
    # model = load_model('energy_forecast_model.h5')

    # Prepare new data for prediction
    # Ensure the new data is preprocessed in the same way

    # Example: Predicting on the test set
    new_predictions = model.predict(X_test)

if __name__ == '__main__':
    main()
