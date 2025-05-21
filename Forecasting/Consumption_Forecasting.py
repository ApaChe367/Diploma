# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, LSTM, Conv1D, MaxPooling1D, Dropout, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib
import requests
import mlflow
import mlflow.keras

def fetch_nasa_power_data(latitude, longitude, start_date, end_date, parameters):
    import requests
    import pandas as pd

    # Build the API request URL
    api_url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters={','.join(parameters)}&"
        f"community=RE&"
        f"longitude={longitude}&"
        f"latitude={latitude}&"
        f"&start={start_date.replace('-', '')}&"
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

    # Extract parameters data
    parameters_data = data['properties']['parameter']

    # Initialize an empty list to collect data
    records = []

    # Iterate over datetime keys
    datetime_keys = list(next(iter(parameters_data.values())).keys())
    datetime_keys.sort()  # Ensure the keys are sorted

    for datetime_str in datetime_keys:
        record = {'datetime': datetime_str}
        for param in parameters:
            value = parameters_data[param][datetime_str]
            record[param] = value
        records.append(record)

    # Create DataFrame from records
    df_nasa = pd.DataFrame(records)

    # Convert datetime strings to datetime objects
    df_nasa['datetime'] = pd.to_datetime(df_nasa['datetime'], format='%Y%m%d%H')

    # Set 'datetime' as index and localize to UTC
    df_nasa.set_index('datetime', inplace=True)
    df_nasa.index = df_nasa.index.tz_localize('UTC')

    # Convert to local time zone (no ambiguous parameter)
    df_nasa.index = df_nasa.index.tz_convert('Europe/Athens')

    return df_nasa

def load_and_preprocess_data(local_data_path, latitude, longitude, start_date, end_date, parameters):
    # Load your local CSV data
    data = pd.read_csv(local_data_path)
    
    # Ensure correct data types
    data['Load (kW)'] = data['Load (kW)'].astype(float)
    data['Air Temp'] = data['Air Temp'].astype(float)
    data['SolRad_Hor'] = data['SolRad_Hor'].astype(float)
    data['SolRad_Dif'] = data['SolRad_Dif'].astype(float)
    data['WS_10m'] = data['WS_10m'].astype(float)
    data['RH'] = data['RH'].astype(float)
    
     # Create a datetime index
    data['datetime'] = pd.date_range(start=start_date, periods=len(data), freq='h')
    data.set_index('datetime', inplace=True)
    
    # Localize to 'Europe/Athens' with ambiguous and nonexistent time handling
    data.index = data.index.tz_localize('Europe/Athens', ambiguous='NaT', nonexistent='shift_forward')
    data.dropna(inplace=True)  # Drop NaT values resulting from ambiguous times
    
    # Fetch NASA data
    df_nasa = fetch_nasa_power_data(latitude, longitude, start_date, end_date, parameters)
    
    # Merge local data with NASA data
    data_combined = data.merge(df_nasa, left_index=True, right_index=True, how='left')
    
    # Handle missing values
    data_combined.dropna(inplace=True)
    
    
    return data_combined

def feature_engineering(data):
    # Time-Based Features
    data['day_of_week'] = data.index.dayofweek  # Monday=0, Sunday=6
    data['is_weekend'] = data['day_of_week'] >= 5  # 1 if Saturday or Sunday
    data['day_of_year'] = data.index.dayofyear
    data['hour_of_day'] = data.index.hour
    
    # Lagged Features for Hourly Data
    for lag in range(1, 25):
        data[f'lag_load_{lag}'] = data['Load (kW)'].shift(lag)
    
   # Create daily aggregated data
    daily_data = data.resample('D').agg({
        'Load (kW)': 'mean',
        'Air Temp': 'mean',
        'SolRad_Hor': 'sum',
        'SolRad_Dif': 'sum',
        'WS_10m': 'mean',
        'RH': 'mean',
        'T2M': 'mean',
        'ALLSKY_SFC_SW_DWN': 'sum',
        'WS2M': 'mean',
        'RH2M': 'mean'
    })

    # Add time-based features
    daily_data['day_of_week'] = daily_data.index.dayofweek
    daily_data['is_weekend'] = (daily_data['day_of_week'] >= 5).astype(int)
    daily_data['day_of_year'] = daily_data.index.dayofyear

    # **Add lagged features**
    for lag in range(1, 8):
        daily_data[f'lag_load_{lag}'] = daily_data['Load (kW)'].shift(lag)

    # Drop rows with NaN values due to lagging
    daily_data.dropna(inplace=True)

    return data, daily_data

def prepare_sequences(data, daily_data, seq_length_hourly=24, seq_length_daily=7):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    # Define feature columns
    hourly_features = [
        'Load (kW)', 'Air Temp', 'SolRad_Hor', 'SolRad_Dif',
        'WS_10m', 'RH', 'T2M', 'ALLSKY_SFC_SW_DWN', 'WS2M', 'RH2M',
        'hour_of_day'
    ] + [f'lag_load_{lag}' for lag in range(1, 25)]

    daily_features = [
        'Load (kW)', 'Air Temp', 'SolRad_Hor', 'SolRad_Dif',
        'WS_10m', 'RH', 'T2M', 'ALLSKY_SFC_SW_DWN', 'WS2M', 'RH2M',
        'day_of_week', 'is_weekend', 'day_of_year'
    ] + [f'lag_load_{lag}' for lag in range(1, 8)]

    # Prepare input features
    X_hourly = data[hourly_features]
    y_hourly = data['Load (kW)'].shift(-1)  # Shift target to predict next hour
    X_hourly = X_hourly[:-1]  # Remove last row to align with shifted target
    y_hourly = y_hourly[:-1]

    X_daily = daily_data[daily_features]
    y_daily = daily_data['Load (kW)'].shift(-1)  # Shift target to predict next day
    X_daily = X_daily[:-1]  # Remove last row to align with shifted target
    y_daily = y_daily[:-1]

    # Scale the features
    hourly_scaler = MinMaxScaler()
    X_hourly_scaled = hourly_scaler.fit_transform(X_hourly)

    daily_scaler = MinMaxScaler()
    X_daily_scaled = daily_scaler.fit_transform(X_daily)

    # Create sequences
    def create_sequences(X, seq_length):
        Xs = []
        for i in range(len(X) - seq_length + 1):
            Xs.append(X[i:(i + seq_length)])
        return np.array(Xs)

    # Create sequences for hourly data
    X_hourly_seq = create_sequences(X_hourly_scaled, seq_length_hourly)
    y_hourly_seq = y_hourly.values[seq_length_hourly - 1:]  # Align target variable
    indices_hourly = y_hourly.index[seq_length_hourly - 1:]

    # Create sequences for daily data
    X_daily_seq = create_sequences(X_daily_scaled, seq_length_daily)
    y_daily_seq = y_daily.values[seq_length_daily - 1:]  # Align target variable
    indices_daily = y_daily.index[seq_length_daily - 1:]

    # Align sequences
    min_length = min(len(X_daily_seq), len(X_hourly_seq))
    X_daily_seq = X_daily_seq[-min_length:]
    X_hourly_seq = X_hourly_seq[-min_length:]
    y_seq = y_daily_seq[-min_length:]  # Use daily target variable
    indices_seq = indices_daily[-min_length:]

    # Scale the target variable
    y_scaler = MinMaxScaler()
    y_seq_scaled = y_scaler.fit_transform(y_seq.reshape(-1, 1))

    # Split into training and testing sets
    train_size = int(len(X_daily_seq) * 0.8)
    X_daily_train, X_daily_test = X_daily_seq[:train_size], X_daily_seq[train_size:]
    X_hourly_train, X_hourly_test = X_hourly_seq[:train_size], X_hourly_seq[train_size:]
    y_train_scaled, y_test_scaled = y_seq_scaled[:train_size], y_seq_scaled[train_size:]
    indices_train, indices_test = indices_seq[:train_size], indices_seq[train_size:]

    # Return sequences and scalers
    return (X_daily_train, X_daily_test,
            X_hourly_train, X_hourly_test,
            y_train_scaled, y_test_scaled,
            hourly_scaler, daily_scaler, y_scaler,
            indices_train, indices_test)


def build_model(num_features_daily, num_features_hourly, seq_length_daily=7, seq_length_hourly=24):
    # **Daily Input Branch**
    daily_input = Input(shape=(seq_length_daily, num_features_daily), name='daily_input')
    daily_lstm = LSTM(64, name='daily_lstm', return_sequences=True)(daily_input)
    daily_lstm = LSTM(32, name='daily_lstm_2')(daily_lstm)

    # **Hourly Input Branch**
    hourly_input = Input(shape=(seq_length_hourly, num_features_hourly), name='hourly_input')
    hourly_conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', name='hourly_conv1')(hourly_input)
    hourly_conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', name='hourly_conv2')(hourly_conv1)
    hourly_pool = MaxPooling1D(pool_size=2, name='hourly_pool')(hourly_conv2)
    hourly_dropout = Dropout(0.2, name='hourly_dropout')(hourly_pool)
    hourly_lstm = LSTM(64, name='hourly_lstm', return_sequences=True)(hourly_dropout)
    hourly_lstm = LSTM(32, name='hourly_lstm_2')(hourly_lstm)

    # **Merging Branches**
    merged = Concatenate(name='merge')([daily_lstm, hourly_lstm])

    # **Fully Connected Layers**
    dense1 = Dense(64, activation='relu', name='dense1')(merged)
    dense2 = Dense(32, activation='relu', name='dense2')(dense1)
    dense_dropout = Dropout(0.3, name='dense_dropout')(dense2)
    output = Dense(1, activation='linear', name='output')(dense_dropout)

    # **Define the Model**
    model = Model(inputs=[daily_input, hourly_input], outputs=output)

    # **Compile the Model**
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model

def train_model(model, X_daily_train, X_hourly_train, y_train_scaled):
    """
    Trains the neural network model.
    """
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    
    # Train the Model
    history = model.fit(
        {'daily_input': X_daily_train, 'hourly_input': X_hourly_train},
        y_train_scaled,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    return history

def evaluate_model(model, X_daily_test, X_hourly_test, y_test_scaled, y_scaler, test_indices):
    # Make predictions
    y_pred_scaled = model.predict({'daily_input': X_daily_test, 'hourly_input': X_hourly_test})

    # Inverse transform predictions and test data
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test = y_scaler.inverse_transform(y_test_scaled)

    # **Print lengths for debugging**
    print(f"Length of test_indices: {len(test_indices)}")
    print(f"Length of y_test: {len(y_test.flatten())}")
    print(f"Length of y_pred: {len(y_pred.flatten())}")

    # Create a DataFrame with results
    results_df = pd.DataFrame({
        'Timestamp': test_indices,
        'Actual': y_test.flatten(),
        'Predicted': y_pred.flatten()
    })


    # Save to CSV
    results_df.to_csv('model_predictions.csv', index=False)

    # Plot Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Timestamp'], results_df['Actual'], label='Actual')
    plt.plot(results_df['Timestamp'], results_df['Predicted'], label='Predicted')
    plt.title('Actual vs Predicted Energy Consumption')
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption (kW)')
    plt.legend()
    plt.savefig('test_set_predictions.png')
    plt.show()

    # Calculate metrics
    test_metrics = model.evaluate(
        {'daily_input': X_daily_test, 'hourly_input': X_hourly_test},
        y_test_scaled
    )
    print(f"Test Loss: {test_metrics[0]}, Test MAE: {test_metrics[1]}")

    return y_pred, y_test

def save_artifacts(model, hourly_scaler, daily_scaler, y_scaler):
    """
    Saves the model and scalers for future use.
    """
    # Save the model
    model.save('energy_consumption_model.h5')

    # Save the scalers
    joblib.dump(hourly_scaler, 'hourly_scaler.save')
    joblib.dump(daily_scaler, 'daily_scaler.save')
    joblib.dump(y_scaler, 'y_scaler.save')

def predict_next_day(model, daily_scaler, hourly_scaler, daily_data_input, hourly_data_input):
    """
    Makes a prediction for the next day.
    """
    # Scale the input data
    daily_data_scaled = daily_scaler.transform(daily_data_input)
    hourly_data_scaled = hourly_scaler.transform(hourly_data_input)
    
    # Reshape the data
    X_daily_seq = daily_data_scaled.reshape((1, daily_data_scaled.shape[0], daily_data_scaled.shape[1]))
    X_hourly_seq = hourly_data_scaled.reshape((1, hourly_data_scaled.shape[0], hourly_data_scaled.shape[1]))
    
    # Make prediction
    prediction = model.predict({'daily_input': X_daily_seq, 'hourly_input': X_hourly_seq})
    
    # Return the prediction
    return prediction[0][0]

def predict_future(model, data, daily_data, hourly_scaler, daily_scaler, y_scaler,
                   seq_length_hourly=24, seq_length_daily=7, future_steps=24):
    """
    Predicts future energy consumption for the next 'future_steps' hours.

    Parameters:
    - model: Trained model.
    - data: Preprocessed DataFrame containing the latest data.
    - daily_data: Daily aggregated data.
    - hourly_scaler: Scaler used for hourly features.
    - daily_scaler: Scaler used for daily features.
    - y_scaler: Scaler used for the target variable.
    - seq_length_hourly: Sequence length for hourly data.
    - seq_length_daily: Sequence length for daily data.
    - future_steps: Number of future hours to predict.

    Returns:
    - future_predictions: List of predicted values.
    - future_timestamps: Corresponding timestamps for the predictions.
    """
    import copy
    # Copy the data to avoid modifying the original DataFrame
    data_future = copy.deepcopy(data)
    daily_data_future = copy.deepcopy(daily_data)

    future_predictions = []

    # Get the last known timestamp
    last_timestamp = data.index[-1]

    # Lists to collect new rows
    data_future_rows = []
    daily_data_future_rows = []

    for step in range(future_steps):
        # Prepare hourly input
        recent_data = data_future.tail(seq_length_hourly)
        hourly_features = [
            'Load (kW)', 'Air Temp', 'SolRad_Hor', 'SolRad_Dif',
            'WS_10m', 'RH', 'T2M', 'ALLSKY_SFC_SW_DWN', 'WS2M', 'RH2M',
            'hour_of_day'
        ] + [f'lag_load_{lag}' for lag in range(1, 25)]

        X_hourly = recent_data[hourly_features].values
        # Assuming 'hourly_features' is a list of column names
        
        X_hourly_scaled = hourly_scaler.transform(X_hourly)

        X_hourly_seq = X_hourly_scaled.reshape((1, seq_length_hourly, -1))

        # Prepare daily input
        recent_daily_data = daily_data_future.tail(seq_length_daily)
        daily_features = [
            'Load (kW)', 'Air Temp', 'SolRad_Hor', 'SolRad_Dif',
            'WS_10m', 'RH', 'T2M', 'ALLSKY_SFC_SW_DWN', 'WS2M', 'RH2M',
            'day_of_week', 'is_weekend', 'day_of_year'
        ] + [f'lag_load_{lag}' for lag in range(1, 8)]

        X_daily = recent_daily_data[daily_features].values
        X_daily_scaled = daily_scaler.transform(X_daily)
        X_daily_seq = X_daily_scaled.reshape((1, seq_length_daily, -1))

        # Make prediction
        y_pred_scaled = model.predict({'daily_input': X_daily_seq, 'hourly_input': X_hourly_seq})
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        predicted_value = y_pred[0][0]
        future_predictions.append(predicted_value)

        # Update data_future with the new prediction
        next_timestamp = last_timestamp + pd.Timedelta(hours=step + 1)
        new_row = {
            'Load (kW)': predicted_value,
            'Air Temp': data_future['Air Temp'][-1],  # Assuming same temperature (adjust as needed)
            'SolRad_Hor': data_future['SolRad_Hor'][-1],  # Adjust as needed
            'SolRad_Dif': data_future['SolRad_Dif'][-1],  # Adjust as needed
            'WS_10m': data_future['WS_10m'][-1],  # Adjust as needed
            'RH': data_future['RH'][-1],  # Adjust as needed
            'T2M': data_future['T2M'][-1],  # Adjust as needed
            'ALLSKY_SFC_SW_DWN': data_future['ALLSKY_SFC_SW_DWN'][-1],  # Adjust as needed
            'WS2M': data_future['WS2M'][-1],  # Adjust as needed
            'RH2M': data_future['RH2M'][-1],  # Adjust as needed
            'hour_of_day': next_timestamp.hour,
            'day_of_week': next_timestamp.dayofweek,
            'is_weekend': int(next_timestamp.dayofweek >= 5),
            'day_of_year': next_timestamp.dayofyear,
        }
        # Update lagged features
        for lag in range(1, 25):
            if lag == 1:
                new_row[f'lag_load_{lag}'] = data_future['Load (kW)'][-1]
            else:
                new_row[f'lag_load_{lag}'] = data_future[f'lag_load_{lag - 1}'][-1]

        # Collect new row
        new_row_df = pd.DataFrame(new_row, index=[next_timestamp])
        data_future_rows.append(new_row_df)

        # Update daily_data_future if a new day starts
        if next_timestamp.hour == 0:
            # Prepare new daily row
            new_daily_row = {
                'Load (kW)': data_future['Load (kW)'].resample('D').mean()[-1],
                'Air Temp': data_future['Air Temp'].resample('D').mean()[-1],
                'SolRad_Hor': data_future['SolRad_Hor'].resample('D').sum()[-1],
                'SolRad_Dif': data_future['SolRad_Dif'].resample('D').sum()[-1],
                'WS_10m': data_future['WS_10m'].resample('D').mean()[-1],
                'RH': data_future['RH'].resample('D').mean()[-1],
                'T2M': data_future['T2M'].resample('D').mean()[-1],
                'ALLSKY_SFC_SW_DWN': data_future['ALLSKY_SFC_SW_DWN'].resample('D').sum()[-1],
                'WS2M': data_future['WS2M'].resample('D').mean()[-1],
                'RH2M': data_future['RH2M'].resample('D').mean()[-1],
                'day_of_week': next_timestamp.dayofweek,
                'is_weekend': int(next_timestamp.dayofweek >= 5),
                'day_of_year': next_timestamp.dayofyear,
            }
            # Update lagged features for daily data
            for lag in range(1, 8):
                if lag == 1:
                    new_daily_row[f'lag_load_{lag}'] = daily_data_future['Load (kW)'][-1]
                else:
                    new_daily_row[f'lag_load_{lag}'] = daily_data_future[f'lag_load_{lag - 1}'][-1]

            # Collect new daily row
            new_daily_row_df = pd.DataFrame(new_daily_row, index=[next_timestamp.date()])
            daily_data_future_rows.append(new_daily_row_df)

    # After the loop, concatenate all new rows at once
    if data_future_rows:
        data_future_new = pd.concat(data_future_rows, axis=0)
        data_future = pd.concat([data_future, data_future_new], axis=0)

    if daily_data_future_rows:
        daily_data_future_new = pd.concat(daily_data_future_rows, axis=0)
        daily_data_future = pd.concat([daily_data_future, daily_data_future_new], axis=0)

    # Generate future timestamps
    future_timestamps = [last_timestamp + pd.Timedelta(hours=i + 1) for i in range(future_steps)]

    return future_predictions, future_timestamps

def plot_last_24_hours(data, model, hourly_scaler, daily_scaler, y_scaler, seq_length_hourly, seq_length_daily):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Prepare recent data
    recent_data = data.copy()

    # Compute lag features for hourly data
    for lag in range(1, 25):
        recent_data[f'lag_load_{lag}'] = recent_data['Load (kW)'].shift(lag)
    recent_data['hour_of_day'] = recent_data.index.hour

    # Drop NaNs resulting from lag features
    recent_data = recent_data.dropna()

    # Define hourly features
    hourly_features = [
        'Load (kW)', 'Air Temp', 'SolRad_Hor', 'SolRad_Dif',
        'WS_10m', 'RH', 'T2M', 'ALLSKY_SFC_SW_DWN',
        'WS2M', 'RH2M', 'hour_of_day'
    ] + [f'lag_load_{lag}' for lag in range(1, 25)]

    # Prepare hourly inputs
    X_hourly = recent_data[hourly_features]
    X_hourly_scaled = pd.DataFrame(hourly_scaler.transform(X_hourly), columns=hourly_features, index=X_hourly.index)

    # Create hourly sequences
    X_hourly_seq = []
    for i in range(len(X_hourly_scaled) - seq_length_hourly + 1):
        X_hourly_seq.append(X_hourly_scaled.iloc[i:i + seq_length_hourly].values)
    X_hourly_seq = np.array(X_hourly_seq)

    # Prepare daily data
    recent_daily_data = recent_data.resample('D').mean()

    # Compute lag features for daily data
    for lag in range(1, 8):
        recent_daily_data[f'lag_load_{lag}'] = recent_daily_data['Load (kW)'].shift(lag)

    # Compute time-based features
    recent_daily_data['day_of_week'] = recent_daily_data.index.dayofweek
    recent_daily_data['is_weekend'] = recent_daily_data['day_of_week'].isin([5, 6]).astype(int)
    recent_daily_data['day_of_year'] = recent_daily_data.index.dayofyear

    # Drop NaNs resulting from lag features
    recent_daily_data = recent_daily_data.dropna()

    # Define daily features
    daily_features = [
        'Load (kW)', 'Air Temp', 'SolRad_Hor', 'SolRad_Dif',
        'WS_10m', 'RH', 'T2M', 'ALLSKY_SFC_SW_DWN',
        'WS2M', 'RH2M', 'day_of_week', 'is_weekend', 'day_of_year'
    ] + [f'lag_load_{lag}' for lag in range(1, 8)]

    # Prepare daily inputs
    X_daily = recent_daily_data[daily_features]
    X_daily_scaled = pd.DataFrame(daily_scaler.transform(X_daily), columns=daily_features, index=X_daily.index)

    # Create daily sequences
    X_daily_seq = []
    for i in range(len(X_daily_scaled) - seq_length_daily + 1):
        X_daily_seq.append(X_daily_scaled.iloc[i:i + seq_length_daily].values)
    X_daily_seq = np.array(X_daily_seq)

    # Align sequences
    min_length = min(len(X_hourly_seq), len(X_daily_seq))
    X_hourly_seq = X_hourly_seq[-min_length:]
    X_daily_seq = X_daily_seq[-min_length:]

    # Make predictions
    # Corrected order
    y_pred_scaled = model.predict([X_daily_seq, X_hourly_seq])
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    # Prepare actual values for plotting
    actual_load = recent_data['Load (kW)'].values[-min_length:]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(recent_data.index[-min_length:], actual_load[-min_length:], label='Actual')
    plt.plot(recent_data.index[-min_length:], y_pred.flatten(), label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Load (kW)')
    plt.title('Actual vs Predicted Load for Last 24 Hours')
    plt.legend()
    plt.show()


def save_predictions_to_csv(y_test, y_pred, test_indices, future_predictions, future_timestamps):
    # Create DataFrame for test set predictions
    test_results_df = pd.DataFrame({
        'Timestamp': test_indices,
        'Actual': y_test.flatten(),
        'Predicted': y_pred.flatten()
    })

    # Create DataFrame for future predictions
    future_results_df = pd.DataFrame({
        'Timestamp': future_timestamps,
        'Predicted': future_predictions
    })

    # Save to CSV
    test_results_df.to_csv(r'C:\Users\Lospsy\Desktop\Thesis\Results\Forecasting\test_set_predictions.csv', index=False)
    future_results_df.to_csv(r'C:\Users\Lospsy\Desktop\Thesis\Results\Forecasting\future_predictions.csv', index=False)

    # Combine both DataFrames for convenience
    combined_results_df = pd.concat([test_results_df, future_results_df], ignore_index=True)
    combined_results_df.to_csv(r'C:\Users\Lospsy\Desktop\Thesis\Results\Forecasting\combined_predictions.csv', index=False)

def save_plots(y_test, y_pred, test_indices, future_predictions, future_timestamps):
    # Plot Actual vs Predicted for Test Set
    plt.figure(figsize=(12, 6))
    plt.plot(test_indices, y_test.flatten(), label='Actual')
    plt.plot(test_indices, y_pred.flatten(), label='Predicted')
    plt.title('Actual vs Predicted Energy Consumption (Test Set)')
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption (kW)')
    plt.legend()
    plt.savefig('test_set_predictions.png')
    plt.show()

    # Plot Future Predictions
    plt.figure(figsize=(12, 6))
    plt.plot(future_timestamps, future_predictions, label='Predicted Future Consumption')
    plt.title('Predicted Energy Consumption for Next 24 Hours')
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption (kW)')
    plt.legend()
    plt.savefig('future_predictions.png')
    plt.show()

    # Plot Combined Actual and Future Predictions
    plt.figure(figsize=(12, 6))
    plt.plot(test_indices, y_test.flatten(), label='Actual (Test Set)')
    plt.plot(test_indices, y_pred.flatten(), label='Predicted (Test Set)')
    plt.plot(future_timestamps, future_predictions, label='Predicted Future Consumption')
    plt.title('Actual and Predicted Energy Consumption')
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption (kW)')
    plt.legend()
    plt.savefig('combined_predictions.png')
    plt.show()

def main():
    import mlflow
    import mlflow.keras
    # Set up MLflow experiment
    mlflow.set_experiment("Energy Consumption Forecasting")

    with mlflow.start_run():
        # Define hyperparameters
        seq_length_hourly = 24
        seq_length_daily = 7
        lstm_units = 32
        batch_size = 32
        epochs = 50

        # Log hyperparameters
        mlflow.log_param("seq_length_hourly", seq_length_hourly)
        mlflow.log_param("seq_length_daily", seq_length_daily)
        mlflow.log_param("lstm_units", lstm_units)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        # File paths and parameters
        local_data_path = r'C:\Users\Lospsy\Desktop\Thesis\DATA_kkav.csv'  
        latitude = 37.9838    # Example: Athens, Greece
        longitude = 23.7275
        start_date = '2020-01-01'
        end_date = '2020-12-31'
        parameters = ['T2M', 'ALLSKY_SFC_SW_DWN', 'WS2M', 'RH2M']

        # Load and preprocess data
        data = load_and_preprocess_data(local_data_path, latitude, longitude, start_date, end_date, parameters)

        # Feature engineering
        data, daily_data = feature_engineering(data)

        

        (X_daily_train, X_daily_test, X_hourly_train, X_hourly_test,
        y_train_scaled, y_test_scaled,
        hourly_scaler, daily_scaler, y_scaler,
        indices_train, indices_test) = prepare_sequences(data, daily_data)

        # Assign test_indices
        test_indices = indices_test

        # Input shapes
        num_features_daily = X_daily_train.shape[2]
        num_features_hourly = X_hourly_train.shape[2]

        # Build the model
        model = build_model(num_features_daily, num_features_hourly)

        # Train the model
        history = train_model(model, X_daily_train, X_hourly_train, y_train_scaled)

        # Log metrics at each epoch
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

        # Evaluate the model and save test predictions
        y_pred, y_test = evaluate_model(model, X_daily_test, X_hourly_test, y_test_scaled, y_scaler, indices_test)

        # Predict future values
        future_predictions, future_timestamps = predict_future(
            model, data, daily_data, hourly_scaler, daily_scaler, y_scaler,
            seq_length_hourly, seq_length_daily, future_steps=24
        )

        # Save predictions to CSV
        save_predictions_to_csv(y_test, y_pred, test_indices, future_predictions, future_timestamps)

        # Save plots
        save_plots(y_test, y_pred, test_indices, future_predictions, future_timestamps)

        # Log the model
        mlflow.keras.log_model(model, "model")

        # Save the model and scalers
        save_artifacts(model, hourly_scaler, daily_scaler, y_scaler)

        # Plot last 24 hours
        plot_last_24_hours(data, model, hourly_scaler, daily_scaler, y_scaler, seq_length_hourly, seq_length_daily)



    
    # Example of making a prediction
    # Prepare new data for prediction
    # daily_data_input = ...  # Your new daily data input
    # hourly_data_input = ...  # Your new hourly data input
    # next_day_prediction = predict_next_day(model, daily_scaler, hourly_scaler, daily_data_input, hourly_data_input)
    # print(f"Predicted energy consumption for the next day: {next_day_prediction} kW")

if __name__ == "__main__":
    main()
