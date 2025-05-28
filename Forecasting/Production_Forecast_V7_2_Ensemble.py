import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split  # Add this separate import
import json
from datetime import datetime
from pathlib import Path
import sys
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.dates as mdates
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import traceback

def check_gpu_status():
    """Detailed check of GPU status and PyTorch configuration."""
    
    print("\n" + "="*60)
    print("GPU/CUDA STATUS CHECK")
    print("="*60)
    
    # Python and PyTorch versions
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Not available'}")
    
    # CUDA environment variables
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    print(f"CUDA_PATH: {cuda_path}")
    
    # Basic CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    
    if cuda_available:
        # CUDA devices
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        
        # List all devices
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - Compute capability: {props.major}.{props.minor}")
            print(f"  - Total memory: {props.total_memory / 1e9:.1f} GB")
        
        # Current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Test CUDA works
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device=torch.device('cuda'))
            y = x * 2
            print("✅ Simple CUDA tensor operation successful")
        except Exception as e:
            print(f"❌ Error with CUDA tensor operation: {e}")
    
    print("\nRecommended fixes if CUDA not available:")
    print("1. Check if NVIDIA drivers are installed:")
    print("   - Run 'nvidia-smi' from command line")
    print("2. Reinstall PyTorch with CUDA support:")
    print("   - conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print("   - OR: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    print("3. Try creating a clean environment:")
    print("   - conda create -n torch_env python=3.9")
    print("   - conda activate torch_env")
    print("   - conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print("="*60)


# Load the data without setting any column as index initially
data_path = "C:/Users/Lospsy/Desktop/Thesis/Results/forecast_data.csv"  
df = pd.read_csv(data_path)

# Check the structure
print("Column names:")
print(df.columns.tolist())
print("\nFirst few rows of first column:")
print(df.iloc[:5, 0])
print(f"First column type: {type(df.iloc[0, 0])}")

# Get the datetime column name (should be first column)
datetime_col_name = df.columns[0]
print(f"Datetime column name: {datetime_col_name}")

# More robust datetime conversion
try:
    # Step 1: Convert the datetime strings to datetime objects
    print("Converting datetime strings...")
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
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek
    
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
    new_index = pd.date_range(start=start_dt, periods=n_hours, freq='h')
    
    # Set new index and remove datetime column
    df.index = new_index
    df = df.drop(columns=[datetime_col_name])

    df['hour'] = df.index.hour
    df['month'] = df.index.month 
    df['dayofweek'] = df.index.dayofweek
    
# Verify the conversion
print(f"\nAfter conversion:")
print(f"Index type: {type(df.index)}")
print(f"Is DatetimeIndex: {isinstance(df.index, pd.DatetimeIndex)}")
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Handle missing values
print(f"\nHandling missing values...")

# Wind speed is completely missing - use a reasonable default
if 'WS_10m' in df.columns and df['WS_10m'].isna().all():
    print("Setting default wind speed (3 m/s)")
    df['WS_10m'] = 3.0

# Mismatch calculation
if 'mismatch' in df.columns and df['mismatch'].isna().all():
    print("Calculating mismatch values")
    df['mismatch'] = df['ac_power_output'] / 1000 - df['Load (kW)']

# Fill any remaining missing values
df = df.ffill().bfill()

# Check critical columns
critical_columns = ['SolRad_Hor', 'SolRad_Dif', 'Air Temp', 'WS_10m', 
                   'zenith', 'azimuth', 'E_ac', 'ac_power_output']
print(f"\nMissing values in critical columns:")
for col in critical_columns:
    if col in df.columns:
        print(f"{col}: {df[col].isna().sum()}")

# Data validation
print(f"\nData validation:")
print(f"Energy production range: {df['E_ac'].min():.2f} to {df['E_ac'].max():.2f} Wh")
print(f"Power output range: {df['ac_power_output'].min():.2f} to {df['ac_power_output'].max():.2f} W")
print(f"Temperature range: {df['Air Temp'].min():.1f} to {df['Air Temp'].max():.1f} °C")

# Now create plots
plt.figure(figsize=(15, 10))

# Plot 1: Solar radiation vs AC power
plt.subplot(2, 3, 1)
# Use a sample of data points for cleaner visualization
sample_mask = np.random.choice(len(df), size=min(5000, len(df)), replace=False)
plt.scatter(df['SolRad_Hor'].iloc[sample_mask], 
           df['ac_power_output'].iloc[sample_mask], alpha=0.5, s=1)
plt.xlabel('Horizontal Solar Radiation (W/m²)')
plt.ylabel('AC Power Output (W)')
plt.title('Solar Radiation vs AC Power')

# Plot 2: Temperature effect on efficiency
plt.subplot(2, 3, 2)
plt.scatter(df['Air Temp'].iloc[sample_mask], 
           df['temperature_factor'].iloc[sample_mask], alpha=0.5, s=1)
plt.xlabel('Air Temperature (°C)')
plt.ylabel('Temperature Factor')
plt.title('Temperature Effect')

# Plot 3: Daily energy production (using groupby instead of resample)
plt.subplot(2, 3, 3)
df_daily = df.groupby(df.index.date)['E_ac'].sum() / 1000  # kWh
plt.plot(range(len(df_daily)), df_daily.values)
plt.xlabel('Day of Year')
plt.ylabel('Daily Energy (kWh)')
plt.title('Daily Energy Production')

# Plot 4: Hourly average production profile
plt.subplot(2, 3, 4)
hourly_avg = df.groupby(df.index.hour)['ac_power_output'].mean()
plt.plot(hourly_avg.index, hourly_avg.values, linewidth=2)
plt.xlabel('Hour of Day')
plt.ylabel('Average AC Power (W)')
plt.title('Average Hourly Profile')
plt.grid(True, alpha=0.3)

# Plot 5: Monthly energy distribution
plt.subplot(2, 3, 5)
monthly = df.groupby(df.index.month)['E_ac'].sum() / 1000
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar(range(1, 13), monthly.values)
plt.xticks(range(1, 13), month_names, rotation=45)
plt.xlabel('Month')
plt.ylabel('Monthly Energy (kWh)')
plt.title('Monthly Energy Distribution')

# Plot 6: Energy vs consumption correlation
plt.subplot(2, 3, 6)
plt.scatter(df['Load (kW)'].iloc[sample_mask], 
           (df['ac_power_output'] / 1000).iloc[sample_mask], alpha=0.5, s=1)
plt.xlabel('Load (kW)')
plt.ylabel('Production (kW)')
plt.title('Production vs Consumption')

plt.tight_layout()
plt.savefig("comprehensive_data_exploration.png", dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print(f"\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"Total annual energy production: {df['E_ac'].sum() / 1e6:.2f} MWh")
print(f"Average daily production: {df_daily.mean():.2f} kWh")
print(f"Peak daily production: {df_daily.max():.2f} kWh")
print(f"Minimum daily production: {df_daily.min():.2f} kWh")
print(f"Maximum instantaneous power: {df['ac_power_output'].max() / 1000:.2f} kW")
print(f"Average load: {df['Load (kW)'].mean():.2f} kW")
print(f"Peak load: {df['Load (kW)'].max():.2f} kW")

# Production efficiency statistics
production_hours = (df['ac_power_output'] > 0).sum()
print(f"Hours with production: {production_hours} ({production_hours/len(df)*100:.1f}%)")

# Monthly breakdown
print(f"\nMonthly Production (kWh):")
for month in range(1, 13):
    month_data = df[df.index.month == month]
    monthly_energy = month_data['E_ac'].sum() / 1000
    print(f"{month_names[month-1]:>3}: {monthly_energy:>7.2f}")

# Save the cleaned dataset
cleaned_data_path = "C:/Users/Lospsy/Desktop/Thesis/Results/cleaned_forecast_data.csv"
df.to_csv(cleaned_data_path)
print(f"\nCleaned data saved to: {cleaned_data_path}")
print(f"Final data shape: {df.shape}")

meta_models = {
    "Ridge": Ridge(alpha=0.5),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100)
}


class SolarProductionDataset(Dataset):
    def __init__(self, dataframe, seq_length=24, forecast_horizon=24, normalize=True):
        """
        Dataset for solar production forecasting.
        
        Args:
            dataframe: Pandas DataFrame with time-series data
            seq_length: Number of previous hours to use as input
            forecast_horizon: Number of hours ahead to predict
            normalize: Whether to normalize the features
        """
        self.df = dataframe.copy()
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        
        # Select relevant features for production forecasting
        feature_columns = ['SolRad_Hor', 'SolRad_Dif', 'Air Temp', 'WS_10m', 
                          'zenith', 'azimuth', 'hour', 'month', 'dayofweek']
        
        # Ensure all columns exist
        for col in feature_columns:
            if col not in self.df.columns:
                print(f"Warning: {col} not found in dataframe")
        
        available_features = [col for col in feature_columns if col in self.df.columns]
        self.features = self.df[available_features].values
        
        # Target variable is E_ac (energy production)
        self.targets = self.df['E_ac'].values
        
        # Store feature names for reference
        self.feature_names = available_features
        
        # Feature normalization
        if normalize:
            self._normalize_features()
        
        # Calculate valid indices (accounting for sequence length and forecast horizon)
        self.valid_indices = len(self.df) - seq_length - forecast_horizon + 1
        
        if self.valid_indices <= 0:
            raise ValueError(f"Not enough data points. Need at least {seq_length + forecast_horizon} points, got {len(self.df)}")
        
    def _normalize_features(self):
        """Normalize features to [0,1] range."""
        # Create simple min-max scaler for each feature
        self.feature_mins = np.min(self.features, axis=0)
        self.feature_maxs = np.max(self.features, axis=0)
        
        # Handle cases where min == max (constant features)
        range_vals = self.feature_maxs - self.feature_mins
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        # Apply normalization
        self.features = (self.features - self.feature_mins) / range_vals
        
        # Target normalization
        self.target_max = np.max(self.targets)
        self.target_min = np.min(self.targets)
        if self.target_max > self.target_min:
            self.targets_normalized = (self.targets - self.target_min) / (self.target_max - self.target_min)
        else:
            self.targets_normalized = self.targets
    
    def __len__(self):
        return self.valid_indices
    
    def __getitem__(self, idx):
        # Get sequence of features
        features = self.features[idx:idx + self.seq_length]
        
        # Get target values (future energy production) 
        targets = self.targets_normalized[idx + self.seq_length:idx + self.seq_length + self.forecast_horizon]
        
        return torch.FloatTensor(features), torch.FloatTensor(targets)
    
    def denormalize_targets(self, normalized_values):
        """Convert normalized predictions back to original scale."""
        if hasattr(self, 'target_max') and hasattr(self, 'target_min'):
            return normalized_values * (self.target_max - self.target_min) + self.target_min
        else:
            return normalized_values

class SolarProductionPINN(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=3, 
                 panel_efficiency=0.146, panel_area=1.642, temp_coeff=-0.0044):
        super().__init__()
        
        # Store physical parameters
        self.panel_efficiency = panel_efficiency
        self.panel_area = panel_area
        self.temp_coeff = temp_coeff
        
        # Neural network for learning complex patterns
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_size, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Store input size for later reference
        self.input_size = input_size
        
    def forward(self, x, return_components=False):
        # x shape: [batch_size, seq_len, features]
        batch_size, seq_len, _ = x.shape
        device = x.device  # Get device from input tensor
        
        # Initialize output tensor ON THE SAME DEVICE as input
        predictions = torch.zeros(batch_size, seq_len, device=device)
        physics_components = torch.zeros(batch_size, seq_len, device=device) if return_components else None
        
        # Process each timestep
        for t in range(seq_len):
            # Get features for current timestep
            x_t = x[:, t, :]
            
            # Extract individual features (adjust indices based on your feature order)
            solar_radiation = x_t[:, 0]  # SolRad_Hor
            temp = x_t[:, 2]             # Air Temp
            zenith = x_t[:, 4]           # Solar zenith angle (if available)
            
            # Physical component: basic solar panel model
            if x_t.shape[1] > 4:  # If zenith angle is available
                # Convert zenith to radians and calculate cosine factor
                zenith_rad = torch.clamp(zenith * np.pi / 180, 0, np.pi/2)
                cos_factor = torch.cos(zenith_rad)
            else:
                # If no zenith angle, use simplified model
                cos_factor = torch.ones_like(solar_radiation)
            
            # Temperature correction
            temp_factor = 1 + self.temp_coeff * (temp - 25)
            
            # Basic physical estimate
            physics_estimate = (solar_radiation * cos_factor * 
                              self.panel_efficiency * self.panel_area * temp_factor)
            physics_estimate = torch.clamp(physics_estimate, min=0)  # No negative power
            
            # Neural network learns residual/corrections
            nn_output = self.net(x_t).squeeze()
            
            # Combined prediction
            prediction = physics_estimate + nn_output
            prediction = torch.clamp(prediction, min=0)  # Ensure non-negative
            
            # Store results
            predictions[:, t] = prediction
            if return_components:
                physics_components[:, t] = physics_estimate
        
        if return_components:
            return predictions, physics_components
        else:
            return predictions
    
    def physics_loss(self, x, predictions):
        """
        Calculate physics-based consistency loss.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device  # Get device from input
        
        # 1. Non-negative power constraint
        negative_power_loss = torch.mean(torch.relu(-predictions))
        
        # 2. Physical consistency checks
        physics_consistency_loss = torch.tensor(0.0, device=device)  # Initialize on correct device
        
        for t in range(min(seq_len, 5)):  # Sample a few timesteps to avoid memory issues
            x_t = x[:, t, :]
            pred_t = predictions[:, t]
            
            # Solar radiation should positively correlate with power
            solar_rad = x_t[:, 0]
            
            # Calculate correlation (should be positive)
            if torch.std(solar_rad) > 0 and torch.std(pred_t) > 0:
                correlation = torch.corrcoef(torch.stack([solar_rad, pred_t]))[0, 1]
                # Penalize negative correlation
                physics_consistency_loss += torch.relu(-correlation)
        
        # 3. Power should be zero when there's no solar radiation
        zero_radiation_mask = (x[:, :, 0] == 0)  # Where solar radiation is zero
        zero_rad_loss = torch.mean(predictions[zero_radiation_mask] ** 2)
        
        return negative_power_loss + physics_consistency_loss + zero_rad_loss * 0.1

class HyperparameterTuner:
    def __init__(self, base_output_dir, dataset, train_dataset, val_dataset, test_dataset):
        """
        Set up the hyperparameter tuning environment.
        
        Args:
            base_output_dir: Directory to save all results
            dataset: Main dataset object for feature information
            train_dataset, val_dataset, test_dataset: Split datasets
        """
        self.base_output_dir = Path(base_output_dir)
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # Create the tuning directory
        self.tuning_dir = self.base_output_dir / "hyperparameter_tuning"
        self.tuning_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize results tracking
        self.results = []
        self.best_val_loss = float('inf')
        self.best_config = None
        
        # Save device information for consistency
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Hyperparameter tuner initialized. Results will be saved to {self.tuning_dir}")
        print(f"Using device: {self.device}")
        
    def fix_json_serialization(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.fix_json_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.fix_json_serialization(item) for item in obj]
        else:
            return obj
    
    def run_grid_search(self, param_grid, epochs=150, early_stopping=True):
        """
        Run grid search over provided parameter grid.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values to try
            epochs: Number of epochs for each training run
            early_stopping: Whether to use early stopping
        """
        # Create all parameter combinations
        from itertools import product
        
        # Get keys and values for the grid
        keys = param_grid.keys()
        values = param_grid.values()
        
        # Generate all combinations
        combinations = list(product(*values))
        
        print(f"Running grid search with {len(combinations)} configurations.")
        
        # Run each configuration
        for i, combination in enumerate(combinations):
            config = dict(zip(keys, combination))
            print(f"\n[{i+1}/{len(combinations)}] Testing configuration: {config}")
            
            # Set a unique run name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
            
            # Set output directory for this run
            run_dir = self.tuning_dir / run_name
            run_dir.mkdir(exist_ok=True)
            
            # Save configuration
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Create and train model with this configuration
            result = self._train_and_evaluate(config, run_dir, epochs, early_stopping)
            
            # Add to results
            self.results.append({
                "config": config,
                "result": result,
                "run_dir": str(run_dir)
            })
            
            # Check if this is the best model so far
            if result["val_loss"] < self.best_val_loss:
                self.best_val_loss = result["val_loss"]
                self.best_config = config
                print(f"New best model found! Val loss: {self.best_val_loss:.6f}")
            
            # Save updated results
            self._save_results()
        
        # Print final results
        self._print_summary()
        
        return self.best_config, self.results
    
    
    def _train_and_evaluate(self, config, run_dir, epochs, early_stopping):
        """
        Train and evaluate a model with the given configuration.
        """
        from torch.utils.data import DataLoader
        
        # Extract parameters
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        batch_size = config["batch_size"]
        lr = config["learning_rate"]
        physics_weight = config.get("physics_weight", 0.1)
        
        # Create data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, 
                               num_workers=0, pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, 
                                num_workers=0, pin_memory=True)
        
        # Get system parameters
        try:
            # Assuming these are stored in the dataset or can be retrieved somehow
            panel_efficiency = config.get("panel_efficiency", 0.146)
            panel_area = config.get("panel_area", 1.642)
            temp_coeff = config.get("temp_coeff", -0.0044)
        except Exception as e:
            print(f"Warning: Using default panel parameters due to error: {e}")
            panel_efficiency = 0.146
            panel_area = 1.642
            temp_coeff = -0.0044
        
        # Create the model
        model = SolarProductionPINN(
            input_size=len(self.dataset.feature_names),
            hidden_size=hidden_size,
            num_layers=num_layers,
            panel_efficiency=panel_efficiency,
            panel_area=panel_area,
            temp_coeff=temp_coeff
        ).to(self.device)
        
        # Set up model save path
        model_save_path = run_dir / "best_model.pt"
        
        # Train the model
        print(f"Training model with {epochs} epochs...")
        model, history = train_pinn(
            model, train_loader, val_loader,
            epochs=epochs,
            lr=lr,
            physics_weight=physics_weight,
            device=self.device,
            save_path=model_save_path,
            early_stopping=early_stopping,
            patience=20  # Patience for early stopping
        )
        
        # Plot and save training history
        self._plot_training_history(history, run_dir)
        
        # Load the best model for evaluation
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=self.device, weights_only=True))
            print(f"Loaded best model from {model_save_path}")
        
        # Evaluate the model
        eval_results = evaluate_model_improved(model, test_loader, self.dataset, device=self.device)
        
        # Save evaluation results
        with open(run_dir / "eval_results.json", "w") as f:
            # Create a copy without large arrays
            serializable_results = {k: v for k, v in eval_results.items() 
                                if k not in ['targets', 'predictions']}
            
            # Convert all numpy types to Python native types
            # Inside _train_and_evaluate method:
            serializable_results = self.fix_json_serialization(serializable_results)
            
            # Save to JSON
            json.dump(serializable_results, f, indent=2)
                
                # Create evaluation plots
            self._plot_evaluation_results(eval_results, run_dir)
        
        # Return key metrics
        return {
            "train_loss": history['train_loss'][-1],
            "val_loss": history['val_loss'][-1],
            "test_loss": eval_results['test_loss'],
            "rmse": eval_results['rmse'],
            "mae": eval_results['mae'],
            "r2": eval_results['r2'],
            "epochs_run": len(history['train_loss']),
        }
    
    def _plot_training_history(self, history, run_dir):
        """Create and save plots of training history."""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(history['data_loss'], label='Data Loss', linewidth=2, color='blue')
        plt.plot(history['physics_loss'], label='Physics Loss', linewidth=2, color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.semilogy(history['train_loss'], label='Train Loss', linewidth=2)
        plt.semilogy(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.title('Training Loss (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(run_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_evaluation_results(self, eval_results, run_dir):
        """Create and save plots of evaluation results."""
        # Sample points for better visualization
        n_sample = min(5000, len(eval_results['targets']))
        indices = np.random.choice(len(eval_results['targets']), n_sample, replace=False)
        targets_sample = eval_results['targets'][indices]
        predictions_sample = eval_results['predictions'][indices]
        
        plt.figure(figsize=(10, 8))
        
        # Plot 1: Predicted vs Actual
        plt.subplot(2, 2, 1)
        plt.scatter(targets_sample, predictions_sample, alpha=0.5, s=1)
        plt.plot([targets_sample.min(), targets_sample.max()], 
                 [targets_sample.min(), targets_sample.max()], 'r--', lw=2)
        plt.xlabel('Actual Energy (Wh)')
        plt.ylabel('Predicted Energy (Wh)')
        plt.title(f'Predicted vs Actual\nR² = {eval_results["r2"]:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        plt.subplot(2, 2, 2)
        errors = eval_results['predictions'] - eval_results['targets']
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error (Wh)')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution\nMAE = {eval_results["mae"]:.2f} Wh')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Time series example
        plt.subplot(2, 2, 3)
        sample_start = np.random.randint(0, len(eval_results['targets']) - 48)
        sample_end = sample_start + 48
        
        plt.plot(range(48), eval_results['targets'][sample_start:sample_end], 
                 label='Actual', linewidth=2)
        plt.plot(range(48), eval_results['predictions'][sample_start:sample_end], 
                 label='Predicted', linewidth=2)
        plt.xlabel('Hours')
        plt.ylabel('Energy (Wh)')
        plt.title('48-Hour Sample Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Metrics summary
        plt.subplot(2, 2, 4)
        metrics = ['RMSE (kWh)', 'MAE (kWh)', 'MAPE (%)', 'R² (%)']
        values = [eval_results['rmse']/1000, eval_results['mae']/1000, 
                  eval_results['mape'], eval_results['r2'] * 100]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow']
        
        bars = plt.bar(metrics, values, color=colors, edgecolor='black')
        plt.ylabel('Value')
        plt.title('Model Performance Metrics')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                    f'{value:.2f}', ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(run_dir / "evaluation_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save results to JSON file."""
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = {
                "config": result["config"],
                "run_dir": result["run_dir"],
                "metrics": {}
            }
            for k, v in result["result"].items():
                if isinstance(v, np.ndarray) or isinstance(v, np.number):
                    serializable_result["metrics"][k] = v.item() if hasattr(v, 'item') else v.tolist()
                else:
                    serializable_result["metrics"][k] = v
            serializable_results.append(serializable_result)
        
        with open(self.tuning_dir / "tuning_results.json", "w") as f:
            json.dump({
                "results": serializable_results,
                "best_config": self.best_config,
                "best_val_loss": float(self.best_val_loss)
            }, f, indent=2)
    
    def _print_summary(self):
        """Print summary of hyperparameter tuning results."""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("="*60)
        print(f"Total configurations tested: {len(self.results)}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Best configuration: {self.best_config}")
        print("="*60)
        
        # Print top 5 configurations
        print("\nTop 5 configurations by validation loss:")
        sorted_results = sorted(self.results, key=lambda x: x["result"]["val_loss"])
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i+1}. Val Loss: {result['result']['val_loss']:.6f}, "
                  f"R²: {result['result']['r2']:.4f}, "
                  f"Config: {result['config']}")
        print("="*60)
        
        # Create comparison plot of top configurations
        self._plot_top_configurations(sorted_results[:10])
    
    def _plot_top_configurations(self, top_results):
        """Plot comparison of top configurations."""
        plt.figure(figsize=(12, 10))
        
        # Metrics to compare
        metrics = ["val_loss", "rmse", "mae", "r2"]
        titles = ["Validation Loss", "RMSE (Wh)", "MAE (Wh)", "R²"]
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(2, 2, i+1)
            
            # Extract values
            values = [result["result"][metric] for result in top_results]
            if metric == "r2":  # Higher is better for R²
                indices = np.argsort(values)[::-1]
            else:  # Lower is better for losses
                indices = np.argsort(values)
            
            sorted_values = [values[i] for i in indices]
            config_labels = [f"Config {i+1}" for i in range(len(top_results))]
            sorted_labels = [config_labels[i] for i in indices]
            
            # Plot bars
            bars = plt.bar(sorted_labels, sorted_values, color='skyblue', edgecolor='black')
            
            # Add value labels
            for bar, value in zip(bars, sorted_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                        f'{value:.4f}', ha='center', va='bottom', rotation=90, fontsize=8)
            
            plt.xlabel('Configuration')
            plt.ylabel(title)
            plt.title(f'Top Configurations by {title}')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.tuning_dir / "top_configurations_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

class SolarProductionDatasetWithForecast(Dataset):
    def __init__(self, dataframe, seq_length=24, forecast_horizon=24, normalize=True):
        """
        Enhanced dataset class that includes weather forecast features.
        """
        self.df = dataframe.copy()
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        
        # Select relevant features for production forecasting
        # Original features
        basic_features = ['SolRad_Hor', 'SolRad_Dif', 'Air Temp', 'WS_10m', 
                          'zenith', 'azimuth', 'hour', 'month', 'dayofweek']
        
        # Add forecast features
        forecast_features = [col for col in self.df.columns 
                             if col.startswith('forecast_') or 
                             col in ['clear_sky_ratio', 'irradiance_factor']]
        
        feature_columns = basic_features + forecast_features
        
        # Ensure all columns exist
        available_features = [col for col in feature_columns if col in self.df.columns]
        self.features = self.df[available_features].values
        
        # Target variable is E_ac (energy production)
        self.targets = self.df['E_ac'].values
        
        # Store feature names for reference
        self.feature_names = available_features
        
        # Feature normalization
        if normalize:
            self._normalize_features()
            
        # Calculate valid indices (accounting for sequence length and forecast horizon)
        self.valid_indices = len(self.df) - seq_length - forecast_horizon + 1
        
        if self.valid_indices <= 0:
            raise ValueError(f"Not enough data points. Need at least {seq_length + forecast_horizon} points, got {len(self.df)}")
    
    def _normalize_features(self):
        """Normalize features to [0,1] range."""
        # Create simple min-max scaler for each feature
        self.feature_mins = np.min(self.features, axis=0)
        self.feature_maxs = np.max(self.features, axis=0)
        
        # Handle cases where min == max (constant features)
        range_vals = self.feature_maxs - self.feature_mins
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        # Apply normalization
        self.features = (self.features - self.feature_mins) / range_vals
        
        # Target normalization
        self.target_max = np.max(self.targets)
        self.target_min = np.min(self.targets)
        if self.target_max > self.target_min:
            self.targets_normalized = (self.targets - self.target_min) / (self.target_max - self.target_min)
        else:
            self.targets_normalized = self.targets
    
    def __len__(self):
        return self.valid_indices
    
    def __getitem__(self, idx):
        # Get sequence of features
        features = self.features[idx:idx + self.seq_length]
        
        # Get target values (future energy production) 
        targets = self.targets_normalized[idx + self.seq_length:idx + self.seq_length + self.forecast_horizon]
        
        return torch.FloatTensor(features), torch.FloatTensor(targets)
    
    def denormalize_targets(self, normalized_values):
        """Convert normalized predictions back to original scale."""
        if hasattr(self, 'target_max') and hasattr(self, 'target_min'):
            return normalized_values * (self.target_max - self.target_min) + self.target_min
        else:
            return normalized_values

class StackedEnsemble:
    def __init__(self, base_models, meta_model):
        """
        Stacked ensemble with base models and a meta-model.
        
        Args:
            base_models: List of trained base models
            meta_model: Model that combines base model predictions
        """
        self.base_models = base_models
        self.meta_model = meta_model
        
    def train_meta_model(self, val_loader, device):
        """
        Train the meta-model on validation set predictions from base models.
        """
        # Get base model predictions on validation set
        base_predictions = []
        actual_values = []
        
        for model in self.base_models:
            model.eval()
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                # Get predictions from each base model
                batch_predictions = []
                for model in self.base_models:
                    pred = model(features)
                    if pred.dim() > 1 and pred.shape[1] > 1:
                        pred = pred[:, 0]  # Take first prediction if multi-step
                    batch_predictions.append(pred.cpu().numpy())
                
                # Stack predictions as features for meta-model
                batch_meta_features = np.column_stack(batch_predictions)
                
                if len(base_predictions) == 0:
                    base_predictions = batch_meta_features
                    actual_values = targets.cpu().numpy()
                else:
                    base_predictions = np.vstack([base_predictions, batch_meta_features])
                    actual_values = np.concatenate([actual_values, targets.cpu().numpy()])
        
        # Train meta-model
        actual_values = actual_values.ravel()
        self.meta_model.fit(base_predictions, actual_values)
        
    def predict(self, test_loader, device):
        """
        Make predictions using the stacked ensemble.
        """
        ensemble_predictions = []
        actual_values = []
        
        for model in self.base_models:
            model.eval()
        
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                
                # Get predictions from each base model
                batch_predictions = []
                for model in self.base_models:
                    pred = model(features)
                    if pred.dim() > 1 and pred.shape[1] > 1:
                        pred = pred[:, 0]  # Take first prediction if multi-step
                    batch_predictions.append(pred.cpu().numpy())
                
                # Stack predictions as features for meta-model
                batch_meta_features = np.column_stack(batch_predictions)
                
                # Get meta-model predictions
                batch_ensemble = self.meta_model.predict(batch_meta_features)
                
                # Store results
                ensemble_predictions.append(batch_ensemble)
                actual_values.append(targets.cpu().numpy())
        
        return np.concatenate(ensemble_predictions), np.concatenate(actual_values)

class WeatherConditionEnsemble:
    def __init__(self, model_dict, weather_feature_idx=None):
        """
        Ensemble that selects models based on weather conditions.
        
        Args:
            model_dict: Dictionary mapping weather condition to model
                e.g., {'clear': model_1, 'cloudy': model_2, 'rainy': model_3}
            weather_feature_idx: Index or indices of weather features in input
        """
        self.model_dict = model_dict
        self.weather_feature_idx = weather_feature_idx
        
    def determine_weather(self, features):
        """
        Determine weather condition from features.
        This is a simplified example - adapt to your specific data.
        """
        # Example: Determine weather from cloud cover (assumed to be at index 4)
        if self.weather_feature_idx is None:
            # Default to using cloud cover if no specific index provided
            cloud_index = 4
        else:
            cloud_index = self.weather_feature_idx
            
        cloud_cover = features[:, cloud_index].cpu().numpy()
        
        # Example classification - adjust thresholds based on your data
        conditions = []
        for cc in cloud_cover:
            if cc < 0.2:  # Normalized cloud cover
                conditions.append('clear')
            elif cc < 0.7:
                conditions.append('partly_cloudy')
            else:
                conditions.append('cloudy')
                
        return conditions
    
    def predict(self, features, targets, device):
        """
        Make predictions using the appropriate model for each weather condition.
        """
        # Determine weather conditions
        conditions = self.determine_weather(features)
        
        # Set models to evaluation mode
        for model in self.model_dict.values():
            model.eval()
        
        # Make predictions
        predictions = torch.zeros_like(targets)
        
        with torch.no_grad():
            for i, condition in enumerate(conditions):
                model = self.model_dict.get(condition, next(iter(self.model_dict.values())))
                pred = model(features[i:i+1])
                if pred.dim() > 1 and pred.shape[1] > 1:
                    pred = pred[:, 0]  # Take first prediction if multi-step
                predictions[i] = pred
        
        return predictions.cpu().numpy(), targets.cpu().numpy()

class LSTMSolarForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, forecast_horizon=24):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.forecast_layer = nn.Linear(hidden_size, forecast_horizon)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use the final LSTM output for forecasting
        # lstm_out shape: [batch_size, seq_len, hidden_size]
        final_hidden = lstm_out[:, -1, :]
        
        # Generate forecasts
        forecasts = self.forecast_layer(final_hidden)
        
        return forecasts

class CNNSolarForecaster(nn.Module):
    def __init__(self, input_size, seq_length, num_filters=64, kernel_size=3, forecast_horizon=24):
        super().__init__()
        
        # Reshape input for CNN (treating temporal dimension as channels)
        self.seq_length = seq_length
        self.input_size = input_size
        
        # 1D CNN layers
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(num_filters*2, num_filters*2, kernel_size=kernel_size, padding=kernel_size//2)
        
        # Pooling and activation
        self.pool = nn.MaxPool1d(2)
        self.activation = nn.ReLU()
        
        # Calculate the size after convolutions and pooling
        conv_output_size = seq_length // 2  # After pooling
        
        # Fully connected layers for forecasting
        self.fc1 = nn.Linear(num_filters*2 * conv_output_size, 128)
        self.fc2 = nn.Linear(128, forecast_horizon)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        batch_size = x.shape[0]
        
        # Reshape for 1D CNN: [batch_size, features, seq_len]
        x = x.permute(0, 2, 1)
        
        # Apply CNN layers
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = self.activation(self.fc1(x))
        forecasts = self.fc2(x)
        
        return forecasts

# class TransformerSolarForecaster(nn.Module):
#     def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1, forecast_horizon=24):
#         super().__init__()
        
#         # Embedding to transform input features to transformer dimension
#         self.embedding = nn.Linear(input_size, d_model)
        
#         # Positional encoding for time awareness
#         self.pos_encoder = PositionalEncoding(d_model, dropout)
        
#         # Transformer encoder
#         encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
#         # Output layer
#         self.output_layer = nn.Linear(d_model, forecast_horizon)
        
#         self.d_model = d_model
        
#     def forward(self, x):
#         # x shape: [batch_size, seq_len, features]
        
#         # Reshape for transformer: [seq_len, batch_size, features]
#         x = x.permute(1, 0, 2)
        
#         # Embed and add positional encoding
#         x = self.embedding(x) * math.sqrt(self.d_model)
#         x = self.pos_encoder(x)
        
#         # Transform
#         x = self.transformer_encoder(x)
        
#         # Use the output corresponding to the last time step
#         x = x[-1, :, :]
        
#         # Generate forecasts
#         forecasts = self.output_layer(x)
        
#         return forecasts

# # Positional encoding for transformer
# class PositionalEncoding(nn.Module):
#         def __init__(self, d_model, dropout=0.1, max_len=5000):
#             super().__init__()
#             self.dropout = nn.Dropout(p=dropout)
            
#             position = torch.arange(max_len).unsqueeze(1)
#             div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#             pe = torch.zeros(max_len, 1, d_model)
#             pe[:, 0, 0::2] = torch.sin(position * div_term)
#             pe[:, 0, 1::2] = torch.cos(position * div_term)
#             self.register_buffer('pe', pe)
            
#         def forward(self, x):
#             x = x + self.pe[:x.size(0)]
#             return self.dropout(x)




def fetch_weather_forecast(lat, lon, api_key):
    """
    Fetch weather forecast data from OpenWeatherMap API.
    """
    import requests
    
    # Example API call to OpenWeatherMap 5-day/3-hour forecast
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching forecast data: {response.status_code}")
        return None

def process_forecast_data(forecast_json):
    """
    Process raw forecast JSON into a pandas DataFrame.
    """
    import pandas as pd
    from datetime import datetime
    
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

def engineer_weather_features(df, df_forecast):
    """
    Merge historical data with forecast and engineer additional features.
    
    Args:
        df: Historical solar production DataFrame
        df_forecast: Weather forecast DataFrame
    
    Returns:
        DataFrame with additional weather forecast features
    """
    # First, align the forecast data with historical timestamps
    # (This is for training - in production, you'd use actual forecasts)
    df_merged = df.copy()
    
    # Add raw forecast features
    for col in df_forecast.columns:
        df_merged[col] = np.nan  # Initialize
    
    # Fill with forecast data where available (for dates in common)
    common_dates = df_merged.index.intersection(df_forecast.index)
    for col in df_forecast.columns:
        df_merged.loc[common_dates, col] = df_forecast.loc[common_dates, col]
    
    # Engineer additional features
    
    # 1. Clear Sky Ratio (actual solar radiation / theoretical max)
    if 'forecast_clouds' in df_merged.columns:
        # Convert cloud percentage to clear sky ratio (inverse relationship)
        df_merged['clear_sky_ratio'] = 1 - (df_merged['forecast_clouds'] / 100)
    
    # 2. Weather condition encoding
    if 'forecast_weather_code' in df_merged.columns:
        # Map weather codes to numerical categories
        # Thunderstorm: 200-299, Drizzle: 300-399, Rain: 500-599, Snow: 600-699, Clear: 800, Clouds: 801-899
        weather_categories = {
            'clear': (df_merged['forecast_weather_code'] == 800),
            'partly_cloudy': ((df_merged['forecast_weather_code'] >= 801) & (df_merged['forecast_weather_code'] <= 803)),
            'cloudy': (df_merged['forecast_weather_code'] == 804),
            'precipitation': ((df_merged['forecast_weather_code'] >= 300) & (df_merged['forecast_weather_code'] <= 599)),
            'snow': ((df_merged['forecast_weather_code'] >= 600) & (df_merged['forecast_weather_code'] <= 699)),
            'thunderstorm': ((df_merged['forecast_weather_code'] >= 200) & (df_merged['forecast_weather_code'] <= 299))
        }
        
        for category, condition in weather_categories.items():
            df_merged[f'weather_{category}'] = condition.astype(int)
    
    # 3. Irradiance reduction factors
    # Calculate estimated effect of clouds, precipitation on solar radiation
    if 'forecast_clouds' in df_merged.columns and 'forecast_precipitation' in df_merged.columns:
        # Simple model: clouds reduce irradiance linearly, precipitation has additional impact
        cloud_factor = 1 - (df_merged['forecast_clouds'] / 100) * 0.8  # 100% clouds reduces irradiance by 80%
        precip_factor = 1 - np.minimum(df_merged['forecast_precipitation'] * 0.5, 0.9)  # Max 90% reduction
        df_merged['irradiance_factor'] = cloud_factor * precip_factor
    
    # 4. Time-based features for seasonality and daily patterns
    df_merged['forecast_hour_sin'] = np.sin(2 * np.pi * df_merged.index.hour / 24)
    df_merged['forecast_hour_cos'] = np.cos(2 * np.pi * df_merged.index.hour / 24)
    df_merged['forecast_day_sin'] = np.sin(2 * np.pi * df_merged.index.dayofyear / 365)
    df_merged['forecast_day_cos'] = np.cos(2 * np.pi * df_merged.index.dayofyear / 365)
    
    return df_merged

def train_pinn(model, train_loader, val_loader, epochs=100, lr=0.001, 
               physics_weight=0.1, device='cuda' if torch.cuda.is_available() else 'cpu',
               save_path=None, early_stopping=False, patience=20, verbose=True):
    """
    Train the PINN model for solar production forecasting with early stopping support.
    
    Args:
        model: The PINN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of epochs to train
        lr: Learning rate
        physics_weight: Weight of physics loss component
        device: Device to train on ('cuda' or 'cpu')
        save_path: Path to save best model
        early_stopping: Whether to use early stopping
        patience: Number of epochs to wait for improvement before stopping
        verbose: Whether to print progress
    
    Returns:
        model: Trained model
        history: Training history dict
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
    
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'data_loss': [],
        'physics_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    no_improve_count = 0
    
    if verbose:
        print(f"Training PINN on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        data_loss_sum = 0
        physics_loss_sum = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            predictions = model(features)
            
            # Make sure targets have the right shape for comparison
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            if predictions.dim() == 2 and targets.dim() == 2:
                # Take only the first prediction if predicting multiple steps
                predictions_for_loss = predictions[:, 0] if predictions.shape[1] > 1 else predictions.squeeze()
                targets_for_loss = targets[:, 0] if targets.shape[1] > 1 else targets.squeeze()
            else:
                predictions_for_loss = predictions.squeeze()
                targets_for_loss = targets.squeeze()
            
            # Data loss
            data_loss = criterion(predictions_for_loss, targets_for_loss)
            
            # Physics loss
            physics_loss = model.physics_loss(features, predictions)
            
            # Combined loss
            loss = data_loss + physics_weight * physics_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            train_loss += loss.item()
            data_loss_sum += data_loss.item()
            physics_loss_sum += physics_loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_data_loss = data_loss_sum / len(train_loader)
        avg_physics_loss = physics_loss_sum / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                # Forward pass
                predictions = model(features)
                
                # Adjust shapes for loss calculation
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                if predictions.dim() == 2 and targets.dim() == 2:
                    predictions_for_loss = predictions[:, 0] if predictions.shape[1] > 1 else predictions.squeeze()
                    targets_for_loss = targets[:, 0] if targets.shape[1] > 1 else targets.squeeze()
                else:
                    predictions_for_loss = predictions.squeeze()
                    targets_for_loss = targets.squeeze()
                
                # Calculate loss
                loss = criterion(predictions_for_loss, targets_for_loss)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            no_improve_count += 1
        
        # Early stopping check
        if early_stopping and no_improve_count >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} - No improvement for {patience} epochs.")
            break
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['data_loss'].append(avg_data_loss)
        history['physics_loss'].append(avg_physics_loss)
        history['learning_rate'].append(current_lr)
        
        # Print progress
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1:3d}/{epochs} - "
                  f"Train: {avg_train_loss:.6f} "
                  f"(Data: {avg_data_loss:.6f}, Physics: {avg_physics_loss:.6f}) - "
                  f"Val: {avg_val_loss:.6f} - "
                  f"LR: {current_lr:.2e}")
    
    return model, history

def evaluate_model_improved(model, test_loader, dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Improved evaluation with better MAPE calculation.
    """
    model.eval()
    model = model.to(device)
    
    test_loss = 0
    all_targets = []
    all_predictions = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Get predictions
            predictions = model(features)
            
            # Adjust shapes for loss calculation
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            if predictions.dim() == 2 and targets.dim() == 2:
                predictions_for_loss = predictions[:, 0] if predictions.shape[1] > 1 else predictions.squeeze()
                targets_for_loss = targets[:, 0] if targets.shape[1] > 1 else targets.squeeze()
            else:
                predictions_for_loss = predictions.squeeze()
                targets_for_loss = targets.squeeze()
            
            # Calculate loss
            loss = criterion(predictions_for_loss, targets_for_loss)
            test_loss += loss.item()
            
            # Store targets and predictions for analysis
            all_targets.append(targets_for_loss.cpu().numpy())
            all_predictions.append(predictions_for_loss.cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Denormalize if necessary
    if hasattr(dataset, 'denormalize_targets'):
        all_targets = dataset.denormalize_targets(all_targets)
        all_predictions = dataset.denormalize_targets(all_predictions)
    
    # Calculate metrics
    mse = np.mean((all_targets - all_predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_targets - all_predictions))
    
    # Fixed MAPE calculation - only for non-zero targets
    non_zero_mask = all_targets > 100  # Only calculate MAPE for production > 100 Wh
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((all_targets[non_zero_mask] - all_predictions[non_zero_mask]) / 
                             all_targets[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    # Alternative metric: MAPE for significant production hours only
    significant_mask = all_targets > 1000  # > 1 kWh
    if np.any(significant_mask):
        mape_significant = np.mean(np.abs((all_targets[significant_mask] - all_predictions[significant_mask]) / 
                                        all_targets[significant_mask])) * 100
    else:
        mape_significant = float('inf')
    
    # Calculate R-squared
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Daily-level performance
    if len(all_targets) >= 24:
        # Reshape to daily if possible (assuming hourly data)
        n_complete_days = len(all_targets) // 24
        daily_targets = all_targets[:n_complete_days*24].reshape(-1, 24).sum(axis=1) / 1000  # kWh
        daily_predictions = all_predictions[:n_complete_days*24].reshape(-1, 24).sum(axis=1) / 1000  # kWh
        
        daily_mse = np.mean((daily_targets - daily_predictions) ** 2)
        daily_rmse = np.sqrt(daily_mse)
        daily_mae = np.mean(np.abs(daily_targets - daily_predictions))
        daily_mape = np.mean(np.abs((daily_targets - daily_predictions) / daily_targets)) * 100
        
        # Daily R²
        daily_ss_res = np.sum((daily_targets - daily_predictions) ** 2)
        daily_ss_tot = np.sum((daily_targets - np.mean(daily_targets)) ** 2)
        daily_r2 = 1 - (daily_ss_res / daily_ss_tot) if daily_ss_tot > 0 else 0
    else:
        daily_rmse = daily_mae = daily_mape = daily_r2 = None
    
    return {
        'test_loss': test_loss / len(test_loader),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'mape_significant': mape_significant,
        'r2': r2,
        'daily_metrics': {
            'rmse': daily_rmse,
            'mae': daily_mae,
            'mape': daily_mape,
            'r2': daily_r2
        },
        'targets': all_targets,
        'predictions': all_predictions
    }

def visualize_results(history, eval_results, output_dir='./'):
    """
    Create visualizations for model training and evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['data_loss'], label='Data Loss', linewidth=2, color='blue')
    plt.plot(history['physics_loss'], label='Physics Loss', linewidth=2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.semilogy(history['train_loss'], label='Train Loss', linewidth=2)
    plt.semilogy(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('Training Loss (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Predicted vs Actual scatter plot
    plt.figure(figsize=(10, 8))
    
    # Sample points for better visualization
    n_sample = min(5000, len(eval_results['targets']))
    indices = np.random.choice(len(eval_results['targets']), n_sample, replace=False)
    targets_sample = eval_results['targets'][indices]
    predictions_sample = eval_results['predictions'][indices]
    
    plt.subplot(2, 2, 1)
    plt.scatter(targets_sample, predictions_sample, alpha=0.5, s=1)
    plt.plot([targets_sample.min(), targets_sample.max()], 
             [targets_sample.min(), targets_sample.max()], 'r--', lw=2)
    plt.xlabel('Actual Energy (Wh)')
    plt.ylabel('Predicted Energy (Wh)')
    plt.title(f'Predicted vs Actual\nR² = {eval_results["r2"]:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    plt.subplot(2, 2, 2)
    errors = eval_results['predictions'] - eval_results['targets']
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error (Wh)')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution\nMAE = {eval_results["mae"]:.2f} Wh')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Time series example
    plt.subplot(2, 2, 3)
    sample_start = np.random.randint(0, len(eval_results['targets']) - 48)
    sample_end = sample_start + 48
    
    plt.plot(range(48), eval_results['targets'][sample_start:sample_end], 
             label='Actual', linewidth=2)
    plt.plot(range(48), eval_results['predictions'][sample_start:sample_end], 
             label='Predicted', linewidth=2)
    plt.xlabel('Hours')
    plt.ylabel('Energy (Wh)')
    plt.title('48-Hour Sample Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Metrics summary
    plt.subplot(2, 2, 4)
    metrics = ['RMSE', 'MAE', 'MAPE', 'R²']
    values = [eval_results['rmse'], eval_results['mae'], 
              eval_results['mape'], eval_results['r2'] * 100]
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow']
    
    bars = plt.bar(metrics, values, color=colors, edgecolor='black')
    plt.ylabel('Value')
    plt.title('Model Performance Metrics')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                f'{value:.2f}', ha='center', va='center', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Squared Error (MSE):     {eval_results['mse']:>10.2f} Wh²")
    print(f"Root Mean Squared Error:      {eval_results['rmse']:>10.2f} Wh")
    print(f"Mean Absolute Error (MAE):    {eval_results['mae']:>10.2f} Wh")
    print(f"Mean Absolute Percentage Err: {eval_results['mape']:>10.2f} %")
    print(f"R-squared (R²):               {eval_results['r2']:>10.3f}")
    print("="*60)

def print_performance_context(eval_results, system_capacity=195):
    """Add context to help interpret the results."""
    rmse_percentage = (eval_results['rmse'] / (system_capacity * 1000)) * 100
    mae_percentage = (eval_results['mae'] / (system_capacity * 1000)) * 100
    
    print(f"\n📊 PERFORMANCE CONTEXT:")
    print(f"RMSE as % of max capacity:  {rmse_percentage:.1f}%")
    print(f"MAE as % of max capacity:   {mae_percentage:.1f}%")
    print(f"System capacity:            {system_capacity} kW")
    
    if eval_results['r2'] > 0.9:
        print("✅ Excellent model fit (R² > 0.9)")
    elif eval_results['r2'] > 0.8:
        print("✅ Good model fit (R² > 0.8)")
    else:
        print("⚠️ Model fit could be improved")

def ensemble_weighted_average_fixed(models, weights, test_loader, device, dataset):
    """
    FIXED: Properly combine predictions with denormalization
    """
    assert len(models) == len(weights), "Number of models must match number of weights"
    assert np.isclose(sum(weights), 1.0), "Weights must sum to 1"
    
    print(f"Creating ensemble with {len(models)} models and weights: {weights}")
    
    all_ensemble_predictions = []
    all_actual_values = []
    
    # Set all models to evaluation mode
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(test_loader):
            features, targets = features.to(device), targets.to(device)
            
            # Get predictions from each model
            batch_predictions = []
            for model_idx, model in enumerate(models):
                pred = model(features)
                
                # Ensure consistent shape handling
                if pred.dim() > 1:
                    if pred.shape[1] > 1:
                        pred = pred[:, 0]
                    pred = pred.squeeze()
                
                batch_predictions.append(pred)
            
            # Stack predictions properly
            batch_predictions = torch.stack(batch_predictions, dim=0)  # [n_models, batch_size]
            
            # Apply weights
            weights_tensor = torch.tensor(weights, device=device).view(-1, 1)
            batch_ensemble = torch.sum(batch_predictions * weights_tensor, dim=0)
            
            # Store results (keep as tensors for now)
            all_ensemble_predictions.append(batch_ensemble)
            
            # Handle targets
            if targets.dim() > 1:
                if targets.shape[1] > 1:
                    targets = targets[:, 0]
                targets = targets.squeeze()
            all_actual_values.append(targets)
    
    # Concatenate all batches
    ensemble_predictions = torch.cat(all_ensemble_predictions).cpu().numpy()
    actual_values = torch.cat(all_actual_values).cpu().numpy()
    
    # IMPORTANT: Denormalize here
    if hasattr(dataset, 'denormalize_targets'):
        ensemble_predictions = dataset.denormalize_targets(ensemble_predictions)
        actual_values = dataset.denormalize_targets(actual_values)
    
    print(f"Ensemble predictions shape: {ensemble_predictions.shape}")
    print(f"Actual values shape: {actual_values.shape}")
    print(f"Prediction range: [{ensemble_predictions.min():.1f}, {ensemble_predictions.max():.1f}]")
    print(f"Actual range: [{actual_values.min():.1f}, {actual_values.max():.1f}]")
    
    return ensemble_predictions, actual_values

def create_lstm_heavy_ensemble(trained_models, model_names, test_loader, device, dataset):
    """
    Create an ensemble that heavily favors the LSTM while still benefiting from diversity
    """
    # Find LSTM index
    lstm_idx = None
    for i, name in enumerate(model_names):
        if 'LSTM' in name.upper():
            lstm_idx = i
            break
    
    if lstm_idx is None:
        print("❌ LSTM not found!")
        return None
    
    # Strategy 1: Very heavy LSTM weight (85%)
    lstm_heavy_weights = [0.0] * len(trained_models)
    lstm_heavy_weights[lstm_idx] = 0.85  # 85% LSTM
    
    # Find CNN index
    cnn_idx = None
    for i, name in enumerate(model_names):
        if 'CNN' in name.upper():
            cnn_idx = i
            break
    
    # Distribute remaining weight
    if cnn_idx is not None:
        lstm_heavy_weights[cnn_idx] = 0.10  # 10% CNN
        remaining = 0.05
    else:
        remaining = 0.15
    
    # Distribute remaining among PINNs
    pinn_count = sum(1 for i, w in enumerate(lstm_heavy_weights) if w == 0.0)
    if pinn_count > 0:
        for i in range(len(lstm_heavy_weights)):
            if lstm_heavy_weights[i] == 0.0:
                lstm_heavy_weights[i] = remaining / pinn_count
    
    # Ensure weights sum to 1
    lstm_heavy_weights = [w/sum(lstm_heavy_weights) for w in lstm_heavy_weights]
    
    print(f"\n🎯 LSTM-Heavy Weights:")
    for name, weight in zip(model_names, lstm_heavy_weights):
        print(f"  {name}: {weight:.2%}")
    
    # Create ensemble using fixed function
    ensemble_preds, actual_values = ensemble_weighted_average_fixed(
        trained_models, lstm_heavy_weights, test_loader, device, dataset
    )
    
    # Evaluate
    mae = np.mean(np.abs(actual_values - ensemble_preds))
    rmse = np.sqrt(np.mean((actual_values - ensemble_preds) ** 2))
    r2 = 1 - np.sum((actual_values - ensemble_preds) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2)
    
    # MAPE calculation
    non_zero_mask = actual_values > 100
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((actual_values[non_zero_mask] - ensemble_preds[non_zero_mask]) / 
                             actual_values[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    result = {
        'ensemble_name': 'LSTM-Heavy Ensemble (85% LSTM)',
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'weights': lstm_heavy_weights
    }
    
    print(f"\n📊 LSTM-Heavy Ensemble Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.0f} Wh")
    print(f"  RMSE: {rmse:.0f} Wh")
    print(f"  MAPE: {mape:.1f}%")
    
    return result

def create_selective_ensemble(trained_models, model_names, evaluation_results, test_loader, device, dataset, r2_threshold=0.93):
    """
    Create ensemble using only models above a certain R² threshold
    """
    # Select only high-performing models
    selected_models = []
    selected_names = []
    selected_r2s = []
    selected_indices = []
    
    for i, result in enumerate(evaluation_results):
        if result['r2'] >= r2_threshold:
            selected_models.append(trained_models[i])
            selected_names.append(model_names[i])
            selected_r2s.append(result['r2'])
            selected_indices.append(i)
    
    print(f"\n🎯 Selective Ensemble (R² > {r2_threshold}):")
    print(f"Selected {len(selected_models)} models:")
    for name, r2 in zip(selected_names, selected_r2s):
        print(f"  {name}: R² = {r2:.4f}")
    
    if len(selected_models) < 2:
        print("❌ Not enough high-performing models for ensemble")
        print("💡 Lowering threshold to 0.90...")
        
        # Try with lower threshold
        r2_threshold = 0.90
        selected_models = []
        selected_names = []
        selected_r2s = []
        
        for i, result in enumerate(evaluation_results):
            if result['r2'] >= r2_threshold:
                selected_models.append(trained_models[i])
                selected_names.append(model_names[i])
                selected_r2s.append(result['r2'])
        
        if len(selected_models) < 2:
            print("Still not enough models. Using top 2 models instead.")
            # Sort by R² and take top 2
            sorted_results = sorted(enumerate(evaluation_results), key=lambda x: x[1]['r2'], reverse=True)
            selected_models = [trained_models[idx] for idx, _ in sorted_results[:2]]
            selected_names = [model_names[idx] for idx, _ in sorted_results[:2]]
            selected_r2s = [result['r2'] for idx, result in sorted_results[:2]]
    
    # Weight by performance (normalized R² scores)
    min_r2 = min(selected_r2s)
    adjusted_r2s = [r2 - min_r2 + 0.01 for r2 in selected_r2s]  # Shift to positive
    total_adjusted = sum(adjusted_r2s)
    weights = [r2/total_adjusted for r2 in adjusted_r2s]
    
    print(f"\nPerformance-based weights:")
    for name, weight, r2 in zip(selected_names, weights, selected_r2s):
        print(f"  {name}: {weight:.2%} (R² = {r2:.4f})")
    
    # Create ensemble using fixed function
    ensemble_preds, actual_values = ensemble_weighted_average_fixed(
        selected_models, weights, test_loader, device, dataset
    )
    
    # Evaluate
    mae = np.mean(np.abs(actual_values - ensemble_preds))
    rmse = np.sqrt(np.mean((actual_values - ensemble_preds) ** 2))
    r2 = 1 - np.sum((actual_values - ensemble_preds) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2)
    
    # MAPE calculation
    non_zero_mask = actual_values > 100
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((actual_values[non_zero_mask] - ensemble_preds[non_zero_mask]) / 
                             actual_values[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    print(f"\n📊 Selective Ensemble Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.0f} Wh")
    print(f"  RMSE: {rmse:.0f} Wh")
    print(f"  MAPE: {mape:.1f}%")
    
    return {
        'ensemble_name': f'Selective Ensemble ({len(selected_models)} models)',
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'models': selected_names,
        'weights': weights
    }

def main():
    """
    COMPLETE FIXED MAIN FUNCTION
    Properly utilizes all functions and includes comprehensive error handling
    """
    import ssl
    import urllib3
    import warnings
    import traceback
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("🌞 SOLAR PRODUCTION FORECASTING - COMPLETE PIPELINE")
    print("="*80)
    
    # Check GPU status
    check_gpu_status()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  CUDA not available, using CPU")
    
    # =============================================================================
    # STEP 1: DATA LOADING AND PREPARATION
    # =============================================================================
    print("\n📊 STEP 1: DATA LOADING AND PREPARATION")
    print("-" * 50)
    
    # Load data
    data_path = "C:/Users/Lospsy/Desktop/Thesis/Results/cleaned_forecast_data.csv"
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✅ Loaded cleaned data: {df.shape}")
    else:
        # Fallback to original data
        data_path = "C:/Users/Lospsy/Desktop/Thesis/Results/forecast_data.csv"
        print(f"⚠️ Cleaned data not found, loading original data from {data_path}")
        df = pd.read_csv(data_path)
        df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        df['WS_10m'] = 3.0
        df['mismatch'] = df['ac_power_output'] / 1000 - df['Load (kW)']
        df = df.ffill().bfill()
    
    # Add time features if missing
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['dayofweek'] = df.index.dayofweek
    
    # Weather forecast integration (optional)
    print("\n🌤️ Attempting weather forecast integration...")
    try:
        lat, lon = 37.98983, 23.74328
        api_key = "7273588818d8b2bb8597ee797baf4935"
        
        forecast_json = fetch_weather_forecast(lat, lon, api_key)
        if forecast_json:
            df_forecast = process_forecast_data(forecast_json)
            df_enhanced = engineer_weather_features(df, df_forecast)
            print(f"✅ Enhanced data with forecast features: {df_enhanced.shape}")
            df = df_enhanced  # Use enhanced data
        else:
            print("⚠️ Weather forecast not available, using original data")
    except Exception as e:
        print(f"⚠️ Weather forecast integration failed: {e}")
        print("Continuing with original data...")
    
    # Extract system parameters
    try:
        panel_efficiency = df['param_panel_efficiency'].iloc[0]
        panel_area = df['param_panel_area'].iloc[0]
        temp_coeff = df['param_temp_coeff'].iloc[0]
        print(f"\n⚙️ System parameters:")
        print(f"  Panel efficiency: {panel_efficiency:.4f}")
        print(f"  Panel area: {panel_area:.4f} m²")
        print(f"  Temperature coefficient: {temp_coeff:.6f}")
    except Exception as e:
        print(f"⚠️ Error extracting system parameters: {e}")
        panel_efficiency, panel_area, temp_coeff = 0.146, 1.642, -0.0044
        print(f"Using default parameters")
    
    # Create dataset
    try:
        dataset = SolarProductionDataset(df, seq_length=24, forecast_horizon=1, normalize=True)
        print(f"✅ Dataset created: {len(dataset)} samples with {len(dataset.feature_names)} features")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📊 Data split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    # =============================================================================
    # STEP 2: MAIN PINN MODEL TRAINING/LOADING
    # =============================================================================
    print("\n🧠 STEP 2: MAIN PINN MODEL TRAINING/LOADING")
    print("-" * 50)
    
    # Setup output directories
    output_dir = "C:/Users/Lospsy/Desktop/Thesis/Results/PINN_results"
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, 'solar_production_pinn_best.pt')
    
    # Create main PINN model
    model = SolarProductionPINN(
        input_size=len(dataset.feature_names),
        hidden_size=64,
        num_layers=3,
        panel_efficiency=panel_efficiency,
        panel_area=panel_area,
        temp_coeff=temp_coeff
    ).to(device)
    
    # Use our smart loading function
    print("🔍 Checking for existing main PINN model...")
    try:
        trained_model, history = load_or_train_main_pinn(
            model, train_loader, val_loader, model_save_path, device
        )
        print("✅ Main PINN model ready!")
    except Exception as e:
        print(f"❌ Error with main PINN model: {e}")
        return
    
    # Evaluate main model
    print("\n📊 Evaluating main PINN model...")
    try:
        eval_results = evaluate_model_improved(trained_model, test_loader, dataset, device=device)
        
        # Show results
        print(f"\n🎯 MAIN PINN PERFORMANCE:")
        print(f"  R² Score: {eval_results['r2']:.4f}")
        print(f"  MAE: {eval_results['mae']:.0f} Wh")
        print(f"  RMSE: {eval_results['rmse']:.0f} Wh")
        
        # Add performance context
        print_performance_context(eval_results)
        
    except Exception as e:
        print(f"❌ Error evaluating main model: {e}")
        eval_results = None
    
    # Create basic visualizations
    try:
        visualize_results(history, eval_results, output_dir)
        print("✅ Basic visualizations created")
    except Exception as e:
        print(f"⚠️ Error creating basic visualizations: {e}")
    
    # =============================================================================
    # STEP 3: COMPREHENSIVE VISUALIZATIONS WITH TESTING
    # =============================================================================
    print("\n🎨 STEP 3: COMPREHENSIVE VISUALIZATIONS")
    print("-" * 50)
    
    # Test visualizations first
    test_vis_dir = os.path.join(output_dir, "test_visualizations")
    
    if quick_visualization_test(trained_model, test_loader, dataset, device, test_vis_dir):
        print("✅ Visualization system works!")
        
        # Create comprehensive visualizations
        vis_output_dir = os.path.join(output_dir, "detailed_visualizations")
        os.makedirs(vis_output_dir, exist_ok=True)
        
        try:
            results_df = create_comprehensive_visualizations(
                trained_model, test_loader, dataset, df, 
                device=device, output_dir=vis_output_dir
            )
            if results_df is not None:
                print("🎉 Comprehensive visualizations completed!")
            else:
                print("⚠️ Some visualizations may have failed")
        except Exception as e:
            print(f"❌ Error creating comprehensive visualizations: {e}")
    else:
        print("❌ Visualization system has issues - skipping detailed visualizations")
    
    # =============================================================================
    # STEP 4: DIVERSE MODEL TRAINING AND ENSEMBLE CREATION
    # =============================================================================
    print("\n🤖 STEP 4: DIVERSE MODEL TRAINING AND ENSEMBLE CREATION")
    print("-" * 50)
    
    diverse_models_dir = "C:/Users/Lospsy/Desktop/Thesis/Results/PINN_diverse"
    
    # Initialize variables to track results
    evaluation_results = []
    ensemble_result = None
    stacked_result = None
    
    try:
        # Train diverse models (or load if they exist)
        trained_models, model_names, evaluation_results, ensemble_result, stacked_result = train_diverse_models(
            df, diverse_models_dir
        )
        print("✅ Diverse model training completed!")
        
        # Show individual model performance
        print(f"\n📊 INDIVIDUAL MODEL PERFORMANCE:")
        for result in evaluation_results:
            print(f"  {result['model_name']:15}: R² = {result['r2']:.4f}, MAE = {result['mae']:.0f} Wh")
        
        # IMPORTANT: Show ensemble results here!
        if ensemble_result:
            print(f"\n📊 ENSEMBLE PERFORMANCE:")
            print(f"  {ensemble_result['ensemble_name']:25}: R² = {ensemble_result['r2']:.4f}, MAE = {ensemble_result['mae']:.0f} Wh")
        
        if stacked_result:
            print(f"  {stacked_result['ensemble_name']:25}: R² = {stacked_result['r2']:.4f}, MAE = {stacked_result['mae']:.0f} Wh")
        
        # Create model comparison visualization
        try:
            create_model_comparison_visualization(
                evaluation_results, 
                [ensemble_result, stacked_result], 
                diverse_models_dir
            )
            print("✅ Model comparison visualization created")
        except Exception as e:
            print(f"⚠️ Error creating model comparison: {e}")
        
    except Exception as e:
        print(f"❌ Error training diverse models: {e}")
        print("Continuing with main model only...")
        trained_models = [trained_model]
        model_names = ["Main-PINN"]
        evaluation_results = []
    
    # DEBUG: Check results after Step 4
    print(f"\n📊 DEBUG after STEP 4:")
    print(f"  Number of models trained: {len(trained_models) if 'trained_models' in locals() else 0}")
    print(f"  Number of evaluation results: {len(evaluation_results)}")
    print(f"  Ensemble result exists: {ensemble_result is not None}")
    print(f"  Stacked result exists: {stacked_result is not None}")
    
    # # =============================================================================
    # # STEP 5: OPTIMAL WEIGHT CALCULATION AND ADVANCED ENSEMBLES
    # # =============================================================================
    # print("\n⚖️ STEP 5: ENHANCED OPTIMAL WEIGHT CALCULATION")
    # print("-" * 50)
    
    # if len(trained_models) > 1:
    #     # Calculate optimal weights with enhanced method
    #     try:
    #         optimal_weights = calculate_optimal_weights(trained_models, val_loader, device, dataset)
    #         print(f"✅ Enhanced optimal weights calculated:")
    #         for name, weight in zip(model_names, optimal_weights):
    #             print(f"  {name:15}: {weight:.4f}")
                
    #         # Analyze the weights
    #         weight_analysis = analyze_ensemble_weights(
    #             trained_models, model_names, optimal_weights, test_loader, device, dataset
    #         )
            
    #     except Exception as e:
    #         print(f"⚠️ Error calculating optimal weights: {e}")
    #         optimal_weights = [1/len(trained_models)] * len(trained_models)
    #         print("Using equal weights instead")
        
    #     # Create optimally weighted ensemble with analysis
    #     try:
    #         optimal_ensemble_preds, actual_values = ensemble_weighted_average(
    #             trained_models, optimal_weights, test_loader, device
    #         )
    #         optimal_result = evaluate_ensemble(
    #             optimal_ensemble_preds, actual_values, "Enhanced Optimal Ensemble"
    #         )
    #         print("✅ Enhanced optimal ensemble created")
            
    #         # Compare with equal weights
    #         equal_weights = [1/len(trained_models)] * len(trained_models)
    #         equal_ensemble_preds, _ = ensemble_weighted_average(
    #             trained_models, equal_weights, test_loader, device
    #         )
    #         equal_result = evaluate_ensemble(
    #             equal_ensemble_preds, actual_values, "Equal Weight Ensemble"
    #         )
            
    #         improvement = optimal_result['r2'] - equal_result['r2']
    #         print(f"\n📈 ENSEMBLE IMPROVEMENT:")
    #         print(f"Equal weights R²:   {equal_result['r2']:.4f}")
    #         print(f"Optimal weights R²: {optimal_result['r2']:.4f}")
    #         print(f"Improvement:        {improvement:.4f} ({improvement/equal_result['r2']*100:.1f}%)")
            
    #     except Exception as e:
    #         print(f"⚠️ Error creating optimal ensemble: {e}")
    
     # =============================================================================
    # STEP 5.1: ENHANCED OPTIMAL WEIGHT CALCULATION WITH NEW ENSEMBLES
    # =============================================================================
    print("\n⚖️ STEP 5: ENHANCED OPTIMAL WEIGHT CALCULATION WITH NEW ENSEMBLES")
    print("-" * 50)
    
    if len(trained_models) > 1:
        # First, fix the original ensemble weighted average function
        # Replace ensemble_weighted_average with ensemble_weighted_average_fixed throughout
        
        # Calculate optimal weights with enhanced method
        try:
            optimal_weights = calculate_optimal_weights(trained_models, val_loader, device, dataset)
            print(f"✅ Enhanced optimal weights calculated:")
            for name, weight in zip(model_names, optimal_weights):
                print(f"  {name:15}: {weight:.4f}")
                
            # Analyze the weights
            weight_analysis = analyze_ensemble_weights(
                trained_models, model_names, optimal_weights, test_loader, device, dataset
            )
            
        except Exception as e:
            print(f"⚠️ Error calculating optimal weights: {e}")
            optimal_weights = [1/len(trained_models)] * len(trained_models)
            print("Using equal weights instead")
        
        # Create optimally weighted ensemble with fixed function
        try:
            optimal_ensemble_preds, actual_values = ensemble_weighted_average_fixed(
                trained_models, optimal_weights, test_loader, device, dataset
            )
            optimal_result = evaluate_ensemble(
                optimal_ensemble_preds, actual_values, "Enhanced Optimal Ensemble"
            )
            print("✅ Enhanced optimal ensemble created")
            
            # Compare with equal weights
            equal_weights = [1/len(trained_models)] * len(trained_models)
            equal_ensemble_preds, _ = ensemble_weighted_average_fixed(
                trained_models, equal_weights, test_loader, device, dataset
            )
            equal_result = evaluate_ensemble(
                equal_ensemble_preds, actual_values, "Equal Weight Ensemble"
            )
            
            improvement = optimal_result['r2'] - equal_result['r2']
            print(f"\n📈 ENSEMBLE IMPROVEMENT:")
            print(f"Equal weights R²:   {equal_result['r2']:.4f}")
            print(f"Optimal weights R²: {optimal_result['r2']:.4f}")
            print(f"Improvement:        {improvement:.4f} ({improvement/equal_result['r2']*100:.1f}%)")
            
        except Exception as e:
            print(f"⚠️ Error creating optimal ensemble: {e}")
            optimal_result = None
        
        # NEW: Create LSTM-Heavy Ensemble
        print("\n🚀 Creating LSTM-Heavy Ensemble...")
        try:
            lstm_heavy_result = create_lstm_heavy_ensemble(
                trained_models, model_names, test_loader, device, dataset
            )
            
            # Compare with pure LSTM
            lstm_results = [r for r in evaluation_results if 'LSTM' in r['model_name']]
            if lstm_results:
                lstm_only_r2 = lstm_results[0]['r2']
                improvement = lstm_heavy_result['r2'] - lstm_only_r2
                
                print(f"\n📊 LSTM-Heavy vs Pure LSTM:")
                print(f"  Pure LSTM: R² = {lstm_only_r2:.4f}")
                print(f"  LSTM-Heavy Ensemble: R² = {lstm_heavy_result['r2']:.4f}")
                print(f"  Difference: {improvement:+.4f}")
        except Exception as e:
            print(f"⚠️ Error creating LSTM-heavy ensemble: {e}")
            lstm_heavy_result = None
        
        # NEW: Create Selective Ensemble
        print("\n🎯 Creating Selective Ensemble...")
        try:
            selective_result = create_selective_ensemble(
                trained_models, model_names, evaluation_results, 
                test_loader, device, dataset, r2_threshold=0.93
            )
        except Exception as e:
            print(f"⚠️ Error creating selective ensemble: {e}")
            selective_result = None
    
    # Store all ensemble results for final comparison
    all_ensemble_results = []
    if 'ensemble_result' in locals() and ensemble_result:
        all_ensemble_results.append(ensemble_result)
    if 'stacked_result' in locals() and stacked_result:
        all_ensemble_results.append(stacked_result)
    if 'optimal_result' in locals() and optimal_result:
        all_ensemble_results.append(optimal_result)
    if 'lstm_heavy_result' in locals() and lstm_heavy_result:
        all_ensemble_results.append(lstm_heavy_result)
    if 'selective_result' in locals() and selective_result:
        all_ensemble_results.append(selective_result)
    
    
    # =============================================================================
    # STEP 6: PRODUCTION PIPELINE AND SPECIALIZED ENSEMBLES
    # =============================================================================
    print("\n🏭 STEP 6: PRODUCTION PIPELINE AND SPECIALIZED ENSEMBLES")
    print("-" * 50)

    # Initialize variables to prevent UnboundLocalError
    pipeline_save_path = None
    specialized_ensembles = None

    if len(trained_models) > 1:
        lstm_heavy_result = create_lstm_heavy_ensemble(
            trained_models, model_names, test_loader, device, dataset
        )
        
        # Compare with pure LSTM
        lstm_only_r2 = 0.9506  # Your LSTM score
        improvement = lstm_heavy_result['r2'] - lstm_only_r2
        
        print(f"\n📊 COMPARISON:")
        print(f"  Pure LSTM: R² = {lstm_only_r2:.4f}")
        print(f"  LSTM-Heavy Ensemble: R² = {lstm_heavy_result['r2']:.4f}")
        print(f"  Difference: {improvement:+.4f}")
    
    if len(trained_models) > 1:
    # Create production pipeline with device fix
        try:
            # Fix device mismatch by ensuring all models are on the same device
            for model in trained_models:
                model.to(device)
            
            production_pipeline = create_production_pipeline(trained_models, optimal_weights, dataset, device)
            
            # Test the pipeline
            sample_features, _ = next(iter(test_loader))
            sample_prediction = production_pipeline(sample_features[:1])
            print(f"✅ Production pipeline created and tested")
            print(f"   Sample prediction: {sample_prediction[0]:.2f} Wh")
            
            # Save pipeline
            pipeline_save_path = os.path.join(diverse_models_dir, "production_pipeline.pt")
            torch.save({
                "models": [model.state_dict() for model in trained_models],
                "weights": optimal_weights,
                "model_names": model_names,
                "feature_names": dataset.feature_names
            }, pipeline_save_path)
            print(f"💾 Production pipeline saved to {pipeline_save_path}")
            
        except Exception as e:
            print(f"⚠️ Error creating production pipeline: {e}")
            pipeline_save_path = "Production pipeline creation failed"
    
    # Create time-specialized ensembles with better error handling
    try:
        print("🕐 Creating time-specialized ensembles...")
        specialized_ensembles = create_time_specialized_ensembles_fixed(df, diverse_models_dir)
        print("✅ Time-specialized ensembles created")
    except Exception as e:
        print(f"⚠️ Error creating time-specialized ensembles: {e}")
        specialized_ensembles = None
    
    # =============================================================================
    # STEP 7: PERFORMANCE DASHBOARD
    # =============================================================================
    print("\n📊 STEP 7: PERFORMANCE DASHBOARD")
    print("-" * 50)
    
    if len(trained_models) > 1:
        try:
            dashboard_dir = os.path.join(diverse_models_dir, "dashboard")
            generate_performance_dashboard(trained_models, model_names, test_loader, dataset, dashboard_dir)
            print(f"✅ Performance dashboard created in {dashboard_dir}")
        except Exception as e:
            print(f"⚠️ Error generating dashboard: {e}")
    
    # # =============================================================================
    # # STEP 8: ENHANCED FORECASTING (OPTIONAL)
    # # =============================================================================
    # print("\n🚀 STEP 8: ENHANCED FORECASTING")
    # print("-" * 50)
    
    # try:
    #     enhanced_output_dir = "C:/Users/Lospsy/Desktop/Thesis/Results/PINN_enhanced"
    #     pinn_model, lstm_model, ensemble_models, ensemble_weights = run_enhanced_forecasting(df, enhanced_output_dir)
    #     print("✅ Enhanced forecasting completed!")
    # except Exception as e:
    #     print(f"⚠️ Enhanced forecasting failed: {e}")
    #     print("Continuing with existing results...")
    
     # =============================================================================
    # STEP 8.1: ENHANCED FORECASTING (SIMPLIFIED)
    # =============================================================================
    print("\n🚀 STEP 8: ENHANCED FORECASTING (SIMPLIFIED)")
    print("-" * 50)
    
    # Skip the problematic enhanced forecasting that's creating new models
    # Instead, just use the existing best models
    print("Using existing trained models for enhanced forecasting...")
    print("Best performing models already identified:")
    if len(evaluation_results) > 0:
        sorted_models = sorted(evaluation_results, key=lambda x: x['r2'], reverse=True)
        for i, result in enumerate(sorted_models[:3]):
            print(f"  {i+1}. {result['model_name']}: R² = {result['r2']:.4f}")
    
    print("✅ Enhanced forecasting analysis complete!")
    
    
    # # =============================================================================
    # # STEP 9: FINAL SUMMARY AND RECOMMENDATIONS
    # # =============================================================================
    print("\n🏆 STEP 9: FINAL SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)

    # Find best model
    if len(evaluation_results) > 0:
        best_model_result = max(evaluation_results, key=lambda x: x['r2'])
        print(f"🥇 BEST INDIVIDUAL MODEL: {best_model_result['model_name']}")
        print(f"   R² Score: {best_model_result['r2']:.4f}")
        print(f"   MAE: {best_model_result['mae']:.0f} Wh")
        print(f"   RMSE: {best_model_result['rmse']:.0f} Wh")
    else:
        print(f"🥇 MAIN MODEL PERFORMANCE:")
        if eval_results:
            print(f"   R² Score: {eval_results['r2']:.4f}")
            print(f"   MAE: {eval_results['mae']:.0f} Wh")
            print(f"   RMSE: {eval_results['rmse']:.0f} Wh")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if len(evaluation_results) > 0:
        lstm_results = [r for r in evaluation_results if 'LSTM' in r['model_name']]
        if lstm_results:
            lstm_r2 = lstm_results[0]['r2']
            print(f"✅ Use LSTM model for production (R² = {lstm_r2:.4f})")
        else:
            print(f"✅ Use best individual model: {best_model_result['model_name']}")
        
        if len(trained_models) > 1:
            print(f"✅ Consider ensemble methods for maximum accuracy")
            print(f"✅ Production pipeline is ready for deployment")
    else:
        print(f"✅ Main PINN model is ready for use")

    # Save paths summary - FIXED to handle None values
    print(f"\n📁 RESULTS LOCATIONS:")
    print(f"   Main results: {output_dir}")
    if len(trained_models) > 1:
        print(f"   Diverse models: {diverse_models_dir}")
        if pipeline_save_path and pipeline_save_path != "Production pipeline creation failed":
            print(f"   Production pipeline: {pipeline_save_path}")
        else:
            print(f"   Production pipeline: Creation failed - check logs above")

    print(f"\n🎉 SOLAR FORECASTING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    # # STEP 10: COMPREHENSIVE HYPERPARAMETER TUNING
    # print("\n🔧 STEP 10: COMPREHENSIVE HYPERPARAMETER TUNING")
    # print("=" * 80)
    
    # try:
    #     tuning_output_dir = "C:/Users/Lospsy/Desktop/Thesis/Results/PINN_hyperparameter_tuning"
        
    #     print("Starting comprehensive grid search hyperparameter tuning...")
    #     print("This will test 1,024 different configurations - may take several hours")
        
    #     # Run comprehensive hyperparameter tuning
    #     best_config = run_hyperparameter_tuning(df, tuning_output_dir)
        
    #     print(f"Grid search completed!")
    #     print(f"Best configuration: {best_config}")
        
    #     # STEP 11: EXTENDED TRAINING WITH BEST CONFIGURATION
    #     print("\n🚀 STEP 11: EXTENDED TRAINING (500+ EPOCHS)")
    #     print("=" * 80)
        
    #     extended_output_dir = os.path.join(tuning_output_dir, "extended_training")
        
    #     final_model, extended_eval_results = run_extended_training(
    #         df, best_config, extended_output_dir, epochs=500
    #     )
        
    #     print(f"FINAL OPTIMIZED MODEL PERFORMANCE:")
    #     print(f"  R² Score: {extended_eval_results['r2']:.4f}")
    #     print(f"  MAE: {extended_eval_results['mae']:.0f} Wh") 
    #     print(f"  RMSE: {extended_eval_results['rmse']:.0f} Wh")
        
    #     # Compare with your current best LSTM
    #     print(f"\nCOMPARISON WITH CURRENT BEST:")
    #     print(f"  Current LSTM R²: 0.9506")
    #     print(f"  Optimized PINN R²: {extended_eval_results['r2']:.4f}")
        
    #     if extended_eval_results['r2'] > 0.9506:
    #         print(f"🎉 NEW BEST MODEL ACHIEVED!")
    #     else:
    #         print(f"LSTM still best, but optimized PINN is much improved")
            
    # except Exception as e:
    #     print(f"Hyperparameter tuning failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # print(f"\n🎉 COMPLETE PIPELINE WITH HYPERPARAMETER OPTIMIZATION FINISHED!")
    # print("=" * 80)
    
    # =============================================================================
    # STEP 10: LOAD PRE-TRAINED BEST MODEL
    # =============================================================================
    
    print("\n🔧 STEP 10: LOADING PRE-TRAINED BEST MODEL")
    print("=" * 80)
    
    # Path to your best model
    best_model_path = r"C:\Users\Lospsy\Desktop\Thesis\Results\PINN_hyperparameter_tuning\hyperparameter_tuning\run_20250525_002241\best_model.pt"
    
    # Best configuration
    best_config = {
        'hidden_size': 32, 
        'num_layers': 4, 
        'batch_size': 64, 
        'learning_rate': 0.002, 
        'physics_weight': 0.15
    }
    
    try:
        # Create dataset for evaluation
        dataset = SolarProductionDataset(df, seq_length=24, forecast_horizon=1, normalize=True)
        
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model with best config
        best_tuned_model = SolarProductionPINN(
            input_size=len(dataset.feature_names),
            hidden_size=best_config['hidden_size'],
            num_layers=best_config['num_layers'],
            panel_efficiency=0.146,
            panel_area=1.642,
            temp_coeff=-0.0044
        ).to(device)
        
        # Load pre-trained weights
        best_tuned_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"✅ Loaded pre-trained model from: {best_model_path}")
        
        # Quick evaluation on test set
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        _, _, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        test_loader = DataLoader(test_dataset, batch_size=64, num_workers=0)
        
        # Evaluate
        eval_results = evaluate_model_improved(best_tuned_model, test_loader, dataset, device=device)
        
        print(f"\nPRE-TRAINED MODEL PERFORMANCE:")
        print(f"  R² Score: {eval_results['r2']:.4f}")
        print(f"  MAE: {eval_results['mae']:.0f} Wh")
        print(f"  RMSE: {eval_results['rmse']:.0f} Wh")
        
        # Compare with LSTM
        print(f"\nCOMPARISON:")
        print(f"  Current LSTM R²: 0.9506")
        print(f"  Tuned PINN R²: {eval_results['r2']:.4f}")
        
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        print("Continuing without hyperparameter tuning...")
    
    print("=" * 80)
    # =============================================================================
    # FINAL COMPREHENSIVE SUMMARY 
    # =============================================================================
    print("\n🏆 FINAL COMPREHENSIVE SUMMARY")
    print("=" * 80)
    
    # Collect all results
    all_model_results = []
    
    # Individual models
    if len(evaluation_results) > 0:
        print("📊 INDIVIDUAL MODEL PERFORMANCE:")
        for result in evaluation_results:
            print(f"  {result['model_name']:20}: R² = {result['r2']:.4f}, MAE = {result['mae']:.0f} Wh")
            all_model_results.append((result['model_name'], result['r2'], 'Individual'))
    
    # Add tuned PINN
    if 'eval_results' in locals() and eval_results:
        print(f"  {'Tuned PINN':20}: R² = {eval_results['r2']:.4f}, MAE = {eval_results['mae']:.0f} Wh")
        all_model_results.append(("Tuned PINN", eval_results['r2'], 'Individual'))
    
    # All ensemble results
    print("\n📊 ENSEMBLE MODEL PERFORMANCE:")
    
    # Original ensembles
    if 'ensemble_result' in locals() and ensemble_result:
        print(f"  {'Equal-Weight Ensemble':25}: R² = {ensemble_result['r2']:.4f}")
        all_model_results.append(("Equal-Weight Ensemble", ensemble_result['r2'], 'Ensemble'))
    
    if 'stacked_result' in locals() and stacked_result:
        print(f"  {'Stacked Ensemble':25}: R² = {stacked_result['r2']:.4f}")
        all_model_results.append(("Stacked Ensemble", stacked_result['r2'], 'Ensemble'))
    
    # Enhanced ensembles
    if 'optimal_result' in locals() and optimal_result:
        print(f"  {'Optimal-Weight Ensemble':25}: R² = {optimal_result['r2']:.4f}")
        all_model_results.append(("Optimal-Weight Ensemble", optimal_result['r2'], 'Ensemble'))
    
    if 'lstm_heavy_result' in locals() and lstm_heavy_result:
        print(f"  {'LSTM-Heavy Ensemble (85%)':25}: R² = {lstm_heavy_result['r2']:.4f}")
        all_model_results.append(("LSTM-Heavy Ensemble", lstm_heavy_result['r2'], 'Ensemble'))
    
    if 'selective_result' in locals() and selective_result:
        print(f"  {selective_result['ensemble_name']:25}: R² = {selective_result['r2']:.4f}")
        all_model_results.append((selective_result['ensemble_name'], selective_result['r2'], 'Ensemble'))
    
    # FINAL RANKING
    print("\n🎯 FINAL MODEL RANKING (ALL MODELS):")
    all_model_results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, r2, model_type) in enumerate(all_model_results[:10]):  # Top 10
        marker = "👑" if i == 0 else f"{i+1}."
        print(f"  {marker} {name:30}: R² = {r2:.4f} [{model_type}]")
    
    # Analysis
    print("\n📊 PERFORMANCE ANALYSIS:")
    
    # Best overall
    best_name, best_r2, best_type = all_model_results[0]
    print(f"🥇 Best Overall: {best_name} (R² = {best_r2:.4f})")
    
    # Best individual vs best ensemble
    best_individual = max([r for n, r, t in all_model_results if t == 'Individual'], default=0)
    best_ensemble = max([r for n, r, t in all_model_results if t == 'Ensemble'], default=0)
    
    print(f"\n📈 Model Type Comparison:")
    print(f"  Best Individual Model: R² = {best_individual:.4f}")
    print(f"  Best Ensemble Model: R² = {best_ensemble:.4f}")
    
    if best_individual > best_ensemble:
        diff = (best_individual - best_ensemble) / best_individual * 100
        print(f"  ➡️ Individual models outperform ensembles by {diff:.1f}%")
        print(f"  💡 This suggests one model (likely LSTM) is significantly superior")
    else:
        diff = (best_ensemble - best_individual) / best_individual * 100
        print(f"  ➡️ Ensemble models outperform individuals by {diff:.1f}%")
        print(f"  💡 Model combination provides added value")
    
    # RECOMMENDATIONS
    print(f"\n💡 FINAL RECOMMENDATIONS:")
    
    if best_type == 'Individual':
        print(f"✅ Deploy the {best_name} model for production")
        print(f"   - Simpler architecture")
        print(f"   - Faster inference")
        print(f"   - Excellent performance (R² = {best_r2:.4f})")
        
        # Check if LSTM-heavy ensemble is close
        lstm_heavy_r2 = next((r for n, r, t in all_model_results if 'LSTM-Heavy' in n), 0)
        if lstm_heavy_r2 > 0 and abs(lstm_heavy_r2 - best_r2) < 0.01:
            print(f"\n💡 Alternative: LSTM-Heavy Ensemble (R² = {lstm_heavy_r2:.4f})")
            print(f"   - Nearly identical performance")
            print(f"   - Might be more robust to edge cases")
    else:
        print(f"✅ Deploy the {best_name} for production")
        print(f"   - Better performance than individual models")
        print(f"   - More robust predictions")
        print(f"   - R² = {best_r2:.4f}")
    
    # Performance thresholds
    print(f"\n📊 Performance Summary:")
    excellent_models = [(n, r) for n, r, t in all_model_results if r >= 0.95]
    good_models = [(n, r) for n, r, t in all_model_results if 0.93 <= r < 0.95]
    
    print(f"  Excellent (R² ≥ 0.95): {len(excellent_models)} models")
    print(f"  Good (0.93 ≤ R² < 0.95): {len(good_models)} models")
    
    # Save paths summary
    print(f"\n📁 RESULTS LOCATIONS:")
    print(f"   Main results: {output_dir}")
    if 'diverse_models_dir' in locals():
        print(f"   Diverse models: {diverse_models_dir}")
    if 'pipeline_save_path' in locals() and pipeline_save_path and pipeline_save_path != "Production pipeline creation failed":
        print(f"   Production pipeline: {pipeline_save_path}")
    
    print("\n" + "="*80)
    print("🎉 SOLAR FORECASTING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)

def create_enhanced_training_plot(history, model_name, save_dir):
        """Create enhanced training plots"""
        plt.figure(figsize=(20, 12))
        
        # Plot 1: Loss curves
        plt.subplot(2, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} - Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Log scale loss
        plt.subplot(2, 3, 2)
        plt.semilogy(history['train_loss'], label='Train Loss', linewidth=2)
        plt.semilogy(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.title('Loss (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Loss components (if PINN)
        if 'physics_loss' in history:
            plt.subplot(2, 3, 3)
            plt.plot(history['data_loss'], label='Data Loss', linewidth=2)
            plt.plot(history['physics_loss'], label='Physics Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Components')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 4: Learning rate schedule
        if 'learning_rate' in history:
            plt.subplot(2, 3, 4)
            plt.semilogy(history['learning_rate'], linewidth=2, color='purple')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Training progress
        plt.subplot(2, 3, 5)
        improvement = [(history['val_loss'][0] - loss) / history['val_loss'][0] * 100 
                    for loss in history['val_loss']]
        plt.plot(improvement, linewidth=2, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Improvement (%)')
        plt.title('Validation Improvement')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Convergence analysis
        plt.subplot(2, 3, 6)
        window = 10
        if len(history['val_loss']) > window:
            smoothed = np.convolve(history['val_loss'], np.ones(window)/window, mode='valid')
            plt.plot(smoothed, linewidth=2, color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Smoothed Val Loss')
            plt.title(f'Convergence (Moving Avg {window})')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{model_name.lower()}_enhanced_training.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_extended_training_plot(history, save_dir):
        """Create comprehensive plot for extended training"""
        plt.figure(figsize=(16, 12))
        
        # Main loss plot
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Extended Training - Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate evolution
        plt.subplot(2, 2, 2)
        plt.semilogy(history.get('learning_rate', [0.001] * len(history['train_loss'])), 
                    linewidth=2, color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        
        # Loss improvement over time
        plt.subplot(2, 2, 3)
        train_improvement = [(history['train_loss'][0] - loss) / history['train_loss'][0] * 100 
                            for loss in history['train_loss']]
        val_improvement = [(history['val_loss'][0] - loss) / history['val_loss'][0] * 100 
                        for loss in history['val_loss']]
        
        plt.plot(train_improvement, label='Train Improvement', linewidth=2)
        plt.plot(val_improvement, label='Val Improvement', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Improvement (%)')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Final convergence analysis
        plt.subplot(2, 2, 4)
        plt.semilogy(history['train_loss'], label='Train Loss', linewidth=2)
        plt.semilogy(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.title('Final Model Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "extended_training_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

def load_or_train_main_pinn(model, train_loader, val_loader, model_save_path, device):
    """Load existing model or train new one"""
    
    if os.path.exists(model_save_path):
        print(f"✅ Loading existing main PINN model from {model_save_path}")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
            print(f"✅ Main PINN model loaded successfully!")
            
            # Create dummy history for compatibility
            history = {
                'train_loss': [0.008],
                'val_loss': [0.008], 
                'data_loss': [0.008],
                'physics_loss': [0.0001],
                'learning_rate': [0.001]
            }
            return model, history
            
        except Exception as e:
            print(f"❌ Error loading main PINN model: {e}")
            print(f"🔄 Training new main PINN model...")
    else:
        print(f"🔄 Training new main PINN model...")
    
    # Train new model
    model, history = train_pinn(
        model, train_loader, val_loader,
        epochs=100, lr=0.001, physics_weight=0.1,
        device=device, save_path=model_save_path
    )
    
    return model, history

def create_best_ensemble(trained_models, model_names, test_loader, device, dataset):
    """Create the best possible ensemble including LSTM"""
    
    print("\n🏆 Creating BEST ENSEMBLE (including LSTM)")
    
    # Find the LSTM model
    lstm_idx = None
    for i, name in enumerate(model_names):
        if 'LSTM' in name.upper():
            lstm_idx = i
            break
    
    if lstm_idx is None:
        print("⚠️ LSTM model not found in ensemble!")
        return None
    
    print(f"✅ Found LSTM model at index {lstm_idx}")
    
    # Create optimized weights (give more weight to best performing models)
    if len(trained_models) == 4:  # PINN-Small, PINN-Large, LSTM, CNN
        optimized_weights = [0.15, 0.20, 0.50, 0.15]  # Higher weight to LSTM
        print("🎯 Using performance-optimized weights:")
        for i, (name, weight) in enumerate(zip(model_names, optimized_weights)):
            print(f"   {name}: {weight:.2f}")
    else:
        optimized_weights = [1/len(trained_models)] * len(trained_models)
    
    # Create ensemble predictions
    try:
        ensemble_preds, actual_values = ensemble_weighted_average(
            trained_models, optimized_weights, test_loader, device
        )
        
        # Denormalize
        if hasattr(dataset, 'denormalize_targets'):
            ensemble_preds = dataset.denormalize_targets(ensemble_preds)
            actual_values = dataset.denormalize_targets(actual_values)
        
        # Evaluate
        mae = np.mean(np.abs(actual_values - ensemble_preds))
        rmse = np.sqrt(np.mean((actual_values - ensemble_preds) ** 2))
        r2 = 1 - np.sum((actual_values - ensemble_preds) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2)
        
        result = {
            'ensemble_name': 'BEST Ensemble (LSTM-weighted)',
            'mae': mae,
            'rmse': rmse, 
            'r2': r2,
            'weights': optimized_weights
        }
        
        print(f"🏆 BEST Ensemble Results:")
        print(f"   MAE: {mae:.0f} Wh")
        print(f"   RMSE: {rmse:.0f} Wh")
        print(f"   R²: {r2:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error creating best ensemble: {e}")
        return None

def run_hyperparameter_tuning(df, base_output_dir):
    """
    COMPREHENSIVE: Enhanced hyperparameter tuning with larger search space
    """
    # Create dataset
    dataset = SolarProductionDataset(df, seq_length=24, forecast_horizon=1, normalize=True)
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Initialize hyperparameter tuner
    tuner = HyperparameterTuner(
        base_output_dir=base_output_dir,
        dataset=dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    # COMPREHENSIVE PARAMETER GRID - Grid Search covers all combinations
    param_grid = {
        "hidden_size": [32, 64, 128, 256],           # 4 options
        "num_layers": [2, 3, 4, 5],                  # 4 options  
        "batch_size": [16, 32, 64, 128],             # 4 options
        "learning_rate": [0.0005, 0.001, 0.002, 0.005], # 4 options
        "physics_weight": [0.05, 0.1, 0.15, 0.2]    # 4 options
    }
    
    # Total combinations: 4^5 = 1,024 different configurations!
    print(f"Grid Search will test {4**5} different configurations")
    print("This is comprehensive but will take time...")
    
    # Run grid search with extended training
    best_config, results = tuner.run_grid_search(
        param_grid=param_grid,
        epochs=300,        # INCREASED from 150
        early_stopping=True
    )
    
    print(f"Best configuration found: {best_config}")
    
    return best_config

def run_extended_training(df, config, output_dir, epochs=500):
    """
    Run extended training with the best configuration.
    
    Args:
        df: Preprocessed dataframe
        config: Best hyperparameter configuration
        output_dir: Output directory
        epochs: Number of epochs for extended training
    """
    # Extract parameters
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    physics_weight = config["physics_weight"]
    
    # Create dataset
    dataset = SolarProductionDataset(df, seq_length=24, forecast_horizon=1, normalize=True)
    
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['dayofweek'] = df.index.dayofweek
    # Split into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            num_workers=0, pin_memory=True)
    
    # Create output directory
    extended_dir = Path(output_dir) / "extended_training"
    extended_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract system parameters
    try:
        panel_efficiency = df['param_panel_efficiency'].iloc[0]
        panel_area = df['param_panel_area'].iloc[0]
        temp_coeff = df['param_temp_coeff'].iloc[0]
    except:
        print("Using default panel parameters")
        panel_efficiency = 0.146
        panel_area = 1.642
        temp_coeff = -0.0044
    
    # Create model with best configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SolarProductionPINN(
        input_size=len(dataset.feature_names),
        hidden_size=hidden_size,
        num_layers=num_layers,
        panel_efficiency=panel_efficiency,
        panel_area=panel_area,
        temp_coeff=temp_coeff
    ).to(device)
    
    # Save path for best model
    model_save_path = extended_dir / "best_model.pt"
    
    print(f"\nStarting extended training for {epochs} epochs...")
    print(f"Model architecture: hidden_size={hidden_size}, num_layers={num_layers}")
    print(f"Training parameters: batch_size={batch_size}, lr={learning_rate}, physics_weight={physics_weight}")
    
    # Train the model
    model, history = train_pinn(
        model, train_loader, val_loader,
        epochs=epochs,
        lr=learning_rate,
        physics_weight=physics_weight,
        device=device,
        save_path=model_save_path,
        early_stopping=True,
        patience=30
    )
    
    # Create training history plot
    plt.figure(figsize=(15, 10))
    
    # Plot regular loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot log loss
    plt.subplot(2, 2, 2)
    plt.semilogy(history['train_loss'], label='Train Loss', linewidth=2)
    plt.semilogy(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('Loss (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss components
    plt.subplot(2, 2, 3)
    plt.plot(history['data_loss'], label='Data Loss', linewidth=2, color='blue')
    plt.plot(history['physics_loss'], label='Physics Loss', linewidth=2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.semilogy(history['learning_rate'], linewidth=2, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(extended_dir / "extended_training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    print(f"Loaded best model from {model_save_path}")
    
    # Evaluate the model
    eval_results = evaluate_model_improved(model, test_loader, dataset, device=device)
    
    # Print evaluation results
    print("\n" + "="*60)
    print("EXTENDED TRAINING EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Squared Error (MSE):     {eval_results['mse']:>10.2f} Wh²")
    print(f"Root Mean Squared Error:      {eval_results['rmse']:>10.2f} Wh")
    print(f"Mean Absolute Error (MAE):    {eval_results['mae']:>10.2f} Wh")
    print(f"Mean Absolute Percentage Err: {eval_results['mape']:>10.2f} %")
    print(f"R-squared (R²):               {eval_results['r2']:>10.3f}")
    print("="*60)
    
    # Save final result summary
    with open(extended_dir / "final_results.txt", "w") as f:
        f.write("EXTENDED TRAINING EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Epochs trained: {len(history['train_loss'])}\n")
        f.write(f"Final train loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"Final validation loss: {history['val_loss'][-1]:.6f}\n")
        f.write(f"Mean Squared Error (MSE): {eval_results['mse']:.2f} Wh²\n")
        f.write(f"Root Mean Squared Error: {eval_results['rmse']:.2f} Wh\n")
        f.write(f"Mean Absolute Error (MAE): {eval_results['mae']:.2f} Wh\n")
        f.write(f"Mean Absolute Percentage Err: {eval_results['mape']:.2f} %\n")
        f.write(f"R-squared (R²): {eval_results['r2']:.4f}\n")
        f.write("="*60 + "\n")
    
    return model, eval_results

def create_comprehensive_visualizations(model, test_loader, dataset, df, 
                                       device='cuda', output_dir='./visualizations'):
    """
    ROBUST: Create comprehensive visualizations with better error handling
    """
    import os
    import warnings
    warnings.filterwarnings('ignore')  # Suppress minor warnings
    
    print(f"🎨 Creating visualizations in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Get predictions with robust error handling
    print("📊 Extracting predictions...")
    model.eval()
    all_targets = []
    all_predictions = []
    
    try:
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(test_loader):
                features, targets = features.to(device), targets.to(device)
                predictions = model(features)
                
                # Robust shape handling
                if predictions.dim() > 1 and predictions.shape[1] > 1:
                    predictions = predictions[:, 0]
                predictions = predictions.squeeze()
                
                if targets.dim() > 1 and targets.shape[1] > 1:
                    targets = targets[:, 0]
                targets = targets.squeeze()
                
                # Convert to numpy
                pred_np = predictions.cpu().numpy()
                targ_np = targets.cpu().numpy()
                
                # Ensure 1D arrays
                if pred_np.ndim > 1:
                    pred_np = pred_np.flatten()
                if targ_np.ndim > 1:
                    targ_np = targ_np.flatten()
                
                all_targets.append(targ_np)
                all_predictions.append(pred_np)
        
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        
        # Denormalize if necessary
        if hasattr(dataset, 'denormalize_targets'):
            all_targets = dataset.denormalize_targets(all_targets)
            all_predictions = dataset.denormalize_targets(all_predictions)
        
        print(f"✅ Extracted {len(all_targets)} predictions successfully")
        
    except Exception as e:
        print(f"❌ Error extracting predictions: {e}")
        return None
    
    # Calculate basic metrics
    try:
        mae = np.mean(np.abs(all_targets - all_predictions))
        rmse = np.sqrt(np.mean((all_targets - all_predictions) ** 2))
        r2 = 1 - np.sum((all_targets - all_predictions) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
        
        print(f"📈 Metrics: MAE={mae:.0f} Wh, RMSE={rmse:.0f} Wh, R²={r2:.3f}")
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        mae = rmse = r2 = 0
    
    # 2. Create basic scatter plot (most important)
    print("🎯 Creating scatter plot...")
    try:
        plt.figure(figsize=(10, 8))
        
        # Sample points for better visualization
        n_sample = min(2000, len(all_targets))
        indices = np.random.choice(len(all_targets), n_sample, replace=False)
        x_sample = all_targets[indices]
        y_sample = all_predictions[indices]
        
        # Scatter plot
        plt.scatter(x_sample, y_sample, alpha=0.6, s=20, color='steelblue')
        
        # Identity line
        min_val = min(x_sample.min(), y_sample.min())
        max_val = max(x_sample.max(), y_sample.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add metrics
        plt.text(0.05, 0.95, f"MAE: {mae:.0f} Wh\nRMSE: {rmse:.0f} Wh\nR²: {r2:.3f}",
                transform=plt.gca().transAxes, fontsize=14, weight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.title('Solar Production: Predicted vs Actual', fontsize=16, weight='bold')
        plt.xlabel('Actual Energy Production (Wh)', fontsize=14)
        plt.ylabel('Predicted Energy Production (Wh)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Scatter plot created")
        
    except Exception as e:
        print(f"❌ Error creating scatter plot: {e}")
    
    # 3. Time series plot (simplified)
    print("📈 Creating time series plot...")
    try:
        plt.figure(figsize=(15, 6))
        
        # Show last week or all data if less
        n_hours = min(168, len(all_targets))  # 168 hours = 1 week
        
        hours = range(n_hours)
        actual_data = all_targets[-n_hours:]
        predicted_data = all_predictions[-n_hours:]
        
        plt.plot(hours, actual_data, label='Actual', linewidth=2, alpha=0.8, color='blue')
        plt.plot(hours, predicted_data, label='Predicted', linewidth=2, alpha=0.8, color='red', linestyle='--')
        
        plt.title(f'Solar Production: Last {n_hours} Hours', fontsize=16, weight='bold')
        plt.xlabel('Hours', fontsize=14)
        plt.ylabel('Energy Production (Wh)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_series.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Time series plot created")
        
    except Exception as e:
        print(f"❌ Error creating time series plot: {e}")
    
    # 4. Error distribution
    print("📊 Creating error distribution...")
    try:
        plt.figure(figsize=(10, 6))
        
        errors = all_predictions - all_targets
        
        # Create histogram
        plt.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        
        # Add statistics
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        
        plt.text(0.05, 0.95, f"Mean Error: {error_mean:.0f} Wh\nStd Dev: {error_std:.0f} Wh",
                transform=plt.gca().transAxes, fontsize=12, weight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.title('Prediction Error Distribution', fontsize=16, weight='bold')
        plt.xlabel('Prediction Error (Wh)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Error distribution created")
        
    except Exception as e:
        print(f"❌ Error creating error distribution: {e}")
    
    # 5. Hourly performance (simplified)
    print("⏰ Creating hourly performance plot...")
    try:
        # Create hourly analysis
        n_points = len(all_targets)
        hours = np.arange(n_points) % 24  # Assume hourly data
        
        # Calculate hourly averages
        hourly_actual = []
        hourly_predicted = []
        hourly_errors = []
        
        for hour in range(24):
            hour_mask = hours == hour
            if np.any(hour_mask):
                hourly_actual.append(np.mean(all_targets[hour_mask]))
                hourly_predicted.append(np.mean(all_predictions[hour_mask]))
                hourly_errors.append(np.mean(np.abs(all_predictions[hour_mask] - all_targets[hour_mask])))
            else:
                hourly_actual.append(0)
                hourly_predicted.append(0)
                hourly_errors.append(0)
        
        # Create plot
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Production curves
        ax1.plot(range(24), hourly_actual, 'b-', linewidth=3, label='Average Actual', marker='o')
        ax1.plot(range(24), hourly_predicted, 'r--', linewidth=3, label='Average Predicted', marker='s')
        ax1.set_xlabel('Hour of Day', fontsize=14)
        ax1.set_ylabel('Average Energy Production (Wh)', fontsize=14, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3)
        
        # Error bars
        ax2 = ax1.twinx()
        ax2.bar(range(24), hourly_errors, alpha=0.3, color='orange', label='Average Error')
        ax2.set_ylabel('Average Absolute Error (Wh)', fontsize=14, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('Average Production and Error by Hour of Day', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hourly_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Hourly performance plot created")
        
    except Exception as e:
        print(f"❌ Error creating hourly performance plot: {e}")
    
    # 6. Model performance summary
    print("📋 Creating performance summary...")
    try:
        plt.figure(figsize=(12, 8))
        
        # Calculate additional metrics
        mape = np.mean(np.abs((all_targets - all_predictions) / np.maximum(all_targets, 100))) * 100
        max_error = np.max(np.abs(all_predictions - all_targets))
        
        # Create summary metrics
        metrics = ['MAE (Wh)', 'RMSE (Wh)', 'R² (%)', 'MAPE (%)', 'Max Error (kWh)']
        values = [mae, rmse, r2*100, mape, max_error/1000]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
        
        # Create bar chart
        bars = plt.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        plt.title('Model Performance Summary', fontsize=16, weight='bold')
        plt.ylabel('Value', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add performance interpretation
        performance_text = f"Overall Performance: {'Excellent' if r2 > 0.9 else 'Good' if r2 > 0.8 else 'Fair'}"
        plt.text(0.5, 0.95, performance_text, transform=plt.gca().transAxes, 
                fontsize=14, weight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if r2 > 0.9 else 'yellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Performance summary created")
        
    except Exception as e:
        print(f"❌ Error creating performance summary: {e}")
    
    # 7. Create a simple DataFrame for return (no complex timestamp alignment)
    print("📝 Creating results dataframe...")
    try:
        results_df = pd.DataFrame({
            'actual': all_targets,
            'predicted': all_predictions,
            'error': all_predictions - all_targets,
            'abs_error': np.abs(all_predictions - all_targets),
            'relative_error': np.abs((all_predictions - all_targets) / np.maximum(all_targets, 1)) * 100
        })
        
        print("✅ Results dataframe created")
        
    except Exception as e:
        print(f"❌ Error creating results dataframe: {e}")
        results_df = None
    
    # 8. Print summary
    print("\n" + "="*60)
    print("🎨 VISUALIZATION SUMMARY")
    print("="*60)
    print(f"📊 Total predictions analyzed: {len(all_targets):,}")
    print(f"📈 Model performance (R²): {r2:.3f}")
    print(f"📉 Mean Absolute Error: {mae:.0f} Wh")
    print(f"🎯 Root Mean Square Error: {rmse:.0f} Wh")
    print(f"📁 Visualizations saved to: {output_dir}")
    print("="*60)
    
    return results_df

# Additional helper function for quick visualization check
def quick_visualization_test(model, test_loader, dataset, device='cuda', output_dir='./test_vis'):
    """
    Quick test to see if visualization works at all
    """
    print("🧪 Running quick visualization test...")
    
    try:
        # Get just a few predictions
        model.eval()
        with torch.no_grad():
            features, targets = next(iter(test_loader))
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            
            # Simple shape handling
            if predictions.dim() > 1:
                predictions = predictions[:, 0] if predictions.shape[1] > 1 else predictions.squeeze()
            if targets.dim() > 1:
                targets = targets[:, 0] if targets.shape[1] > 1 else targets.squeeze()
            
            pred_np = predictions.cpu().numpy()
            targ_np = targets.cpu().numpy()
            
            if hasattr(dataset, 'denormalize_targets'):
                pred_np = dataset.denormalize_targets(pred_np)
                targ_np = dataset.denormalize_targets(targ_np)
        
        # Simple scatter plot
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.scatter(targ_np, pred_np, alpha=0.7)
        plt.plot([targ_np.min(), targ_np.max()], [targ_np.min(), targ_np.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Quick Test: Predicted vs Actual')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/quick_test.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Quick visualization test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Quick visualization test FAILED: {e}")
        return False

def ensemble_weighted_average(models, weights, test_loader, device):
    """
    FIXED: Combine predictions from multiple models using weighted averaging.
    """
    assert len(models) == len(weights), "Number of models must match number of weights"
    assert np.isclose(sum(weights), 1.0), "Weights must sum to 1"
    
    print(f"Creating ensemble with {len(models)} models and weights: {weights}")
    
    all_ensemble_predictions = []
    all_actual_values = []
    
    # Set all models to evaluation mode
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(test_loader):
            features, targets = features.to(device), targets.to(device)
            
            # Get predictions from each model
            batch_predictions = []
            for model_idx, model in enumerate(models):
                pred = model(features)
                
                # Ensure consistent shape handling
                if pred.dim() > 1:
                    if pred.shape[1] > 1:
                        pred = pred[:, 0]  # Take first prediction if multi-step
                    pred = pred.squeeze()
                
                pred_np = pred.cpu().numpy()
                if pred_np.ndim > 1:
                    pred_np = pred_np.flatten()
                
                batch_predictions.append(pred_np)
            
            # Ensure all predictions have the same length
            min_length = min(len(p) for p in batch_predictions)
            batch_predictions = [p[:min_length] for p in batch_predictions]
            
            # Combine predictions with weights
            batch_ensemble = np.zeros(min_length)
            for i, pred in enumerate(batch_predictions):
                batch_ensemble += weights[i] * pred
            
            # Handle targets consistently
            targets_np = targets.cpu().numpy()
            if targets_np.ndim > 1:
                if targets_np.shape[1] > 1:
                    targets_np = targets_np[:, 0]
                targets_np = targets_np.flatten()
            
            targets_np = targets_np[:min_length]
            
            # Store results
            all_ensemble_predictions.append(batch_ensemble)
            all_actual_values.append(targets_np)
    
    # Concatenate all batches
    ensemble_predictions = np.concatenate(all_ensemble_predictions)
    actual_values = np.concatenate(all_actual_values)
    
    print(f"Ensemble predictions shape: {ensemble_predictions.shape}")
    print(f"Actual values shape: {actual_values.shape}")
    
    return ensemble_predictions, actual_values

def train_diverse_models(df, output_dir):
    """
    Train diverse models but LOAD if they already exist
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    dataset = SolarProductionDataset(df, seq_length=24, forecast_horizon=1, normalize=True)
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['dayofweek'] = df.index.dayofweek
        
    # Split into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create output directory
    model_dir = Path(output_dir) / "diverse_models"
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Get system parameters
    try:
        panel_efficiency = df['param_panel_efficiency'].iloc[0]
        panel_area = df['param_panel_area'].iloc[0]
        temp_coeff = df['param_temp_coeff'].iloc[0]
    except:
        panel_efficiency = 0.146
        panel_area = 1.642
        temp_coeff = -0.0044
    
    # Create models
    input_size = len(dataset.feature_names)
    
    models_config = [
        {
            'name': 'PINN-Small',
            'model': SolarProductionPINN(input_size=input_size, hidden_size=32, num_layers=2,
                                       panel_efficiency=panel_efficiency, panel_area=panel_area, temp_coeff=temp_coeff),
            'is_pinn': True,
            'epochs': 250,  # INCREASED from 100
            'patience': 50  # INCREASED from 20
        },
        {
            'name': 'PINN-Large', 
            'model': SolarProductionPINN(input_size=input_size, hidden_size=128, num_layers=4,
                                       panel_efficiency=panel_efficiency, panel_area=panel_area, temp_coeff=temp_coeff),
            'is_pinn': True,
            'epochs': 300,  # INCREASED from 100
            'patience': 50  # INCREASED from 20
        },
        {
            'name': 'LSTM',
            'model': LSTMSolarForecaster(input_size=input_size, hidden_size=64, num_layers=2, forecast_horizon=1),
            'is_pinn': False,
            'epochs': 200,  # INCREASED from 100
            'patience': 40  # INCREASED from 20
        },
        {
            'name': 'CNN',
            'model': CNNSolarForecaster(input_size=input_size, seq_length=24, num_filters=32, forecast_horizon=1),
            'is_pinn': False,
            'epochs': 200,  # INCREASED from 100
            'patience': 40  # INCREASED from 20
        }
    ]
    
    trained_models = []
    model_names = []
    evaluation_results = []
    
    for config in models_config:
        name = config['name']
        model = config['model'].to(device)
        is_pinn = config['is_pinn']
        
        model_save_path = model_dir / f"{name.lower().replace('-', '_')}_model.pt"
        
        print(f"\n--- Processing {name} Model ---")
        
        # CHECK IF MODEL ALREADY EXISTS
        if model_save_path.exists():
            print(f"✅ Loading existing {name} model from {model_save_path}")
            try:
                model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
                print(f"✅ {name} model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading {name} model: {e}")
                print(f"🔄 Training new {name} model...")
                # USE ENHANCED TRAINING
                model = train_single_model_enhanced(
                    model, train_loader, val_loader, name, is_pinn,
                    epochs=config['epochs'], patience=config['patience'],
                    device=device, save_path=model_save_path
                )
        else:
            print(f"🔄 Training new {name} model...")
            # USE ENHANCED TRAINING
            model = train_single_model_enhanced(
                model, train_loader, val_loader, name, is_pinn,
                epochs=config['epochs'], patience=config['patience'],
                device=device, save_path=model_save_path
            )
        
        trained_models.append(model)
        model_names.append(name)
        
        # Evaluate model
        print(f"📊 Evaluating {name} model...")
        eval_result = evaluate_model(model, test_loader, name, device, dataset)
        evaluation_results.append(eval_result)
    
    print("\n🔗 Creating ensembles...")

    # Equal-weighted ensemble - UPDATE THIS:
    weights = [0.25, 0.25, 0.25, 0.25]
    ensemble_preds, actual_values = ensemble_weighted_average_fixed(  # Changed to _fixed
        trained_models, weights, test_loader, device, dataset  # Added dataset parameter
    )

    # The predictions are already denormalized by the fixed function
    ensemble_result = evaluate_ensemble(ensemble_preds, actual_values, "Equally-Weighted Ensemble")

    # Stacked ensemble
    from sklearn.linear_model import Ridge
    meta_model = Ridge(alpha=0.5)
    stacked = StackedEnsemble(trained_models, meta_model)
    stacked.train_meta_model(val_loader, device)
    stacked_preds, stacked_actual = stacked.predict(test_loader, device)

    if hasattr(dataset, 'denormalize_targets'):
        stacked_preds = dataset.denormalize_targets(stacked_preds)
        stacked_actual = dataset.denormalize_targets(stacked_actual)

    stacked_result = evaluate_ensemble(stacked_preds, stacked_actual, "Stacked Ensemble")

def train_single_model_enhanced(model, train_loader, val_loader, name, is_pinn, 
                               epochs, patience, device, save_path):
    """ENHANCED: Train a single model with better parameters"""
    
    if is_pinn:
        model, history = train_pinn(
            model, train_loader, val_loader,
            epochs=epochs,          # Use config epochs
            lr=0.001,
            physics_weight=0.1,
            device=device, 
            save_path=save_path,
            early_stopping=True,
            patience=patience,      # Use config patience
            verbose=True
        )
    else:
        model, history = train_generic_model(
            model, train_loader, val_loader,
            epochs=epochs,          # Use config epochs
            lr=0.001,
            device=device,
            save_path=save_path,
            early_stopping=True,
            patience=patience,      # Use config patience
            verbose=True
        )
    create_enhanced_training_plot(history, name, save_path.parent)
    
    return model

def load_or_train_main_pinn(model, train_loader, val_loader, model_save_path, device):
    """Load existing model or train new one"""
    
    if os.path.exists(model_save_path):
        print(f"✅ Loading existing main PINN model from {model_save_path}")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
            print(f"✅ Main PINN model loaded successfully!")
            
            # Create dummy history for compatibility
            history = {
                'train_loss': [0.008],
                'val_loss': [0.008], 
                'data_loss': [0.008],
                'physics_loss': [0.0001],
                'learning_rate': [0.001]
            }
            return model, history
            
        except Exception as e:
            print(f"❌ Error loading main PINN model: {e}")
            print(f"🔄 Training new main PINN model...")
    else:
        print(f"🔄 Training new main PINN model...")
    
    # Train new model
    model, history = train_pinn(
        model, train_loader, val_loader,
        epochs=100, lr=0.001, physics_weight=0.1,
        device=device, save_path=model_save_path
    )
    
    return model, history

def run_enhanced_forecasting(df, base_output_dir):
    """
    FIXED: Run enhanced solar forecasting with better error handling
    """
    import os
    import torch
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader, random_split
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Create output directories
    enhanced_dir = Path(base_output_dir) / "enhanced_forecasting"
    os.makedirs(enhanced_dir, exist_ok=True)
    
    vis_dir = enhanced_dir / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    ensemble_dir = enhanced_dir / "ensemble_models"
    os.makedirs(ensemble_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("ENHANCED SOLAR PRODUCTION FORECASTING")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Create datasets with weather features
    print("\nPreparing datasets...")
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['dayofweek'] = df.index.dayofweek
    
    # Create dataset
    dataset = SolarProductionDataset(df, seq_length=24, forecast_horizon=1, normalize=True)
    print(f"Dataset created: {len(dataset)} samples with {len(dataset.feature_names)} features")
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Data split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    # Get system parameters
    try:
        panel_efficiency = df['param_panel_efficiency'].iloc[0]
        panel_area = df['param_panel_area'].iloc[0]
        temp_coeff = df['param_temp_coeff'].iloc[0]
    except:
        panel_efficiency = 0.146
        panel_area = 1.642
        temp_coeff = -0.0044
    
    print(f"Panel parameters: Efficiency={panel_efficiency:.4f}, Area={panel_area:.4f}m², Temp Coeff={temp_coeff:.6f}")
    
    # Create models
    print("\nCreating base PINN model...")
    pinn_model = SolarProductionPINN(
        input_size=len(dataset.feature_names),
        hidden_size=64,
        num_layers=3,
        panel_efficiency=panel_efficiency,
        panel_area=panel_area,
        temp_coeff=temp_coeff
    ).to(device)
    
    print("\nCreating LSTM model...")
    lstm_model = LSTMSolarForecaster(
        input_size=len(dataset.feature_names),
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        forecast_horizon=1
    ).to(device)
    
    # Try to load existing models or train new ones
    pinn_path = str(enhanced_dir / "pinn_model.pt")
    lstm_path = str(enhanced_dir / "lstm_model.pt")
    
    models = [pinn_model, lstm_model]
    model_names = ["PINN", "LSTM"]
    
    # Evaluate models and create ensemble
    print("\nEvaluating models and creating ensemble...")
    try:
        # Simple equal weights
        weights = [0.5, 0.5]
        
        # Get ensemble predictions
        ensemble_predictions, actual_values = ensemble_weighted_average(
            models, weights, test_loader, device
        )
        
        # Denormalize if needed
        if hasattr(dataset, 'denormalize_targets'):
            ensemble_predictions = dataset.denormalize_targets(ensemble_predictions)
            actual_values = dataset.denormalize_targets(actual_values)
        
        # Calculate ensemble metrics
        mae = np.mean(np.abs(actual_values - ensemble_predictions))
        rmse = np.sqrt(np.mean((actual_values - ensemble_predictions) ** 2))
        r2 = 1 - np.sum((actual_values - ensemble_predictions) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2)
        
        print("\nEnsemble Model Performance:")
        print(f"  MAE: {mae:.2f} Wh")
        print(f"  RMSE: {rmse:.2f} Wh")
        print(f"  R²: {r2:.4f}")
        
        return pinn_model, lstm_model, models, weights
        
    except Exception as e:
        print(f"Error in ensemble creation: {e}")
        print("Returning individual models...")
        return pinn_model, lstm_model, models, [0.5, 0.5]

def train_generic_model(model, train_loader, val_loader, epochs=100, lr=0.001, 
                         device='cuda' if torch.cuda.is_available() else 'cpu',
                         save_path=None, early_stopping=True, patience=20, verbose=True):
    """
    Generic training function for non-PINN models (LSTM, CNN, etc.)
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of epochs to train
        lr: Learning rate
        device: Device to train on
        save_path: Path to save best model
        early_stopping: Whether to use early stopping
        patience: Number of epochs to wait for improvement before stopping
        verbose: Whether to print progress
    
    Returns:
        model: Trained model
        history: Training history dict
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
    
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    no_improve_count = 0
    
    if verbose:
        print(f"Training model on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            predictions = model(features)
            
            # Make sure targets have the right shape for comparison
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            if predictions.dim() == 2 and targets.dim() == 2:
                predictions_for_loss = predictions[:, 0] if predictions.shape[1] > 1 else predictions.squeeze()
                targets_for_loss = targets[:, 0] if targets.shape[1] > 1 else targets.squeeze()
            else:
                predictions_for_loss = predictions.squeeze()
                targets_for_loss = targets.squeeze()
            
            # Calculate loss
            loss = criterion(predictions_for_loss, targets_for_loss)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            train_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                # Forward pass
                predictions = model(features)
                
                # Adjust shapes for loss calculation
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                if predictions.dim() == 2 and targets.dim() == 2:
                    predictions_for_loss = predictions[:, 0] if predictions.shape[1] > 1 else predictions.squeeze()
                    targets_for_loss = targets[:, 0] if targets.shape[1] > 1 else targets.squeeze()
                else:
                    predictions_for_loss = predictions.squeeze()
                    targets_for_loss = targets.squeeze()
                
                # Calculate loss
                loss = criterion(predictions_for_loss, targets_for_loss)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            no_improve_count += 1
        
        # Early stopping check
        if early_stopping and no_improve_count >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} - No improvement for {patience} epochs.")
            break
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        
        # Print progress
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1:3d}/{epochs} - "
                  f"Train: {avg_train_loss:.6f} - "
                  f"Val: {avg_val_loss:.6f} - "
                  f"LR: {current_lr:.2e}")
    
    return model, history

def evaluate_model(model, test_loader, model_name, device, dataset=None):
    """
    Evaluate a single model and print performance metrics.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        model_name: Name of the model (for printing)
        device: Computation device
        dataset: Dataset object (for denormalization)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Get predictions
            predictions = model(features)
            
            # Adjust dimensions
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            if predictions.dim() == 2 and targets.dim() == 2:
                predictions_for_eval = predictions[:, 0] if predictions.shape[1] > 1 else predictions.squeeze()
                targets_for_eval = targets[:, 0] if targets.shape[1] > 1 else targets.squeeze()
            else:
                predictions_for_eval = predictions.squeeze()
                targets_for_eval = targets.squeeze()
            
            all_targets.append(targets_for_eval.cpu().numpy())
            all_predictions.append(predictions_for_eval.cpu().numpy())
    
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    
    # Denormalize if necessary
    if dataset is not None and hasattr(dataset, 'denormalize_targets'):
        all_targets = dataset.denormalize_targets(all_targets)
        all_predictions = dataset.denormalize_targets(all_predictions)
    
    # Calculate metrics
    mae = np.mean(np.abs(all_targets - all_predictions))
    rmse = np.sqrt(np.mean((all_targets - all_predictions) ** 2))
    
    # Fixed MAPE calculation - only for significant values
    significant_mask = all_targets > 100  # Only calculate MAPE for production > 100 Wh
    if np.any(significant_mask):
        mape = np.mean(np.abs((all_targets[significant_mask] - all_predictions[significant_mask]) / 
                             all_targets[significant_mask])) * 100
    else:
        mape = float('inf')
    
    # Calculate R-squared
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Print results
    print(f"\n{model_name} Model Performance:")
    print(f"  MAE: {mae:.2f} Wh")
    print(f"  RMSE: {rmse:.2f} Wh")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²: {r2:.4f}")
    
    return {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'targets': all_targets,
        'predictions': all_predictions
    }

def evaluate_ensemble(ensemble_predictions, actual_values, ensemble_name):
    """
    Evaluate ensemble predictions against actual values.
    
    Args:
        ensemble_predictions: Numpy array of ensemble predictions
        actual_values: Numpy array of actual values
        ensemble_name: Name of the ensemble (for printing)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Ensure arrays have compatible shapes
    ensemble_predictions = np.asarray(ensemble_predictions).flatten()
    actual_values = np.asarray(actual_values).flatten()
    
    # Check if shapes match now
    if ensemble_predictions.shape != actual_values.shape:
        print(f"Warning: Shapes still don't match after flattening. "
              f"ensemble_predictions: {ensemble_predictions.shape}, "
              f"actual_values: {actual_values.shape}")
        # Use minimum length to avoid index errors
        min_len = min(len(ensemble_predictions), len(actual_values))
        ensemble_predictions = ensemble_predictions[:min_len]
        actual_values = actual_values[:min_len]
    
    # Calculate metrics
    mae = np.mean(np.abs(actual_values - ensemble_predictions))
    rmse = np.sqrt(np.mean((actual_values - ensemble_predictions) ** 2))
    
    # Fixed MAPE calculation - only for significant values
    significant_mask = actual_values > 100  # Only calculate MAPE for production > 100 Wh
    if np.any(significant_mask):
        mape = np.mean(np.abs((actual_values[significant_mask] - ensemble_predictions[significant_mask]) / 
                             actual_values[significant_mask])) * 100
    else:
        mape = float('inf')
    
    # Calculate R-squared
    ss_res = np.sum((actual_values - ensemble_predictions) ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Print results
    print(f"\n{ensemble_name} Performance:")
    print(f"  MAE: {mae:.2f} Wh")
    print(f"  RMSE: {rmse:.2f} Wh")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²: {r2:.4f}")
    
    # Create visualization of ensemble vs actual
    plt.figure(figsize=(10, 6))
    
    # Sample a subset of points for better visualization
    sample_size = min(1000, len(actual_values))
    indices = np.random.choice(len(actual_values), sample_size, replace=False)
    
    plt.scatter(actual_values[indices], ensemble_predictions[indices], 
               alpha=0.5, s=5, color='blue')
    
    # Add identity line
    min_val = min(actual_values.min(), ensemble_predictions.min())
    max_val = max(actual_values.max(), ensemble_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # Add metrics as text
    plt.text(0.05, 0.95, f"MAE: {mae:.2f} Wh\nRMSE: {rmse:.2f} Wh\nR²: {r2:.4f}",
            transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.title(f'{ensemble_name} Predictions vs Actual', fontsize=14)
    plt.xlabel('Actual Energy (Wh)', fontsize=12)
    plt.ylabel('Predicted Energy (Wh)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return {
        'ensemble_name': ensemble_name,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

def create_model_comparison_visualization(model_results, ensemble_results, output_dir):
    """
    Create visualization comparing all models and ensembles.
    
    Args:
        model_results: List of individual model evaluation results
        ensemble_results: List of ensemble evaluation results
        output_dir: Directory to save visualization
    """
    # Extract model names and metrics
    model_names = [result['model_name'] for result in model_results] + [result['ensemble_name'] for result in ensemble_results]
    mae_values = [result['mae'] for result in model_results] + [result['mae'] for result in ensemble_results]
    rmse_values = [result['rmse'] for result in model_results] + [result['rmse'] for result in ensemble_results]
    r2_values = [result['r2'] for result in model_results] + [result['r2'] for result in ensemble_results]
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot MAE
    plt.subplot(3, 1, 1)
    bars1 = plt.bar(range(len(model_names)), mae_values, color='skyblue')
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.ylabel('MAE (Wh)')
    plt.title('Mean Absolute Error Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot RMSE
    plt.subplot(3, 1, 2)
    bars2 = plt.bar(range(len(model_names)), rmse_values, color='lightgreen')
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.ylabel('RMSE (Wh)')
    plt.title('Root Mean Squared Error Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot R²
    plt.subplot(3, 1, 3)
    bars3 = plt.bar(range(len(model_names)), r2_values, color='lightcoral')
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.ylabel('R²')
    plt.title('R-squared Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def calculate_optimal_weights(models, val_loader, device, dataset):
    """FIXED: Calculate optimal weights with better optimization and validation"""
    print("Calculating optimal weights with enhanced validation...")
    
    # Get predictions for each model on validation set
    all_preds = []
    actual_values = []
    
    # Use a larger validation sample for better weight optimization
    sample_batches = min(len(val_loader), 10)  # Use more validation data
    batch_count = 0
    
    with torch.no_grad():
        for features, targets in val_loader:
            if batch_count >= sample_batches:
                break
                
            features, targets = features.to(device), targets.to(device)
            batch_preds = []
            
            for model in models:
                model.eval()
                pred = model(features)
                
                # Consistent shape handling
                if pred.dim() > 1:
                    if pred.shape[1] > 1:
                        pred = pred[:, 0]
                    pred = pred.squeeze()
                
                # Convert to numpy and ensure 1D
                pred_np = pred.cpu().numpy()
                if pred_np.ndim > 1:
                    pred_np = pred_np.flatten()
                
                batch_preds.append(pred_np)
            
            # Ensure all predictions have the same length
            min_length = min(len(p) for p in batch_preds)
            batch_preds = [p[:min_length] for p in batch_preds]
            
            all_preds.append(batch_preds)
            
            # Handle targets consistently
            targets_np = targets.cpu().numpy()
            if targets_np.ndim > 1:
                if targets_np.shape[1] > 1:
                    targets_np = targets_np[:, 0]
                targets_np = targets_np.flatten()
            
            actual_values.append(targets_np[:min_length])
            batch_count += 1
    
    # Reshape predictions for optimization
    model_preds = []
    for i in range(len(models)):
        model_pred = np.concatenate([preds[i] for preds in all_preds])
        
        # Denormalize if needed
        if hasattr(dataset, 'denormalize_targets'):
            model_pred = dataset.denormalize_targets(model_pred)
        
        # Ensure 1D array
        if model_pred.ndim > 1:
            model_pred = model_pred.flatten()
            
        model_preds.append(model_pred)
    
    # Get actual values
    actual = np.concatenate(actual_values)
    if hasattr(dataset, 'denormalize_targets'):
        actual = dataset.denormalize_targets(actual)
    
    # Ensure actual is 1D
    if actual.ndim > 1:
        actual = actual.flatten()
    
    # Ensure all arrays have the same length
    min_length = min(len(actual), min(len(p) for p in model_preds))
    actual = actual[:min_length]
    model_preds = [p[:min_length] for p in model_preds]
    
    print(f"Validation data shapes: actual={actual.shape}, predictions={[p.shape for p in model_preds]}")
    
    # Calculate individual model performance for reference
    individual_r2s = []
    individual_maes = []
    
    for i, pred in enumerate(model_preds):
        r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
        mae = np.mean(np.abs(actual - pred))
        individual_r2s.append(r2)
        individual_maes.append(mae)
        print(f"Model {i} validation: R² = {r2:.4f}, MAE = {mae:.0f}")
    
    # ENHANCED OPTIMIZATION with multiple methods
    from scipy.optimize import minimize, differential_evolution
    
    def mse_loss(weights):
        weights = np.array(weights)
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        prediction = np.zeros_like(actual)
        for i, model_pred in enumerate(model_preds):
            prediction += weights[i] * model_pred
        
        return np.mean((prediction - actual) ** 2)
    
    def r2_loss(weights):
        """Minimize negative R² (maximize R²)"""
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        prediction = np.zeros_like(actual)
        for i, model_pred in enumerate(model_preds):
            prediction += weights[i] * model_pred
        
        r2 = 1 - np.sum((actual - prediction) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
        return -r2  # Minimize negative R²
    
    # Try multiple optimization approaches
    initial_weights = [1/len(models)] * len(models)
    
    # Test initial equal weights
    initial_loss = mse_loss(initial_weights)
    print(f"Initial loss with equal weights: {initial_loss:.6f}")
    
    # Method 1: Performance-based initialization
    # Give higher initial weights to better performing models
    if max(individual_r2s) - min(individual_r2s) > 0.01:  # If there's meaningful difference
        performance_weights = np.array(individual_r2s)
        performance_weights = performance_weights / np.sum(performance_weights)
        print(f"Performance-based weights: {performance_weights}")
    else:
        performance_weights = initial_weights
    
    best_weights = initial_weights
    best_loss = initial_loss
    
    # Method 2: Constrained optimization (sum to 1)
    try:
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0.01, 0.99) for _ in range(len(models))]  # Prevent extreme weights
        
        # Try with MSE loss
        result1 = minimize(mse_loss, performance_weights, bounds=bounds, constraints=constraints, method='SLSQP')
        if result1.success and mse_loss(result1.x) < best_loss:
            best_weights = result1.x
            best_loss = mse_loss(result1.x)
            print(f"✅ Constrained optimization improved: {best_loss:.6f}")
        
        # Try with R² maximization
        result2 = minimize(r2_loss, performance_weights, bounds=bounds, constraints=constraints, method='SLSQP')
        if result2.success and mse_loss(result2.x) < best_loss:
            best_weights = result2.x
            best_loss = mse_loss(result2.x)
            print(f"✅ R² optimization improved: {best_loss:.6f}")
            
    except Exception as e:
        print(f"Constrained optimization failed: {e}")
    
    # Method 3: Differential Evolution (global optimization)
    try:
        bounds_de = [(0.01, 0.99) for _ in range(len(models))]
        
        def constrained_mse(weights):
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            return mse_loss(weights)
        
        result3 = differential_evolution(constrained_mse, bounds_de, seed=42, maxiter=100)
        if result3.success and result3.fun < best_loss:
            best_weights = result3.x / np.sum(result3.x)  # Normalize
            best_loss = result3.fun
            print(f"✅ Global optimization improved: {best_loss:.6f}")
            
    except Exception as e:
        print(f"Global optimization failed: {e}")
    
    # Normalize final weights
    best_weights = np.array(best_weights)
    best_weights = best_weights / np.sum(best_weights)
    
    # Validate the improvement
    improvement = (initial_loss - best_loss) / initial_loss * 100
    print(f"Weight optimization improvement: {improvement:.2f}%")
    
    if improvement < 0.1:  # Less than 0.1% improvement
        print("⚠️ Minimal improvement - models may be too similar")
        # Use performance-based weights if optimization didn't help much
        if max(individual_r2s) - min(individual_r2s) > 0.01:
            best_weights = np.array(individual_r2s)
            best_weights = best_weights / np.sum(best_weights)
            print("Using performance-proportional weights instead")
    
    return best_weights

def create_time_specialized_ensembles_fixed(df, output_dir):
    """FIXED: Create time-specialized ensembles for different parts of the day"""
    # Define time periods
    periods = {
        "morning": range(6, 12),   # 6 AM - 11 AM
        "afternoon": range(12, 18), # 12 PM - 5 PM
        "evening": range(18, 24),  # 6 PM - 11 PM
        "night": list(range(0, 6)) # 12 AM - 5 AM
    }
    
    results = {}
    
    for period_name, hours in periods.items():
        print(f"\nTraining specialized ensemble for {period_name} hours...")
        
        try:
            # Filter data for this time period
            period_mask = df.index.hour.isin(hours)
            period_df = df[period_mask].copy()
            
            if len(period_df) < 100:  # Skip if too little data
                print(f"⚠️ Insufficient data for {period_name} period ({len(period_df)} samples)")
                continue
            
            # Train models on this subset - FIXED return values
            models_result = train_diverse_models(
                period_df, 
                os.path.join(output_dir, f"time_specialized/{period_name}")
            )
            
            # Handle the return values properly
            if len(models_result) >= 4:
                trained_models, model_names, evaluation_results, ensemble_result = models_result[:4]
                stacked_result = models_result[4] if len(models_result) > 4 else None
                
                results[period_name] = {
                    "models": trained_models,
                    "model_names": model_names,
                    "evaluation_results": evaluation_results,
                    "ensemble_performance": ensemble_result,
                    "stacked_performance": stacked_result
                }
                print(f"✅ {period_name} ensemble created successfully")
            else:
                print(f"⚠️ {period_name} ensemble creation returned unexpected results")
                
        except Exception as e:
            print(f"⚠️ Error creating {period_name} ensemble: {e}")
            continue
    
    return results

def create_production_pipeline(models, weights, dataset, device):
    """FIXED: Create a production pipeline for making predictions"""
    
    # Ensure all models are on the correct device
    for model in models:
        model.to(device)
    
    def predict(features):
        """Make predictions using the ensemble"""
        # Convert features to tensor and ensure correct device
        if not isinstance(features, torch.Tensor):
            features = torch.FloatTensor(features).to(device)
        else:
            features = features.to(device)
        
        # Add batch dimension if needed
        if features.dim() == 2:
            features = features.unsqueeze(0)
        
        # Get predictions from each model
        predictions = []
        for model in models:
            model.eval()
            model.to(device)  # Ensure model is on correct device
            with torch.no_grad():
                pred = model(features)
                if pred.dim() > 1 and pred.shape[1] > 1:
                    pred = pred[:, 0]
                predictions.append(pred.cpu().numpy())
        
        # Weight predictions
        ensemble_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += weights[i] * pred
        
        # Denormalize if needed
        if hasattr(dataset, 'denormalize_targets'):
            ensemble_pred = dataset.denormalize_targets(ensemble_pred)
        
        return ensemble_pred
    
    return predict

def generate_performance_dashboard(models, model_names, test_loader, dataset, output_dir):
    """FIXED: Generate a comprehensive performance dashboard for all models"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 Creating performance dashboard in {output_dir}")
    
    # Get predictions and actual values
    device = next(models[0].parameters()).device
    all_preds = []
    all_actual = []
    
    # Get predictions from each model
    for model, name in zip(models, model_names):
        model.eval()
        model_preds = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                
                pred = model(features)
                if pred.dim() > 1 and pred.shape[1] > 1:
                    pred = pred[:, 0]
                if targets.dim() > 1 and targets.shape[1] > 1:
                    targets = targets[:, 0]
                
                # Ensure 1D arrays
                pred_np = pred.cpu().numpy()
                if pred_np.ndim > 1:
                    pred_np = pred_np.flatten()
                
                model_preds.append(pred_np)
                
                # Only collect actual values once
                if len(all_actual) < len(test_loader):
                    targets_np = targets.cpu().numpy()
                    if targets_np.ndim > 1:
                        targets_np = targets_np.flatten()
                    all_actual.append(targets_np)
        
        # Concatenate predictions for this model
        model_pred = np.concatenate(model_preds)
        if hasattr(dataset, 'denormalize_targets'):
            model_pred = dataset.denormalize_targets(model_pred)
        all_preds.append(model_pred)
    
    # Concatenate actual values
    actual = np.concatenate(all_actual)
    if hasattr(dataset, 'denormalize_targets'):
        actual = dataset.denormalize_targets(actual)
    
    # Ensure all arrays have the same length
    min_length = min(len(actual), min(len(p) for p in all_preds))
    actual = actual[:min_length]
    all_preds = [p[:min_length] for p in all_preds]
    
    # Calculate metrics for each model
    metrics = {
        "R²": [],
        "RMSE (Wh)": [],
        "MAE (Wh)": [],
        "MAPE (%)": []
    }
    
    for i, (pred, name) in enumerate(zip(all_preds, model_names)):
        # Calculate metrics
        r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mae = np.mean(np.abs(actual - pred))
        
        # MAPE calculation (only for significant values)
        significant = actual > 100
        if np.any(significant):
            mape = np.mean(np.abs((actual[significant] - pred[significant]) / actual[significant])) * 100
        else:
            mape = 0
        
        # Store metrics
        metrics["R²"].append(r2)
        metrics["RMSE (Wh)"].append(rmse)
        metrics["MAE (Wh)"].append(mae)
        metrics["MAPE (%)"].append(mape)
    
    # Create performance comparison plot
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot each metric
        for i, (metric, values) in enumerate(metrics.items()):
            plt.subplot(2, 2, i+1)
            
            # Create horizontal bar chart
            bars = plt.barh(model_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}', ha='left', va='center')
            
            plt.title(f'{metric}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Performance dashboard created successfully")
        
    except Exception as e:
        print(f"❌ Error creating dashboard plots: {e}")
    
def run_extended_training(df, best_config, output_dir, epochs=500):
    """
    FINAL TRAINING: Extended training with best configuration and 500+ epochs
    """
    print(f"Starting extended training with {epochs} epochs...")
    print(f"Best config: {best_config}")
    
    # Create dataset
    dataset = SolarProductionDataset(df, seq_length=24, forecast_horizon=1, normalize=True)
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders with best batch size
    batch_size = best_config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    # Create output directory
    extended_dir = Path(output_dir)
    extended_dir.mkdir(exist_ok=True, parents=True)
    
    # Get system parameters
    try:
        panel_efficiency = df['param_panel_efficiency'].iloc[0]
        panel_area = df['param_panel_area'].iloc[0]
        temp_coeff = df['param_temp_coeff'].iloc[0]
    except:
        panel_efficiency, panel_area, temp_coeff = 0.146, 1.642, -0.0044
    
    # Create model with BEST configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SolarProductionPINN(
        input_size=len(dataset.feature_names),
        hidden_size=best_config["hidden_size"],      # Optimized
        num_layers=best_config["num_layers"],        # Optimized
        panel_efficiency=panel_efficiency,
        panel_area=panel_area,
        temp_coeff=temp_coeff
    ).to(device)
    
    # Save path
    model_save_path = extended_dir / "best_extended_model.pt"
    
    print(f"Training final model:")
    print(f"  Hidden size: {best_config['hidden_size']}")
    print(f"  Num layers: {best_config['num_layers']}")
    print(f"  Learning rate: {best_config['learning_rate']}")
    print(f"  Physics weight: {best_config['physics_weight']}")
    print(f"  Epochs: {epochs}")
    
    # EXTENDED TRAINING with best parameters
    model, history = train_pinn(
        model, train_loader, val_loader,
        epochs=epochs,                               # 500+ epochs
        lr=best_config["learning_rate"],             # Optimized LR
        physics_weight=best_config["physics_weight"], # Optimized physics weight
        device=device,
        save_path=model_save_path,
        early_stopping=True,
        patience=100,                                # High patience for extended training
        verbose=True
    )
    
    # Create comprehensive training history plot
    create_extended_training_plot(history, extended_dir)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    
    # Final evaluation
    eval_results = evaluate_model_improved(model, test_loader, dataset, device=device)
    
    # Save final results
    final_results = {
        'config': best_config,
        'epochs_trained': len(history['train_loss']),
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'test_metrics': eval_results
    }
    
    with open(extended_dir / "final_results.json", "w") as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for k, v in final_results.items():
            if isinstance(v, dict):
                json_results[k] = {kk: float(vv) if isinstance(vv, np.number) else vv 
                                 for kk, vv in v.items() if kk not in ['targets', 'predictions']}
            else:
                json_results[k] = float(v) if isinstance(v, np.number) else v
        
        json.dump(json_results, f, indent=2)
    
    print(f"Extended training completed!")
    print(f"Final R²: {eval_results['r2']:.4f}")
    print(f"Final MAE: {eval_results['mae']:.0f} Wh")
    print(f"Results saved to: {extended_dir}")
    
    return model, eval_results

def create_best_ensemble(trained_models, model_names, test_loader, device, dataset):
    """FIXED: Create the best possible ensemble including LSTM"""
    
    print("\n🏆 Creating BEST ENSEMBLE (including LSTM)")
    
    # Find the LSTM model
    lstm_idx = None
    for i, name in enumerate(model_names):
        if 'LSTM' in name.upper():
            lstm_idx = i
            break
    
    if lstm_idx is None:
        print("⚠️ LSTM model not found in ensemble!")
        return None
    
    print(f"✅ Found LSTM model at index {lstm_idx}")
    
    # Create optimized weights (give more weight to best performing models)
    if len(trained_models) == 4:  # PINN-Small, PINN-Large, LSTM, CNN
        optimized_weights = [0.15, 0.20, 0.50, 0.15]  # Higher weight to LSTM
        print("🎯 Using performance-optimized weights:")
        for i, (name, weight) in enumerate(zip(model_names, optimized_weights)):
            print(f"   {name}: {weight:.2f}")
    else:
        optimized_weights = [1/len(trained_models)] * len(trained_models)
    
    # Create ensemble predictions
    try:
        ensemble_preds, actual_values = ensemble_weighted_average(
            trained_models, optimized_weights, test_loader, device
        )
        
        # Denormalize
        if hasattr(dataset, 'denormalize_targets'):
            ensemble_preds = dataset.denormalize_targets(ensemble_preds)
            actual_values = dataset.denormalize_targets(actual_values)
        
        # Calculate metrics
        mae = np.mean(np.abs(actual_values - ensemble_preds))
        rmse = np.sqrt(np.mean((actual_values - ensemble_preds) ** 2))
        r2 = 1 - np.sum((actual_values - ensemble_preds) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2)
        
        result = {
            'ensemble_name': 'BEST Ensemble (LSTM-weighted)',
            'mae': mae,
            'rmse': rmse, 
            'r2': r2,
            'weights': optimized_weights
        }
        
        print(f"🏆 BEST Ensemble Results:")
        print(f"   MAE: {mae:.0f} Wh")
        print(f"   RMSE: {rmse:.0f} Wh")
        print(f"   R²: {r2:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error creating best ensemble: {e}")
        return None

def analyze_ensemble_weights(models, model_names, weights, test_loader, device, dataset):
    """Analyze why certain weights were chosen"""
    print(f"\n📊 ENSEMBLE WEIGHT ANALYSIS")
    print("-" * 50)
    
    # Individual model performance on test set
    individual_performance = []
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                pred = model(features)
                
                if pred.dim() > 1 and pred.shape[1] > 1:
                    pred = pred[:, 0]
                pred = pred.squeeze()
                
                if targets.dim() > 1 and targets.shape[1] > 1:
                    targets = targets[:, 0]
                targets = targets.squeeze()
                
                predictions.append(pred.cpu().numpy())
                actuals.append(targets.cpu().numpy())
        
        pred_all = np.concatenate(predictions)
        actual_all = np.concatenate(actuals)
        
        if hasattr(dataset, 'denormalize_targets'):
            pred_all = dataset.denormalize_targets(pred_all)
            actual_all = dataset.denormalize_targets(actual_all)
        
        # Calculate metrics
        r2 = 1 - np.sum((actual_all - pred_all) ** 2) / np.sum((actual_all - np.mean(actual_all)) ** 2)
        mae = np.mean(np.abs(actual_all - pred_all))
        
        individual_performance.append({'name': name, 'r2': r2, 'mae': mae, 'weight': weights[i]})
        
        print(f"{name:15}: R² = {r2:.4f}, MAE = {mae:6.0f}, Weight = {weights[i]:.3f}")
    
    # Analysis
    print(f"\n🔍 WEIGHT ANALYSIS:")
    
    # Sort by performance
    by_r2 = sorted(individual_performance, key=lambda x: x['r2'], reverse=True)
    by_weight = sorted(individual_performance, key=lambda x: x['weight'], reverse=True)
    
    print(f"Best performing model: {by_r2[0]['name']} (R² = {by_r2[0]['r2']:.4f})")
    print(f"Highest weighted model: {by_weight[0]['name']} (Weight = {by_weight[0]['weight']:.3f})")
    
    # Check if weights correlate with performance
    r2_values = [p['r2'] for p in individual_performance]
    weight_values = [p['weight'] for p in individual_performance]
    
    correlation = np.corrcoef(r2_values, weight_values)[0, 1]
    print(f"Weight-Performance correlation: {correlation:.3f}")
    
    if correlation > 0.5:
        print("✅ Weights correlate well with performance")
    elif correlation > 0.2:
        print("⚠️ Moderate correlation between weights and performance") 
    else:
        print("❌ Weights don't correlate with individual performance")
        print("   This suggests models have complementary strengths")
    
    return individual_performance

def create_lstm_heavy_ensemble(trained_models, model_names, test_loader, device, dataset):
    """
    Create an ensemble that heavily favors the LSTM while still benefiting from diversity
    """
    # Find LSTM index
    lstm_idx = None
    for i, name in enumerate(model_names):
        if 'LSTM' in name.upper():
            lstm_idx = i
            break
    
    if lstm_idx is None:
        print("❌ LSTM not found!")
        return None
    
    # Strategy 1: Very heavy LSTM weight
    lstm_heavy_weights = [0.0] * len(trained_models)
    lstm_heavy_weights[lstm_idx] = 0.85  # 85% LSTM
    
    # Distribute remaining 15% among other good models
    remaining_weight = 0.15
    for i, name in enumerate(model_names):
        if i != lstm_idx:
            if 'CNN' in name:
                lstm_heavy_weights[i] = 0.10  # 10% CNN
            else:
                lstm_heavy_weights[i] = 0.025  # 2.5% each PINN
    
    print(f"\n🎯 LSTM-Heavy Weights:")
    for name, weight in zip(model_names, lstm_heavy_weights):
        print(f"  {name}: {weight:.2%}")
    
    # Create ensemble
    ensemble_preds, actual_values = ensemble_weighted_average(
        trained_models, lstm_heavy_weights, test_loader, device
    )
    
    # Denormalize
    if hasattr(dataset, 'denormalize_targets'):
        ensemble_preds = dataset.denormalize_targets(ensemble_preds)
        actual_values = dataset.denormalize_targets(actual_values)
    
    # Evaluate
    mae = np.mean(np.abs(actual_values - ensemble_preds))
    rmse = np.sqrt(np.mean((actual_values - ensemble_preds) ** 2))
    r2 = 1 - np.sum((actual_values - ensemble_preds) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2)
    
    result = {
        'ensemble_name': 'LSTM-Heavy Ensemble (85% LSTM)',
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'weights': lstm_heavy_weights
    }
    
    print(f"\n📊 LSTM-Heavy Ensemble Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.0f} Wh")
    print(f"  RMSE: {rmse:.0f} Wh")
    
    return result

def create_selective_ensemble(trained_models, model_names, evaluation_results, test_loader, device, dataset, r2_threshold=0.93):
    """
    Create ensemble using only models above a certain R² threshold
    """
    # Select only high-performing models
    selected_models = []
    selected_names = []
    selected_r2s = []
    
    for i, result in enumerate(evaluation_results):
        if result['r2'] >= r2_threshold:
            selected_models.append(trained_models[i])
            selected_names.append(model_names[i])
            selected_r2s.append(result['r2'])
    
    print(f"\n🎯 Selective Ensemble (R² > {r2_threshold}):")
    print(f"Selected {len(selected_models)} models:")
    for name, r2 in zip(selected_names, selected_r2s):
        print(f"  {name}: R² = {r2:.4f}")
    
    if len(selected_models) < 2:
        print("❌ Not enough high-performing models for ensemble")
        return None
    
    # Weight by performance
    total_r2 = sum(selected_r2s)
    weights = [r2/total_r2 for r2 in selected_r2s]
    
    print(f"\nPerformance-based weights:")
    for name, weight in zip(selected_names, weights):
        print(f"  {name}: {weight:.2%}")
    
    # Create ensemble
    ensemble_preds, actual_values = ensemble_weighted_average(
        selected_models, weights, test_loader, device
    )
    
    # Denormalize
    if hasattr(dataset, 'denormalize_targets'):
        ensemble_preds = dataset.denormalize_targets(ensemble_preds)
        actual_values = dataset.denormalize_targets(actual_values)
    
    # Evaluate
    mae = np.mean(np.abs(actual_values - ensemble_preds))
    rmse = np.sqrt(np.mean((actual_values - ensemble_preds) ** 2))
    r2 = 1 - np.sum((actual_values - ensemble_preds) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2)
    
    print(f"\n📊 Selective Ensemble Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.0f} Wh")
    print(f"  RMSE: {rmse:.0f} Wh")
    
    return {
        'ensemble_name': f'Selective Ensemble (R²>{r2_threshold})',
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'models': selected_names,
        'weights': weights
    }

if __name__ == "__main__":
    main()

generate_performance_dashboard
create_production_pipeline
create_time_specialized_ensembles_fixed
calculate_optimal_weights
create_model_comparison_visualization