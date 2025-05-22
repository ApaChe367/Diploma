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
data_path = "C:/Users/kigeorgiou/Desktop/Forcasting/Results/forecast_data.csv"
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
cleaned_data_path = "C:/Users/kigeorgiou/Desktop/Forcasting/Results/cleaned_forecast_data.csv"
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

def main():
    import ssl
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    ssl._create_default_https_context = ssl._create_unverified_context
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    check_gpu_status()
    
    # Force GPU usage check
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  CUDA not available, using CPU")
    
    # Load the cleaned data (should already exist from exploration)
    data_path = "C:/Users/kigeorgiou/Desktop/Forcasting/Results/cleaned_forecast_data.csv"
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Loaded cleaned data: {df.shape}")
    else:
        # If cleaned data doesn't exist, use the original and clean it
        data_path = "C:/Users/kigeorgiou/Desktop/Forcasting/Results/forecast_data.csv"
        df = pd.read_csv(data_path)
        # Apply the same cleaning process as in exploration
        df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        df['WS_10m'] = 3.0  # Default wind speed
        df['mismatch'] = df['ac_power_output'] / 1000 - df['Load (kW)']
        df = df.ffill().bfill()
    
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['dayofweek'] = df.index.dayofweek
    
     # Add weather forecast integration
    # Define location (example coordinates)
    lat = 37.98983
    lon = 23.74328
    api_key = "7273588818d8b2bb8597ee797baf4935" #API key
    
    print("\nFetching weather forecast data...")
    try:
        forecast_json = fetch_weather_forecast(lat, lon, api_key)
        if forecast_json:
            df_forecast = process_forecast_data(forecast_json)
            # Combine with historical data
            df_enhanced = engineer_weather_features(df, df_forecast)
            print(f"Enhanced data with forecast features: {df_enhanced.shape}")
            
            # Use enhanced data for forecasting
            dataset = SolarProductionDatasetWithForecast(df_enhanced, seq_length=24, forecast_horizon=1)
        else:
            print("Using original data without forecast features")
            dataset = SolarProductionDataset(df, seq_length=24, forecast_horizon=1)
    except Exception as e:
        print(f"Error fetching forecast data: {e}")
        print("Using original data without forecast features")
        dataset = SolarProductionDataset(df, seq_length=24, forecast_horizon=1)
    
    
    # Extract system parameters
    panel_efficiency = df['param_panel_efficiency'].iloc[0]
    panel_area = df['param_panel_area'].iloc[0]
    temp_coeff = df['param_temp_coeff'].iloc[0]
    
    print(f"System parameters:")
    print(f"  Panel efficiency: {panel_efficiency:.4f}")
    print(f"  Panel area: {panel_area:.4f} m²")
    print(f"  Temperature coefficient: {temp_coeff:.6f}")
    
    # Create dataset with shorter sequences for initial testing
    try:
        dataset = SolarProductionDataset(df, seq_length=24, forecast_horizon=1, normalize=True)
        print(f"Dataset created successfully: {len(dataset)} samples")
        print(f"Features used: {dataset.feature_names}")
        print(f"Input size: {len(dataset.feature_names)}")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
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
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    # Prepare output directory BEFORE creating model
    output_dir = "C:/Users/kigeorgiou/Desktop/Forcasting/Results/PINN_results"   
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, 'solar_production_pinn_best.pt')
    
    # Create model and move to device
    model = SolarProductionPINN(
        input_size=len(dataset.feature_names),
        hidden_size=64,
        num_layers=3,
        panel_efficiency=panel_efficiency,
        panel_area=panel_area,
        temp_coeff=temp_coeff
    ).to(device)
    
    # Train model (single call, with device parameter)
    print(f"\nStarting PINN training on {device}...")
    trained_model, history = train_pinn(
        model, train_loader, val_loader,
        epochs=100,
        lr=0.001,
        physics_weight=0.1,
        device=device,  # Pass device explicitly
        save_path=model_save_path
    )
    
    # Load best model for evaluation
    if os.path.exists(model_save_path):
        trained_model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded best model from {model_save_path}")
    
    # Evaluate model with improved function
    print("\nEvaluating model...")
    eval_results = evaluate_model_improved(trained_model, test_loader, dataset, device=device)
    
    # Create visualizations
    visualize_results(history, eval_results, output_dir)
    
    # Analysis of prediction quality by time of day 
    targets = eval_results['targets']
    predictions = eval_results['predictions']

    # Assuming hourly data, analyze by hour of day
    if len(targets) >= 24:
        hourly_errors = []
        hours = np.arange(24)
        
        for hour in hours:
            hour_mask = np.arange(len(targets)) % 24 == hour
            if np.any(hour_mask):
                hour_targets = targets[hour_mask]
                hour_predictions = predictions[hour_mask]
                hour_mae = np.mean(np.abs(hour_targets - hour_predictions))
                hourly_errors.append(hour_mae)
            else:
                hourly_errors.append(0)
        
        # Plot hourly error pattern
        plt.figure(figsize=(12, 6))
        plt.plot(hours, hourly_errors, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Absolute Error (Wh)')
        plt.title('Prediction Error by Hour of Day')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        plt.savefig(f"{output_dir}/hourly_error_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nHOURLY ANALYSIS:")
        print(f"Best prediction hours: {hours[np.argsort(hourly_errors)[:3]]}")
        print(f"Worst prediction hours: {hours[np.argsort(hourly_errors)[-3:]]}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'solar_production_pinn_final.pt')
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    print("\nPINN training and evaluation completed successfully!")

    # Comprehensive visualizations for the basic model
    vis_output_dir = os.path.join(output_dir, "detailed_visualizations")
    os.makedirs(vis_output_dir, exist_ok=True)
    
    print("\nCreating comprehensive visualizations...")
    try:
        results_df = create_comprehensive_visualizations(
            trained_model, test_loader, dataset, df, 
            device=device, output_dir=vis_output_dir
        )
        print(f"Detailed visualizations saved to {vis_output_dir}")
    except Exception as e:
        print(f"Error creating comprehensive visualizations: {e}")

    # Run enhanced forecasting
    print("\nStarting enhanced forecasting with ensemble models...")
    enhanced_output_dir = "C:/Users/kigeorgiou/Desktop/Forcasting/Results/PINN_enhanced"
    
    try:
        pinn_model, lstm_model, ensemble_models, ensemble_weights = run_enhanced_forecasting(df, enhanced_output_dir)
        print("\nEnhanced forecasting with ensemble models completed successfully!")
    except Exception as e:
        print(f"Error during enhanced forecasting: {e}")
        print("Continuing with basic model results...")
    
    print("\nAll processing completed successfully!")
    
    print("\nTraining diverse models for ensemble comparison...")
    
    try:
        diverse_models_dir = "C:/Users/kigeorgiou/Desktop/Forcasting/Results/PINN_diverse"
        trained_models, model_results, ensemble_result, stacked_result = train_diverse_models(df, diverse_models_dir)
        print("Diverse model training completed!")
    except Exception as e:
        print(f"Error training diverse models: {e}")

    print("\nTraining diverse ensemble models...")
    diverse_models_dir = "C:/Users/kigeorgiou/Desktop/Forcasting/Results/PINN_diverse"
    trained_models, evaluation_results, ensemble_result, stacked_result = train_diverse_models(df, diverse_models_dir)
    
    print("\nCalculating optimal model weights...")
    optimal_weights = calculate_optimal_weights(trained_models, val_loader, device, dataset)
    print(f"Optimal weights for ensemble models:")
    for name, weight in zip(model_names, optimal_weights):
        print(f"  {name}: {weight:.4f}")
    
    # Create optimally weighted ensemble predictions
    print("\nCreating optimally weighted ensemble predictions...")
    optimal_ensemble_preds, actual_values = ensemble_weighted_average(
        trained_models, optimal_weights, test_loader, device
    )
    # Evaluate the optimally weighted ensemble
    optimal_result = evaluate_ensemble(optimal_ensemble_preds, actual_values, "Optimally Weighted Ensemble")
    
    print("\nTraining time-specialized ensemble models...")
    specialized_ensembles = create_time_specialized_ensembles(df, diverse_models_dir)
    
    print("\nCreating production prediction pipeline...")
    production_pipeline = create_production_pipeline(trained_models, optimal_weights, dataset, device)
    
    # Save the production pipeline (optional)
    pipeline_save_path = os.path.join(diverse_models_dir, "production_pipeline.pt")
    torch.save({
        "models": [model.state_dict() for model in trained_models],
        "weights": optimal_weights,
        "model_names": model_names
    }, pipeline_save_path)
    print(f"Production pipeline saved to {pipeline_save_path}")
    
    # Test the production pipeline with a sample
    sample_features, _ = next(iter(test_loader))
    sample_prediction = production_pipeline(sample_features[:1])
    print(f"Sample prediction from production pipeline: {sample_prediction[0]:.2f} Wh")
    
    print("\nGenerating performance dashboard...")
    dashboard_dir = os.path.join(diverse_models_dir, "dashboard")
    generate_performance_dashboard(trained_models, model_names, test_loader, dataset, dashboard_dir)
    
    print(f"\nAll analysis complete! Results saved to {diverse_models_dir}")
    
    # output_dir = "C:/Users/kigeorgiou/Desktop/Forcasting/Results/PINN_improved"
    # best_config = run_hyperparameter_tuning(df, output_dir)
    
    # # Run extended training with best configuration
    # final_model, eval_results = run_extended_training(df, best_config, output_dir, epochs=500)
    
    # print("\nTraining and tuning complete!")
    # print(f"Final R² score: {eval_results['r2']:.4f}")
    # print(f"Results saved to: {output_dir}")
    
def run_hyperparameter_tuning(df, base_output_dir):
    """
    Run hyperparameter tuning for the PINN model.
    
    Args:
        df: Preprocessed dataframe
        base_output_dir: Base directory for outputs
    """
    # Create dataset with current parameters
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
    
    # Initialize hyperparameter tuner
    tuner = HyperparameterTuner(
        base_output_dir=base_output_dir,
        dataset=dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    # Define parameter grid for tuning
    param_grid = {
        "hidden_size": [32, 64, 128],
        "num_layers": [2, 3, 4],
        "batch_size": [32, 64],
        "learning_rate": [0.001, 0.0005],
        "physics_weight": [0.05, 0.1, 0.2]
    }
    
    # Run grid search
    best_config, results = tuner.run_grid_search(
        param_grid=param_grid,
        epochs=200,  # Extended training
        early_stopping=True  # Use early stopping
    )
    
    print("\nBest configuration found:")
    print(best_config)
    
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
    Create comprehensive visualizations for model evaluation.
    
    Args:
        model: Trained PINN model
        test_loader: DataLoader for test data
        dataset: Dataset object (for denormalization)
        df: Original dataframe (for time series visualization)
        device: Computation device
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Get predictions
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
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
    if hasattr(dataset, 'denormalize_targets'):
        all_targets = dataset.denormalize_targets(all_targets)
        all_predictions = dataset.denormalize_targets(all_predictions)
    
    # Calculate error metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    
    # 2. Create a DataFrame with test set timestamps
    # This is approximate since we need to match the test indices to original data
    # In practice, you'd need to ensure proper alignment between test data and timestamps
    
    # For demonstration, we'll create synthetic timestamps
    # In reality, you'd need to track the actual test set indices
    timestamps = df.index[-len(all_targets):]
    
    # Create result DataFrame
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'actual': all_targets,
        'predicted': all_predictions,
        'error': all_predictions - all_targets
    })
    
    # Add derived columns
    results_df['hour'] = results_df['timestamp'].dt.hour
    results_df['day'] = results_df['timestamp'].dt.day
    results_df['month'] = results_df['timestamp'].dt.month
    results_df['dayofweek'] = results_df['timestamp'].dt.dayofweek  # 0=Monday
    results_df['weekend'] = results_df['dayofweek'].isin([5, 6]).astype(int)
    
    # Add relative error (where actual > threshold to avoid division by zero issues)
    threshold = 100  # Only calculate relative error for production > 100 Wh
    results_df['relative_error'] = np.nan
    mask = results_df['actual'] > threshold
    results_df.loc[mask, 'relative_error'] = (
        (results_df.loc[mask, 'predicted'] - results_df.loc[mask, 'actual']) / 
        results_df.loc[mask, 'actual'] * 100  # As percentage
    )
    
    # 3. Visualization suite
    
    # 3.1 Time series comparison (last 7 days)
    plt.figure(figsize=(20, 10))
    
    # Plot the last 7 days or all data if less than 7 days
    plot_days = 7
    last_n_points = min(24 * plot_days, len(results_df))
    
    ax = plt.subplot(111)
    ax.plot(results_df['timestamp'].iloc[-last_n_points:], 
            results_df['actual'].iloc[-last_n_points:], 
            label='Actual', linewidth=2.5)
    ax.plot(results_df['timestamp'].iloc[-last_n_points:], 
            results_df['predicted'].iloc[-last_n_points:], 
            label='Predicted', linewidth=2, linestyle='--')
    
    # Add shading for nighttime (assuming no production at night)
    night_mask = results_df['actual'].iloc[-last_n_points:] < threshold
    night_indices = np.where(night_mask)[0]
    for idx in night_indices:
        if idx < len(results_df.iloc[-last_n_points:]):
            ax.axvspan(results_df['timestamp'].iloc[-last_n_points+idx], 
                      results_df['timestamp'].iloc[-last_n_points+idx+1] if idx+1 < last_n_points 
                      else results_df['timestamp'].iloc[-1],
                      alpha=0.1, color='gray')
    
    # Format the plot
    plt.title(f'Solar Production: Actual vs Predicted (Last {plot_days} Days)', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Energy Production (Wh)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format x-axis to show dates clearly
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_series_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.2 Scatter plot with identity line and density
    plt.figure(figsize=(12, 10))
    
    # Create custom colormap from seaborn deep palette
    cmap = LinearSegmentedColormap.from_list("deep", sns.color_palette("deep", 10), N=10)
    
    # Determine point density for coloring
    from scipy.stats import gaussian_kde
    xy = np.vstack([all_targets, all_predictions])
    density = gaussian_kde(xy)(xy)
    
    # Sort points by density for better visualization
    idx = density.argsort()
    x, y, z = all_targets[idx], all_predictions[idx], density[idx]
    
    # Create scatter plot
    plt.scatter(x, y, c=z, s=20, cmap=cmap, alpha=0.8)
    
    # Add identity line
    min_val = min(all_targets.min(), all_predictions.min())
    max_val = max(all_targets.max(), all_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_targets, all_predictions)
    plt.plot(all_targets, slope*all_targets + intercept, 'r-', linewidth=2,
            label=f'Regression Line (slope={slope:.3f})')
    
    # Add metrics as text
    plt.text(0.05, 0.95, f"MAE: {mae:.2f} Wh\nRMSE: {rmse:.2f} Wh\nR²: {r2:.4f}",
            transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Formatting
    plt.title('Predicted vs Actual Energy Production', fontsize=16)
    plt.xlabel('Actual Energy Production (Wh)', fontsize=14)
    plt.ylabel('Predicted Energy Production (Wh)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.3 Error distribution
    plt.figure(figsize=(12, 8))
    
    # Calculate error bins
    bins = 50
    
    # Plot error histogram
    plt.hist(results_df['error'], bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add vertical line at zero
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    
    # Add error statistics
    error_mean = results_df['error'].mean()
    error_std = results_df['error'].std()
    error_skew = results_df['error'].skew()
    
    plt.text(0.05, 0.95, 
            f"Mean Error: {error_mean:.2f} Wh\nStd Dev: {error_std:.2f} Wh\nSkewness: {error_skew:.3f}",
            transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Formatting
    plt.title('Error Distribution', fontsize=16)
    plt.xlabel('Prediction Error (Wh)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.4 Heat map of errors by hour of day and month
    plt.figure(figsize=(14, 8))
    
    # Calculate average absolute error by hour and month
    hourly_monthly_error = results_df.pivot_table(
        values='error', 
        index='hour', 
        columns='month', 
        aggfunc=lambda x: np.abs(x).mean()
    )
    
    # Create heat map
    sns.heatmap(hourly_monthly_error, cmap='YlOrRd', annot=True, fmt='.0f',
               linewidths=0.5, cbar_kws={'label': 'Mean Absolute Error (Wh)'})
    
    # Formatting
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.title('Mean Absolute Error by Hour and Month', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Hour of Day', fontsize=14)
    plt.yticks(np.arange(0.5, 24.5), range(24))
    plt.xticks(np.arange(0.5, 12.5), month_names, rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.5 Relative error by production level
    plt.figure(figsize=(12, 8))
    
    # Filter for actual production above threshold
    filtered_results = results_df[results_df['actual'] > threshold].copy()
    
    # Create bins for actual production
    max_production = filtered_results['actual'].max()
    bin_edges = np.linspace(threshold, max_production, 10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate relative errors by bin
    rel_errors_by_bin = []
    for i in range(len(bin_edges) - 1):
        mask = (filtered_results['actual'] >= bin_edges[i]) & (filtered_results['actual'] < bin_edges[i+1])
        bin_data = filtered_results.loc[mask, 'relative_error']
        if len(bin_data) > 0:
            rel_errors_by_bin.append(bin_data.abs().mean())
        else:
            rel_errors_by_bin.append(np.nan)
    
    # Plot
    plt.bar(bin_centers, rel_errors_by_bin, width=(bin_edges[1] - bin_edges[0]) * 0.8, 
           alpha=0.7, color='skyblue', edgecolor='black')
    
    # Formatting
    plt.title('Mean Absolute Percentage Error by Production Level', fontsize=16)
    plt.xlabel('Actual Production (Wh)', fontsize=14)
    plt.ylabel('Mean Absolute Percentage Error (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_by_production_level.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.6 Performance by time of day
    plt.figure(figsize=(14, 10))
    
    # Calculate hourly metrics
    hourly_metrics = results_df.groupby('hour').agg({
        'actual': 'mean',
        'predicted': 'mean',
        'error': lambda x: np.abs(x).mean(),
        'relative_error': lambda x: np.abs(x[~np.isnan(x)]).mean()
    }).reset_index()
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    
    # Plot production curves
    ax1.plot(hourly_metrics['hour'], hourly_metrics['actual'], 'b-', 
            label='Average Actual Production', linewidth=2.5)
    ax1.plot(hourly_metrics['hour'], hourly_metrics['predicted'], 'g--', 
            label='Average Predicted Production', linewidth=2.5)
    
    # Plot error
    ax2.bar(hourly_metrics['hour'], hourly_metrics['error'], 
           alpha=0.3, color='r', label='Mean Absolute Error')
    
    # Formatting
    ax1.set_xlabel('Hour of Day', fontsize=14)
    ax1.set_ylabel('Energy Production (Wh)', fontsize=14, color='b')
    ax2.set_ylabel('Error (Wh)', fontsize=14, color='r')
    
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax1.set_xticks(range(24))
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.title('Average Production and Error by Hour of Day', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hourly_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.7 Weekly pattern analysis
    plt.figure(figsize=(14, 8))
    
    # Calculate daily metrics
    daily_metrics = results_df.groupby('dayofweek').agg({
        'error': lambda x: np.abs(x).mean(),
        'relative_error': lambda x: np.abs(x[~np.isnan(x)]).mean()
    }).reset_index()
    
    # Create bar chart
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    x = np.arange(len(days))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width/2, daily_metrics['error'], width, label='Mean Absolute Error (Wh)', color='skyblue')
    
    # Add second axis for relative error
    ax2 = ax.twinx()
    ax2.bar(x + width/2, daily_metrics['relative_error'], width, label='Mean Absolute Percentage Error (%)', 
           color='lightcoral')
    
    # Add labels and legend
    ax.set_xlabel('Day of Week', fontsize=14)
    ax.set_ylabel('Absolute Error (Wh)', fontsize=14, color='skyblue')
    ax2.set_ylabel('Absolute Percentage Error (%)', fontsize=14, color='lightcoral')
    
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    
    ax.tick_params(axis='y', labelcolor='skyblue')
    ax2.tick_params(axis='y', labelcolor='lightcoral')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Prediction Error by Day of Week', fontsize=16)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weekly_pattern.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3.8 Physics component analysis (if model has physics component)
    if hasattr(model, 'forward') and 'return_components' in model.forward.__code__.co_varnames:
        plt.figure(figsize=(14, 8))
        
        # Get a sample of data
        sample_size = min(100, len(test_loader.dataset))
        sample_indices = np.random.choice(len(test_loader.dataset), sample_size, replace=False)
        
        sample_features = []
        sample_targets = []
        
        for idx in sample_indices:
            features, targets = test_loader.dataset[idx]
            sample_features.append(features)
            sample_targets.append(targets)
        
        sample_features = torch.stack(sample_features).to(device)
        sample_targets = torch.stack(sample_targets).to(device)
        
        # Get predictions with physics components
        with torch.no_grad():
            predictions, physics_components = model(sample_features, return_components=True)
        
        # Convert to numpy and denormalize
        physics_np = physics_components.cpu().numpy()
        neural_np = predictions.cpu().numpy() - physics_np
        targets_np = sample_targets.cpu().numpy()
        
        if hasattr(dataset, 'denormalize_targets'):
            physics_np = dataset.denormalize_targets(physics_np)
            neural_np = dataset.denormalize_targets(neural_np)
            targets_np = dataset.denormalize_targets(targets_np)
        
        # Plot stacked components
        sorted_indices = np.argsort(targets_np.flatten())
        
        plt.plot(targets_np.flatten()[sorted_indices], label='Actual', linewidth=2, color='black')
        plt.plot(physics_np.flatten()[sorted_indices], label='Physics Component', linewidth=2, color='blue')
        plt.plot(neural_np.flatten()[sorted_indices], label='Neural Network Component', linewidth=2, color='red')
        plt.plot((physics_np + neural_np).flatten()[sorted_indices], 
                label='Combined Prediction', linewidth=2, color='green', linestyle='--')
        
        # Formatting
        plt.title('PINN Components Analysis', fontsize=16)
        plt.xlabel('Sample (sorted by actual value)', fontsize=14)
        plt.ylabel('Energy Production (Wh)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/physics_component_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Return results DataFrame for further analysis
    return results_df

def ensemble_weighted_average(models, weights, test_loader, device):
    """
    Combine predictions from multiple models using weighted averaging.
    
    Args:
        models: List of trained models
        weights: List of weights for each model (should sum to 1)
        test_loader: DataLoader for test data
        device: Device to use for computation
    
    Returns:
        Numpy arrays of ensemble predictions and actual values
    """
    assert len(models) == len(weights), "Number of models must match number of weights"
    assert np.isclose(sum(weights), 1.0), "Weights must sum to 1"
    
    all_ensemble_predictions = []
    all_actual_values = []
    
    # Set all models to evaluation mode
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Get predictions from each model
            batch_predictions = []
            for model in models:
                pred = model(features)
                if pred.dim() > 1 and pred.shape[1] > 1:
                    pred = pred[:, 0]  # Take first prediction if multi-step
                batch_predictions.append(pred.cpu().numpy())
            
            # Make sure all predictions have the same shape
            for i in range(len(batch_predictions)):
                batch_predictions[i] = batch_predictions[i].flatten()
            
            # Ensure all arrays have the same length
            min_length = min(len(p) for p in batch_predictions)
            batch_predictions = [p[:min_length] for p in batch_predictions]
            
            # Combine predictions with weights (creating a new array)
            batch_ensemble = np.zeros_like(batch_predictions[0])
            for i, pred in enumerate(batch_predictions):
                batch_ensemble += weights[i] * pred
            
            # Make sure targets have matching shape
            targets_np = targets.cpu().numpy().flatten()[:min_length]
            
            # Store results
            all_ensemble_predictions.append(batch_ensemble)
            all_actual_values.append(targets_np)
    
    # Concatenate all batches
    ensemble_predictions = np.concatenate(all_ensemble_predictions)
    actual_values = np.concatenate(all_actual_values)
    
    return ensemble_predictions, actual_values

def train_diverse_models(df, output_dir):
    """
    Train a diverse set of models for ensemble learning and save them.
    
    Args:
        df: Preprocessed dataframe with solar production data
        output_dir: Output directory for saving models and results
        
    Returns:
        List of trained models and evaluation results
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
    
    # Get system parameters for PINN models
    try:
        panel_efficiency = df['param_panel_efficiency'].iloc[0]
        panel_area = df['param_panel_area'].iloc[0]
        temp_coeff = df['param_temp_coeff'].iloc[0]
    except:
        # Use defaults if not available
        panel_efficiency = 0.146
        panel_area = 1.642
        temp_coeff = -0.0044
    
    # 1. Create a set of diverse models
    input_size = len(dataset.feature_names)
    
    # Model 1: Smaller PINN (quicker to converge, may overfit)
    model1 = SolarProductionPINN(
        input_size=input_size,
        hidden_size=32,
        num_layers=2,
        panel_efficiency=panel_efficiency,
        panel_area=panel_area,
        temp_coeff=temp_coeff
    ).to(device)
    
    # Model 2: Larger PINN (more capacity, slower convergence)
    model2 = SolarProductionPINN(
        input_size=input_size,
        hidden_size=128,
        num_layers=4,
        panel_efficiency=panel_efficiency,
        panel_area=panel_area,
        temp_coeff=temp_coeff
    ).to(device)
    
    # Model 3: LSTM-based forecaster (good for temporal patterns)
    model3 = LSTMSolarForecaster(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        forecast_horizon=1
    ).to(device)
    
    # Model 4: CNN-based forecaster (good for pattern recognition)
    model4 = CNNSolarForecaster(
        input_size=input_size,
        seq_length=24,  # Assuming 24-hour history
        num_filters=32,
        forecast_horizon=1
    ).to(device)
    
    # Train each model
    models = [model1, model2, model3, model4]
    model_names = ["PINN-Small", "PINN-Large", "LSTM", "CNN"]
    trained_models = []
    evaluation_results = []
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        model_save_path = model_dir / f"{name.lower()}_model.pt"
        
        print(f"\nTraining {name} model...")
        
        # Train model (implementation depends on model type)
        if name.startswith("PINN"):
            model, history = train_pinn(
                model, train_loader, val_loader,
                epochs=100,
                lr=0.001,
                physics_weight=0.1,
                device=device,
                save_path=model_save_path
            )
        else:
            # Generic training loop for non-PINN models
            model, history = train_generic_model(
                model, train_loader, val_loader,
                epochs=100,
                lr=0.001,
                device=device,
                save_path=model_save_path
            )
        
        trained_models.append(model)
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name} Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(history['train_loss'], label='Train Loss', linewidth=2)
        plt.semilogy(history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.title('Loss (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(model_dir / f"{name.lower()}_training.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Evaluate individual model
        print(f"Evaluating {name} model...")
        eval_result = evaluate_model(model, test_loader, name, device, dataset)
        evaluation_results.append(eval_result)
    
    # Create and evaluate ensemble
    print("\nCreating ensemble...")
    
    # Option 1: Simple equally-weighted ensemble
    weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
    ensemble_preds, actual_values = ensemble_weighted_average(
        trained_models, weights, test_loader, device
    )
    
    # Denormalize if necessary
    if hasattr(dataset, 'denormalize_targets'):
        ensemble_preds = dataset.denormalize_targets(ensemble_preds)
        actual_values = dataset.denormalize_targets(actual_values)
    
    ensemble_result = evaluate_ensemble(ensemble_preds, actual_values, "Equally-Weighted Ensemble")
    
    # Option 2: Create a stacked ensemble
    from sklearn.linear_model import Ridge
    meta_model = Ridge(alpha=0.5)
    stacked = StackedEnsemble(trained_models, meta_model)
    
    # Train meta-model on validation set
    print("Training stacked ensemble...")
    stacked.train_meta_model(val_loader, device)
    
    # Test stacked ensemble
    print("Evaluating stacked ensemble...")
    stacked_preds, stacked_actual = stacked.predict(test_loader, device)
    
    # Denormalize if necessary
    if hasattr(dataset, 'denormalize_targets'):
        stacked_preds = dataset.denormalize_targets(stacked_preds)
        stacked_actual = dataset.denormalize_targets(stacked_actual)
    
    stacked_result = evaluate_ensemble(stacked_preds, stacked_actual, "Stacked Ensemble")
    
    # Compare all models and ensembles
    create_model_comparison_visualization(
        evaluation_results, 
        [ensemble_result, stacked_result],
        model_dir
    )
    
    for name, meta_model in meta_models.items():
        stacked = StackedEnsemble(trained_models, meta_model)
        stacked.train_meta_model(val_loader, device)
        stacked_preds, stacked_actual = stacked.predict(test_loader, device)
        results = evaluate_ensemble(stacked_preds, stacked_actual, f"Stacked Ensemble ({name})")
        print(f"{name} Meta-Model: R² = {results['r2']:.4f}, MAE = {results['mae']:.2f} Wh")

    print(f"\nAll models and visualizations saved to {model_dir}")
    
    model_names = ["PINN-Small", "PINN-Large", "LSTM", "CNN"]  # Add this line
    return trained_models, model_names, evaluation_results, ensemble_result, stacked_result

def run_enhanced_forecasting(df, base_output_dir):
    """
    Run enhanced solar forecasting with visualizations and ensemble methods.
    
    Args:
        df: Preprocessed dataframe with solar production data
        base_output_dir: Base directory for saving results
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
    
    # Step 2: Load best hyperparameters from tuning (if available)
    try:
        # Extract system parameters from data
        panel_efficiency = df['param_panel_efficiency'].iloc[0]
        panel_area = df['param_panel_area'].iloc[0]
        temp_coeff = df['param_temp_coeff'].iloc[0]
    except:
        # Use defaults if not available
        panel_efficiency = 0.146
        panel_area = 1.642
        temp_coeff = -0.0044
    
    print(f"Panel parameters: Efficiency={panel_efficiency:.4f}, Area={panel_area:.4f}m², Temp Coeff={temp_coeff:.6f}")
    
    # Step 3: Create base PINN model
    print("\nCreating base PINN model...")
    pinn_model = SolarProductionPINN(
        input_size=len(dataset.feature_names),
        hidden_size=64,
        num_layers=3,
        panel_efficiency=panel_efficiency,
        panel_area=panel_area,
        temp_coeff=temp_coeff
    ).to(device)
    
    # Load previous model or train a new one
    model_path = str(enhanced_dir / "pinn_model.pt")
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        pinn_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training new model...")
        pinn_model, _ = train_pinn(
            pinn_model, train_loader, val_loader,
            epochs=100,
            lr=0.001,
            physics_weight=0.1,
            device=device,
            save_path=model_path
        )
    
    # Step 4: Create comprehensive visualizations
    print("\nGenerating visualizations...")
    try:
        # Create visualizations for the PINN model
        results_df = create_comprehensive_visualizations(
            pinn_model, test_loader, dataset, df, 
            device=device, output_dir=str(vis_dir)
        )
        print(f"Visualizations saved to {vis_dir}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Step 5: Create LSTM model for ensemble
    print("\nCreating LSTM model for ensemble...")
    lstm_model = LSTMSolarForecaster(
        input_size=len(dataset.feature_names),
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        forecast_horizon=1
    ).to(device)
    
    # Train LSTM model
    lstm_path = str(ensemble_dir / "lstm_model.pt")
    
    if os.path.exists(lstm_path):
        print(f"Loading existing LSTM model from {lstm_path}")
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    else:
        print("Training LSTM model...")
        # Simple training loop
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        
        lstm_model.train()
        for epoch in range(50):  # Fewer epochs for demonstration
            epoch_loss = 0
            for features, targets in train_loader:
                features, targets = features.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = lstm_model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"LSTM Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.6f}")
        
        torch.save(lstm_model.state_dict(), lstm_path)
    
    # Step 6: Create ensemble and evaluate
    print("\nCreating and evaluating ensemble...")
    models = [pinn_model, lstm_model]
    model_names = ["PINN", "LSTM"]
    
    # Evaluate individual models first
    for i, (model, name) in enumerate(zip(models, model_names)):
        model.eval()
        with torch.no_grad():
            total_loss = 0
            all_targets = []
            all_predictions = []
            
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                predictions = model(features)
                
                # Ensure correct shape
                if predictions.dim() > 1 and predictions.shape[1] > 1:
                    predictions = predictions[:, 0]
                if targets.dim() > 1 and targets.shape[1] > 1:
                    targets = targets[:, 0]
                
                # Store predictions and targets
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
            
            all_targets = np.concatenate(all_targets)
            all_predictions = np.concatenate(all_predictions)
            
            # Denormalize if needed
            if hasattr(dataset, 'denormalize_targets'):
                all_targets = dataset.denormalize_targets(all_targets)
                all_predictions = dataset.denormalize_targets(all_predictions)
            
            # Calculate metrics
            mae = np.mean(np.abs(all_targets - all_predictions))
            rmse = np.sqrt(np.mean((all_targets - all_predictions) ** 2))
            r2 = 1 - np.sum((all_targets - all_predictions) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
            
            print(f"{name} Model Performance:")
            print(f"  MAE: {mae:.2f} Wh")
            print(f"  RMSE: {rmse:.2f} Wh")
            print(f"  R²: {r2:.4f}")
    
    # Create weighted ensemble
    # Equal weights for now
    weights = [0.5, 0.5]  # 50% PINN, 50% LSTM 
    
    print("\nCreating weighted ensemble with weights:", weights)
    
    # Get predictions from both models
    all_model_predictions = []
    ground_truth = []

    for i, model in enumerate(models):
        model_predictions = []
        model.eval()
        
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                predictions = model(features)
                
                # Ensure correct shape
                if predictions.dim() > 1 and predictions.shape[1] > 1:
                    predictions = predictions[:, 0]
                if targets.dim() > 1 and targets.shape[1] > 1:
                    targets = targets[:, 0]
                
                model_predictions.append(predictions.cpu().numpy())
                
                # Only store ground truth for the first model
                if i == 0:
                    ground_truth.append(targets.cpu().numpy())
    
    # Concatenate ground truth
    ground_truth = np.concatenate(ground_truth)
    
    # Create ensemble predictions
    ensemble_predictions = np.zeros_like(all_model_predictions[0])
    for i, pred in enumerate(all_model_predictions):
        ensemble_predictions += weights[i] * pred
    
    # Denormalize if needed
    if hasattr(dataset, 'denormalize_targets'):
        ensemble_predictions = dataset.denormalize_targets(ensemble_predictions)
        ground_truth = dataset.denormalize_targets(ground_truth)
    
    # Calculate ensemble metrics
    mae = np.mean(np.abs(ground_truth - ensemble_predictions))
    rmse = np.sqrt(np.mean((ground_truth - ensemble_predictions) ** 2))
    r2 = 1 - np.sum((ground_truth - ensemble_predictions) ** 2) / np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    
    print("\nEnsemble Model Performance:")
    print(f"  MAE: {mae:.2f} Wh")
    print(f"  RMSE: {rmse:.2f} Wh")
    print(f"  R²: {r2:.4f}")
    
    # Create a final visualization comparing models
    plt.figure(figsize=(12, 6))
    
    # Sample a slice of data points to visualize
    sample_size = min(48, len(ground_truth))  # Show 2 days worth of data if available
    start_idx = len(ground_truth) - sample_size  # Show the last points
    
    # Plot actual values
    plt.plot(range(sample_size), ground_truth[start_idx:start_idx+sample_size], 
            'k-', label='Actual', linewidth=2)
    
    # Plot PINN predictions
    plt.plot(range(sample_size), all_model_predictions[0][start_idx:start_idx+sample_size], 
            'b--', label='PINN', linewidth=2)
    
    # Plot LSTM predictions
    plt.plot(range(sample_size), all_model_predictions[1][start_idx:start_idx+sample_size], 
            'g--', label='LSTM', linewidth=2)
    
    # Plot ensemble predictions
    plt.plot(range(sample_size), ensemble_predictions[start_idx:start_idx+sample_size], 
            'r-', label='Ensemble', linewidth=2)
    
    plt.title('Model Comparison: Last 48 Hours')
    plt.xlabel('Hours')
    plt.ylabel('Energy Production (Wh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(str(ensemble_dir / "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison visualization saved to {ensemble_dir / 'model_comparison.png'}")
    print("\nEnhanced forecasting complete!")
    
    return pinn_model, lstm_model, models, weights

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
    """Calculate optimal weights based on validation performance"""
    # Get predictions for each model on validation set
    all_preds = []
    
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            batch_preds = []
            
            for model in models:
                pred = model(features)
                if pred.dim() > 1 and pred.shape[1] > 1:
                    pred = pred[:, 0]
                batch_preds.append(pred.cpu().numpy())
            
            all_preds.append(batch_preds)
    
    # Reshape predictions for optimization
    model_preds = []
    for i in range(len(models)):
        model_pred = np.concatenate([preds[i] for preds in all_preds])
        if hasattr(dataset, 'denormalize_targets'):
            model_pred = dataset.denormalize_targets(model_pred)
        model_preds.append(model_pred)
    
    # Get actual values
    actual = []
    for _, targets in val_loader:
        actual.append(targets.cpu().numpy())
    actual = np.concatenate(actual)
    if hasattr(dataset, 'denormalize_targets'):
        actual = dataset.denormalize_targets(actual)
    
    # Define weight constraints (must sum to 1 and be non-negative)
    from scipy.optimize import minimize
    
    def mse_loss(weights):
        weights = np.array(weights)
        prediction = np.zeros_like(model_preds[0])
        for i, model_pred in enumerate(model_preds):
            prediction += weights[i] * model_pred
        return np.mean((prediction - actual) ** 2)
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(len(models))]
    result = minimize(mse_loss, [1/len(models)] * len(models), bounds=bounds, constraints=constraints)
    
    return result.x

def create_time_specialized_ensembles(df, output_dir):
    """Create time-specialized ensembles for different parts of the day"""
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
        
        # Filter data for this time period
        period_mask = df.index.hour.isin(hours)
        period_df = df[period_mask].copy()
        
        # Train models on this subset
        models, _, ensemble_result, _ = train_diverse_models(
            period_df, 
            os.path.join(output_dir, f"time_specialized/{period_name}")
        )
        
        results[period_name] = {
            "models": models,
            "performance": ensemble_result
        }
    
    return results

def create_production_pipeline(models, weights, dataset, device):
    """Create a production pipeline for making predictions"""
    
    def predict(features):
        """Make predictions using the ensemble"""
        # Convert features to tensor
        if not isinstance(features, torch.Tensor):
            features = torch.FloatTensor(features).to(device)
        
        # Add batch dimension if needed
        if features.dim() == 2:
            features = features.unsqueeze(0)
        
        # Get predictions from each model
        predictions = []
        for model in models:
            model.eval()
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
    """Generate a comprehensive performance dashboard for all models"""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
                
                model_preds.append(pred.cpu().numpy())
                
                # Only collect actual values once
                if len(all_actual) < len(test_loader):
                    all_actual.append(targets.cpu().numpy())
        
        # Concatenate predictions
        model_pred = np.concatenate(model_preds)
        if hasattr(dataset, 'denormalize_targets'):
            model_pred = dataset.denormalize_targets(model_pred)
        all_preds.append(model_pred)
    
    # Concatenate actual values
    actual = np.concatenate(all_actual)
    if hasattr(dataset, 'denormalize_targets'):
        actual = dataset.denormalize_targets(actual)
    
    # 1. Overall performance comparison
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
        mape = np.mean(np.abs((actual[significant] - pred[significant]) / actual[significant])) * 100
        
        # Store metrics
        metrics["R²"].append(r2)
        metrics["RMSE (Wh)"].append(rmse)
        metrics["MAE (Wh)"].append(mae)
        metrics["MAPE (%)"].append(mape)
    
    # Create performance comparison plot
    plt.figure(figsize=(15, 10))
    
    # Plot each metric
    for i, (metric, values) in enumerate(metrics.items()):
        plt.subplot(2, 2, i+1)
        
        # Create horizontal bar chart
        bars = plt.barh(model_names, values, color=sns.color_palette("viridis", len(model_names)))
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center')
        
        plt.title(f'{metric}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
    
    # Additional visualizations...
if __name__ == "__main__":
    main()

generate_performance_dashboard
create_production_pipeline
create_time_specialized_ensembles
calculate_optimal_weights
create_model_comparison_visualization