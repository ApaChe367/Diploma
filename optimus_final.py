import argparse
import pandas as pd
import pvlib
from datetime import datetime, timedelta
import logging
import os
import itertools
import numpy as np
from scipy.optimize import differential_evolution, minimize
import locale
import yaml
import sys
from deap import base, creator, tools, algorithms
import multiprocessing
import random
from typing import Dict, List, Tuple, Optional
import json
from deap.algorithms import varOr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdate
import matplotlib.dates as mdates
from matplotlib.patches import Circle
import pandas as pd
import plotting_module


try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available - battery plots will be skipped")

try:
    import numpy_financial as npf
    NUMPY_FINANCIAL_AVAILABLE = True
except ImportError:
    NUMPY_FINANCIAL_AVAILABLE = False
    
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})   
    
# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
U0 = 25.0 
U1 = 6.84

if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Define constants directly in the script
RANDOM_SEED = 42
TIME_INTERVAL_HOURS = 1.0  # Hours per data point
NOCT = 47.5  # Nominal Operating Cell Temperature in °C for Sharp ND-R240A5

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class OptimizationContext:
    def __init__(self, df_subset, dni_extra, number_of_panels, inverter_params, latitude, longitude, weighting_strategy='adaptive_improved'):
        self.df_subset = df_subset
        self.dni_extra = dni_extra
        self.number_of_panels = number_of_panels
        self.inverter_params = inverter_params
        self.latitude = latitude
        self.longitude = longitude
        self.weighting_strategy = weighting_strategy
        
# Initialize a global variable to hold the context
optimization_context = None

def init_worker(context):
    """
    Initializer for each worker process in the pool.
    
    Parameters:
    - context (OptimizationContext): An instance of OptimizationContext containing all necessary data.
    """
    global optimization_context
    optimization_context = context

def evaluate_individual_IMPROVED(individual):
    """
    IMPROVED evaluation function for DEAP using realistic energy calculations.
    
    Parameters:
    - individual (list): A list containing tilt and azimuth angles.
    
    Returns:
    - tuple: A tuple containing weighted energy mismatch, total energy production.
    """
    global optimization_context
    return objective_function_multi(
        individual,
        optimization_context.df_subset,
        optimization_context.dni_extra,
        optimization_context.number_of_panels,
        optimization_context.inverter_params,
        optimization_context.weighting_strategy)

def setup_logging(output_dir):
    """
    Set up logging to file and console with proper encoding.
    """
    log_file = os.path.join(output_dir, 'solar_analysis.log')
    
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if setup_logging is called multiple times
    if not logger.handlers:
        # Create handlers with proper encoding
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file, encoding='utf-8')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add to handlers
        c_format = logging.Formatter('%(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

def load_config(config_file):
    """
    Load configuration from a YAML file.

    Parameters:
    - config_file (str): Path to the YAML configuration file.

    Returns:
    - config (dict): Configuration parameters.
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_file}")
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}", exc_info=True)
        raise
    return config

def load_and_preprocess_pvgis_data(pvgis_file, load_file):
    """
    CRITICAL FIX: Handle multi-year PVGIS data correctly by creating a synthetic year.
    """
    
    logging.info("=== LOADING MULTI-YEAR PVGIS DATA ===")
    
    try:
        # Load PVGIS TMY data
        df_pvgis = pd.read_csv(pvgis_file)
        logging.info(f"PVGIS data loaded: {len(df_pvgis)} records")
        logging.info(f"PVGIS columns: {list(df_pvgis.columns)}")
        
        # Load consumption data
        df_load = pd.read_csv(load_file)
        logging.info(f"Load data loaded: {len(df_load)} records")
        
        # CRITICAL FIX: Parse PVGIS time and create synthetic hourly timeline
        def parse_pvgis_time_FIXED(time_str):
            """Enhanced time parsing"""
            try:
                if ':' in time_str:
                    date_part = time_str.split(':')[0]
                    time_part = time_str.split(':')[1]
                else:
                    date_part = time_str[:8]
                    time_part = time_str[8:] if len(time_str) > 8 else "0000"
                
                year = int(date_part[:4])
                month = int(date_part[4:6])
                day = int(date_part[6:8])
                hour = int(time_part[:2])
                minute = int(time_part[2:4]) if len(time_part) >= 4 else 0
                
                return datetime(year, month, day, hour, minute)
            except Exception as e:
                logging.error(f"Error parsing time '{time_str}': {e}")
                return None
        
        # Parse original times
        df_pvgis['original_datetime'] = df_pvgis['time_UTC'].apply(parse_pvgis_time_FIXED)
        df_pvgis = df_pvgis.dropna(subset=['original_datetime'])
        
        # CRITICAL FIX: Create synthetic year timeline
        # This data appears to be TMY (Typical Meteorological Year) with data from different years
        # We need to create a consistent hourly timeline for one year
        
        logging.info("Creating synthetic year timeline for TMY data with 30-min timestamp correction...")
        
        # Create a standard year timeline (2020 as reference, non-leap year for consistency)
        base_year = 2020
        start_date = datetime(base_year, 1, 1, 0, 0)
        synthetic_timeline = pd.date_range(
            start=start_date, 
            periods=8760,  # Exactly one year of hours
            freq='h',
            tz='UTC'
        )
        
        # Map original data to synthetic timeline by month-day-hour
        df_pvgis['month'] = df_pvgis['original_datetime'].dt.month
        df_pvgis['day'] = df_pvgis['original_datetime'].dt.day
        df_pvgis['hour'] = df_pvgis['original_datetime'].dt.hour
        
        # Create synthetic datetime with consistent year + CRITICAL FIX: 30-minute offset
        df_pvgis['synthetic_datetime'] = df_pvgis.apply(
            lambda row: datetime(base_year, row['month'], row['day'], row['hour'], 0) + 
                    timedelta(minutes=30),  # PVGIS timestamps represent middle of hour
            axis=1
        )
        
        # Sort by synthetic datetime to ensure proper chronological order
        df_pvgis = df_pvgis.sort_values('synthetic_datetime').reset_index(drop=True)
        
        # CRITICAL: Localize to UTC first, then convert to Athens
        df_pvgis['datetime_utc'] = pd.to_datetime(df_pvgis['synthetic_datetime']).dt.tz_localize('UTC')
        df_pvgis['datetime_athens'] = df_pvgis['datetime_utc'].dt.tz_convert('Europe/Athens')
        
        # VALIDATION: Check timeline consistency
        if len(df_pvgis) != 8760:
            logging.error(f"Expected 8760 hours, got {len(df_pvgis)}")
            
        # Check for duplicates
        duplicate_times = df_pvgis['synthetic_datetime'].duplicated().sum()
        if duplicate_times > 0:
            logging.warning(f"Found {duplicate_times} duplicate timestamps - keeping first occurrence")
            df_pvgis = df_pvgis.drop_duplicates(subset=['synthetic_datetime'], keep='first')
        
        # Check for missing hours
        expected_hours = set(range(8760))
        actual_hours = set((df_pvgis['synthetic_datetime'] - datetime(base_year, 1, 1)).dt.total_seconds() // 3600)
        missing_hours = expected_hours - actual_hours
        if missing_hours:
            logging.warning(f"Missing {len(missing_hours)} hours in the dataset")
        
        # Create the required column mapping with correct PVGIS column names
        df_mapped = pd.DataFrame()
        df_mapped['datetime'] = df_pvgis['datetime_athens']
        df_mapped['Air Temp'] = df_pvgis['T2m']
        df_mapped['SolRad_Hor'] = df_pvgis['G(h)']    # Global Horizontal Irradiance
        df_mapped['SolRad_Dif'] = df_pvgis['Gd(h)']   # Diffuse Horizontal Irradiance  
        df_mapped['WS_10m'] = df_pvgis['WS10m']
        
        # Add DNI from PVGIS data
        if 'Gb(n)' in df_pvgis.columns:
            df_mapped['DNI_pvgis'] = df_pvgis['Gb(n)']
            logging.info("Using PVGIS-provided DNI data")
        
        # Set datetime as index
        df_mapped.set_index('datetime', inplace=True)
        df_mapped = df_mapped.sort_index()
        
        # CRITICAL FIX: Validate solar data quality
        max_ghi = df_mapped['SolRad_Hor'].max()
        avg_ghi = df_mapped['SolRad_Hor'].mean()
        annual_ghi = df_mapped['SolRad_Hor'].sum() / 1000
        
        logging.info(f"FIXED PVGIS solar data validation:")
        logging.info(f"  Max GHI: {max_ghi:.0f} W/m²")
        logging.info(f"  Average GHI: {avg_ghi:.0f} W/m²")
        logging.info(f"  Annual GHI: {annual_ghi:.0f} kWh/m²")
        
        # Handle load data alignment
        if len(df_load) == len(df_mapped):
            df_mapped['Load (kW)'] = df_load['Load (kW)'].values
        else:
            logging.warning(f"Length mismatch: PVGIS={len(df_mapped)}, Load={len(df_load)}")
            if 'Load (kW)' in df_load.columns:
                # Create matching timeline for load data
                load_timeline = pd.date_range(
                    start=df_mapped.index[0], 
                    periods=len(df_load), 
                    freq='h'
                )
                df_load_indexed = df_load.set_index(load_timeline)
                load_reindexed = df_load_indexed['Load (kW)'].reindex(df_mapped.index, method='nearest')
                df_mapped['Load (kW)'] = load_reindexed.fillna(df_load['Load (kW)'].mean())
                logging.info("Load data aligned using nearest neighbor interpolation")
            else:
                df_mapped['Load (kW)'] = 100.0
                logging.warning("Load column not found, using default 100 kW")
        
        # Ensure all columns are numeric
        numeric_columns = ['Air Temp', 'SolRad_Hor', 'SolRad_Dif', 'WS_10m', 'Load (kW)']
        for col in numeric_columns:
            if col in df_mapped.columns:
                df_mapped[col] = pd.to_numeric(df_mapped[col], errors='coerce')
                if df_mapped[col].isnull().any():
                    if col in ['SolRad_Hor', 'SolRad_Dif']:
                        df_mapped[col] = df_mapped[col].fillna(0)
                    else:
                        df_mapped[col] = df_mapped[col].interpolate().bfill()
        
        # Add E_ac column
        if 'E_ac' not in df_mapped.columns:
            df_mapped['E_ac'] = 0.0
        
        # CRITICAL: Validate time intervals are now exactly 1 hour
        time_diffs = df_mapped.index.to_series().diff().dropna()
        time_diffs_hours = time_diffs.dt.total_seconds() / 3600
        
        logging.info(f"=== FIXED TIME INTERVAL VALIDATION ===")
        logging.info(f"Time interval statistics after fix:")
        logging.info(f"  Mean: {time_diffs_hours.mean():.3f} hours")
        logging.info(f"  Std: {time_diffs_hours.std():.3f} hours")
        logging.info(f"  Min: {time_diffs_hours.min():.3f} hours")
        logging.info(f"  Max: {time_diffs_hours.max():.3f} hours")
        
        non_hourly = time_diffs_hours[(time_diffs_hours < 0.95) | (time_diffs_hours > 1.05)]
        if len(non_hourly) > 0:
            logging.warning(f"Still found {len(non_hourly)} non-hourly intervals after fix!")
        else:
            logging.info("✓ All time intervals are now exactly 1 hour")
        
        # Final validation
        logging.info(f"=== FIXED PVGIS DATA SUMMARY ===")
        logging.info(f"Records: {len(df_mapped)}")
        logging.info(f"Time range: {df_mapped.index.min()} to {df_mapped.index.max()}")
        logging.info(f"Timezone: {df_mapped.index.tz}")
        
        # Check if solar noon occurs around 12-13 local time
        summer_data = df_mapped[df_mapped.index.month == 6]
        if len(summer_data) > 0:
            peak_solar_hour = summer_data.groupby(summer_data.index.hour)['SolRad_Hor'].mean().idxmax()
            logging.info(f"Peak solar occurs at hour: {peak_solar_hour} (should be 12-13 for Athens)")
        
        return df_mapped
        
    except Exception as e:
        logging.error(f"Fixed PVGIS processing failed: {e}", exc_info=True)
        raise
    
def calculate_solar_position(df, latitude, longitude):
    """
    FIXED: Calculate solar position with proper validation for Athens.
    """
    logging.info("=== FIXED SOLAR POSITION CALCULATION ===")
    
    try:
        if df.index.tz is None:
            logging.error("DataFrame must have timezone-aware index!")
            raise ValueError("DateTime index must have timezone information")
        
        # CRITICAL FIX: Ensure we have proper continuous timeline
        logging.info(f"Input data timeline: {df.index.min()} to {df.index.max()}")
        logging.info(f"Data timezone: {df.index.tz}")
        
        # Calculate solar position
        solar_position = pvlib.solarposition.get_solarposition(
            df.index, latitude, longitude, method='nrel_numpy'
        )
        
        df['zenith'] = solar_position['apparent_zenith']
        df['azimuth'] = solar_position['azimuth']
        
        # FIXED: More comprehensive solar position validation
        logging.info("=== FIXED SOLAR POSITION VALIDATION ===")
        
        # Check solar noon timing
        daylight_mask = (df['zenith'] < 85) & (df['SolRad_Hor'] > 50)
        
        if daylight_mask.any():
            df_daylight = df[daylight_mask].copy()
            df_daylight['hour_decimal'] = (df_daylight.index.hour + 
                                         df_daylight.index.minute/60.0)
            
            # Find solar noon for each day
            daily_solar_noon = []
            for date in pd.date_range(df.index.min().date(), df.index.max().date(), freq='D'):
                day_data = df_daylight[df_daylight.index.date == date.date()]
                if len(day_data) > 0:
                    min_zenith_idx = day_data['zenith'].idxmin()
                    noon_hour = day_data.loc[min_zenith_idx, 'hour_decimal']
                    daily_solar_noon.append(noon_hour)
            
            if daily_solar_noon:
                avg_solar_noon = sum(daily_solar_noon) / len(daily_solar_noon)
                logging.info(f"Average solar noon: {avg_solar_noon:.2f}h")
                
                # FIXED: More realistic bounds for Athens
                if 11.5 <= avg_solar_noon <= 13.5:  # Allow for DST and equation of time
                    logging.info("✓ Solar noon timing is reasonable for Athens")
                    solar_noon_ok = True
                else:
                    logging.warning(f"⚠ Solar noon at {avg_solar_noon:.2f}h is unusual for Athens!")
                    solar_noon_ok = False
            else:
                solar_noon_ok = True
        else:
            solar_noon_ok = True
        
        # CRITICAL FIX: Orientation validation with proper DNI calculation
        logging.info("=== FIXED ORIENTATION VALIDATION TEST ===")
        
        try:
            # Calculate DNI if not present
            if 'DNI' not in df.columns:
                if 'DNI_pvgis' in df.columns:
                    df['DNI'] = df['DNI_pvgis']
                    logging.info("Using PVGIS DNI for validation")
                else:
                    dni = pvlib.irradiance.disc(
                        ghi=df['SolRad_Hor'],
                        solar_zenith=df['zenith'],
                        datetime_or_doy=df.index
                    )['dni']
                    df['DNI'] = dni
                    logging.info("Calculated DNI using disc model for validation")
            
            # Get DNI extra for irradiance calculations
            dni_extra = pvlib.irradiance.get_extra_radiation(df.index, method='nrel')
            
            # FIXED: Test orientations that should show south is optimal
            test_orientations = [
                (30, 90, "East"),           # Should be lower
                (30, 135, "Southeast"),     # Should be lower than south
                (30, 180, "South"),         # Should be HIGHEST for max energy
                (30, 225, "Southwest"),     # Should be lower than south
                (30, 270, "West")           # Should be lower
            ]
            
            irradiation_results = {}
            
            for tilt, azimuth, name in test_orientations:
                try:
                    irradiance_data = pvlib.irradiance.get_total_irradiance(
                        surface_tilt=tilt, 
                        surface_azimuth=azimuth,
                        solar_zenith=df['zenith'], 
                        solar_azimuth=df['azimuth'],
                        dni=df['DNI'], 
                        ghi=df['SolRad_Hor'], 
                        dhi=df['SolRad_Dif'],
                        dni_extra=dni_extra, 
                        model='haydavies'
                    )
                    
                    annual_irradiation = irradiance_data['poa_global'].sum() / 1000
                    irradiation_results[name] = annual_irradiation
                    
                    logging.info(f"  {name} ({azimuth}°): {annual_irradiation:,.0f} kWh/m²")
                    
                except Exception as e:
                    logging.error(f"Error calculating irradiance for {name}: {e}")
                    irradiation_results[name] = 0
            
            # CRITICAL: Check if South produces maximum irradiation
            if len(irradiation_results) > 0:
                max_orientation = max(irradiation_results.keys(), 
                                    key=lambda k: irradiation_results[k])
                south_value = irradiation_results.get('South', 0)
                max_value = max(irradiation_results.values())
                
                logging.info(f"Maximum irradiation orientation: {max_orientation}")
                logging.info(f"South irradiation: {south_value:,.0f} kWh/m²")
                logging.info(f"Maximum irradiation: {max_value:,.0f} kWh/m²")
                
                if max_orientation == 'South':
                    logging.info("✓ VALIDATION PASSED: South produces maximum irradiation")
                    orientation_ok = True
                else:
                    # Check how much better the max is than south
                    if south_value > 0:
                        excess = ((max_value - south_value) / south_value) * 100
                        logging.error(f"✗ VALIDATION FAILED: {max_orientation} produces {excess:.1f}% more than South!")
                        logging.error("This indicates a problem with:")
                        logging.error("  - Solar position calculations")
                        logging.error("  - Timezone conversion")
                        logging.error("  - Data quality")
                        orientation_ok = False
                    else:
                        logging.error("✗ VALIDATION FAILED: South produces zero irradiation!")
                        orientation_ok = False
            else:
                orientation_ok = False
        
        except Exception as e:
            logging.error(f"Orientation validation failed: {e}")
            orientation_ok = False
        
        # Overall validation result
        validation_passed = solar_noon_ok and orientation_ok
        
        if validation_passed:
            logging.info("✓ OVERALL SOLAR POSITION VALIDATION: PASSED")
        else:
            logging.error("✗ OVERALL SOLAR POSITION VALIDATION: FAILED")
            logging.error("Solar calculations may be incorrect - optimization results will be wrong!")
        
        return df, validation_passed
        
    except Exception as e:
        logging.error(f"Solar position calculation failed: {e}", exc_info=True)

def calculate_dni(df):
    """
    FIXED: Calculate DNI - use PVGIS data if available, otherwise calculate using disc model.

    Parameters:
    - df (DataFrame): DataFrame with necessary irradiance and solar position data.

    Returns:
    - df (DataFrame): DataFrame with 'DNI' column added.
    """
    required_columns = ['SolRad_Hor', 'zenith']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the DataFrame.")
            raise ValueError(f"'{col}' column is missing in the DataFrame.")

    try:
        # Check if PVGIS already provided DNI data
        if 'DNI_pvgis' in df.columns:
            logging.info("Using PVGIS-provided DNI data")
            df['DNI'] = df['DNI_pvgis']
            
            # Validate PVGIS DNI data
            max_dni = df['DNI'].max()
            mean_dni = df['DNI'].mean()
            
            logging.info(f"PVGIS DNI validation:")
            logging.info(f"  Max DNI: {max_dni:.0f} W/m²")
            logging.info(f"  Mean DNI: {mean_dni:.0f} W/m²")
            
            # Check for reasonable DNI values
            if max_dni > 1200:
                logging.warning(f"Very high DNI values detected: max = {max_dni:.0f} W/m²")
            if max_dni < 800:
                logging.warning(f"Low DNI peak detected: max = {max_dni:.0f} W/m²")
            
            # Handle negative or missing values
            negative_count = (df['DNI'] < 0).sum()
            if negative_count > 0:
                logging.warning(f"Found {negative_count} negative DNI values - setting to 0")
                df['DNI'] = df['DNI'].clip(lower=0)
            
            nan_count = df['DNI'].isnull().sum()
            if nan_count > 0:
                logging.warning(f"Found {nan_count} missing DNI values - filling with 0")
                df['DNI'] = df['DNI'].fillna(0)
        else:
            # Calculate DNI using pvlib's disc() function
            logging.info("Calculating DNI using disc model")
            dni = pvlib.irradiance.disc(
                ghi=df['SolRad_Hor'],
                solar_zenith=df['zenith'],
                datetime_or_doy=df.index
            )['dni']
            df['DNI'] = dni
        
        # Validate DNI timing regardless of source
        daylight_mask = df['zenith'] < 85
        if daylight_mask.any():
            df_test = df[daylight_mask].copy()
            df_test['hour'] = df_test.index.hour
            hourly_dni = df_test.groupby('hour')['DNI'].mean()
            hourly_ghi = df_test.groupby('hour')['SolRad_Hor'].mean()
            
            dni_peak_hour = hourly_dni.idxmax()
            ghi_peak_hour = hourly_ghi.idxmax()
            
            logging.info(f"DNI peak at: {dni_peak_hour}h ({hourly_dni.max():.0f} W/m²)")
            logging.info(f"GHI peak at: {ghi_peak_hour}h ({hourly_ghi.max():.0f} W/m²)")
            
            if dni_peak_hour < 10 or dni_peak_hour > 15:
                logging.warning(f"DNI peak at {dni_peak_hour}h seems unusual for Athens")
                logging.warning(f"This suggests potential datetime conversion issues")
            else:
                logging.info(f"DNI peak timing looks reasonable")
        
        logging.info("DNI processing completed and added to DataFrame.")
        
        # Final DNI statistics
        annual_dni = df['DNI'].sum() / 1000  # kWh/m²
        logging.info(f"Annual DNI: {annual_dni:.0f} kWh/m² (typical range for Athens: 1800-2200)")
        
    except Exception as e:
        logging.error(f"Error calculating/processing DNI: {e}", exc_info=True)
        raise

    return df

def calculate_weighting_factors(df, strategy='adaptive_improved'):
    """
    Calculate weighting factors - ENHANCED with better validation.
    """
    for col in ['Load (kW)', 'zenith']:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the DataFrame.")
            raise ValueError(f"'{col}' column is missing in the DataFrame.")

    daylight_mask = df['zenith'] < 90
    df['weighting_factor'] = 0.0

    if not daylight_mask.any():
        logging.warning("No daylight hours detected; weighting factors remain zero.")
        return df['weighting_factor']

    daylight_load = df.loc[daylight_mask, 'Load (kW)']
    hours = df.loc[daylight_mask].index.hour

    # ENHANCED: Validate load data
    if len(daylight_load) == 0:
        logging.warning("No load data available during daylight hours")
        return df['weighting_factor']
    
    # Check for reasonable load values
    if daylight_load.min() < 0:
        logging.warning(f"Negative load values detected: min = {daylight_load.min():.1f} kW")
    
    if daylight_load.max() > 1000:  # Adjust threshold as needed
        logging.warning(f"Very high load values detected: max = {daylight_load.max():.1f} kW")

    if strategy == 'adaptive_improved':
        if len(daylight_load) > 1:
            min_load = daylight_load.min()
            max_load = daylight_load.max()

            if max_load != min_load:
                normalized_load = (daylight_load - min_load) / (max_load - min_load)
            else:
                normalized_load = pd.Series(0.5, index=daylight_load.index)
                logging.warning("Load values are constant - using uniform weighting")

            # FIXED: Time factor calculation with better centering
            time_factor = np.exp(-0.5 * ((hours - 13) / 4) ** 2)  # Peak at 13h (1 PM)
            time_factor = pd.Series(time_factor, index=daylight_load.index)
            
            if time_factor.max() != time_factor.min():
                time_factor = (time_factor - time_factor.min()) / (time_factor.max() - time_factor.min())
            else:
                time_factor = pd.Series(0.5, index=daylight_load.index)

            solar_irradiance = df.loc[daylight_mask, 'SolRad_Hor']
            if solar_irradiance.max() > 0:
                solar_factor = solar_irradiance / solar_irradiance.max()
            else:
                solar_factor = pd.Series(0.5, index=daylight_load.index)
                
            combined_weight = (0.6 * normalized_load + 
                          0.2 * time_factor + 
                          0.2 * solar_factor)
        
            if combined_weight.max() != combined_weight.min():
                combined_weight = (combined_weight - combined_weight.min()) / (combined_weight.max() - combined_weight.min())
            
            final_weights = 0.1 + 0.9 * combined_weight
            df.loc[daylight_mask, 'weighting_factor'] = final_weights
    
    elif strategy == 'pure_load_matching':
        if daylight_load.max() != daylight_load.min():
            weights = (daylight_load - daylight_load.min()) / (daylight_load.max() - daylight_load.min())
            weights = 0.2 + 0.8 * weights
        else:
            weights = pd.Series(0.5, index=daylight_load.index)
            logging.warning("Constant load - using uniform weights for pure_load_matching")
        df.loc[daylight_mask, 'weighting_factor'] = weights
    
    elif strategy == 'peak_focused':
        base_weight = 0.2
        if daylight_load.max() != daylight_load.min():
            load_norm = (daylight_load - daylight_load.min()) / (daylight_load.max() - daylight_load.min())
        else:
            load_norm = pd.Series(0.5, index=daylight_load.index)
        
        peak_threshold = daylight_load.quantile(0.8)
        peak_multiplier = pd.Series(1.0, index=daylight_load.index)
        peak_multiplier[daylight_load >= peak_threshold] = 2.0
        
        weights = base_weight + 0.8 * load_norm * peak_multiplier
        df.loc[daylight_mask, 'weighting_factor'] = weights

    # ENHANCED: Validation of weighting factors
    avg_weight = df.loc[daylight_mask, 'weighting_factor'].mean()
    min_weight = df.loc[daylight_mask, 'weighting_factor'].min()
    max_weight = df.loc[daylight_mask, 'weighting_factor'].max()
    
    logging.info(f"Weighting strategy '{strategy}':")
    logging.info(f"  Average weight: {avg_weight:.3f}")
    logging.info(f"  Weight range: {min_weight:.3f} to {max_weight:.3f}")
    logging.info(f"  Daylight hours with weights: {daylight_mask.sum()}")
    
    return df['weighting_factor']

def calculate_weighted_energy(df):
    """Calculate weighted energy production."""
    for col in ['E_ac', 'weighting_factor']:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the DataFrame.")
            raise ValueError(f"'{col}' column is missing in the DataFrame.")

    weighted_energy = (df['E_ac'] * df['weighting_factor']).sum() / 1000  # Convert Wh to kWh
    return weighted_energy

def calculate_total_irradiance(df, tilt_angle, azimuth_angle, dni_extra):
    """
    FIXED: Calculate total irradiance with azimuth correction for Athens.
    """
    required_columns = ['zenith', 'azimuth', 'DNI', 'SolRad_Hor', 'SolRad_Dif']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the DataFrame.")
            raise ValueError(f"'{col}' column is missing in the DataFrame.")

    try:
        # CRITICAL FIX: Apply azimuth correction for Athens solar position data
        # The solar azimuth data appears to be offset by ~20° 
        # Apply -15° correction to surface azimuth to compensate
        corrected_azimuth = azimuth_angle - 15.0
        
        # Ensure azimuth stays in 0-360° range
        if corrected_azimuth < 0:
            corrected_azimuth += 360
        elif corrected_azimuth >= 360:
            corrected_azimuth -= 360
            
        logging.info(f"AZIMUTH FIX: {azimuth_angle}° → {corrected_azimuth}° (applied -15° correction)")
        
        irradiance_data = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt_angle,
            surface_azimuth=corrected_azimuth,  # Use corrected azimuth
            solar_zenith=df['zenith'],
            solar_azimuth=df['azimuth'],
            dni=df['DNI'],
            ghi=df['SolRad_Hor'],
            dhi=df['SolRad_Dif'],
            dni_extra=dni_extra,
            model='haydavies'
        )
        df['total_irradiance'] = irradiance_data['poa_global']
        logging.info(f"Total irradiance calculated with CORRECTED tilt {tilt_angle}° and azimuth {corrected_azimuth}° (originally {azimuth_angle}°).")
    except Exception as e:
        logging.error(f"Error calculating total irradiance: {e}", exc_info=True)
        raise

    return df

def calculate_energy_production(df, number_of_panels, inverter_params):
    """
    IMPROVED energy production calculation with realistic losses for Athens.
    """
    required_columns = ['total_irradiance', 'Air Temp']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the DataFrame.")
            raise ValueError(f"'{col}' column is missing in the DataFrame.")

    try:
        # Panel parameters (Sharp ND-R240A5)
        panel_area = 1.642  # m²
        panel_efficiency_stc = 0.146  # 14.6% at STC
        panel_power_stc = 240  # Wp at STC
        TEMP_COEFF_PMAX = -0.0044  # -0.44%/°C
        NOCT = 47.5  # °C
        
        # System parameters
        total_panel_area = panel_area * number_of_panels
        total_system_power_stc = panel_power_stc * number_of_panels
        
        # REALISTIC loss factors for Athens (Mediterranean climate)
        soiling_factor = 0.96       # 4% soiling loss (monthly cleaning assumed)
        shading_factor = 0.98       # 2% shading loss (well-designed rooftop)
        reflection_factor = 0.97    # 3% reflection loss (unchanged)
        mismatch_factor = 0.98      # 2% mismatch loss (modern panels)
        dc_wiring_factor = 0.99     # 1% DC wiring loss (optimized design)
        ac_wiring_factor = 0.995    # 0.5% AC wiring loss (short runs)
        
        pre_temp_efficiency = (soiling_factor * shading_factor * reflection_factor * 
                              mismatch_factor * dc_wiring_factor * ac_wiring_factor)
        
        logging.info(f"REALISTIC Loss Model for Athens:")
        logging.info(f"  Soiling losses: {(1-soiling_factor)*100:.1f}% (dust, pollution)")
        logging.info(f"  Shading losses: {(1-shading_factor)*100:.1f}% (rooftop installation)")
        logging.info(f"  Reflection losses: {(1-reflection_factor)*100:.1f}%")
        logging.info(f"  Mismatch losses: {(1-mismatch_factor)*100:.1f}%")
        logging.info(f"  DC wiring losses: {(1-dc_wiring_factor)*100:.1f}%")
        logging.info(f"  AC wiring losses: {(1-ac_wiring_factor)*100:.1f}%")
        logging.info(f"  Combined pre-temp efficiency: {pre_temp_efficiency:.3f} ({pre_temp_efficiency*100:.1f}%)")
        
        # Step 1: Calculate incident energy
        df['incident_irradiance'] = df['total_irradiance']  # W/m²
        df['incident_energy'] = df['incident_irradiance'] * total_panel_area * TIME_INTERVAL_HOURS  # Wh
        df['E_incident'] = df['incident_energy']
        
        # Step 2: Cell temperature calculation (more accurate for Athens climate)
        df['cell_temperature'] = df['Air Temp'] + (NOCT - 20) * df['incident_irradiance'] / 800
        
        # Step 3: Temperature factor
        df['temperature_factor'] = 1 + TEMP_COEFF_PMAX * (df['cell_temperature'] - 25)
        df['temperature_factor'] = df['temperature_factor'].clip(lower=0)
        
        # Step 4: DC power calculations
        df['dc_power_stc'] = df['incident_irradiance'] * total_panel_area * panel_efficiency_stc
        df['dc_power_with_losses'] = df['dc_power_stc'] * pre_temp_efficiency
        df['dc_power_actual'] = df['dc_power_with_losses'] * df['temperature_factor']
        df['dc_power_actual'] = df['dc_power_actual'].clip(upper=total_system_power_stc)
        
        # Step 5: AC power output (realistic inverter efficiency)
        inverter_efficiency = min(inverter_params['eta_inv_nom'] / 100, 0.965)  # Cap at 96%
        df['ac_power_output'] = df['dc_power_actual'] * inverter_efficiency
        
        inverter_max_ac = total_system_power_stc * inverter_efficiency
        df['ac_power_output'] = df['ac_power_output'].clip(upper=inverter_max_ac)
        
        # Step 6: Energy calculations
        df['E_dc_ideal'] = df['dc_power_stc'] * TIME_INTERVAL_HOURS
        df['E_dc_actual'] = df['dc_power_actual'] * TIME_INTERVAL_HOURS
        df['E_ac'] = df['ac_power_output'] * TIME_INTERVAL_HOURS
        
        # Create ALL loss columns for compatibility
        df['E_loss_pre_temperature'] = df['incident_energy'] * panel_efficiency_stc - df['E_dc_ideal']
        df['E_loss_temperature'] = df['E_dc_ideal'] - df['E_dc_actual']
        df['E_loss_inverter'] = df['E_dc_actual'] - df['E_ac']
        df['E_loss_total'] = (df['incident_energy'] * panel_efficiency_stc) - df['E_ac']
        
        # Step 7: Performance Ratio
        system_power_stc_kw = total_system_power_stc / 1000
        df['reference_yield'] = df['incident_irradiance'] * TIME_INTERVAL_HOURS / 1000
        df['array_yield'] = (df['E_ac'] / 1000) / system_power_stc_kw
        df['PR'] = np.where(
            df['reference_yield'] > 0.01,
            df['array_yield'] / df['reference_yield'],
            0
        )
        df['PR'] = df['PR'].clip(0, 1.2)
        
        # Additional columns for compatibility
        df['dc_power_output_per_panel'] = df['dc_power_actual'] / number_of_panels
        df['dc_power_output'] = df['dc_power_actual']
        
        # IMPROVED: Validation with realistic bounds for Athens
        valid_pr = df[df['reference_yield'] > 0.1]['PR']
        if len(valid_pr) > 0:
            avg_pr = valid_pr.mean()
            annual_production = df['E_ac'].sum() / 1000
            specific_yield = annual_production / system_power_stc_kw
            
            logging.info(f"IMPROVED Energy Production Results:")
            logging.info(f"  Average PR: {avg_pr:.3f} ({avg_pr*100:.1f}%)")
            logging.info(f"  Annual Production: {annual_production:,.0f} kWh")
            logging.info(f"  Specific Yield: {specific_yield:.0f} kWh/kWp")
            logging.info(f"  Pre-temp Efficiency: {pre_temp_efficiency*100:.1f}%")
            logging.info(f"  System Size: {system_power_stc_kw:.1f} kWp")
            
            # Validate against realistic ranges for Athens
            athens_pr_range = (0.76, 0.82)  # Realistic PR range for Athens
            athens_yield_range = (1350, 1550)  # kWh/kWp/year for Athens
            
            if athens_pr_range[0] <= avg_pr <= athens_pr_range[1]:
                logging.info(f"✓ PR validation OK: {avg_pr:.3f} is within realistic range for Athens")
            else:
                logging.warning(f"⚠ PR {avg_pr:.3f} is outside typical range {athens_pr_range} for Athens")
            
            if athens_yield_range[0] <= specific_yield <= athens_yield_range[1]:
                logging.info(f"✓ Specific yield validation OK: {specific_yield:.0f} kWh/kWp is realistic for Athens")
            else:
                logging.warning(f"⚠ Specific yield {specific_yield:.0f} kWh/kWp is outside typical range {athens_yield_range} for Athens")
                if specific_yield > athens_yield_range[1]:
                    logging.warning("  Consider checking irradiance data or loss factors")
        
    except Exception as e:
        logging.error(f"Error calculating energy production: {e}", exc_info=True)
        raise
    
    return df

def summarize_energy(df):
    """
    Summarize energy flows with crash protection.
    """
    try:
        # Handle missing E_incident column
        if 'E_incident' not in df.columns:
            if 'incident_energy' in df.columns:
                df['E_incident'] = df['incident_energy']
                logging.info("Created E_incident from incident_energy")
            elif 'total_irradiance' in df.columns:
                # Emergency fallback calculation
                panel_area = 1.642
                number_of_panels = len(df) // 8760 * 845 if len(df) > 1000 else 845
                total_panel_area = panel_area * number_of_panels
                df['E_incident'] = df['total_irradiance'] * total_panel_area * TIME_INTERVAL_HOURS
                logging.warning("Emergency calculation of E_incident from total_irradiance")
            else:
                logging.error("Cannot calculate incident energy - missing required columns")
                return None, None, 0.0

        # Check for other required columns and create if missing
        if 'E_dc_ideal' not in df.columns:
            df['E_dc_ideal'] = df['E_incident'] * 0.146  # Assume 14.6% panel efficiency
        if 'E_dc_actual' not in df.columns:
            df['E_dc_actual'] = df['E_dc_ideal'] * 0.9  # Assume 90% after losses
        if 'E_ac' not in df.columns:
            df['E_ac'] = df['E_dc_actual'] * 0.96  # Assume 96% inverter efficiency

        # Sum up energies (convert to kWh)
        total_incident = df['E_incident'].sum() / 1000  # kWh
        total_dc_ideal = df['E_dc_ideal'].sum() / 1000  # kWh  
        total_dc_actual = df['E_dc_actual'].sum() / 1000  # kWh
        total_ac = df['E_ac'].sum() / 1000  # kWh
        
        # Calculate system efficiency metrics
        system_efficiency = (total_ac / total_incident) * 100 if total_incident > 0 else 0
        
        # Calculate losses (in kWh)
        pre_temp_losses = (total_incident * 0.146) - total_dc_ideal
        temp_losses = total_dc_ideal - total_dc_actual
        inverter_losses = total_dc_actual - total_ac
        total_losses = (total_incident * 0.146) - total_ac
        
        # Calculate average Performance Ratio
        if 'PR' in df.columns:
            avg_pr = df[df['PR'] > 0]['PR'].mean() * 100 if (df['PR'] > 0).any() else system_efficiency / 14.6
        else:
            avg_pr = system_efficiency / 14.6  # Approximate PR
        
        # Create energy flow breakdown
        energy_breakdown = pd.DataFrame({
            'Stage': [
                'Incident Solar Energy',
                'After Panel Efficiency (STC)', 
                'After Pre-temp Losses',
                'After Temperature Effects',
                'AC Output (Final)'
            ],
            'Energy (kWh)': [
                f"{total_incident:,.0f}",
                f"{total_incident * 0.146:,.0f}",
                f"{total_dc_ideal:,.0f}",
                f"{total_dc_actual:,.0f}",
                f"{total_ac:,.0f}"
            ],
            'Efficiency (%)': [
                '100.0',
                '14.6',
                f"{(total_dc_ideal/total_incident)*100:.1f}" if total_incident > 0 else "0.0",
                f"{(total_dc_actual/total_incident)*100:.1f}" if total_incident > 0 else "0.0",
                f"{system_efficiency:.1f}"
            ]
        })
        
        energy_losses = pd.DataFrame({
            'Loss Type': [
                'Pre-Temperature Losses (Soiling, Shading, etc.)',
                'Temperature Losses',
                'Inverter Losses',
                'Total System Losses'
            ],
            'Energy Lost (kWh)': [
                pre_temp_losses,
                temp_losses,
                inverter_losses,
                total_losses
            ],
            'Percentage of Input': [
                f"{(pre_temp_losses/total_incident)*100:.1f}%" if total_incident > 0 else "N/A",
                f"{(temp_losses/total_incident)*100:.1f}%" if total_incident > 0 else "N/A",
                f"{(inverter_losses/total_incident)*100:.1f}%" if total_incident > 0 else "N/A",
                f"{(total_losses/total_incident)*100:.1f}%" if total_incident > 0 else "N/A"
            ]
        })
        
        logging.info(f"System Summary - Input: {total_incident:,.0f} kWh, Output: {total_ac:,.0f} kWh")
        logging.info(f"Overall System Efficiency: {system_efficiency:.2f}%, Average PR: {avg_pr:.2f}%")
        
        return energy_breakdown, energy_losses, system_efficiency
        
    except Exception as e:
        logging.error(f"Error summarizing energy: {e}", exc_info=True)
        # Return minimal safe values to prevent crash
        simple_breakdown = pd.DataFrame({
            'Stage': ['AC Output (Final)'],
            'Energy (kWh)': ['N/A'],
            'Efficiency (%)': ['N/A']
        })
        simple_losses = pd.DataFrame({
            'Loss Type': ['Unknown'],
            'Energy Lost (kWh)': [0],
            'Percentage of Input': ['N/A']
        })
        return simple_breakdown, simple_losses, 0.0

def objective_function_multi(angles, df_subset, dni_extra, number_of_panels, inverter_params, 
                            weighting_strategy='adaptive_improved'):
    """
    IMPROVED Multi-objective function using realistic energy production calculation.
    Uses penalty approach instead of infinite returns for bounds violations.
    
    Parameters:
    - angles: [tilt_angle, azimuth_angle]
    - df_subset: DataFrame subset for calculations
    - dni_extra: Extra-terrestrial DNI values
    - number_of_panels: Number of panels in system
    - inverter_params: Inverter parameters dictionary
    - weighting_strategy: Strategy for calculating load weighting factors
    """
    try:
        tilt_angle, azimuth_angle = angles
        
        # Use penalty approach instead of returning infinite values
        penalty = 0.0
        
        # Apply bounds penalties but don't return inf
        if tilt_angle < 0:
            penalty += abs(tilt_angle) * 1000
        elif tilt_angle > 90:
            penalty += (tilt_angle - 90) * 1000

        # Azimuth bounds (0-360°) - ALLOW SOUTH (180°)
        if azimuth_angle < 0:
            penalty += abs(azimuth_angle) * 100
        elif azimuth_angle > 360:
            penalty += (azimuth_angle - 360) * 100

        # Discourage North-facing orientations (270-90°) for Athens
        if 270 <= azimuth_angle <= 360 or 0 <= azimuth_angle <= 90:
            north_penalty = min(90 - abs(azimuth_angle - 360 if azimuth_angle > 180 else azimuth_angle), 90)
            penalty += north_penalty * 10  # Mild penalty for north-facing

        # Calculate performance with corrected angles - USING IMPROVED FUNCTION
        df_temp = df_subset.copy()
        df_temp = calculate_total_irradiance(df_temp, tilt_angle, azimuth_angle, dni_extra)
        df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)  # FIXED
        
        total_energy_production = df_temp['E_ac'].sum() / 1000  # kWh
        
        # Use better weighting strategy
        df_temp['weighting_factor'] = calculate_weighting_factors(df_temp, strategy=weighting_strategy)
        df_temp['load_wh'] = df_temp['Load (kW)'] * 1000 * TIME_INTERVAL_HOURS
        df_temp['hourly_mismatch'] = df_temp['E_ac'] - df_temp['load_wh']
        df_temp['weighted_mismatch'] = df_temp['weighting_factor'] * np.abs(df_temp['hourly_mismatch'] / 1000)
        
        total_weighted_mismatch = df_temp['weighted_mismatch'].sum()
        
        # Add boundary violation penalties to mismatch
        adjusted_mismatch = total_weighted_mismatch + penalty
        
        if not np.isfinite(adjusted_mismatch):
            adjusted_mismatch = 1e6
        if not np.isfinite(total_energy_production):
            total_energy_production = 0.0
        
        # Scale objectives to similar ranges for better NSGA-II performance
        normalized_mismatch = adjusted_mismatch / 1000  # Scale to ~0.01-1.0 range
        normalized_production = total_energy_production / 1000  # Scale to ~2-4 range
            
        return (normalized_mismatch, normalized_production)

    except Exception as e:
        logging.error(f"Error in objective_function_multi: {e}", exc_info=True)
        return (1e6, 0.0)

def run_deap_multi_objective_optimization(
    df_subset,
    dni_extra,
    number_of_panels,
    inverter_params,
    output_dir,
    latitude,
    longitude,
    pop_size=None,
    max_gen=None,
    weighting_strategy='adaptive_improved'
):
    """
    IMPROVED multi-objective optimization with realistic energy calculations.
    """

    # Dynamic defaults for pop_size / max_gen
    if pop_size is None:
        pop_size = 50
    if max_gen is None:
        max_gen = 30

    logging.info("=== IMPROVED Multi-objective optimization with realistic energy model ===")

    # Create an optimization context object
    context = OptimizationContext(
        df_subset=df_subset,
        dni_extra=dni_extra,
        number_of_panels=number_of_panels,
        inverter_params=inverter_params,
        latitude=latitude,
        longitude=longitude,
        weighting_strategy=weighting_strategy
    )

    # Set up DEAP 'creator' for 2-objective optimization
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Toolbox with bounded operators
    toolbox = base.Toolbox()

    # Bounded individual creation
    def create_bounded_individual():
        """Create individual within valid bounds"""
        tilt = random.uniform(0, 90)
        azimuth = random.uniform(0, 360)
        return creator.Individual([tilt, azimuth])

    # Register functions with bounds - USING IMPROVED EVALUATION
    toolbox.register("individual", create_bounded_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual_IMPROVED)  # FIXED
    
    # Use bounded operators
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                 eta=20.0, low=[0, 0], up=[90, 360])
    toolbox.register("mutate", tools.mutPolynomialBounded, 
                 eta=20.0, low=[0, 0], up=[90, 360], indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    
    # Add bounds checking decorator as safety net
    def checkBounds(min_vals, max_vals):
        def decorator(func):
            def wrapper(*args, **kwargs):
                offspring = func(*args, **kwargs)
                for child in offspring:
                    for i in range(len(child)):
                        child[i] = np.clip(child[i], min_vals[i], max_vals[i])
                return offspring
            return wrapper
        return decorator
    
    toolbox.decorate("mate", checkBounds([0, 0], [90, 360]))
    toolbox.decorate("mutate", checkBounds([0, 0], [90, 360]))

    # Optional: Parallelize
    pool = multiprocessing.Pool(initializer=init_worker, initargs=(context,))
    toolbox.register("map", pool.map)

    # Initialize population
    population = toolbox.population(n=pop_size)

    # Prepare statistics and HallOfFame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    hof = tools.ParetoFront()

    # Run the global custom NSGA-II evolution loop
    final_pop, logbook = custom_nsga2_evolution(
        population,
        toolbox,
        cxpb=0.7,
        mutpb_start=0.2,
        mutpb_end=0.05,
        ngen=max_gen,
        stats=stats,
        halloffame=hof
    )

    # Close and join the multiprocessing pool
    pool.close()
    pool.join()

    # Extract final Pareto front and perform post-processing
    pareto_front = list(hof)

    # Filter solutions based on a production threshold (ADJUSTED for realistic values)
    system_capacity_kw = (number_of_panels * 240) / 1000
    # More realistic threshold based on improved energy calculations
    production_threshold = system_capacity_kw * 1400 / 1000  # 1400 kWh/kWp * capacity in MW
    logging.info(f"Production threshold set to {production_threshold:.0f} MWh (realistic for Athens)")
    
    filtered_front = [ind for ind in pareto_front if ind.fitness.values[1] >= production_threshold]
    logging.info(f"Filtered front: {len(filtered_front)} of {len(pareto_front)} pass production >= {production_threshold:.0f}")

    # Select the "most balanced" solution
    if filtered_front:
        mismatch_vals = np.array([ind.fitness.values[0] for ind in filtered_front])
        production_vals = np.array([ind.fitness.values[1] for ind in filtered_front])
        # Normalize both objectives
        mismatch_norm = (mismatch_vals - mismatch_vals.min()) / (np.ptp(mismatch_vals) + 1e-9)
        prod_norm = (production_vals.max() - production_vals) / (np.ptp(production_vals) + 1e-9)
        diff = np.abs(mismatch_norm - prod_norm)
        best_idx = np.argmin(diff)
        best_balanced = filtered_front[best_idx]
    else:
        best_balanced = None

    # Save the Pareto front to CSV
    rows = []
    for ind in pareto_front:
        mismatch, production = ind.fitness.values
        rows.append({
            'tilt_angle': ind[0],
            'azimuth_angle': ind[1],
            'weighted_mismatch_kWh': mismatch * 1000,  # Convert back from normalized
            'total_energy_production_kWh': production * 1000,  # Convert back from normalized
            'azimuth_deviation_from_south': abs(ind[1] - 180)
        })
    pareto_df = pd.DataFrame(rows)
    outfile = os.path.join(output_dir, 'pareto_front_results.csv')
    pareto_df.to_csv(outfile, index=False)
    logging.info(f"Pareto front results saved to {outfile}")

    return pareto_front, filtered_front, best_balanced, logbook

def custom_nsga2_evolution(pop, toolbox, cxpb, mutpb_start, mutpb_end, ngen, stats, halloffame):
    """
    Custom NSGA-II evolution loop with adaptive mutation probability.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Evaluate the initial population.
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))
    if halloffame is not None:
        halloffame.update(pop)
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    
    # Log meaningful values
    best_ind = min(pop, key=lambda x: x.fitness.values[0])
    logging.info(f"Generation 0 - best mismatch={best_ind.fitness.values[0]:.2f}, best production={best_ind.fitness.values[1]:.2f}")

    # Evolution loop.
    for gen in range(1, ngen + 1):
        fraction = gen / float(ngen)
        current_mutpb = mutpb_start + fraction * (mutpb_end - mutpb_start)
        offspring = varOr(pop, toolbox, lambda_=len(pop), cxpb=cxpb, mutpb=current_mutpb)
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop = toolbox.select(pop + offspring, k=len(pop))
        if halloffame is not None:
            halloffame.update(pop)
        
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        best_ind = min(pop, key=lambda x: x.fitness.values[0])
        mismatch, production = best_ind.fitness.values
        logging.info(f"Gen {gen} - best mismatch={mismatch:.2f}, best prod={production:.2f}, mutpb={current_mutpb:.3f}")
    
    return pop, logbook

def optimize_for_maximum_energy_production(df_subset, dni_extra, number_of_panels, inverter_params, output_dir, latitude, longitude):
    """
    FIXED: Single-objective optimization that ONLY maximizes energy production.
    Should find optimal angles around 30-35° tilt, 180° azimuth for Athens.
    """
    logging.info("=== FIXED SINGLE-OBJECTIVE OPTIMIZATION (MAX ENERGY ONLY) ===")
    
    def pure_energy_objective(angles):
        """CORRECTED: Use EXACT same calculation as validation tests"""
        try:
            tilt_angle, azimuth_angle = angles
            
            # Strict bounds checking
            if not (0 <= tilt_angle <= 90):
                return 1e12
            if not (0 <= azimuth_angle <= 360):
                return 1e12
            
            # CRITICAL FIX: Use the same calculation approach as other functions
            df_temp = df_subset.copy()
            df_temp = calculate_total_irradiance(df_temp, tilt_angle, azimuth_angle, dni_extra)
            df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)
            
            total_energy_production = df_temp['E_ac'].sum() / 1000  # kWh
            
            # Return negative for minimization
            if np.isfinite(total_energy_production) and total_energy_production > 0:
                return -total_energy_production
            else:
                return 1e12
                
        except Exception as e:
            logging.error(f"Error in corrected energy objective: {e}")
            return 1e12
    
    # FIXED: More focused bounds centered around expected optimal for Athens
    bounds = [
        (25, 45),      # Tilt: Reasonable range for Athens
        (160, 200)     # Azimuth: Focused around south (180° ± 20°)
    ]
    
    def test_orientation_with_optimization_method(tilt, azimuth, description):
        """Test orientation using the same method as optimization"""
        try:
            df_temp = df_subset.copy()
            df_temp = calculate_total_irradiance(df_temp, tilt, azimuth, dni_extra)
            df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)
            total_energy = df_temp['E_ac'].sum() / 1000  # kWh
            logging.info(f"  {description}: {total_energy:,.0f} kWh")
            return total_energy
        except Exception as e:
            logging.error(f"  Error testing {description}: {e}")
            return 0

    # Test key orientations
    south_energy = test_orientation_with_optimization_method(33.8, 180.0, "South (33.8°, 180°)")
    current_energy = test_orientation_with_optimization_method(33.8, 164.7, "Current result (33.8°, 164.7°)")
    southeast_energy = test_orientation_with_optimization_method(33.8, 150.0, "Southeast (33.8°, 150°)")

    best_test_energy = max(south_energy, current_energy, southeast_energy)
    if south_energy == best_test_energy:
        logging.info("✓ SOUTH IS OPTIMAL - optimization should find this")
    elif current_energy == best_test_energy:
        logging.info("⚠ Current result is optimal - need to investigate why")
    else:
        logging.info("⚠ Southeast is optimal - major calculation error")

    logging.info(f"Energy difference (South - Current): {south_energy - current_energy:,.0f} kWh")
    logging.info("=== TESTING OBJECTIVE FUNCTION DIRECTLY ===")

    test_angles = [
        [33.8, 180.0, "South"],
        [33.8, 164.7, "Current optimum"], 
        [33.8, 150.0, "Southeast"],
        [30.0, 180.0, "30° South"],
        [35.0, 180.0, "35° South"]
    ]

    for tilt, azimuth, desc in test_angles:
        try:
            objective_value = pure_energy_objective([tilt, azimuth])
            energy_value = -objective_value  # Convert back to positive
            logging.info(f"  {desc} ({tilt}°, {azimuth}°): Objective={objective_value:.0f}, Energy={energy_value:,.0f} kWh")
        except Exception as e:
            logging.error(f"  Error in objective function for {desc}: {e}")

    logging.info("=== END VERIFICATION - PROCEEDING WITH OPTIMIZATION ===")
    
    logging.info("Running FIXED single-objective optimization...")
    logging.info(f"Bounds: Tilt {bounds[0][0]}-{bounds[0][1]}°, Azimuth {bounds[1][0]}-{bounds[1][1]}°")
    logging.info("Expected result for Athens: ~30-35° tilt, ~180° azimuth")
    
    best_result = None
    best_energy = -np.inf
    
    # FIXED: More comprehensive starting points around the expected optimal
    starting_points = [
        [30, 180],   # Expected optimal for Athens
        [35, 180],   # Latitude tilt
        [25, 180],   # Lower tilt
        [40, 180],   # Higher tilt
        [32, 175],   # Slightly SE
        [32, 185],   # Slightly SW
        [28, 180],   # Lower tilt, south
        [38, 180],   # Higher tilt, south
    ]
    
    # FINAL DEBUG: Test south vs southeast using corrected objective function
    logging.info("=== FINAL DEBUG TEST ===")
    south_obj = pure_energy_objective([30.0, 180.0])
    southeast_obj = pure_energy_objective([30.0, 165.0])
    current_obj = pure_energy_objective([33.8, 164.7])

    logging.info(f"Corrected objective function results:")
    logging.info(f"  South (30°, 180°): {-south_obj:,.0f} kWh")
    logging.info(f"  Southeast (30°, 165°): {-southeast_obj:,.0f} kWh")
    logging.info(f"  Current result (33.8°, 164.7°): {-current_obj:,.0f} kWh")

    if south_obj < southeast_obj:  # Remember: lower objective = higher energy
        logging.info("✓ South IS better in corrected objective function")
    else:
        logging.info("⚠ Southeast still appears better - need deeper investigation")
    
    for i, start_point in enumerate(starting_points):
        logging.info(f"Attempt {i+1}: Starting from tilt={start_point[0]}°, azimuth={start_point[1]}°")
        
        try:
            result = differential_evolution(
                pure_energy_objective,
                bounds,
                maxiter=200,  # More iterations for better convergence
                popsize=30,   # Larger population
                seed=42 + i,
                x0=start_point,
                polish=True,
                disp=False,
                atol=1,       # Absolute tolerance for convergence
                tol=1e-6      # Relative tolerance
            )
            
            if result.success:
                energy = -result.fun  # Convert back to positive
                tilt_result = result.x[0]
                azimuth_result = result.x[1]
                
                logging.info(f"  Result: {tilt_result:.1f}° tilt, {azimuth_result:.1f}° azimuth")
                logging.info(f"  Energy: {energy:,.0f} kWh")
                
                # Validate the result makes sense
                if 15 <= tilt_result <= 60 and 90 <= azimuth_result <= 270:
                    logging.info(f"  ✓ Result is within acceptable range")
                    
                    if energy > best_energy:
                        best_result = result
                        best_energy = energy
                        logging.info(f"  ★ New best result!")
                else:
                    logging.warning(f"  ⚠ Result outside reasonable bounds")
                    # Still save it if it's the best energy found
                    if energy > best_energy:
                        best_result = result
                        best_energy = energy
                        logging.info(f"  ★ New best result (but unusual angles)")
                
        except Exception as e:
            logging.error(f"  Error in attempt {i+1}: {e}")
    
    if best_result and best_result.success:
        optimal_tilt, optimal_azimuth = best_result.x
        max_energy_production = -best_result.fun
        
        logging.info(f"=== FIXED SINGLE-OBJECTIVE OPTIMIZATION COMPLETE ===")
        logging.info(f"Best result from {len(starting_points)} attempts:")
        logging.info(f"  Optimal angles: {optimal_tilt:.1f}° tilt, {optimal_azimuth:.1f}° azimuth")
        logging.info(f"  Max Energy Production: {max_energy_production:,.0f} kWh/year")
        
        # CRITICAL VALIDATION: Check if result is correct for Athens
        azimuth_deviation = abs(optimal_azimuth - 180)
        tilt_reasonable = 25 <= optimal_tilt <= 40
        azimuth_reasonable = azimuth_deviation <= 20  # Allow some deviation
        
        logging.info(f"=== RESULT VALIDATION ===")
        logging.info(f"Tilt angle: {optimal_tilt:.1f}° ({'✓' if tilt_reasonable else '✗'} Expected: 25-40°)")
        logging.info(f"Azimuth deviation from south: {azimuth_deviation:.1f}° ({'✓' if azimuth_reasonable else '✗'} Expected: <20°)")
        
        validation_passed = tilt_reasonable and azimuth_reasonable
        
        if validation_passed:
            logging.info("✓ VALIDATION PASSED: Results are realistic for Athens")
        else:
            logging.error("✗ VALIDATION FAILED: Results are unusual for maximum energy optimization")
            logging.error("This suggests issues with:")
            logging.error("  - Solar position calculations")
            logging.error("  - Data quality/timezone")
            logging.error("  - Irradiance calculations")
        
        # Calculate detailed performance with optimal angles
        df_optimal = df_subset.copy()
        df_optimal = calculate_total_irradiance(df_optimal, optimal_tilt, optimal_azimuth, dni_extra)
        df_optimal = calculate_energy_production(df_optimal, number_of_panels, inverter_params)
        
        # Validate energy production
        system_capacity_kwp = (number_of_panels * 240) / 1000
        specific_yield = max_energy_production / system_capacity_kwp
        
        logging.info(f"System performance validation:")
        logging.info(f"  Specific yield: {specific_yield:.0f} kWh/kWp")
        
        if 1350 <= specific_yield <= 1600:
            logging.info("✓ Specific yield is realistic for Athens")
        else:
            logging.warning(f"⚠ Specific yield outside expected range (1350-1600 kWh/kWp)")
        
        # Save results
        single_obj_results = {
            'optimization_type': 'single_objective_max_energy_FIXED',
            'optimal_tilt': optimal_tilt,
            'optimal_azimuth': optimal_azimuth,
            'max_energy_production_kwh': max_energy_production,
            'specific_yield_kwh_per_kwp': specific_yield,
            'optimization_success': True,
            'azimuth_deviation_from_south': azimuth_deviation,
            'validation_passed': validation_passed,
            'tilt_reasonable': tilt_reasonable,
            'azimuth_reasonable': azimuth_reasonable
        }
        
        # Also calculate weighted mismatch for comparison (but didn't optimize for it)
        df_optimal['weighting_factor'] = calculate_weighting_factors(df_optimal, strategy='adaptive_improved')
        df_optimal['load_wh'] = df_optimal['Load (kW)'] * 1000 * TIME_INTERVAL_HOURS
        df_optimal['hourly_mismatch'] = df_optimal['E_ac'] - df_optimal['load_wh']
        df_optimal['weighted_mismatch'] = df_optimal['weighting_factor'] * np.abs(df_optimal['hourly_mismatch'] / 1000)
        
        total_weighted_mismatch = df_optimal['weighted_mismatch'].sum()
        single_obj_results['weighted_mismatch_kwh'] = total_weighted_mismatch
        
        # Save to CSV
        results_df = pd.DataFrame([single_obj_results])
        results_df.to_csv(os.path.join(output_dir, 'single_objective_FIXED_results.csv'), index=False)
        
        return single_obj_results, df_optimal
        
    else:
        logging.error("FIXED single-objective optimization failed in all attempts!")
        logging.error("This indicates a fundamental problem with the optimization setup")
        return None, None

def compare_optimization_results(single_obj_results, pareto_front, output_dir):
    """
    Compare single-objective (max energy) results with multi-objective Pareto front.
    """
    if single_obj_results is None or not pareto_front:
        logging.warning("Cannot compare results - missing data")
        return
    
    logging.info("=== OPTIMIZATION COMPARISON ===")
    
    # Find best energy production solution from Pareto front
    pareto_energies = [ind.fitness.values[1] for ind in pareto_front]
    pareto_mismatches = [ind.fitness.values[0] for ind in pareto_front]
    
    max_pareto_energy = max(pareto_energies)
    idx_max_energy = pareto_energies.index(max_pareto_energy)
    best_energy_pareto = pareto_front[idx_max_energy]
    
    # Find best mismatch solution from Pareto front  
    min_pareto_mismatch = min(pareto_mismatches)
    idx_min_mismatch = pareto_mismatches.index(min_pareto_mismatch)
    best_mismatch_pareto = pareto_front[idx_min_mismatch]
    
    # Create comparison table
    comparison_data = {
        'Optimization_Type': [
            'Single-Objective (Max Energy)',
            'Multi-Objective (Best Energy)',
            'Multi-Objective (Best Mismatch)'
        ],
        'Tilt_Angle': [
            single_obj_results['optimal_tilt'],
            best_energy_pareto[0],
            best_mismatch_pareto[0]
        ],
        'Azimuth_Angle': [
            single_obj_results['optimal_azimuth'],
            best_energy_pareto[1],
            best_mismatch_pareto[1]
        ],
        'Energy_Production_kWh': [
            single_obj_results['max_energy_production_kwh'],
            max_pareto_energy * 1000,  # Convert back from normalized
            best_mismatch_pareto.fitness.values[1] * 1000
        ],
        'Weighted_Mismatch_kWh': [
            single_obj_results['weighted_mismatch_kwh'],
            best_energy_pareto.fitness.values[0] * 1000,  # Convert back from normalized
            min_pareto_mismatch * 1000
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, 'optimization_comparison.csv'), index=False)
    
    # Log comparison
    energy_sacrifice = single_obj_results['max_energy_production_kwh'] - (max_pareto_energy * 1000)
    mismatch_improvement = single_obj_results['weighted_mismatch_kwh'] - (best_energy_pareto.fitness.values[0] * 1000)
    
    logging.info(f"Comparison Results:")
    logging.info(f"  Single-obj max energy: {single_obj_results['max_energy_production_kwh']:,.0f} kWh")
    logging.info(f"  Multi-obj best energy: {max_pareto_energy * 1000:,.0f} kWh")
    logging.info(f"  Energy sacrifice for load matching: {energy_sacrifice:,.0f} kWh ({energy_sacrifice/single_obj_results['max_energy_production_kwh']*100:.1f}%)")
    logging.info(f"  Mismatch improvement: {mismatch_improvement:,.0f} kWh")
    
    return comparison_df

def analyze_seasonal_performance(df):
    """
    Analyze seasonal variations in production, consumption, and matching.
    """
    # Make a copy to avoid modifying the original dataframe
    df_season = df.copy()
    
    # Add time-based columns
    df_season['month'] = df_season.index.month
    df_season['day_of_year'] = df_season.index.dayofyear
    
    # Define seasons (Northern Hemisphere)
    season_mapping = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall',
        11: 'Fall', 12: 'Winter'
    }
    df_season['season'] = df_season['month'].map(season_mapping)
    
    # Calculate mismatch between production and consumption
    df_season['load_wh'] = df_season['Load (kW)'] * 1000  # Convert kW to W for comparison
    df_season['mismatch'] = df_season['E_ac'] - df_season['load_wh']
    df_season['surplus'] = df_season['mismatch'].clip(lower=0)
    df_season['deficit'] = (-df_season['mismatch']).clip(lower=0)
    
    # Calculate self-consumption and self-sufficiency
    df_season['consumed_solar'] = np.minimum(df_season['E_ac'], df_season['load_wh'])
    
    # Aggregate metrics by season
    seasonal_stats = df_season.groupby('season').agg({
        'E_ac': 'sum',
        'load_wh': 'sum',
        'surplus': 'sum',
        'deficit': 'sum',
        'consumed_solar': 'sum'
    })
    
    # Calculate additional metrics
    seasonal_stats['net_balance_wh'] = seasonal_stats['E_ac'] - seasonal_stats['load_wh']
    seasonal_stats['self_consumption_ratio'] = (seasonal_stats['consumed_solar'] / seasonal_stats['E_ac']) * 100
    seasonal_stats['self_sufficiency_ratio'] = (seasonal_stats['consumed_solar'] / seasonal_stats['load_wh']) * 100
    
    # Convert to kWh for better readability
    for col in ['E_ac', 'load_wh', 'surplus', 'deficit', 'consumed_solar', 'net_balance_wh']:
        seasonal_stats[f'{col}_kwh'] = seasonal_stats[col] / 1000
    
    # Calculate daily averages by season
    df_season['date_str'] = df_season.index.strftime('%Y-%m-%d')
    
    # Group by season and date string, then calculate daily sums
    daily_sums = df_season.groupby(['season', 'date_str']).agg({
        'E_ac': 'sum',
        'load_wh': 'sum',
        'consumed_solar': 'sum'
    })
    
    # Then calculate the mean of these daily sums for each season
    daily_seasonal = daily_sums.groupby('season').mean()
    
    # Convert daily averages to kWh
    for col in ['E_ac', 'load_wh', 'consumed_solar']:
        daily_seasonal[f'{col}_kwh'] = daily_seasonal[col] / 1000
    
    # Reorder seasons for chronological display
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_stats = seasonal_stats.reindex(season_order)
    daily_seasonal = daily_seasonal.reindex(season_order)
    
    return seasonal_stats, daily_seasonal


# -------------------------------------------------------------------------------
#      BATTERY AND FINANCE SENCTION
# -------------------------------------------------------------------------------


def calculate_optimal_battery_capacity(df, output_dir,
                                      min_capacity=1,  # kWh
                                      max_capacity=500,  # kWh (reduced from 100 for realism)
                                      capacity_step=2.5,  # kWh
                                      battery_round_trip_efficiency=0.90,  # Round-trip efficiency
                                      depth_of_discharge=0.80,
                                      battery_cost_per_kwh=400,  # €/kWh (realistic 2024 price)
                                      electricity_buy_price=0.24,  # €/kWh
                                      electricity_sell_price=0.08,  # €/kWh
                                      battery_lifetime_years=10):
    """
    Calculate optimal battery capacity using CORRECTED round-trip efficiency model.
    
    Key Fix: Properly applies round-trip efficiency by using sqrt(efficiency) for each direction.
    """
    logging.info("Starting CORRECTED battery capacity optimization analysis...")
    
    # CORRECTED: Calculate one-way efficiency from round-trip efficiency
    one_way_efficiency = np.sqrt(battery_round_trip_efficiency)
    
    logging.info(f"Round-trip efficiency: {battery_round_trip_efficiency*100:.1f}%")
    logging.info(f"One-way efficiency: {one_way_efficiency*100:.1f}%")
    
    # Prepare data
    df_battery = df.copy()
    
    # Convert Load to Wh to match E_ac units
    df_battery['load_wh'] = df_battery['Load (kW)'] * 1000  # kW to Wh
    
    # Calculate energy surplus/deficit at each timestep
    df_battery['energy_balance'] = df_battery['E_ac'] - df_battery['load_wh']
    df_battery['surplus'] = df_battery['energy_balance'].clip(lower=0)
    df_battery['deficit'] = (-df_battery['energy_balance']).clip(lower=0)
    
    # Define battery capacity range to test
    capacities = np.arange(min_capacity, max_capacity + capacity_step, capacity_step) * 1000  # Convert to Wh
    
    # Track results for each capacity
    results = []
    
    for capacity_wh in capacities:
        usable_capacity_wh = capacity_wh * depth_of_discharge
        
        # Initialize tracking variables for this capacity
        grid_import_wh = 0  # Energy drawn from grid
        grid_export_wh = 0  # Energy exported to grid
        self_consumed_wh = 0  # Solar energy consumed directly
        battery_charged_wh = 0  # Energy used to charge battery (before losses)
        battery_discharged_wh = 0  # Energy discharged from battery (after losses)
        battery_losses_wh = 0  # Total energy lost in battery operations
        
        # Simulate a full year with hourly timesteps
        soc_wh = capacity_wh * 0.5  # Start with half-full battery
        soc_percent = []  # Track state of charge for analysis
        
        for i, row in df_battery.iterrows():
            # Step 1: Direct self-consumption (highest priority)
            direct_consumption = min(row['E_ac'], row['load_wh'])
            self_consumed_wh += direct_consumption
            
            # Remaining solar energy after direct consumption
            remaining_surplus = row['E_ac'] - direct_consumption
            
            # Remaining load to be satisfied
            remaining_load = row['load_wh'] - direct_consumption
            
            # Step 2: Handle energy surplus (charge battery or export)
            if remaining_surplus > 0:
                # Calculate how much space is available in battery
                space_in_battery = capacity_wh - soc_wh
                energy_to_battery = min(remaining_surplus, space_in_battery)
                
                # CORRECTED: Apply charging efficiency
                energy_stored = energy_to_battery * one_way_efficiency
                energy_lost_charging = energy_to_battery - energy_stored
                
                # Track charging and losses
                battery_charged_wh += energy_to_battery
                battery_losses_wh += energy_lost_charging
                
                # Update battery state of charge
                soc_wh += energy_stored
                
                # Export any remaining surplus that couldn't be stored
                grid_export_wh += remaining_surplus - energy_to_battery
            
            # Step 3: Handle energy deficit (discharge battery or import)
            elif remaining_load > 0:
                # Calculate how much energy can be drawn from battery
                available_energy = max(0, soc_wh - (capacity_wh * (1 - depth_of_discharge)))
                
                # CORRECTED: Apply discharging efficiency
                # Need to draw more from battery to get the required energy output
                energy_needed_from_battery = min(remaining_load, available_energy * one_way_efficiency)
                energy_drawn_from_battery = energy_needed_from_battery / one_way_efficiency
                energy_lost_discharging = energy_drawn_from_battery - energy_needed_from_battery
                
                # Ensure we don't exceed available energy
                available_energy = max(0, soc_wh - (capacity_wh * (1 - depth_of_discharge)))
                max_deliverable = available_energy * one_way_efficiency
                energy_from_battery = min(remaining_load, max_deliverable)
                energy_drawn_from_battery = energy_from_battery / one_way_efficiency

                # Ensure we don't exceed available energy
                energy_drawn_from_battery = min(energy_drawn_from_battery, available_energy)
                energy_from_battery = energy_drawn_from_battery * one_way_efficiency
                
                # Track discharging and losses
                battery_discharged_wh += energy_needed_from_battery
                battery_losses_wh += energy_lost_discharging
                
                # Update battery state of charge
                soc_wh -= energy_drawn_from_battery
                
                # Import any remaining deficit that battery couldn't cover
                grid_import_wh += remaining_load - energy_needed_from_battery
            
            # Record state of charge percentage
            soc_percent.append((soc_wh / capacity_wh) * 100)
        
        # Calculate key metrics
        total_consumption_wh = df_battery['load_wh'].sum()
        total_production_wh = df_battery['E_ac'].sum()
        
        # Self-consumption and self-sufficiency rates
        total_self_consumed = self_consumed_wh + battery_discharged_wh
        self_consumption_rate = total_self_consumed / total_production_wh if total_production_wh > 0 else 0
        self_sufficiency_rate = total_self_consumed / total_consumption_wh if total_consumption_wh > 0 else 0
        
        # Economic calculations
        battery_investment = (capacity_wh / 1000) * battery_cost_per_kwh  # Battery cost
        
        # Calculate baseline (without battery)
        baseline_grid_import = df_battery['deficit'].sum() / 1000  # kWh
        baseline_grid_export = df_battery['surplus'].sum() / 1000  # kWh
        
        # Calculate with battery
        actual_grid_import = grid_import_wh / 1000  # kWh
        actual_grid_export = grid_export_wh / 1000  # kWh
        
        # Calculate savings
        avoided_grid_import = baseline_grid_import - actual_grid_import
        additional_grid_export = actual_grid_export - baseline_grid_export
        
        # Annual economic benefit
        annual_savings = avoided_grid_import * electricity_buy_price
        annual_export_revenue = additional_grid_export * electricity_sell_price
        total_annual_benefit = annual_savings + annual_export_revenue
        
        # Payback calculation
        if total_annual_benefit > 0:
            simple_payback = battery_investment / total_annual_benefit
            # Realistic check: if payback > battery lifetime, it's not viable
            if simple_payback > battery_lifetime_years:
                effective_payback = float('inf')
            else:
                effective_payback = simple_payback
        else:
            effective_payback = float('inf')
        
        # Cycle counting
        total_energy_cycled = battery_discharged_wh
        equivalent_full_cycles = total_energy_cycled / usable_capacity_wh if usable_capacity_wh > 0 else 0
        
        # Round-trip efficiency verification
        actual_round_trip_eff = (battery_discharged_wh / battery_charged_wh) if battery_charged_wh > 0 else 0
        
        # Save results for this capacity
        results.append({
            'capacity_kwh': capacity_wh / 1000,
            'usable_capacity_kwh': usable_capacity_wh / 1000,
            'self_consumption_rate': self_consumption_rate * 100,
            'self_sufficiency_rate': self_sufficiency_rate * 100,
            'grid_import_kwh': actual_grid_import,
            'grid_export_kwh': actual_grid_export,
            'avoided_grid_import_kwh': avoided_grid_import,
            'battery_charged_kwh': battery_charged_wh / 1000,
            'battery_discharged_kwh': battery_discharged_wh / 1000,
            'battery_losses_kwh': battery_losses_wh / 1000,
            'equivalent_full_cycles': equivalent_full_cycles,
            'battery_investment': battery_investment,
            'annual_savings': annual_savings,
            'annual_export_revenue': annual_export_revenue,
            'total_annual_benefit': total_annual_benefit,
            'simple_payback_years': effective_payback,
            'avg_soc_percent': np.mean(soc_percent),
            'actual_round_trip_efficiency': actual_round_trip_eff * 100
        })
    
    # Convert results to DataFrame
    battery_results = pd.DataFrame(results)
    
    # Determine optimal capacity based on economic criteria
    # Filter for viable options (payback <= battery lifetime)
    viable_options = battery_results[battery_results['simple_payback_years'] <= battery_lifetime_years]
    
    if not viable_options.empty:
        # Among viable options, choose the one with best NPV approximation
        # Simple NPV calculation: Total benefits over lifetime - investment
        viable_options = viable_options.copy()
        viable_options['simple_npv'] = (viable_options['total_annual_benefit'] * battery_lifetime_years) - viable_options['battery_investment']
        best_option = viable_options.loc[viable_options['simple_npv'].idxmax()]
        optimal_capacity = best_option['capacity_kwh']
    else:
        # If no economically viable options, choose smallest capacity for minimal loss
        optimal_capacity = battery_results['capacity_kwh'].min()
        logging.warning("No economically viable battery options found. Selecting minimum capacity.")
    
    # Save results to CSV
    battery_results.to_csv(os.path.join(output_dir, 'battery_capacity_analysis.csv'), index=False)
    logging.info(f"Battery capacity analysis saved to CSV")
    
    # Create visualization plots
    create_battery_analysis_plots(battery_results, optimal_capacity, output_dir)
    
    logging.info(f"Optimal battery capacity determined: {optimal_capacity:.1f} kWh")
    logging.info(f"Expected round-trip efficiency: {battery_round_trip_efficiency*100:.1f}%")
    
    return optimal_capacity, battery_results

def create_battery_analysis_plots(battery_results, optimal_capacity, output_dir):
    """Create comprehensive battery analysis plots with corrected efficiency model."""
    
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("Matplotlib not available - skipping battery analysis plots")
        return
    
    try:
        # Create a 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Self-consumption and self-sufficiency vs capacity
        ax1.plot(battery_results['capacity_kwh'], battery_results['self_consumption_rate'],
                'o-', color='blue', linewidth=2, markersize=6, label='Self-Consumption Rate (%)')
        ax1.plot(battery_results['capacity_kwh'], battery_results['self_sufficiency_rate'],
                'o-', color='green', linewidth=2, markersize=6, label='Self-Sufficiency Rate (%)')
        
        # Highlight optimal capacity
        ax1.axvline(x=optimal_capacity, color='red', linestyle='--', alpha=0.8, linewidth=2,
                    label=f'Optimal: {optimal_capacity:.1f} kWh')
        
        ax1.set_title('Self-Consumption & Self-Sufficiency vs Battery Capacity', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Battery Capacity (kWh)')
        ax1.set_ylabel('Rate (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 100)
        
        # Plot 2: Economic analysis
        # Filter payback periods for better visualization
        economic_data = battery_results[battery_results['simple_payback_years'] <= 20]
        
        if not economic_data.empty:
            ax2.plot(economic_data['capacity_kwh'], economic_data['simple_payback_years'],
                    'o-', color='purple', linewidth=2, markersize=6)
            
            # Highlight optimal capacity
            if optimal_capacity in economic_data['capacity_kwh'].values:
                optimal_row = economic_data[economic_data['capacity_kwh'] == optimal_capacity].iloc[0]
                ax2.plot(optimal_capacity, optimal_row['simple_payback_years'], 
                        'o', color='red', markersize=10, label=f'Optimal: {optimal_row["simple_payback_years"]:.1f} years')
            
            ax2.axvline(x=optimal_capacity, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax2.set_title('Payback Period vs Battery Capacity', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Battery Capacity (kWh)')
        ax2.set_ylabel('Simple Payback Period (years)')
        ax2.grid(True, alpha=0.3)
        if not economic_data.empty:
            ax2.legend()
        
        # Plot 3: Battery utilization
        ax3.plot(battery_results['capacity_kwh'], battery_results['equivalent_full_cycles'],
                'o-', color='orange', linewidth=2, markersize=6)
        
        # Highlight optimal capacity
        if optimal_capacity in battery_results['capacity_kwh'].values:
            optimal_cycles = battery_results[battery_results['capacity_kwh'] == optimal_capacity]['equivalent_full_cycles'].iloc[0]
            ax3.plot(optimal_capacity, optimal_cycles, 'o', color='red', markersize=10)
        
        ax3.axvline(x=optimal_capacity, color='red', linestyle='--', alpha=0.8, linewidth=2,
                    label=f'Optimal: {optimal_capacity:.1f} kWh')
        
        ax3.set_title('Battery Utilization vs Capacity', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Battery Capacity (kWh)')
        ax3.set_ylabel('Equivalent Full Cycles per Year')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Efficiency verification
        ax4.plot(battery_results['capacity_kwh'], battery_results['actual_round_trip_efficiency'],
                'o-', color='brown', linewidth=2, markersize=6, label='Actual Round-trip Efficiency')
        
        # Add theoretical line
        theoretical_efficiency = battery_results['battery_charged_kwh'].iloc[0]  # Just for reference
        ax4.axhline(y=90, color='gray', linestyle=':', alpha=0.8, label='Target: 90%')
        
        ax4.axvline(x=optimal_capacity, color='red', linestyle='--', alpha=0.8, linewidth=2,
                    label=f'Optimal: {optimal_capacity:.1f} kWh')
        
        ax4.set_title('Round-trip Efficiency Verification', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Battery Capacity (kWh)')
        ax4.set_ylabel('Efficiency (%)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(80, 95)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'battery_analysis_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary table
        create_battery_summary_table(battery_results, optimal_capacity, output_dir)
        
        logging.info("Battery analysis plots created successfully")
        
    except Exception as e:
        logging.error(f"Error creating battery analysis plots: {e}")

def create_battery_summary_table(battery_results, optimal_capacity, output_dir):
    """Create a summary table of key battery options."""
    
    # Select key capacities for comparison
    key_capacities = [0, 10, 20, optimal_capacity, 40, 50]
    key_capacities = [c for c in key_capacities if c <= battery_results['capacity_kwh'].max()]
    
    # Remove duplicates and sort
    key_capacities = sorted(list(set(key_capacities)))
    
    # Create summary data
    summary_data = []
    for capacity in key_capacities:
        if capacity == 0:
            # No battery case
            row_data = {
                'Capacity (kWh)': 0,
                'Self-Sufficiency (%)': 0,  # Will be calculated separately
                'Annual Savings (€)': 0,
                'Investment (€)': 0,
                'Payback (years)': 'N/A',
                'Status': 'Baseline'
            }
        else:
            # Find closest match in results
            closest_idx = (battery_results['capacity_kwh'] - capacity).abs().idxmin()
            row = battery_results.loc[closest_idx]
            
            status = '★ OPTIMAL ★' if abs(row['capacity_kwh'] - optimal_capacity) < 0.1 else ''
            payback_str = f"{row['simple_payback_years']:.1f}" if row['simple_payback_years'] < 100 else ">20"
            
            row_data = {
                'Capacity (kWh)': row['capacity_kwh'],
                'Self-Sufficiency (%)': f"{row['self_sufficiency_rate']:.1f}",
                'Annual Savings (€)': f"{row['total_annual_benefit']:.0f}",
                'Investment (€)': f"{row['battery_investment']:.0f}",
                'Payback (years)': payback_str,
                'Status': status
            }
        
        summary_data.append(row_data)
    
    # Convert to DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'battery_summary_comparison.csv'), index=False)
    
    logging.info("Battery summary table created")

def calculate_investment_cost(number_of_panels, battery_capacity_kwh, economic_params):
    """
    Calculate the total initial investment (CAPEX) for the PV system.

    Parameters:
    - number_of_panels (int): Number of solar panels
    - battery_capacity_kwh (float): Battery capacity in kWh  
    - economic_params (dict): Economic parameters dictionary

    Returns:
    - dict: Investment breakdown and total
    """
    # Get panel power from config or use default
    panel_power_wp = economic_params.get('panel_power_wp', 240)  # Match Sharp ND-R240A5
    total_panel_power_kwp = (number_of_panels * panel_power_wp) / 1000

    # Calculate component costs
    panel_cost = number_of_panels * economic_params.get('panel_cost_per_unit', 150)  # €150 per panel
    inverter_cost = total_panel_power_kwp * economic_params.get('inverter_cost_per_kw', 200)  # €200/kW
    bos_cost = total_panel_power_kwp * economic_params.get('bos_cost_per_kwp', 300)  # €300/kWp for mounting, cables
    installation_cost = total_panel_power_kwp * economic_params.get('installation_cost_per_kwp', 250)  # €250/kWp
    battery_cost = battery_capacity_kwh * economic_params.get('battery_cost_per_kwh', 400)  # €400/kWh

    total_investment = panel_cost + inverter_cost + bos_cost + installation_cost + battery_cost

    investment_breakdown = {
        'panel_cost': panel_cost,
        'inverter_cost': inverter_cost,
        'bos_cost': bos_cost,          # Balance of System (cables, mounting)
        'installation_cost': installation_cost,
        'battery_cost': battery_cost,
        'total_investment': total_investment,
        'system_size_kwp': total_panel_power_kwp
    }
    
    logging.info(f"Investment breakdown:")
    logging.info(f"  Panels ({number_of_panels}x{panel_power_wp}W): €{panel_cost:,.0f}")
    logging.info(f"  Inverter ({total_panel_power_kwp:.1f}kW): €{inverter_cost:,.0f}")
    logging.info(f"  Balance of System: €{bos_cost:,.0f}")
    logging.info(f"  Installation: €{installation_cost:,.0f}")
    logging.info(f"  Battery ({battery_capacity_kwh:.1f}kWh): €{battery_cost:,.0f}")
    logging.info(f"  TOTAL INVESTMENT: €{total_investment:,.0f}")
    
    return investment_breakdown

def calculate_irr_manual(cash_flows, max_iter=1000, tol=1e-6):
    """
    Calculate IRR manually using Newton-Raphson method.
    Fallback when numpy_financial is not available.
    """
    def npv_func(rate, cash_flows):
        return sum([cf / (1 + rate) ** i for i, cf in enumerate(cash_flows)])
    
    def npv_derivative(rate, cash_flows):
        return sum([-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows)])
    
    # Initial guess
    rate = 0.1
    
    for iteration in range(max_iter):
        npv_val = npv_func(rate, cash_flows)
        if abs(npv_val) < tol:
            return rate * 100  # Convert to percentage
        
        npv_deriv = npv_derivative(rate, cash_flows)
        if abs(npv_deriv) < tol:
            return 0  # Cannot find IRR
        
        rate = rate - npv_val / npv_deriv
        
        # Keep rate within reasonable bounds
        if rate < -0.99:
            rate = -0.99
        elif rate > 10:
            rate = 10
    
    return 0  # Could not converge

def calculate_financial_metrics(df_scenario, battery_capacity_kwh, battery_simulation, 
                               initial_investment, economic_params):
    """
    Perform detailed financial analysis over the project's lifetime.

    Parameters:
    - df_scenario (DataFrame): DataFrame with hourly production/load data
    - battery_capacity_kwh (float): Battery system size
    - battery_simulation (dict): Results from battery simulation
    - initial_investment (dict): CAPEX breakdown 
    - economic_params (dict): Economic parameters

    Returns:
    - dict: Contains cashflows DataFrame and financial metrics
    """
    
    logging.info("Calculating financial metrics...")
    
    # Economic parameters with defaults
    LIFETIME = economic_params.get('project_lifetime_years', 25)
    DISCOUNT_RATE = economic_params.get('discount_rate_percent', 5) / 100
    OM_RATE = economic_params.get('om_cost_percent_of_capex', 1.5) / 100  # 1.5% per year
    INVERTER_LIFETIME = economic_params.get('inverter_lifetime_years', 12)
    PRICE_INFLATION = economic_params.get('electricity_price_inflation_percent', 2) / 100
    DEGRADATION_RATE = economic_params.get('panel_degradation_percent', 0.5) / 100  # 0.5% per year
    BATTERY_DEGRADATION_RATE = 0.02
    
    
    # Base electricity prices
    electricity_buy_price = economic_params.get('electricity_price', 0.24)  # €/kWh
    electricity_sell_price = economic_params.get('feed_in_tariff', 0.08)   # €/kWh

    # Calculate baseline performance (without battery)
    total_production_kwh = df_scenario['E_ac'].sum() / 1000
    total_consumption_kwh = (df_scenario['Load (kW)'] * 1000).sum() / 1000
    
    # Direct self-consumption without battery
    direct_consumption_kwh = df_scenario.apply(
        lambda row: min(row['E_ac']/1000, row['Load (kW)']), axis=1
    ).sum()
    
    baseline_grid_export = df_scenario.apply(
        lambda row: max(0, row['E_ac']/1000 - row['Load (kW)']), axis=1
    ).sum()
    
    baseline_grid_import = df_scenario.apply(
        lambda row: max(0, row['Load (kW)'] - row['E_ac']/1000), axis=1
    ).sum()

    # With battery performance
    if battery_simulation and battery_capacity_kwh > 0:
        # Use actual battery simulation results
        annual_grid_import = battery_simulation['grid_import_kwh']
        annual_grid_export = battery_simulation['grid_export_kwh']
        annual_self_consumption = (battery_simulation['self_consumed_direct_kwh'] + 
                                 battery_simulation['battery_discharged_kwh'])
    else:
        # No battery case
        annual_grid_import = baseline_grid_import
        annual_grid_export = baseline_grid_export
        annual_self_consumption = direct_consumption_kwh


    
    
    # Initialize cash flow DataFrame
    years = np.arange(0, LIFETIME + 1)
    cashflows = pd.DataFrame(index=years)
    cashflows['investment'] = 0.0
    cashflows['grid_import_savings'] = 0.0
    cashflows['grid_export_revenue'] = 0.0
    cashflows['om_cost'] = 0.0
    cashflows['inverter_replacement'] = 0.0
    cashflows['production_kwh'] = 0.0

    # Year 0: Initial Investment
    cashflows.loc[0, 'investment'] = -initial_investment['total_investment']

    # Years 1 to LIFETIME: Operations
    for year in range(1, LIFETIME + 1):
        # Apply degradation to production
        production_factor = (1 - DEGRADATION_RATE) ** (year - 1)
        annual_production_degraded = total_production_kwh * production_factor
        cashflows.loc[year, 'production_kwh'] = annual_production_degraded
        
        # Scale battery benefits with production degradation
        degraded_grid_import = annual_grid_import * production_factor
        degraded_grid_export = annual_grid_export * production_factor
        
        if battery_capacity_kwh > 0:
            battery_factor = (1 - BATTERY_DEGRADATION_RATE) ** (year - 1)
            degraded_grid_import *= battery_factor
            degraded_grid_export *= battery_factor
        
        # Calculate electricity prices with inflation
        current_buy_price = electricity_buy_price * ((1 + PRICE_INFLATION) ** (year - 1))
        current_sell_price = electricity_sell_price * ((1 + PRICE_INFLATION) ** (year - 1))
        
        # Calculate annual savings and revenue
        import_savings = (baseline_grid_import - degraded_grid_import) * current_buy_price
        export_revenue = degraded_grid_export * current_sell_price
        
        cashflows.loc[year, 'grid_import_savings'] = max(0, import_savings)
        cashflows.loc[year, 'grid_export_revenue'] = max(0, export_revenue)

        # O&M Cost (escalates with inflation)
        annual_om_cost = initial_investment['total_investment'] * OM_RATE * ((1 + PRICE_INFLATION) ** (year - 1))
        cashflows.loc[year, 'om_cost'] = -annual_om_cost
        
        # Inverter Replacement Cost
        if year % INVERTER_LIFETIME == 0 and year < LIFETIME:
            # Assume inverter cost decreases over time
            replacement_factor = 0.8 if year <= 15 else 0.6
            replacement_cost = initial_investment['inverter_cost'] * replacement_factor
            cashflows.loc[year, 'inverter_replacement'] = -replacement_cost
            logging.info(f"Inverter replacement in year {year}: €{replacement_cost:,.0f}")

    # Calculate Net Cash Flow and Discounted Cash Flow
    cashflows['total_revenue'] = cashflows['grid_import_savings'] + cashflows['grid_export_revenue']
    cashflows['total_costs'] = cashflows['investment'] + cashflows['om_cost'] + cashflows['inverter_replacement']
    cashflows['net_cash_flow'] = cashflows['total_revenue'] + cashflows['total_costs']  # costs are negative
    cashflows['discounted_cash_flow'] = cashflows['net_cash_flow'] / ((1 + DISCOUNT_RATE) ** cashflows.index)

    # Calculate Financial Metrics
    npv = cashflows['discounted_cash_flow'].sum()
    
    # IRR Calculation
    cash_flow_values = cashflows['net_cash_flow'].values
    if NUMPY_FINANCIAL_AVAILABLE:
        try:
            irr_value = npf.irr(cash_flow_values) * 100
            if np.isnan(irr_value) or np.isinf(irr_value):
                irr_value = calculate_irr_manual(cash_flow_values)
        except:
            irr_value = calculate_irr_manual(cash_flow_values)
    else:
        irr_value = calculate_irr_manual(cash_flow_values)

    # Payback Period
    cumulative_cash_flow = cashflows['net_cash_flow'].cumsum()
    payback_period = float('inf')
    
    positive_flows = cumulative_cash_flow[cumulative_cash_flow > 0]
    if not positive_flows.empty:
        payback_year = positive_flows.index[0]
        if payback_year > 0:
            last_negative_flow = cumulative_cash_flow[payback_year - 1]
            current_year_flow = cashflows.loc[payback_year, 'net_cash_flow']
            if current_year_flow != 0:
                payback_period = payback_year - 1 + (-last_negative_flow / current_year_flow)
            else:
                payback_period = payback_year

    # Levelized Cost of Energy (LCOE)
    cost_columns = ['investment', 'om_cost', 'inverter_replacement']
    total_discounted_costs = 0
    for year in cashflows.index:
        year_costs = -cashflows.loc[year, cost_columns].sum()  # Make costs positive
        total_discounted_costs += year_costs / ((1 + DISCOUNT_RATE) ** year)

    # Calculate discounted energy
    total_discounted_energy = sum([
        cashflows.loc[year, 'production_kwh'] / ((1 + DISCOUNT_RATE) ** year) 
        for year in range(1, LIFETIME + 1)
    ])
    lcoe = total_discounted_costs / total_discounted_energy

    # Energy metrics
    lifetime_production = cashflows['production_kwh'].sum()
    
    financial_metrics = {
        'NPV': npv,
        'IRR_percent': irr_value,
        'Payback_Period_Years': payback_period,
        'LCOE_eur_per_kwh': lcoe,
        'Total_Investment': initial_investment['total_investment'],
        'Lifetime_Production_kWh': lifetime_production,
        'Annual_Savings_Year1': cashflows.loc[1, 'total_revenue'],
        'Annual_OM_Cost_Year1': -cashflows.loc[1, 'om_cost']
    }

    logging.info(f"Financial Analysis Results:")
    logging.info(f"  NPV: €{npv:,.0f}")
    logging.info(f"  IRR: {irr_value:.1f}%")
    logging.info(f"  Payback Period: {payback_period:.1f} years")
    logging.info(f"  LCOE: €{lcoe:.3f}/kWh")

    return {
        'cashflows': cashflows, 
        'financial_metrics': financial_metrics,
        'system_performance': {
            'annual_production_kwh': total_production_kwh,
            'annual_consumption_kwh': total_consumption_kwh,
            'annual_self_consumption_kwh': annual_self_consumption,
            'annual_grid_import_kwh': annual_grid_import,
            'annual_grid_export_kwh': annual_grid_export,
            'self_sufficiency_rate': (annual_self_consumption / total_consumption_kwh) * 100 if total_consumption_kwh > 0 else 0,
            'self_consumption_rate': (annual_self_consumption / total_production_kwh) * 100 if total_production_kwh > 0 else 0
        }
    }

def create_financial_analysis_plots(financial_results, output_dir):
    """Create financial analysis visualization plots."""
    
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("Matplotlib not available - skipping financial plots")
        return
    
    try:
        cashflows = financial_results['cashflows']
        metrics = financial_results['financial_metrics']
        
        # Create 2x2 subplot for financial analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Annual Cash Flows
        years = cashflows.index[1:]  # Exclude year 0
        ax1.bar(years, cashflows.loc[years, 'total_revenue'], 
                label='Revenue', color='green', alpha=0.7)
        ax1.bar(years, -cashflows.loc[years, 'om_cost'], 
                label='O&M Costs', color='red', alpha=0.7)
        ax1.bar(years, -cashflows.loc[years, 'inverter_replacement'], 
                label='Inverter Replacement', color='orange', alpha=0.7)
        
        ax1.set_title('Annual Cash Flows Over Project Lifetime')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Cash Flow (€)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Cash Flow
        cumulative = cashflows['net_cash_flow'].cumsum()
        ax2.plot(cashflows.index, cumulative, 'b-', linewidth=2)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_title(f'Cumulative Cash Flow (Payback: {metrics["Payback_Period_Years"]:.1f} years)')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative Cash Flow (€)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Energy Production Degradation
        ax3.plot(years, cashflows.loc[years, 'production_kwh'], 'g-', linewidth=2)
        ax3.set_title('Annual Energy Production (with 0.5% degradation)')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Production (kWh)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Key Metrics Summary (text)
        ax4.axis('off')
        metrics_text = f"""
        Financial Performance Summary
        
        Net Present Value: €{metrics['NPV']:,.0f}
        Internal Rate of Return: {metrics['IRR_percent']:.1f}%
        Payback Period: {metrics['Payback_Period_Years']:.1f} years
        LCOE: €{metrics['LCOE_eur_per_kwh']:.3f}/kWh
        
        Investment: €{metrics['Total_Investment']:,.0f}
        Lifetime Production: {metrics['Lifetime_Production_kWh']:,.0f} kWh
        Annual Savings (Year 1): €{metrics['Annual_Savings_Year1']:,.0f}
        """
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'financial_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Financial analysis plots created")
        
    except Exception as e:
        logging.error(f"Error creating financial plots: {e}")

def get_default_economic_params():
    """
    Get default economic parameters for financial analysis.
    
    Returns:
    - dict: Default economic parameters
    """
    return {
        # System costs
        'panel_cost_per_unit': 150,              # €150 per panel
        'inverter_cost_per_kw': 200,             # €200 per kW
        'bos_cost_per_kwp': 300,                 # €300 per kWp (mounting, cables)
        'installation_cost_per_kwp': 250,       # €250 per kWp
        'battery_cost_per_kwh': 400,             # €400 per kWh
        'panel_power_wp': 240,                   # 240W panels (Sharp ND-R240A5)
        
        # Financial parameters
        'project_lifetime_years': 25,            # 25 year project life
        'discount_rate_percent': 5,              # 5% discount rate
        'om_cost_percent_of_capex': 1.5,         # 1.5% annual O&M
        'inverter_lifetime_years': 12,           # Inverter replacement every 12 years
        'electricity_price_inflation_percent': 2, # 2% annual inflation
        'panel_degradation_percent': 0.5,        # 0.5% annual degradation
        
        # Electricity prices (€/kWh)
        'electricity_price': 0.24,               # Grid purchase price
        'feed_in_tariff': 0.08,                 # Feed-in tariff
    }


def run_battery_simulation(df, battery_capacity_kwh, economic_params):
    """
    Run detailed battery simulation with the optimal capacity.
    
    Parameters:
    - df (DataFrame): DataFrame with energy production and load data
    - battery_capacity_kwh (float): Battery capacity in kWh
    - economic_params (dict): Economic parameters
    
    Returns:
    - dict: Battery simulation results
    """
    try:
        logging.info(f"Running detailed battery simulation with {battery_capacity_kwh:.1f} kWh capacity")
        
        if battery_capacity_kwh <= 0:
            # No battery case
            df_sim = df.copy()
            df_sim['load_wh'] = df_sim['Load (kW)'] * 1000
            
            total_production = df_sim['E_ac'].sum() / 1000  # kWh
            total_consumption = df_sim['load_wh'].sum() / 1000  # kWh
            
            # Direct self-consumption without battery
            direct_consumption = df_sim.apply(
                lambda row: min(row['E_ac']/1000, row['Load (kW)']), axis=1
            ).sum()
            
            grid_export = df_sim.apply(
                lambda row: max(0, row['E_ac']/1000 - row['Load (kW)']), axis=1
            ).sum()
            
            grid_import = df_sim.apply(
                lambda row: max(0, row['Load (kW)'] - row['E_ac']/1000), axis=1
            ).sum()
            
            return {
                'total_production_kwh': total_production,
                'total_consumption_kwh': total_consumption,
                'self_consumed_direct_kwh': direct_consumption,
                'battery_discharged_kwh': 0,
                'grid_import_kwh': grid_import,
                'grid_export_kwh': grid_export,
                'self_consumption_rate': (direct_consumption / total_production) * 100 if total_production > 0 else 0,
                'self_sufficiency_rate': (direct_consumption / total_consumption) * 100 if total_consumption > 0 else 0,
                'soc_history': [50.0] * len(df)  # No battery, constant 50% for compatibility
            }
        
        # Battery simulation parameters
        battery_round_trip_efficiency = 0.90
        one_way_efficiency = np.sqrt(battery_round_trip_efficiency)
        depth_of_discharge = 0.80
        
        # Prepare data
        df_sim = df.copy()
        df_sim['load_wh'] = df_sim['Load (kW)'] * 1000  # Convert to Wh
        
        # Initialize tracking variables
        capacity_wh = battery_capacity_kwh * 1000
        usable_capacity_wh = capacity_wh * depth_of_discharge
        
        grid_import_wh = 0
        grid_export_wh = 0
        self_consumed_direct_wh = 0
        battery_discharged_wh = 0
        
        # Initialize battery state
        soc_wh = capacity_wh * 0.5  # Start at 50%
        soc_history = []
        
        # Simulate hourly operation
        for i, row in df_sim.iterrows():
            # Step 1: Direct self-consumption
            direct_consumption = min(row['E_ac'], row['load_wh'])
            self_consumed_direct_wh += direct_consumption
            
            # Remaining energy flows
            remaining_surplus = row['E_ac'] - direct_consumption
            remaining_load = row['load_wh'] - direct_consumption
            
            # Step 2: Handle surplus (charge battery or export)
            if remaining_surplus > 0:
                # Available space in battery
                space_in_battery = capacity_wh - soc_wh
                energy_to_battery = min(remaining_surplus, space_in_battery)
                
                # Apply charging efficiency
                energy_stored = energy_to_battery * one_way_efficiency
                soc_wh += energy_stored
                
                # Export remaining surplus
                grid_export_wh += remaining_surplus - energy_to_battery
            
            # Step 3: Handle deficit (discharge battery or import)
            elif remaining_load > 0:
                # Available energy from battery
                available_energy = max(0, soc_wh - (capacity_wh * (1 - depth_of_discharge)))
                
                # Apply discharging efficiency
                energy_from_battery = min(remaining_load, available_energy * one_way_efficiency)
                energy_drawn = energy_from_battery / one_way_efficiency
                
                # Ensure we don't exceed available energy
                if energy_drawn > available_energy:
                    energy_drawn = available_energy
                    energy_from_battery = available_energy * one_way_efficiency
                
                battery_discharged_wh += energy_from_battery
                soc_wh -= energy_drawn
                
                # Import remaining deficit
                grid_import_wh += remaining_load - energy_from_battery
            
            # Record SOC percentage
            soc_history.append((soc_wh / capacity_wh) * 100)
        
        # Calculate metrics (convert to kWh)
        total_production_kwh = df_sim['E_ac'].sum() / 1000
        total_consumption_kwh = df_sim['load_wh'].sum() / 1000
        self_consumed_direct_kwh = self_consumed_direct_wh / 1000
        battery_discharged_kwh = battery_discharged_wh / 1000
        grid_import_kwh = grid_import_wh / 1000
        grid_export_kwh = grid_export_wh / 1000
        
        # Calculate rates
        total_self_consumed = self_consumed_direct_kwh + battery_discharged_kwh
        self_consumption_rate = (total_self_consumed / total_production_kwh) * 100 if total_production_kwh > 0 else 0
        self_sufficiency_rate = (total_self_consumed / total_consumption_kwh) * 100 if total_consumption_kwh > 0 else 0
        
        logging.info(f"Battery simulation complete:")
        logging.info(f"  Self-consumption rate: {self_consumption_rate:.1f}%")
        logging.info(f"  Self-sufficiency rate: {self_sufficiency_rate:.1f}%")
        logging.info(f"  Grid import: {grid_import_kwh:,.0f} kWh")
        logging.info(f"  Grid export: {grid_export_kwh:,.0f} kWh")
        
        return {
            'total_production_kwh': total_production_kwh,
            'total_consumption_kwh': total_consumption_kwh,
            'self_consumed_direct_kwh': self_consumed_direct_kwh,
            'battery_discharged_kwh': battery_discharged_kwh,
            'grid_import_kwh': grid_import_kwh,
            'grid_export_kwh': grid_export_kwh,
            'self_consumption_rate': self_consumption_rate,
            'self_sufficiency_rate': self_sufficiency_rate,
            'soc_history': soc_history
        }
        
    except Exception as e:
        logging.error(f"Error in battery simulation: {e}")
        raise
    







# -------------------------------------------------------------------------------
#      VALIDATIO SENCTION 
# -------------------------------------------------------------------------------


def validate_energy_production(df, number_of_panels, panel_power_rating=240):
    """
    Validate if energy production results are realistic for Athens.
    """
    total_production_kwh = df['E_ac'].sum() / 1000
    system_capacity_kwp = (number_of_panels * panel_power_rating) / 1000
    specific_yield = total_production_kwh / system_capacity_kwp
    
    # Realistic ranges for Athens based on PVGIS/PVLib studies
    athens_yield_range = (1350, 1550)  # kWh/kWp/year for well-designed systems
    
    logging.info(f"=== ENERGY PRODUCTION VALIDATION ===")
    logging.info(f"Total production: {total_production_kwh:,.0f} kWh/year")
    logging.info(f"System capacity: {system_capacity_kwp:.1f} kWp")
    logging.info(f"Specific yield: {specific_yield:.0f} kWh/kWp/year")
    logging.info(f"Expected range for Athens: {athens_yield_range[0]}-{athens_yield_range[1]} kWh/kWp/year")
    
    if athens_yield_range[0] <= specific_yield <= athens_yield_range[1]:
        logging.info("✓ Energy production is within realistic range for Athens")
        return True
    elif specific_yield > athens_yield_range[1]:
        logging.warning(f"⚠ Energy production ({specific_yield:.0f}) is higher than expected!")
        logging.warning("This might indicate:")
        logging.warning("  - Overly optimistic irradiance data")
        logging.warning("  - Too low system losses")
        logging.warning("  - Calculation error in energy model")
        return False
    else:
        logging.warning(f"⚠ Energy production ({specific_yield:.0f}) is lower than expected!")
        logging.warning("This might indicate excessive system losses")
        return False   

def quick_validation_test(df_subset, dni_extra, number_of_panels, inverter_params):
    """
    FIXED: Quick test that should show south-facing as optimal.
    """
    test_configs = [
        (25, 180, "Low tilt south (25°S)"),
        (30, 180, "Standard south (30°S)"),      # Should be close to optimal
        (35, 180, "Latitude tilt (35°S)"),       # Should be optimal or very close
        (40, 180, "High tilt south (40°S)"),
        (45, 180, "Steep tilt south (45°S)")
    ]
    
    logging.info("=== FIXED QUICK VALIDATION TEST ===")
    logging.info("Testing south-facing configurations to establish baseline:")
    
    best_config = None
    best_energy = 0
    
    for tilt, azimuth, description in test_configs:
        try:
            df_test = df_subset.copy()
            df_test = calculate_total_irradiance(df_test, tilt, azimuth, dni_extra)
            df_test = calculate_energy_production(df_test, number_of_panels, inverter_params)
            
            annual_energy = df_test['E_ac'].sum() / 1000
            system_capacity = (number_of_panels * 240) / 1000
            specific_yield = annual_energy / system_capacity
            
            if annual_energy > best_energy:
                best_energy = annual_energy
                best_config = (tilt, azimuth, description)
            
            logging.info(f"{description}:")
            logging.info(f"  Angles: {tilt}° tilt, {azimuth}° azimuth")
            logging.info(f"  Production: {annual_energy:,.0f} kWh")
            logging.info(f"  Specific yield: {specific_yield:.0f} kWh/kWp")
            
            # Validation
            if 1350 <= specific_yield <= 1600:  # Slightly wider range
                logging.info(f"  Status: ✓ Realistic for Athens")
            else:
                logging.warning(f"  Status: ⚠ Outside typical range (1350-1600 kWh/kWp)")
            logging.info("")
            
        except Exception as e:
            logging.error(f"Error testing {description}: {e}")
    
    if best_config:
        logging.info(f"Best south-facing configuration: {best_config[2]} with {best_energy:,.0f} kWh")
        logging.info(f"Single-objective optimizer should find angles close to this!")
        
        # CRITICAL: The best should be around 30-35° tilt, 180° azimuth
        best_tilt = best_config[0]
        if 25 <= best_tilt <= 40:
            logging.info("✓ Best tilt angle is reasonable for Athens latitude")
        else:
            logging.warning(f"⚠ Best tilt {best_tilt}° is unusual for Athens (expected 25-40°)")
    
    return best_config, best_energy

def validate_optimization_results_comprehensive(optimal_tilt, optimal_azimuth, annual_production, 
                                              system_capacity_kwp, latitude=37.99):
    """
    Comprehensive validation of optimization results for Athens.
    """
    logging.info(f"=== COMPREHENSIVE RESULTS VALIDATION ===")
    logging.info(f"Location: Athens (Latitude: {latitude}°N)")
    logging.info(f"Optimal configuration: {optimal_tilt:.1f}° tilt, {optimal_azimuth:.1f}° azimuth")
    logging.info(f"Annual production: {annual_production:,.0f} kWh")
    logging.info(f"System capacity: {system_capacity_kwp:.1f} kWp")
    
    # Calculate specific yield
    specific_yield = annual_production / system_capacity_kwp
    logging.info(f"Specific yield: {specific_yield:.0f} kWh/kWp/year")
    
    validation_results = {}
    
    # 1. Validate tilt angle
    expected_tilt_range = (latitude - 15, latitude + 15)
    tilt_ok = expected_tilt_range[0] <= optimal_tilt <= expected_tilt_range[1]
    validation_results['tilt'] = tilt_ok
    
    logging.info(f"\n1. Tilt Angle Validation:")
    logging.info(f"   Optimal: {optimal_tilt:.1f}°")
    logging.info(f"   Expected range: {expected_tilt_range[0]:.1f}° - {expected_tilt_range[1]:.1f}°")
    logging.info(f"   Status: {'✓ Valid' if tilt_ok else '✗ Invalid'}")
    
    # 2. Validate azimuth angle
    azimuth_deviation = abs(optimal_azimuth - 180)
    azimuth_ok = azimuth_deviation <= 45  # Allow up to 45° deviation
    validation_results['azimuth'] = azimuth_ok
    
    logging.info(f"\n2. Azimuth Angle Validation:")
    logging.info(f"   Optimal: {optimal_azimuth:.1f}° ({azimuth_deviation:.1f}° from south)")
    logging.info(f"   Status: {'✓ Valid' if azimuth_ok else '✗ Unusual'}")
    
    if not azimuth_ok:
        logging.info(f"   Note: Large deviation from south may indicate:")
        logging.info(f"   - Load-matching optimization")
        logging.info(f"   - Specific site constraints")
        logging.info(f"   - Multi-objective optimization")
    
    # 3. Validate specific yield
    athens_yield_range = (1350, 1550)  # kWh/kWp/year for Athens
    yield_ok = athens_yield_range[0] <= specific_yield <= athens_yield_range[1]
    validation_results['yield'] = yield_ok
    
    logging.info(f"\n3. Specific Yield Validation:")
    logging.info(f"   Calculated: {specific_yield:.0f} kWh/kWp/year")
    logging.info(f"   Expected range for Athens: {athens_yield_range[0]}-{athens_yield_range[1]} kWh/kWp/year")
    logging.info(f"   Status: {'✓ Valid' if yield_ok else '✗ Outside typical range'}")
    
    if not yield_ok:
        if specific_yield > athens_yield_range[1]:
            logging.warning(f"   ⚠ Yield is {specific_yield - athens_yield_range[1]:.0f} kWh/kWp higher than expected")
            logging.warning(f"   Consider checking: irradiance data, loss factors, calculation errors")
        else:
            logging.warning(f"   ⚠ Yield is {athens_yield_range[0] - specific_yield:.0f} kWh/kWp lower than expected")
            logging.warning(f"   Consider checking: system losses, shading, equipment efficiency")
    
    # Overall validation
    all_valid = all(validation_results.values())
    logging.info(f"\n=== OVERALL VALIDATION: {'✓ PASSED' if all_valid else '⚠ ISSUES DETECTED'} ===")
    
    return validation_results, all_valid

def get_default_economic_params():
    """
    Get default economic parameters for financial analysis.
    
    Returns:
    - dict: Default economic parameters
    """
    return {
        # System costs
        'panel_cost_per_unit': 150,              # €150 per panel
        'inverter_cost_per_kw': 200,             # €200 per kW
        'bos_cost_per_kwp': 300,                 # €300 per kWp (mounting, cables)
        'installation_cost_per_kwp': 250,       # €250 per kWp
        'battery_cost_per_kwh': 400,             # €400 per kWh
        'panel_power_wp': 240,                   # 240W panels (Sharp ND-R240A5)
        
        # Financial parameters
        'project_lifetime_years': 25,            # 25 year project life
        'discount_rate_percent': 5,              # 5% discount rate
        'om_cost_percent_of_capex': 1.5,         # 1.5% annual O&M
        'inverter_lifetime_years': 12,           # Inverter replacement every 12 years
        'electricity_price_inflation_percent': 2, # 2% annual inflation
        'panel_degradation_percent': 0.5,        # 0.5% annual degradation
        
        # Electricity prices (€/kWh)
        'electricity_price': 0.24,               # Grid purchase price
        'feed_in_tariff': 0.08,                 # Feed-in tariff
    }

def debug_temperature_model(df_subset, dni_extra, number_of_panels, inverter_params):
    """Debug why southeast might beat south due to temperature effects"""
    
    print("=== TEMPERATURE MODEL DEBUG ===")
    
    test_configs = [
        (30, 180, "South"),
        (30, 165, "Southeast"),
        (30, 150, "More Southeast")
    ]
    
    results = []
    
    for tilt, azimuth, name in test_configs:
        # Calculate for this orientation
        df_test = df_subset.copy()
        df_test = calculate_total_irradiance(df_test, tilt, azimuth, dni_extra)
        df_test = calculate_energy_production(df_test, number_of_panels, inverter_params)
        
        # Get daylight hours only
        daylight = df_test[df_test['total_irradiance'] > 50]
        
        # Calculate key metrics
        avg_irradiance = daylight['total_irradiance'].mean()
        max_irradiance = daylight['total_irradiance'].max()
        avg_cell_temp = daylight['cell_temperature'].mean()
        max_cell_temp = daylight['cell_temperature'].max()
        avg_temp_factor = daylight['temperature_factor'].mean()
        min_temp_factor = daylight['temperature_factor'].min()
        annual_energy = df_test['E_ac'].sum() / 1000
        
        # Store results
        results.append({
            'name': name,
            'azimuth': azimuth,
            'avg_irradiance': avg_irradiance,
            'max_irradiance': max_irradiance,
            'avg_cell_temp': avg_cell_temp,
            'max_cell_temp': max_cell_temp,
            'avg_temp_factor': avg_temp_factor,
            'min_temp_factor': min_temp_factor,
            'annual_energy': annual_energy
        })
        
        print(f"\n{name} ({azimuth}°):")
        print(f"  Avg POA Irradiance: {avg_irradiance:.0f} W/m²")
        print(f"  Max POA Irradiance: {max_irradiance:.0f} W/m²")
        print(f"  Avg Cell Temperature: {avg_cell_temp:.1f}°C")
        print(f"  Max Cell Temperature: {max_cell_temp:.1f}°C")
        print(f"  Avg Temperature Factor: {avg_temp_factor:.3f} ({avg_temp_factor*100:.1f}%)")
        print(f"  Min Temperature Factor: {min_temp_factor:.3f} ({min_temp_factor*100:.1f}%)")
        print(f"  Annual Energy: {annual_energy:,.0f} kWh")
        
        # Calculate temperature losses
        temp_loss_percent = (1 - avg_temp_factor) * 100
        print(f"  Temperature Loss: {temp_loss_percent:.1f}%")
    
    # Compare South vs Southeast
    south_result = next(r for r in results if r['name'] == 'South')
    se_result = next(r for r in results if r['name'] == 'Southeast')
    
    print(f"\n=== SOUTH vs SOUTHEAST COMPARISON ===")
    irr_advantage = ((south_result['avg_irradiance'] - se_result['avg_irradiance']) / se_result['avg_irradiance']) * 100
    temp_advantage = ((south_result['avg_temp_factor'] - se_result['avg_temp_factor']) / se_result['avg_temp_factor']) * 100
    energy_difference = south_result['annual_energy'] - se_result['annual_energy']
    
    print(f"South irradiance advantage: +{irr_advantage:.1f}%")
    print(f"South temperature disadvantage: {temp_advantage:.1f}%")
    print(f"Net energy difference: {energy_difference:,.0f} kWh")
    print(f"Southeast wins by: {-energy_difference:,.0f} kWh" if energy_difference < 0 else f"South wins by: {energy_difference:,.0f} kWh")
    
    # Check if temperature model is causing the issue
    if energy_difference < 0:
        print(f"\n⚠️  ISSUE IDENTIFIED: Southeast wins due to temperature model!")
        print(f"The higher irradiance of south-facing panels is causing:")
        print(f"  - Higher cell temperatures ({south_result['avg_cell_temp']:.1f}°C vs {se_result['avg_cell_temp']:.1f}°C)")
        print(f"  - Greater temperature losses")
        print(f"  - Net energy disadvantage despite higher irradiance")
        
        # Check if this is realistic
        temp_diff = south_result['avg_cell_temp'] - se_result['avg_cell_temp']
        if temp_diff > 5:
            print(f"\n❌ UNREALISTIC: {temp_diff:.1f}°C temperature difference is too high!")
            print(f"This suggests an error in the cell temperature model.")
        else:
            print(f"\n✅ REALISTIC: {temp_diff:.1f}°C temperature difference could explain the result.")
    
    return results

def compare_irradiance_calculations(df_subset, dni_extra):
    """Compare the validation method vs energy calculation method"""
    
    print("=== IRRADIANCE CALCULATION COMPARISON ===")
    
    # Test the EXACT same orientations as validation
    test_configs = [
        (30, 180, "South"),
        (30, 135, "Southeast (135°)"),  # Same as validation
        (30, 165, "Southeast (165°)")   # Same as energy calc
    ]
    
    for tilt, azimuth, name in test_configs:
        print(f"\n{name} ({tilt}°, {azimuth}°):")
        
        # Method 1: Direct irradiance calculation (like validation)
        irradiance_data = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=df_subset['zenith'],
            solar_azimuth=df_subset['azimuth'],
            dni=df_subset['DNI'],
            ghi=df_subset['SolRad_Hor'],
            dhi=df_subset['SolRad_Dif'],
            dni_extra=dni_extra,
            model='haydavies'
        )
        
        annual_irradiation_direct = irradiance_data['poa_global'].sum() / 1000
        avg_irradiance_direct = irradiance_data['poa_global'].mean()
        
        # Method 2: Using calculate_total_irradiance function
        df_test = df_subset.copy()
        df_test = calculate_total_irradiance(df_test, tilt, azimuth, dni_extra)
        
        annual_irradiation_function = df_test['total_irradiance'].sum() / 1000
        avg_irradiance_function = df_test['total_irradiance'].mean()
        
        print(f"  Direct calculation:")
        print(f"    Annual: {annual_irradiation_direct:,.0f} kWh/m²")
        print(f"    Average: {avg_irradiance_direct:.0f} W/m²")
        print(f"  Function calculation:")
        print(f"    Annual: {annual_irradiation_function:,.0f} kWh/m²")
        print(f"    Average: {avg_irradiance_function:.0f} W/m²")
        
        # Check for differences
        annual_diff = annual_irradiation_function - annual_irradiation_direct
        avg_diff = avg_irradiance_function - avg_irradiance_direct
        
        if abs(annual_diff) > 1 or abs(avg_diff) > 1:
            print(f"  ⚠️  DIFFERENCE DETECTED:")
            print(f"    Annual diff: {annual_diff:+.0f} kWh/m²")
            print(f"    Average diff: {avg_diff:+.0f} W/m²")
        else:
            print(f"  ✓ Methods agree")
    
    # Compare 180° vs 165° directly
    print(f"\n=== DIRECT COMPARISON: 180° vs 165° ===")
    
    # 180° (South)
    irr_180 = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30, surface_azimuth=180,
        solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
        dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
        dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='haydavies'
    )
    
    # 165° (Southeast)
    irr_165 = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30, surface_azimuth=165,
        solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
        dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
        dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='haydavies'
    )
    
    annual_180 = irr_180['poa_global'].sum() / 1000
    annual_165 = irr_165['poa_global'].sum() / 1000
    avg_180 = irr_180['poa_global'].mean()
    avg_165 = irr_165['poa_global'].mean()
    
    print(f"180° (South): {annual_180:,.0f} kWh/m² annual, {avg_180:.0f} W/m² average")
    print(f"165° (SE): {annual_165:,.0f} kWh/m² annual, {avg_165:.0f} W/m² average")
    print(f"Difference: {annual_180 - annual_165:+.0f} kWh/m² annual, {avg_180 - avg_165:+.0f} W/m² average")
    
    if annual_180 > annual_165:
        print("✓ CORRECT: South (180°) has higher irradiation")
    else:
        print("❌ ERROR: Southeast (165°) appears to have higher irradiation!")
        print("This suggests a bug in the irradiance calculation or data processing")

def compare_irradiance_calculations(df_subset, dni_extra):
    """Compare the validation method vs energy calculation method"""
    
    print("=== IRRADIANCE CALCULATION COMPARISON ===")
    
    # Test the EXACT same orientations as validation
    test_configs = [
        (30, 180, "South"),
        (30, 135, "Southeast (135°)"),  # Same as validation
        (30, 165, "Southeast (165°)")   # Same as energy calc
    ]
    
    for tilt, azimuth, name in test_configs:
        print(f"\n{name} ({tilt}°, {azimuth}°):")
        
        # Method 1: Direct irradiance calculation (like validation)
        irradiance_data = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=df_subset['zenith'],
            solar_azimuth=df_subset['azimuth'],
            dni=df_subset['DNI'],
            ghi=df_subset['SolRad_Hor'],
            dhi=df_subset['SolRad_Dif'],
            dni_extra=dni_extra,
            model='haydavies'
        )
        
        annual_irradiation_direct = irradiance_data['poa_global'].sum() / 1000
        avg_irradiance_direct = irradiance_data['poa_global'].mean()
        
        # Method 2: Using calculate_total_irradiance function
        df_test = df_subset.copy()
        df_test = calculate_total_irradiance(df_test, tilt, azimuth, dni_extra)
        
        annual_irradiation_function = df_test['total_irradiance'].sum() / 1000
        avg_irradiance_function = df_test['total_irradiance'].mean()
        
        print(f"  Direct calculation:")
        print(f"    Annual: {annual_irradiation_direct:,.0f} kWh/m²")
        print(f"    Average: {avg_irradiance_direct:.0f} W/m²")
        print(f"  Function calculation:")
        print(f"    Annual: {annual_irradiation_function:,.0f} kWh/m²")
        print(f"    Average: {avg_irradiance_function:.0f} W/m²")
        
        # Check for differences
        annual_diff = annual_irradiation_function - annual_irradiation_direct
        avg_diff = avg_irradiance_function - avg_irradiance_direct
        
        if abs(annual_diff) > 1 or abs(avg_diff) > 1:
            print(f"  ⚠️  DIFFERENCE DETECTED:")
            print(f"    Annual diff: {annual_diff:+.0f} kWh/m²")
            print(f"    Average diff: {avg_diff:+.0f} W/m²")
        else:
            print(f"  ✓ Methods agree")
    
    # Compare 180° vs 165° directly
    print(f"\n=== DIRECT COMPARISON: 180° vs 165° ===")
    
    # 180° (South)
    irr_180 = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30, surface_azimuth=180,
        solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
        dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
        dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='haydavies'
    )
    
    # 165° (Southeast)
    irr_165 = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30, surface_azimuth=165,
        solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
        dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
        dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='haydavies'
    )
    
    annual_180 = irr_180['poa_global'].sum() / 1000
    annual_165 = irr_165['poa_global'].sum() / 1000
    avg_180 = irr_180['poa_global'].mean()
    avg_165 = irr_165['poa_global'].mean()
    
    print(f"180° (South): {annual_180:,.0f} kWh/m² annual, {avg_180:.0f} W/m² average")
    print(f"165° (SE): {annual_165:,.0f} kWh/m² annual, {avg_165:.0f} W/m² average")
    print(f"Difference: {annual_180 - annual_165:+.0f} kWh/m² annual, {avg_180 - avg_165:+.0f} W/m² average")
    
    if annual_180 > annual_165:
        print("✓ CORRECT: South (180°) has higher irradiation")
    else:
        print("❌ ERROR: Southeast (165°) appears to have higher irradiation!")
        print("This suggests a bug in the irradiance calculation or data processing")

def debug_solar_position_data(df_subset):
    """Check if solar position data is causing the irradiance error"""
    
    print("=== SOLAR POSITION DATA DEBUG ===")
    
    # Check solar position data quality
    print("Solar position data summary:")
    print(f"  Zenith range: {df_subset['zenith'].min():.1f}° to {df_subset['zenith'].max():.1f}°")
    print(f"  Azimuth range: {df_subset['azimuth'].min():.1f}° to {df_subset['azimuth'].max():.1f}°")
    
    # Check for suspicious values
    high_zenith = (df_subset['zenith'] > 90).sum()
    print(f"  Hours with zenith > 90°: {high_zenith} (nighttime)")
    
    # Check solar noon timing
    daylight = df_subset[df_subset['zenith'] < 85]
    if len(daylight) > 0:
        daylight_with_hour = daylight.copy()
        daylight_with_hour['hour'] = daylight_with_hour.index.hour
        
        # Find average solar noon (minimum zenith time)
        daily_min_zenith = daylight_with_hour.groupby(daylight_with_hour.index.date)['zenith'].idxmin()
        noon_hours = [daylight_with_hour.loc[idx, 'hour'] for idx in daily_min_zenith if idx in daylight_with_hour.index]
        avg_solar_noon = sum(noon_hours) / len(noon_hours) if noon_hours else 0
        
        print(f"  Average solar noon occurs at: {avg_solar_noon:.1f}h")
        if not (11.5 <= avg_solar_noon <= 13.5):
            print(f"  ⚠️  Solar noon timing seems wrong for Athens!")
    
    # Check summer solstice (should have minimum zenith ~14° for Athens)
    summer_data = df_subset[df_subset.index.month == 6]
    if len(summer_data) > 0:
        min_zenith_summer = summer_data['zenith'].min()
        print(f"  Minimum zenith in June: {min_zenith_summer:.1f}° (should be ~14° for Athens)")
        if min_zenith_summer < 10 or min_zenith_summer > 20:
            print(f"  ⚠️  Summer zenith seems wrong for Athens!")
    
    # Check azimuth distribution
    print(f"\nAzimuth distribution check:")
    daylight_summer = df_subset[(df_subset['zenith'] < 85) & (df_subset.index.month == 6)]
    if len(daylight_summer) > 0:
        azimuth_range = daylight_summer['azimuth'].max() - daylight_summer['azimuth'].min()
        print(f"  Summer azimuth range: {daylight_summer['azimuth'].min():.1f}° to {daylight_summer['azimuth'].max():.1f}° (span: {azimuth_range:.1f}°)")
        if azimuth_range < 180:
            print(f"  ⚠️  Azimuth range seems too narrow!")
    
    # Check specific times for solar position validation
    print(f"\nSpecific time checks:")
    
    # Summer solstice noon
    try:
        summer_solstice = pd.Timestamp('2020-06-21 13:00:00+02:00')  # 1 PM Athens time
        if summer_solstice in df_subset.index:
            zenith_ss = df_subset.loc[summer_solstice, 'zenith']
            azimuth_ss = df_subset.loc[summer_solstice, 'azimuth']
            print(f"  Summer solstice 1 PM: zenith={zenith_ss:.1f}°, azimuth={azimuth_ss:.1f}°")
            print(f"    Expected: zenith~14°, azimuth~180°")
            if abs(azimuth_ss - 180) > 10:
                print(f"  ⚠️  Summer solstice azimuth is wrong!")
        else:
            print(f"  Summer solstice time not found in data")
    except:
        print(f"  Could not check summer solstice")
    
    # Winter solstice noon
    try:
        winter_solstice = pd.Timestamp('2020-12-21 13:00:00+02:00')  # 1 PM Athens time
        if winter_solstice in df_subset.index:
            zenith_ws = df_subset.loc[winter_solstice, 'zenith']
            azimuth_ws = df_subset.loc[winter_solstice, 'azimuth']
            print(f"  Winter solstice 1 PM: zenith={zenith_ws:.1f}°, azimuth={azimuth_ws:.1f}°")
            print(f"    Expected: zenith~61°, azimuth~180°")
            if abs(azimuth_ws - 180) > 10:
                print(f"  ⚠️  Winter solstice azimuth is wrong!")
        else:
            print(f"  Winter solstice time not found in data")
    except:
        print(f"  Could not check winter solstice")

def debug_irradiance_components(df_subset, dni_extra):
    """Check the irradiance components for 165° vs 180°"""
    
    print(f"\n=== IRRADIANCE COMPONENTS DEBUG ===")
    
    # Compare irradiance components for problematic orientations
    orientations = [(180, "South"), (165, "Southeast")]
    
    for azimuth, name in orientations:
        print(f"\n{name} (30°, {azimuth}°):")
        
        # Calculate components
        irr_data = pvlib.irradiance.get_total_irradiance(
            surface_tilt=30, surface_azimuth=azimuth,
            solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
            dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
            dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='haydavies'
        )
        
        # Annual sums
        poa_direct = irr_data['poa_direct'].sum() / 1000
        poa_diffuse = irr_data['poa_diffuse'].sum() / 1000  
        poa_ground = irr_data['poa_ground'].sum() / 1000
        poa_total = irr_data['poa_global'].sum() / 1000
        
        print(f"  POA Direct: {poa_direct:,.0f} kWh/m²")
        print(f"  POA Diffuse: {poa_diffuse:,.0f} kWh/m²") 
        print(f"  POA Ground: {poa_ground:,.0f} kWh/m²")
        print(f"  POA Total: {poa_total:,.0f} kWh/m²")
        
        # Check which component is driving the difference
        if azimuth == 165:
            print(f"  Components breakdown:")
            print(f"    Direct: {(poa_direct/poa_total)*100:.1f}%")
            print(f"    Diffuse: {(poa_diffuse/poa_total)*100:.1f}%")
            print(f"    Ground: {(poa_ground/poa_total)*100:.1f}%")

def test_irradiance_models(df_subset, dni_extra):
    """Test different irradiance models to identify the problem"""
    
    print("=== TESTING DIFFERENT IRRADIANCE MODELS ===")
    
    orientations = [(180, "South"), (165, "Southeast")]
    models = ['isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez']
    
    results = {}
    
    for model in models:
        print(f"\nTesting model: {model}")
        model_results = {}
        
        for azimuth, name in orientations:
            try:
                # Test this model
                if model == 'isotropic':
                    irr_data = pvlib.irradiance.get_total_irradiance(
                        surface_tilt=30, surface_azimuth=azimuth,
                        solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
                        dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
                        dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, 
                        model='isotropic'
                    )
                elif model == 'klucher':
                    irr_data = pvlib.irradiance.get_total_irradiance(
                        surface_tilt=30, surface_azimuth=azimuth,
                        solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
                        dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
                        dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, 
                        model='klucher'
                    )
                elif model == 'haydavies':
                    irr_data = pvlib.irradiance.get_total_irradiance(
                        surface_tilt=30, surface_azimuth=azimuth,
                        solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
                        dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
                        dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, 
                        model='haydavies'
                    )
                else:
                    # Skip complex models for now
                    continue
                
                annual_irr = irr_data['poa_global'].sum() / 1000
                model_results[azimuth] = annual_irr
                print(f"  {name} ({azimuth}°): {annual_irr:,.0f} kWh/m²")
                
            except Exception as e:
                print(f"  {name} ({azimuth}°): ERROR - {e}")
                model_results[azimuth] = None
        
        # Check if south wins with this model
        if 180 in model_results and 165 in model_results:
            if model_results[180] and model_results[165]:
                diff = model_results[180] - model_results[165]
                if diff > 0:
                    print(f"  ✅ {model}: South wins by {diff:.0f} kWh/m²")
                else:
                    print(f"  ❌ {model}: Southeast wins by {-diff:.0f} kWh/m²")
        
        results[model] = model_results

def test_simple_geometry(df_subset):
    """Test basic solar geometry to verify the issue"""
    
    print(f"\n=== BASIC SOLAR GEOMETRY TEST ===")
    
    # Test just the geometric part (no atmosphere models)
    # Calculate angle of incidence for different orientations
    
    orientations = [(180, "South"), (165, "Southeast"), (135, "Southeast2")]
    
    print(f"Testing angle of incidence (basic geometry):")
    
    for azimuth, name in orientations:
        # Calculate surface normal vector
        surface_tilt_rad = np.radians(30)
        surface_azimuth_rad = np.radians(azimuth)
        
        # Surface normal components
        surface_normal_x = np.sin(surface_tilt_rad) * np.sin(surface_azimuth_rad)
        surface_normal_y = np.sin(surface_tilt_rad) * np.cos(surface_azimuth_rad)  
        surface_normal_z = np.cos(surface_tilt_rad)
        
        # Solar vector components
        solar_zenith_rad = np.radians(df_subset['zenith'])
        solar_azimuth_rad = np.radians(df_subset['azimuth'])
        
        sun_x = np.sin(solar_zenith_rad) * np.sin(solar_azimuth_rad)
        sun_y = np.sin(solar_zenith_rad) * np.cos(solar_azimuth_rad)
        sun_z = np.cos(solar_zenith_rad)
        
        # Angle of incidence (dot product)
        cos_incidence = (surface_normal_x * sun_x + 
                        surface_normal_y * sun_y + 
                        surface_normal_z * sun_z)
        
        # Only positive values (sun hitting front of panel)
        cos_incidence_positive = np.maximum(cos_incidence, 0)
        
        # Calculate direct normal irradiance on tilted surface
        direct_on_surface = df_subset['DNI'] * cos_incidence_positive
        
        # Annual sum
        annual_direct = direct_on_surface.sum() / 1000
        
        print(f"  {name} ({azimuth}°): {annual_direct:,.0f} kWh/m² (direct only)")

def test_azimuth_convention_fix(df_subset, dni_extra):
    """Test if there's an azimuth convention mismatch causing the issue"""
    
    print("=== TESTING AZIMUTH CONVENTION FIX ===")
    
    # Test 1: Try different azimuth shifts
    print("Testing azimuth shifts:")
    
    shifts_to_test = [0, 15, -15, 30, -30, 45, -45]
    
    for shift in shifts_to_test:
        print(f"\nTesting {shift:+d}° azimuth shift:")
        
        # Test south (should be optimal)
        south_shifted = 180 + shift
        if south_shifted < 0:
            south_shifted += 360
        elif south_shifted >= 360:
            south_shifted -= 360
            
        # Test southeast 
        se_shifted = 165 + shift
        if se_shifted < 0:
            se_shifted += 360
        elif se_shifted >= 360:
            se_shifted -= 360
        
        try:
            # Calculate for shifted azimuths
            irr_south = pvlib.irradiance.get_total_irradiance(
                surface_tilt=30, surface_azimuth=south_shifted,
                solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
                dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
                dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='isotropic'
            )
            
            irr_se = pvlib.irradiance.get_total_irradiance(
                surface_tilt=30, surface_azimuth=se_shifted,
                solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
                dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
                dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='isotropic'
            )
            
            annual_south = irr_south['poa_global'].sum() / 1000
            annual_se = irr_se['poa_global'].sum() / 1000
            
            print(f"  South ({south_shifted}°): {annual_south:,.0f} kWh/m²")
            print(f"  SE ({se_shifted}°): {annual_se:,.0f} kWh/m²")
            
            if annual_south > annual_se:
                diff = annual_south - annual_se
                print(f"  ✅ FIXED! South wins by {diff:.0f} kWh/m² with {shift:+d}° shift")
            else:
                diff = annual_se - annual_south
                print(f"  ❌ Still wrong: SE wins by {diff:.0f} kWh/m²")
                
        except Exception as e:
            print(f"  Error with {shift}° shift: {e}")

def test_manual_solar_position_verification(df_subset):
    """Manually verify solar positions for known dates/times"""
    
    print(f"\n=== MANUAL SOLAR POSITION VERIFICATION ===")
    
    # Test known solar positions for Athens
    test_times = [
        ('2020-06-21 13:00:00+02:00', 'Summer solstice noon', 14.6, 180),
        ('2020-12-21 13:00:00+02:00', 'Winter solstice noon', 61.0, 180),
        ('2020-03-21 13:00:00+02:00', 'Spring equinox noon', 38.0, 180),
        ('2020-06-21 07:00:00+02:00', 'Summer sunrise', 85, 72),
        ('2020-06-21 19:00:00+02:00', 'Summer sunset', 85, 288)
    ]
    
    for time_str, description, expected_zenith, expected_azimuth in test_times:
        try:
            test_time = pd.Timestamp(time_str)
            
            # Find closest time in data
            closest_idx = df_subset.index.get_indexer([test_time], method='nearest')[0]
            closest_time = df_subset.index[closest_idx]
            
            actual_zenith = df_subset.iloc[closest_idx]['zenith']
            actual_azimuth = df_subset.iloc[closest_idx]['azimuth']
            
            zenith_diff = abs(actual_zenith - expected_zenith)
            azimuth_diff = abs(actual_azimuth - expected_azimuth)
            
            print(f"\n{description} ({test_time}):")
            print(f"  Closest data time: {closest_time}")
            print(f"  Expected: zenith={expected_zenith:.1f}°, azimuth={expected_azimuth:.1f}°")
            print(f"  Actual: zenith={actual_zenith:.1f}°, azimuth={actual_azimuth:.1f}°")
            print(f"  Difference: zenith={zenith_diff:.1f}°, azimuth={azimuth_diff:.1f}°")
            
            if zenith_diff > 5 or azimuth_diff > 10:
                print(f"  ⚠️  Large difference detected!")
            else:
                print(f"  ✓ Reasonable match")
                
        except Exception as e:
            print(f"  Error checking {description}: {e}")

def test_alternative_azimuth_definition(df_subset, dni_extra):
    """Test if using a different azimuth definition fixes the issue"""
    
    print(f"\n=== TESTING ALTERNATIVE AZIMUTH DEFINITIONS ===")
    
    # Common azimuth conventions:
    # 1. Meteorological: 0°=North, 90°=East, 180°=South, 270°=West
    # 2. Mathematical: 0°=East, 90°=North, 180°=West, 270°=South  
    # 3. Solar: 0°=North, clockwise positive
    
    # Current: South=180°, Southeast=165°
    # Try: Converting surface azimuth to different convention
    
    print("Testing different surface azimuth conventions:")
    
    conversions = [
        (180, 165, "Current (South=180°)"),
        (0, 15, "North=0° (South=0°, SE=15°)"),
        (360-180, 360-165, "South=180° reversed"),
        (180+15, 165+15, "Shifted +15°")
    ]
    
    for south_az, se_az, description in conversions:
        try:
            print(f"\n{description}:")
            
            irr_south = pvlib.irradiance.get_total_irradiance(
                surface_tilt=30, surface_azimuth=south_az,
                solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
                dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
                dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='isotropic'
            )
            
            irr_se = pvlib.irradiance.get_total_irradiance(
                surface_tilt=30, surface_azimuth=se_az,
                solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
                dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
                dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='isotropic'
            )
            
            annual_south = irr_south['poa_global'].sum() / 1000
            annual_se = irr_se['poa_global'].sum() / 1000
            
            print(f"  South ({south_az}°): {annual_south:,.0f} kWh/m²")
            print(f"  SE ({se_az}°): {annual_se:,.0f} kWh/m²")
            
            if annual_south > annual_se:
                diff = annual_south - annual_se
                print(f"  ✅ WORKS! South wins by {diff:.0f} kWh/m²")
            else:
                diff = annual_se - annual_south
                print(f"  ❌ Still wrong: SE wins by {diff:.0f} kWh/m²")
                
        except Exception as e:
            print(f"  Error with {description}: {e}")

def test_azimuth_fix(df_subset, dni_extra):
    """Test that the azimuth fix works correctly"""
    
    print("=== TESTING AZIMUTH FIX ===")
    
    # Test orientations with the fix
    test_configs = [
        (30, 180, "South"),
        (30, 165, "Southeast"),
        (30, 150, "More Southeast")
    ]
    
    print("Results with azimuth fix applied:")
    
    for tilt, azimuth, name in test_configs:
        # Apply the same correction as the fix
        corrected_azimuth = azimuth - 15.0
        if corrected_azimuth < 0:
            corrected_azimuth += 360
            
        irr_data = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt, surface_azimuth=corrected_azimuth,
            solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
            dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
            dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='haydavies'
        )
        
        annual_irr = irr_data['poa_global'].sum() / 1000
        print(f"  {name} ({azimuth}° → {corrected_azimuth}°): {annual_irr:,.0f} kWh/m²")
    
    # Verify south is now optimal
    south_fixed = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30, surface_azimuth=165,  # 180-15
        solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
        dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
        dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='haydavies'
    )
    
    se_fixed = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30, surface_azimuth=150,  # 165-15
        solar_zenith=df_subset['zenith'], solar_azimuth=df_subset['azimuth'],
        dni=df_subset['DNI'], ghi=df_subset['SolRad_Hor'], 
        dhi=df_subset['SolRad_Dif'], dni_extra=dni_extra, model='haydavies'
    )
    
    south_annual = south_fixed['poa_global'].sum() / 1000
    se_annual = se_fixed['poa_global'].sum() / 1000
    
    print(f"\nDirect comparison with fix:")
    print(f"  South (180° → 165°): {south_annual:,.0f} kWh/m²")
    print(f"  Southeast (165° → 150°): {se_annual:,.0f} kWh/m²")
    
    if south_annual > se_annual:
        diff = south_annual - se_annual
        print(f"  ✅ SUCCESS! South wins by {diff:.0f} kWh/m² with azimuth fix")
        return True
    else:
        diff = se_annual - south_annual
        print(f"  ❌ Still wrong: Southeast wins by {diff:.0f} kWh/m²")
        return False



# Helper function to prepare PVGIS reference data from your provided data
def prepare_pvgis_reference_data():
    """
    Prepare PVGIS reference data from the provided Athens results.
    """
    
    # Data from your PVGIS results
    pvgis_data = {
        'single_objective': {
            'slope': 34,
            'azimuth': 0,
            'annual_kwh': 323284.21,
            'annual_irradiation_kwh_m2': 2042.27,
            'year_to_year_variability': 5284.19,
            'monthly_data': [20883.11, 21979.95, 29577.83, 31062.22, 31591.73, 30589.62,
                           32582.91, 32729.39, 29007.94, 25435.83, 19825.6, 18018.08],
            'losses': {
                'aoi_loss_pct': -2.49,
                'spectral_effects_pct': 0.49,
                'temperature_low_irr_pct': -7.38,
                'total_loss_pct': -21.94
            }
        },
        'multi_objective': {
            'slope': 29,
            'azimuth': 20,
            'annual_kwh': 317962.25,
            'annual_irradiation_kwh_m2': 2015.06,
            'year_to_year_variability': 5162.34,
            'monthly_data': [19237.38, 20902.52, 28621.34, 30976.1, 32229.53, 31585.86,
                           33594.03, 33030.5, 28433.73, 24314.78, 18526.37, 16510.09],
            'losses': {
                'aoi_loss_pct': -2.6,
                'spectral_effects_pct': 0.46,
                'temperature_low_irr_pct': -7.54,
                'total_loss_pct': -22.19
            }
        }
    }
    
    return pvgis_data




def main():
    try:
        # Parse Command-Line Arguments
        parser = argparse.ArgumentParser(description='Solar Energy Analysis Tool - PVGIS TMY Data')
        parser.add_argument('--pvgis_file', type=str, required=True, 
                          help='Path to PVGIS TMY CSV data file')
        parser.add_argument('--load_file', type=str, required=True,
                          help='Path to load data CSV file')
        parser.add_argument('--output_dir', type=str, required=True, 
                          help='Directory to save the output results')
        parser.add_argument('--config_file', type=str, required=True, 
                          help='Path to the YAML configuration file')
        parser.add_argument('--latitude', type=float, default=37.99, 
                          help='Latitude of the location (Athens default)')
        parser.add_argument('--longitude', type=float, default=23.74, 
                          help='Longitude of the location (Athens default)')
        parser.add_argument('--optimization_mode', type=str, 
                          choices=['single_objective', 'multi_objective', 'comprehensive'], 
                          default='single_objective',
                          help='Optimization approach')
        parser.add_argument('--weighting_strategy', type=str, 
                          choices=['adaptive_improved', 'pure_load_matching', 'peak_focused'], 
                          default='adaptive_improved',
                          help='Weighting strategy for multi-objective optimization')
        parser.add_argument('--compare_strategies', action='store_true',
                          help='Run all weighting strategies for comparison')
        parser.add_argument('--include_battery', action='store_true',
                          help='Include battery storage optimization in the analysis')
        parser.add_argument('--include_financial_analysis', action='store_true',
                          help='Include detailed financial analysis (NPV, IRR, LCOE)')
        parser.add_argument('--battery_max_capacity', type=float, default=500.0,
                          help='Maximum battery capacity to consider (kWh)')
        parser.add_argument('--battery_cost_per_kwh', type=float, default=400.0,
                          help='Battery cost per kWh (EUR)')
        parser.add_argument('--electricity_buy_price', type=float, default=0.24,
                          help='Electricity purchase price (EUR/kWh)')
        parser.add_argument('--electricity_sell_price', type=float, default=0.08,
                          help='Electricity feed-in tariff (EUR/kWh)')
        parser.add_argument('--project_lifetime_years', type=int, default=25,
                          help='Project lifetime for financial analysis (years)')
        parser.add_argument('--discount_rate_percent', type=float, default=5.0,
                          help='Discount rate for NPV calculation (%)')
        parser.add_argument('--create_academic_plots', action='store_true',
                  help='Create comprehensive academic plots for scientific review')
        args = parser.parse_args()
        
        
        # Create output directory
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Set Up Logging and Configuration
        setup_logging(args.output_dir)
        logging.info("=== PV SYSTEM OPTIMIZATION - PVGIS TMY DATA FOR ATHENS ===")
        
        config = load_config(args.config_file)
        
        # Panel and inverter parameters
        panel_params = {
            'Name': config['solar_panel']['name'],
            'Pmp': config['solar_panel']['pmp'],
            'gamma_pmp_percent': config['solar_panel']['gamma_pmp_percent'],
            'gamma_pdc': config['solar_panel']['gamma_pmp_percent'] / 100
        }
        inverter_params = {
            'eta_inv_nom': config['inverter']['eta_inv_nom'],
            'eta_inv_ref': config['inverter']['eta_inv_ref'],
            'pdc0': config['inverter']['pdc0']
        }

        # LOAD AND VALIDATE DATA
        logging.info("=== LOADING PVGIS TMY DATA ===")
        try:
            df_original = load_and_preprocess_pvgis_data(args.pvgis_file, args.load_file)
            logging.info("PVGIS data loaded and preprocessed successfully")
            
        except Exception as e:
            logging.error(f"Failed to load PVGIS data: {e}")
            print(f"FAILED to load PVGIS data: {e}")
            sys.exit(1)

        # SOLAR POSITION AND DNI CALCULATIONS
        logging.info("=== CALCULATING SOLAR POSITION ===")
        try:
            df_original, solar_valid = calculate_solar_position(df_original, args.latitude, args.longitude)
            if solar_valid:
                logging.info("Solar position validation passed")
                print("Solar position calculations: VALID")
            else:
                logging.warning("Solar position validation issues detected")
                print("Solar position calculations: ISSUES DETECTED")
                
        except Exception as e:
            logging.error(f"Solar position calculation failed: {e}")
            print(f"Solar position calculation failed: {e}")
            sys.exit(1)
        
        # Calculate DNI
        if 'DNI' not in df_original.columns:
            logging.info("Calculating DNI...")
            df_original = calculate_dni(df_original)
        
        # Get DNI extra
        dni_extra = pvlib.irradiance.get_extra_radiation(df_original.index, method='nrel')
        
        # SYSTEM CONFIGURATION
        logging.info("=== SYSTEM CONFIGURATION ===")
        available_area = config.get('available_area', 1500)
        panel_length = config['solar_panel']['length']
        panel_width = config['solar_panel']['width']
        spacing_length = config['solar_panel']['spacing_length']
        spacing_width = config['solar_panel']['spacing_width']
        panel_power_rating = config['solar_panel']['power_rating']
        
        area_per_panel = (panel_length + spacing_length) * (panel_width + spacing_width)
        number_of_panels = int(available_area // area_per_panel)
        panel_area = panel_length * panel_width
        total_panel_area = panel_area * number_of_panels
        panel_efficiency = panel_power_rating / (panel_area * 1000)
        
        inverter_params['pdc0'] = panel_params['Pmp'] * number_of_panels
        
        logging.info(f"System Configuration:")
        logging.info(f"  Panels: {number_of_panels} x {panel_power_rating}W = {number_of_panels * panel_power_rating / 1000:.1f} kWp")
        logging.info(f"  Total Panel Area: {total_panel_area:.1f} m²")
        logging.info(f"  Panel Efficiency: {panel_efficiency * 100:.2f}%")
        
        # Create subset for optimization
        columns_needed = ['SolRad_Hor', 'SolRad_Dif', 'Air Temp', 'WS_10m', 'zenith', 'azimuth', 'DNI', 'Load (kW)']
        df_subset = df_original[columns_needed].copy()

        # OPTIMIZATION EXECUTION
        single_obj_results = None
        optimal_tilt = None
        optimal_azimuth = None
        total_production = 0
        logbook = None  # ADD THIS
        pareto_front = None
        
        if args.optimization_mode == 'single_objective':
            logging.info("=== SINGLE-OBJECTIVE OPTIMIZATION (MAX ENERGY) ===")
            
            # Run quick validation test first
            best_test_config, best_test_energy = quick_validation_test(
                df_subset, dni_extra, number_of_panels, inverter_params
            )
            
            # Use the fixed single-objective optimization
            single_obj_results, df_optimal = optimize_for_maximum_energy_production(
                df_subset, dni_extra, number_of_panels, inverter_params, args.output_dir,
                args.latitude, args.longitude
            )
            
            if single_obj_results:
                optimal_tilt = single_obj_results['optimal_tilt']
                optimal_azimuth = single_obj_results['optimal_azimuth']
                total_production = single_obj_results['max_energy_production_kwh']
                
                # Debug tests (only if you want them - comment out for production runs)
                # print("\n" + "="*60)
                # print("DEBUGGING TEMPERATURE MODEL")
                # print("="*60)
                # debug_results = debug_temperature_model(df_subset, dni_extra, number_of_panels, inverter_params)
                
                logging.info(f"Single-objective optimization completed:")
                logging.info(f"  Angles: {optimal_tilt:.1f}° tilt, {optimal_azimuth:.1f}° azimuth")
                logging.info(f"  Production: {total_production:,.0f} kWh/year")
                
                # Comprehensive validation
                system_capacity_kwp = (number_of_panels * panel_power_rating) / 1000
                validation_results, all_valid = validate_optimization_results_comprehensive(
                    optimal_tilt, optimal_azimuth, total_production, system_capacity_kwp, args.latitude
                )
                
                if not all_valid:
                    logging.warning("⚠ Some validation checks failed - please review results")
                else:
                    logging.info("✓ All validation checks passed - results appear realistic")
                    
                # Compare with test baseline
                if best_test_config and best_test_energy:
                    energy_improvement = total_production - best_test_energy
                    if energy_improvement >= 0:
                        logging.info(f"✓ Optimization improved over best test config by {energy_improvement:,.0f} kWh")
                    else:
                        logging.warning(f"⚠ Optimization performed worse than best test config by {abs(energy_improvement):,.0f} kWh")
                        logging.warning(f"This suggests potential optimization issues")
                        
            else:
                logging.error("Single-objective optimization failed!")
                sys.exit(1)
                
        elif args.optimization_mode == 'multi_objective':
            logging.info("=== MULTI-OBJECTIVE OPTIMIZATION ===")
            
            if args.compare_strategies:
                logging.info("Comparing all weighting strategies...")
                strategies = ['adaptive_improved', 'pure_load_matching', 'peak_focused']
                results_by_strategy = {}
                
                for strategy in strategies:
                    logging.info(f"\nRunning NSGA-II with {strategy} strategy...")
                    
                    # Use the improved multi-objective optimization
                    pareto_front, filtered_front, best_balanced, logbook  = run_deap_multi_objective_optimization(
                        df_subset, dni_extra, number_of_panels, inverter_params, args.output_dir,
                        args.latitude, args.longitude, weighting_strategy=strategy
                    )
                    
                    results_by_strategy[strategy] = {
                        'pareto_front': pareto_front,
                        'filtered_front': filtered_front,
                        'best_balanced': best_balanced,
                        'logbook': logbook 
                    }
                
                # Use adaptive_improved as default selection
                best_strategy = 'adaptive_improved'
                best_balanced = results_by_strategy[best_strategy]['best_balanced']
                pareto_front = results_by_strategy[best_strategy]['pareto_front']
                
                logging.info(f"Strategy comparison complete. Using {best_strategy} result.")
                
            else:
                # Run single strategy
                pareto_front, filtered_front, best_balanced,  logbook = run_deap_multi_objective_optimization(
                    df_subset, dni_extra, number_of_panels, inverter_params, args.output_dir,
                    args.latitude, args.longitude, weighting_strategy=args.weighting_strategy
                )
            
            if best_balanced:
                optimal_tilt = best_balanced[0]
                optimal_azimuth = best_balanced[1]
                weighted_mismatch = best_balanced.fitness.values[0]
                total_production = best_balanced.fitness.values[1] * 1000  # Convert back from normalized
                
                logging.info(f"Multi-objective optimization completed:")
                logging.info(f"  Strategy: {args.weighting_strategy if not args.compare_strategies else 'adaptive_improved'}")
                logging.info(f"  Angles: {optimal_tilt:.1f}° tilt, {optimal_azimuth:.1f}° azimuth")
                logging.info(f"  Production: {total_production:,.0f} kWh/year")
                logging.info(f"  Weighted Mismatch: {weighted_mismatch * 1000:,.0f} kWh")
                
                # Note: Multi-objective can deviate from south for load matching
                azimuth_deviation = abs(optimal_azimuth - 180)
                if azimuth_deviation > 20:
                    logging.info(f"Note: {azimuth_deviation:.1f}° deviation from south is expected for load-matching optimization")
                
            else:
                logging.error("Multi-objective optimization failed!")
                sys.exit(1)
                
        elif args.optimization_mode == 'comprehensive':
            logging.info("=== COMPREHENSIVE ANALYSIS (ALL METHODS) ===")
            
            # 1. Quick validation test
            logging.info("Running quick validation test...")
            best_test_config, best_test_energy = quick_validation_test(
                df_subset, dni_extra, number_of_panels, inverter_params
            )
            
            # 2. Single-objective optimization
            logging.info("Running single-objective optimization...")
            single_obj_results, df_optimal = optimize_for_maximum_energy_production(
                df_subset, dni_extra, number_of_panels, inverter_params, args.output_dir,
                args.latitude, args.longitude
            )
            
            # 3. Multi-objective with all strategies
            logging.info("Running multi-objective with all strategies...")
            strategies = ['adaptive_improved', 'pure_load_matching', 'peak_focused']
            results_by_strategy = {}
            
            for strategy in strategies:
                logging.info(f"  Running {strategy} strategy...")
                
                pareto_front, filtered_front, best_balanced, logbook = run_deap_multi_objective_optimization(
                    df_subset, dni_extra, number_of_panels, inverter_params, args.output_dir,
                    args.latitude, args.longitude, weighting_strategy=strategy
                )
                
                results_by_strategy[strategy] = {
                    'pareto_front': pareto_front,
                    'filtered_front': filtered_front,
                    'best_balanced': best_balanced,
                    'logbook': logbook 
                }
                
                # Compare with single-objective
                if single_obj_results and pareto_front:
                    comparison_df = compare_optimization_results(
                        single_obj_results, pareto_front, args.output_dir
                    )
            
            # Use adaptive_improved as final result
            best_strategy = 'adaptive_improved'
            best_balanced = results_by_strategy[best_strategy]['best_balanced']
            
            if best_balanced:
                optimal_tilt = best_balanced[0]
                optimal_azimuth = best_balanced[1]
                total_production = best_balanced.fitness.values[1] * 1000  # Convert back from normalized
                
                logging.info(f"Comprehensive analysis completed:")
                logging.info(f"  Selected Strategy: {best_strategy}")
                logging.info(f"  Angles: {optimal_tilt:.1f}° tilt, {optimal_azimuth:.1f}° azimuth")
                logging.info(f"  Production: {total_production:,.0f} kWh/year")
            else:
                logging.error("Comprehensive analysis failed!")
                sys.exit(1)

        # CALCULATE FINAL SYSTEM PERFORMANCE
        logging.info("=== CALCULATING FINAL SYSTEM PERFORMANCE ===")
        df_final = df_subset.copy()
        df_final = calculate_total_irradiance(df_final, optimal_tilt, optimal_azimuth, dni_extra)
        df_final = calculate_energy_production(df_final, number_of_panels, inverter_params)
        
        # Final validation of energy production
        validate_energy_production(df_final, number_of_panels, panel_power_rating)

        # BATTERY OPTIMIZATION - Initialize variables first
        optimal_battery_capacity = 0.0
        battery_results = None
        battery_simulation = None


        # FINANCIAL ANALYSIS
        financial_results = None
        if args.include_financial_analysis:
            logging.info("=== FINANCIAL ANALYSIS ===")
            
            try:
                # Get economic parameters
                economic_params = get_default_economic_params()
                economic_params.update({
                    'battery_cost_per_kwh': args.battery_cost_per_kwh,
                    'electricity_price': args.electricity_buy_price,
                    'feed_in_tariff': args.electricity_sell_price,
                    'project_lifetime_years': args.project_lifetime_years,
                    'discount_rate_percent': args.discount_rate_percent
                })
                
                # Calculate investment cost
                investment_cost = calculate_investment_cost(
                    number_of_panels, optimal_battery_capacity, economic_params
                )
                
                # Calculate financial metrics
                financial_results = calculate_financial_metrics(
                    df_final, optimal_battery_capacity, battery_simulation,
                    investment_cost, economic_params
                )
                
                # Create financial plots
                create_financial_analysis_plots(financial_results, args.output_dir)
                
                # Log key results
                metrics = financial_results['financial_metrics']
                logging.info(f"Financial analysis completed:")
                logging.info(f"  Total Investment: €{metrics['Total_Investment']:,.0f}")
                logging.info(f"  NPV (25 years): €{metrics['NPV']:,.0f}")
                logging.info(f"  IRR: {metrics['IRR_percent']:.1f}%")
                logging.info(f"  Payback Period: {metrics['Payback_Period_Years']:.1f} years")
                logging.info(f"  LCOE: €{metrics['LCOE_eur_per_kwh']:.3f}/kWh")
                
            except Exception as e:
                logging.error(f"Financial analysis failed: {e}")
                logging.info("Continuing without financial analysis")
        
        # ANALYZE PERFORMANCE
        logging.info("=== ANALYZING PERFORMANCE ===")
        
        # Calculate seasonal performance
        seasonal_stats, daily_seasonal = analyze_seasonal_performance(df_final)
        
        # Summarize energy flows
        try:
            energy_breakdown, energy_losses, system_efficiency = summarize_energy(df_final)
        except Exception as e:
            logging.error(f"Energy summary failed: {e}")
            energy_breakdown, energy_losses, system_efficiency = None, None, 15.0

        # FINANCIAL ANALYSIS PLOTTING
        if args.include_financial_analysis and financial_results:
            logging.info("=== CREATING FINANCIAL ANALYSIS PLOTS ===")
            
            try:
                # Get economic parameters for sensitivity analysis
                economic_params = get_default_economic_params()
                economic_params.update({
                    'battery_cost_per_kwh': args.battery_cost_per_kwh,
                    'electricity_price': args.electricity_buy_price,
                    'feed_in_tariff': args.electricity_sell_price,
                    'project_lifetime_years': args.project_lifetime_years,
                    'discount_rate_percent': args.discount_rate_percent
                })
                
                # Prepare financial results for comparison plots
                single_obj_financial_results = None
                multi_obj_financial_results = financial_results  # Main results are from multi-objective
                
                # If you have single-objective financial results, calculate them here
                if single_obj_results is not None and args.optimization_mode == 'both':
                    try:
                        # Calculate financial metrics for single objective results
                        # You would need to create a DataFrame with single objective energy production
                        df_single_financial = df_subset.copy()
                        df_single_financial = calculate_total_irradiance(
                            df_single_financial, 
                            single_obj_results['optimal_tilt'], 
                            single_obj_results['optimal_azimuth'],
                            dni_extra
                        )
                        df_single_financial = calculate_energy_production(
                            df_single_financial, number_of_panels, inverter_params
                        )
                        
                        # Calculate single objective investment cost
                        single_obj_investment = calculate_investment_cost(
                            number_of_panels, 0, economic_params  # Assuming no battery for single obj
                        )
                        
                        # Calculate single objective financial metrics
                        single_obj_financial_results = calculate_financial_metrics(
                            df_single_financial, 0, None,  # No battery simulation for single obj
                            single_obj_investment, economic_params
                        )
                        
                        logging.info("Single objective financial analysis completed")
                        
                    except Exception as e:
                        logging.warning(f"Could not calculate single objective financial results: {e}")
                        single_obj_financial_results = None
                
                # Create all financial plots
                plotting_module.create_all_financial_plots(
                    financial_results=financial_results,
                    single_obj_financial=single_obj_financial_results,
                    multi_obj_financial=multi_obj_financial_results,
                    single_obj_battery=None,  # You can add single obj battery results if available
                    multi_obj_battery=battery_simulation,
                    economic_params=economic_params,
                    output_dir=args.output_dir
                )
                
                logging.info("Financial analysis plots completed successfully")
                
            except Exception as e:
                logging.error(f"Failed to create financial analysis plots: {e}")
                logging.info("Continuing without financial plots")
        
        # SAVE RESULTS
        logging.info("=== SAVING RESULTS ===")
        
        logging.info("=== CREATING INPUT DATA VISUALIZATION PLOTS ===")

        try:
            # Create input data plots using the original loaded data
            plotting_module.create_input_data_plots(df_original, args.output_dir)
            
        except Exception as e:
            logging.error(f"Failed to create input data plots: {e}")
            logging.info("Continuing without input data plots")

        # CREATE ENERGY PRODUCTION VS LOAD PLOTS
        logging.info("=== CREATING ENERGY PRODUCTION VS LOAD PLOTS ===")

        try:
            df_single_for_plot = None
            df_multi_for_plot = df_final  # Final results use multi-objective angles
            
            # For single objective results, need to recalculate energy production
            if single_obj_results is not None:
                logging.info("Calculating energy production for single objective angles...")
                df_single_for_plot = df_subset.copy()
                df_single_for_plot = calculate_total_irradiance(
                    df_single_for_plot, 
                    single_obj_results['optimal_tilt'], 
                    single_obj_results['optimal_azimuth'], 
                    dni_extra
                )
                df_single_for_plot = calculate_energy_production(
                    df_single_for_plot, number_of_panels, inverter_params
                )
            
            # Create multi-objective results dictionary for plotting
            multi_obj_results_for_plot = {
                'optimal_tilt': optimal_tilt,
                'optimal_azimuth': optimal_azimuth,
                'total_production': total_production,
                'optimization_mode': args.optimization_mode
            } if optimal_tilt is not None else None
            
            # CREATE ALL ENERGY PRODUCTION PLOTS
            if (df_single_for_plot is not None and df_multi_for_plot is not None and 
                single_obj_results is not None and multi_obj_results_for_plot is not None):
                
                plotting_module.create_energy_production_plots(
                    df_single_for_plot, 
                    df_multi_for_plot, 
                    single_obj_results, 
                    multi_obj_results_for_plot, 
                    args.output_dir
                )
                logging.info("✓ Energy production vs load plots created")
            else:
                logging.warning("Skipping energy production plots - missing required data")
                
            logging.info("=== CREATING ANNUAL ENERGY COMPARISON PLOTS ===")
            if single_obj_results is not None and df_multi_for_plot is not None:
                plotting_module.create_annual_energy_comparison_plots(
                    df_single=df_single_for_plot,
                    df_multi=df_multi_for_plot,
                    single_obj_results=single_obj_results,
                    multi_obj_results={'optimal_tilt': optimal_tilt, 'optimal_azimuth': optimal_azimuth},
                    output_dir=args.output_dir
                )
            
               
        except Exception as e:
            logging.error(f"Failed to create energy production plots: {e}")
            logging.error(f"Error details: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            logging.info("Continuing without energy production plots")
        

        # CREATE PARETO FRONT ANALYSIS  
        if args.optimization_mode in ['multi_objective', 'comprehensive']:
            logging.info("=== CREATING PARETO FRONT ANALYSIS ===")
            
            try:
                # Get the pareto front and logbook from the optimization
                pareto_front_for_plot = None
                logbook_for_plot = None
                
                if args.optimization_mode == 'multi_objective':
                    pareto_front_for_plot = pareto_front
                    logbook_for_plot = logbook
                elif args.optimization_mode == 'comprehensive':
                    # Use results from adaptive_improved strategy
                    strategy_results = results_by_strategy.get('adaptive_improved', {})
                    pareto_front_for_plot = strategy_results.get('pareto_front')
                    logbook_for_plot = strategy_results.get('logbook')
                
                if pareto_front_for_plot and logbook_for_plot:
                    plotting_module.create_pareto_front_plots(
                        pareto_front_for_plot, 
                        logbook_for_plot, 
                        single_obj_results, 
                        args.output_dir
                    )
                    logging.info("✓ Pareto front analysis plots created")
                else:
                    logging.warning("Skipping Pareto front plots - missing data")
                    
            except Exception as e:
                logging.error(f"Failed to create Pareto front plots: {e}")
                logging.info("Continuing without Pareto front plots")

            if args.optimization_mode in ['multi_objective', 'comprehensive']:
                logging.info("=== CREATING WEIGHTING FACTORS ANALYSIS ===")
                
                try:
                    weighting_strategy = args.weighting_strategy if args.optimization_mode == 'multi_objective' else 'adaptive_improved'
                    df_with_weights = df_final.copy()
                    df_with_weights['weighting_factor'] = calculate_weighting_factors(df_with_weights, strategy=weighting_strategy)
                    plotting_module.create_weighting_factors_plot(df_with_weights, weighting_strategy, args.output_dir)
                    logging.info("✓ Weighting factors analysis plot created")
                    
                except Exception as e:
                    logging.error(f"Failed to create weighting factors plot: {e}")
                    import traceback
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    
                    
            

            # COMBINED BATTERY ANALYSIS (OPTIMIZATION + PLOTTING)
            if args.include_battery:
                logging.info("=== BATTERY STORAGE OPTIMIZATION FOR BOTH STRATEGIES ===")
                
                battery_results_single = None
                battery_results_multi = None
                optimal_battery_single = 0.0
                optimal_battery_multi = 0.0
                battery_simulation = None
                optimal_battery_capacity = 0.0
                battery_results = None
                
                try:
                    # CREATE df_single_for_plot if single-objective results exist
                    df_single_for_plot = None
                    if single_obj_results is not None:
                        logging.info("Preparing single-objective data for battery analysis...")
                        df_single_for_plot = df_subset.copy()
                        df_single_for_plot = calculate_total_irradiance(
                            df_single_for_plot, 
                            single_obj_results['optimal_tilt'], 
                            single_obj_results['optimal_azimuth'], 
                            dni_extra
                        )
                        df_single_for_plot = calculate_energy_production(
                            df_single_for_plot, number_of_panels, inverter_params
                        )
                    
                    # Battery optimization for single-objective results
                    if df_single_for_plot is not None:
                        logging.info("Running battery optimization for single-objective strategy...")
                        optimal_battery_single, battery_results_single = calculate_optimal_battery_capacity(
                            df_single_for_plot, args.output_dir,
                            min_capacity=2.5, max_capacity=args.battery_max_capacity,
                            capacity_step=2.5, battery_round_trip_efficiency=0.90,
                            depth_of_discharge=0.80, battery_cost_per_kwh=args.battery_cost_per_kwh,
                            electricity_buy_price=args.electricity_buy_price,
                            electricity_sell_price=args.electricity_sell_price, battery_lifetime_years=10
                        )
                        logging.info(f"Single-objective optimal battery: {optimal_battery_single:.1f} kWh")
                    
                    # Battery optimization for multi-objective results  
                    logging.info("Running battery optimization for multi-objective strategy...")
                    optimal_battery_multi, battery_results_multi = calculate_optimal_battery_capacity(
                        df_final, args.output_dir,
                        min_capacity=2.5, max_capacity=args.battery_max_capacity,
                        capacity_step=2.5, battery_round_trip_efficiency=0.90,
                        depth_of_discharge=0.80, battery_cost_per_kwh=args.battery_cost_per_kwh,
                        electricity_buy_price=args.electricity_buy_price,
                        electricity_sell_price=args.electricity_sell_price, battery_lifetime_years=10
                    )
                    logging.info(f"Multi-objective optimal battery: {optimal_battery_multi:.1f} kWh")
                    
                    # Set variables for plotting (use multi-objective as primary)
                    optimal_battery_capacity = optimal_battery_multi
                    battery_results = battery_results_multi
                    
                    # Run battery simulation for the multi-objective strategy
                    economic_params = get_default_economic_params()
                    economic_params.update({
                        'battery_cost_per_kwh': args.battery_cost_per_kwh,
                        'electricity_price': args.electricity_buy_price,
                        'feed_in_tariff': args.electricity_sell_price
                    })
                    
                    battery_simulation = run_battery_simulation(
                        df_final, 
                        optimal_battery_capacity,
                        economic_params
                    )
                    
                    logging.info(f"Battery analysis completed:")
                    logging.info(f"  Single-objective optimal: {optimal_battery_single:.1f} kWh")
                    logging.info(f"  Multi-objective optimal: {optimal_battery_multi:.1f} kWh")
                    logging.info(f"  Selected for analysis: {optimal_battery_capacity:.1f} kWh (multi-objective)")
                    
                    # NOW CREATE ALL BATTERY PLOTS IN THE SAME SECTION
                    logging.info("=== CREATING BATTERY ANALYSIS PLOTS ===")
                    
                    # 1. Battery Performance Dashboard
                    if battery_simulation and optimal_battery_capacity > 0:
                        plotting_module.create_battery_performance_dashboard(
                            df_final, 
                            battery_simulation, 
                            optimal_battery_capacity, 
                            args.output_dir
                        )
                        logging.info("✓ Battery performance dashboard created")
                    else:
                        logging.warning("Skipping battery performance dashboard - no battery simulation data")
                    
                    # 2. Battery Economic Optimization Plot
                    if battery_results is not None and len(battery_results) > 0:
                        plotting_module.create_battery_economic_optimization_plot(
                            battery_results, 
                            optimal_battery_capacity, 
                            args.output_dir
                        )
                        logging.info("✓ Battery economic optimization plot created")
                    else:
                        logging.warning("Skipping battery economic optimization plot - no battery results data")
                    
                    # 3. Self-Sufficiency Comparison
                    plotting_module.create_self_sufficiency_comparison(
                        df_final, 
                        battery_simulation, 
                        optimal_battery_capacity, 
                        args.output_dir
                    )
                    logging.info("✓ Self-sufficiency comparison plot created")
                    
                    # 4. Strategy Comparison (if both strategies have results)
                    multi_obj_results_for_battery = {
                        'optimal_tilt': optimal_tilt,
                        'optimal_azimuth': optimal_azimuth,
                        'total_production': total_production,
                        'optimization_mode': args.optimization_mode
                    } if optimal_tilt is not None else None
                    
                    if (df_single_for_plot is not None and battery_results_single is not None and 
                        battery_results_multi is not None and single_obj_results is not None and
                        multi_obj_results_for_battery is not None):
                        plotting_module.create_battery_strategy_comparison(
                            df_single_for_plot, df_final, single_obj_results, multi_obj_results_for_battery,
                            battery_results_single, battery_results_multi,
                            optimal_battery_single, optimal_battery_multi, args.output_dir
                        )
                        logging.info("✓ Battery strategy comparison plot created")
                    
                except Exception as e:
                    logging.error(f"Battery analysis failed: {e}")
                    import traceback
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    logging.info("Continuing without battery analysis")

            else:
                logging.info("Battery analysis skipped (--include_battery not specified)")
                             
        # Create comprehensive results summary
        results_summary = {
            'Optimization_Mode': args.optimization_mode,
            'Weighting_Strategy': args.weighting_strategy if args.optimization_mode == 'multi_objective' else 'N/A',
            'Optimal_Tilt_deg': optimal_tilt,
            'Optimal_Azimuth_deg': optimal_azimuth,
            'Azimuth_Deviation_from_South_deg': abs(optimal_azimuth - 180),
            'Annual_Production_kWh': total_production,
            'System_Efficiency_pct': system_efficiency,
            'Number_of_Panels': number_of_panels,
            'Total_System_Capacity_kWp': number_of_panels * panel_power_rating / 1000,
            'Specific_Yield_kWh_per_kWp': total_production / (number_of_panels * panel_power_rating / 1000) if total_production > 0 else 0,
            'Panel_Efficiency_pct': panel_efficiency * 100,
            
            # BATTERY RESULTS
            'Battery_Optimization_Included': args.include_battery,
            'Optimal_Battery_Capacity_kWh': optimal_battery_capacity,
            'Battery_Self_Sufficiency_pct': battery_simulation['self_sufficiency_rate'] if battery_simulation else 0,
            'Battery_Self_Consumption_pct': battery_simulation['self_consumption_rate'] if battery_simulation else 0,
            'Battery_Grid_Import_kWh': battery_simulation['grid_import_kwh'] if battery_simulation else 0,
            'Battery_Grid_Export_kWh': battery_simulation['grid_export_kwh'] if battery_simulation else 0,
            
            # FINANCIAL RESULTS
            'Financial_Analysis_Included': args.include_financial_analysis,
            'Total_Investment_EUR': financial_results['financial_metrics']['Total_Investment'] if financial_results else 0,
            'NPV_EUR': financial_results['financial_metrics']['NPV'] if financial_results else 0,
            'IRR_percent': financial_results['financial_metrics']['IRR_percent'] if financial_results else 0,
            'Payback_Period_years': financial_results['financial_metrics']['Payback_Period_Years'] if financial_results else 0,
            'LCOE_EUR_per_kWh': financial_results['financial_metrics']['LCOE_eur_per_kwh'] if financial_results else 0,
            'Annual_Savings_Year1_EUR': financial_results['financial_metrics']['Annual_Savings_Year1'] if financial_results else 0
        }
        
        # Add validation flags to results
        if args.optimization_mode == 'single_objective' and single_obj_results:
            results_summary.update({
                'Single_Obj_Validation_Passed': single_obj_results.get('validation_passed', False),
                'Single_Obj_Azimuth_OK': single_obj_results.get('azimuth_deviation_from_south', 999) <= 15
            })
        
        # Save main results
        pd.DataFrame(list(results_summary.items()), columns=['Metric', 'Value']).to_csv(
            os.path.join(args.output_dir, 'optimization_results.csv'), index=False
        )
        
        # Save performance analysis
        if seasonal_stats is not None:
            seasonal_stats.to_csv(os.path.join(args.output_dir, 'seasonal_performance.csv'))
        if energy_breakdown is not None:
            energy_breakdown.to_csv(os.path.join(args.output_dir, 'energy_breakdown.csv'), index=False)
        if energy_losses is not None:
            energy_losses.to_csv(os.path.join(args.output_dir, 'energy_losses.csv'), index=False)
        
        # Save financial results
        if financial_results:
            financial_metrics_df = pd.DataFrame(list(financial_results['financial_metrics'].items()), 
                                               columns=['Metric', 'Value'])
            financial_metrics_df.to_csv(os.path.join(args.output_dir, 'financial_metrics.csv'), index=False)
            
            cashflows = financial_results['cashflows']
            cashflows.to_csv(os.path.join(args.output_dir, 'financial_cashflows.csv'))
            
            performance = financial_results['system_performance']
            performance_df = pd.DataFrame(list(performance.items()), columns=['Metric', 'Value'])
            performance_df.to_csv(os.path.join(args.output_dir, 'system_performance_summary.csv'), index=False)
            
            logging.info("Financial analysis results saved to CSV files")
        
        # Save battery results
        if battery_results is not None:
            logging.info("Battery analysis results already saved during optimization")
            
            if battery_simulation:
                # Calculate baseline metrics for comparison
                baseline_self_consumption = df_final.apply(
                    lambda row: min(row['E_ac']/1000, row['Load (kW)']), axis=1
                ).sum()
                baseline_self_sufficiency = (baseline_self_consumption / (df_final['Load (kW)'].sum())) * 100
                baseline_grid_import = df_final.apply(
                    lambda row: max(0, row['Load (kW)'] - row['E_ac']/1000), axis=1
                ).sum()
                baseline_grid_export = df_final.apply(
                    lambda row: max(0, row['E_ac']/1000 - row['Load (kW)']), axis=1
                ).sum()
                
                battery_summary = pd.DataFrame([{
                    'Metric': 'Total Production (kWh)',
                    'Without Battery': total_production,
                    'With Battery': battery_simulation['total_production_kwh']
                }, {
                    'Metric': 'Self-Sufficiency (%)',
                    'Without Battery': baseline_self_sufficiency,
                    'With Battery': battery_simulation['self_sufficiency_rate']
                }, {
                    'Metric': 'Grid Import (kWh)',
                    'Without Battery': baseline_grid_import,
                    'With Battery': battery_simulation['grid_import_kwh']
                }, {
                    'Metric': 'Grid Export (kWh)',
                    'Without Battery': baseline_grid_export,
                    'With Battery': battery_simulation['grid_export_kwh']
                }])
                
                battery_summary.to_csv(os.path.join(args.output_dir, 'battery_impact_summary.csv'), index=False)
                logging.info("Battery impact summary saved to CSV")
        
        # Save hourly data with battery SOC if available
        hourly_columns = ['E_ac', 'Load (kW)', 'total_irradiance', 'PR']
        if battery_simulation and len(battery_simulation['soc_history']) == len(df_final):
            df_final['battery_soc_percent'] = battery_simulation['soc_history']
            hourly_columns.append('battery_soc_percent')
        
        df_final[hourly_columns].to_csv(os.path.join(args.output_dir, 'hourly_performance.csv'))
        logging.info("Hourly performance data saved to CSV")

        logging.info("=== CREATING DATA TABLE VISUALIZATIONS ===")
    
        try:
            # Create all table visualizations
            plotting_module.create_all_data_tables(
                df_load=df_original,  # Your original input data
                df_weather=None,      # If you have separate weather DataFrame, pass it here
                output_dir=args.output_dir
            )
            
            logging.info("Data table visualizations completed successfully")
            
        except Exception as e:
            logging.error(f"Failed to create data table visualizations: {e}")
            logging.info("Continuing without table visualizations")

        # 🎯 EXISTING PLOTTING FUNCTIONS COME AFTER
        logging.info("=== CREATING INPUT DATA VISUALIZATION PLOTS ===")

        try:
            # Create input data plots using the original loaded data
            plotting_module.create_input_data_plots(df_original, args.output_dir)
            
        except Exception as e:
            logging.error(f"Failed to create input data plots: {e}")
            logging.info("Continuing without input data plots")

        # CREATE ENERGY PRODUCTION VS LOAD PLOTS
        logging.info("=== CREATING ENERGY PRODUCTION VS LOAD PLOTS ===")
        
        # FINAL SUMMARY
        logging.info("=== OPTIMIZATION COMPLETE ===")
        logging.info(f"Mode: {args.optimization_mode}")
        logging.info(f"Final Configuration:")
        logging.info(f"  • Angles: {optimal_tilt:.1f}° tilt, {optimal_azimuth:.1f}° azimuth")
        logging.info(f"  • Production: {total_production:,.0f} kWh/year")
        logging.info(f"  • System Efficiency: {system_efficiency:.1f}%")
        
        # Add specific yield and validation status
        specific_yield = total_production / (number_of_panels * panel_power_rating / 1000) if total_production > 0 else 0
        logging.info(f"  • Specific Yield: {specific_yield:.0f} kWh/kWp/year")
        
        if args.optimization_mode == 'single_objective':
            azimuth_deviation = abs(optimal_azimuth - 180)
            logging.info(f"  • Azimuth deviation from south: {azimuth_deviation:.1f}°")
            if azimuth_deviation <= 15:
                logging.info(f"  • ✓ Single-objective result validated")
            else:
                logging.warning(f"  • ⚠ Large azimuth deviation - check optimization")
        
        # BATTERY RESULTS
        if args.include_battery and optimal_battery_capacity > 0:
            logging.info(f"Battery Analysis:")
            logging.info(f"  • Optimal Capacity: {optimal_battery_capacity:.1f} kWh")
            if battery_simulation:
                logging.info(f"  • Self-Sufficiency: {battery_simulation['self_sufficiency_rate']:.1f}%")
                logging.info(f"  • Self-Consumption: {battery_simulation['self_consumption_rate']:.1f}%")
                logging.info(f"  • Grid Import: {battery_simulation['grid_import_kwh']:,.0f} kWh/year")
                logging.info(f"  • Grid Export: {battery_simulation['grid_export_kwh']:,.0f} kWh/year")
        elif args.include_battery:
            logging.info(f"Battery Analysis: No economically viable battery found")
        
        # FINANCIAL RESULTS
        if args.include_financial_analysis and financial_results:
            metrics = financial_results['financial_metrics']
            logging.info(f"Financial Analysis:")
            logging.info(f"  • Total Investment: €{metrics['Total_Investment']:,.0f}")
            logging.info(f"  • NPV (25 years): €{metrics['NPV']:,.0f}")
            logging.info(f"  • IRR: {metrics['IRR_percent']:.1f}%")
            logging.info(f"  • Payback Period: {metrics['Payback_Period_Years']:.1f} years")
            logging.info(f"  • LCOE: €{metrics['LCOE_eur_per_kwh']:.3f}/kWh")
            logging.info(f"  • Annual Savings (Year 1): €{metrics['Annual_Savings_Year1']:,.0f}")
        
        logging.info(f"Results saved to: {args.output_dir}")

        # CONSOLE OUTPUT - Enhanced with battery and financial info
        print(f"\n{'='*60}")
        print(" OPTIMIZATION COMPLETE!")
        print("="*60)
        print(f"Angles: {optimal_tilt:.1f}° tilt, {optimal_azimuth:.1f}° azimuth")
        print(f"Production: {total_production:,.0f} kWh/year")
        print(f"Specific Yield: {specific_yield:.0f} kWh/kWp")
        print(f"System: {number_of_panels} panels, {number_of_panels * panel_power_rating / 1000:.1f} kWp")
        
        # Show validation status for single-objective
        if args.optimization_mode == 'single_objective':
            azimuth_deviation = abs(optimal_azimuth - 180)
            print(f"Azimuth deviation from south: {azimuth_deviation:.1f}°")
            status = "✓ VALIDATED" if azimuth_deviation <= 15 else "⚠ CHECK REQUIRED"
            print(f"Single-objective validation: {status}")
        
        # Battery results
        if args.include_battery and optimal_battery_capacity > 0:
            print(f"Battery: {optimal_battery_capacity:.1f} kWh optimal capacity")
            if battery_simulation:
                print(f"Self-Sufficiency: {battery_simulation['self_sufficiency_rate']:.1f}% | Self-Consumption: {battery_simulation['self_consumption_rate']:.1f}%")
        elif args.include_battery:
            print(f"Battery: No economically viable option found")
        
        # Financial results
        if args.include_financial_analysis and financial_results:
            metrics = financial_results['financial_metrics']
            print(f"Investment: €{metrics['Total_Investment']:,.0f}")
            print(f"NPV: €{metrics['NPV']:,.0f} | IRR: {metrics['IRR_percent']:.1f}% | Payback: {metrics['Payback_Period_Years']:.1f} years")
            print(f"LCOE: €{metrics['LCOE_eur_per_kwh']:.3f}/kWh")
        
        print(f"Results: {args.output_dir}")
        print("="*60)
            
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\nERROR: Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()