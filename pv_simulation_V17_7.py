#pv_simulation_V17_2Constrained_FIXED.py

import argparse
import pandas as pd
import pvlib
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import os
import itertools
import numpy as np  # For creating heatmaps
import seaborn as sns  # For enhanced plotting
from scipy.optimize import differential_evolution
import locale
import yaml
import sys
from constants import TOTAL_LOSS_FACTOR, NOCT, TIME_INTERVAL_HOURS
from deap import base, creator, tools, algorithms
import multiprocessing
import random
from deap.algorithms import varOr
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class OptimizationContext:
    def __init__(self, df_subset, dni_extra, number_of_panels, inverter_params):
        self.df_subset = df_subset
        self.dni_extra = dni_extra
        self.number_of_panels = number_of_panels
        self.inverter_params = inverter_params

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

def evaluate_individual(individual):
    """
    Top-level evaluation function for DEAP.
    
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
        optimization_context.inverter_params)

def setup_logging(output_dir):
    """
    Set up logging to file and console.

    Parameters:
    - output_dir (str): Directory where the log file will be saved.
    """
    log_file = os.path.join(output_dir, 'solar_analysis.log')
    
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Prevent adding multiple handlers if setup_logging is called multiple times
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)
        c_handler.setLevel(logging.INFO)  # Console handler set to INFO
        f_handler.setLevel(logging.DEBUG)  # File handler set to DEBUG

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

def load_and_preprocess_data(data_file):
    """
    Load the data from CSV and preprocess it.

    Parameters:
    - data_file (str): Path to the CSV data file.

    Returns:
    - df (DataFrame): Preprocessed DataFrame with datetime index.
    """
    # Load the data
    try:
        df = pd.read_csv(data_file)
        logging.info(f"Data loaded from {data_file}")
    except Exception as e:
        logging.error(f"Error loading data file: {e}", exc_info=True)
        raise
    
    # Ensure correct data types
    numeric_columns = ['hour', 'SolRad_Hor', 'SolRad_Dif', 'Air Temp', 'WS_10m', 'Load (kW)']
    for col in numeric_columns:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the data file.")
            raise ValueError(f"'{col}' column is missing in the data file.")
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Convert 'hour' column to datetime starting from January 1st, 00:00
    start_date = datetime(2023, 1, 1, 0, 0)
    try:
        df['datetime'] = pd.to_datetime(df['hour'] - 1, unit='h', origin=start_date)
        logging.info("'hour' column converted to datetime.")
    except Exception as e:
        logging.error(f'Error converting hour to datetime: {e}', exc_info=True)
        raise
    
    # Set 'datetime' as the index
    df.set_index('datetime', inplace=True)
    
    # Localize to 'Europe/Athens' timezone, handling DST transitions
    try:
        df.index = df.index.tz_localize('Europe/Athens', ambiguous='NaT', nonexistent='shift_forward')
        logging.info("Datetime index localized to 'Europe/Athens' timezone.")
    except Exception as e:
        logging.error(f'Error localizing timezone: {e}', exc_info=True)
        raise
    
    # Drop rows with NaT in the index (ambiguous times)
    df = df[~df.index.isna()]
    if df.empty:
        logging.error("All rows have NaT after timezone localization. Exiting.")
        raise ValueError("All rows have NaT after timezone localization.")
    
    # Sort the index
    df = df.sort_index()
    logging.info("DataFrame index sorted.")
    
    # Detect and handle duplicate timestamps
    duplicates = df.index.duplicated(keep='first')
    if duplicates.any():
        logging.warning(f"Number of duplicate timestamps: {duplicates.sum()}")
        # Remove duplicates by keeping the first occurrence
        df = df[~duplicates]
        logging.info("Duplicate timestamps removed by keeping the first occurrence.")
        # Alternatively, aggregate duplicates:
        # df = df.groupby(df.index).mean()
    
    # Set the frequency of the datetime index to hourly using lowercase 'h'
    try:
        df = df.asfreq('h')  # Lowercase 'h' for hourly frequency
        logging.info("DataFrame frequency set to hourly.")
    except Exception as e:
        logging.error(f"Error setting frequency: {e}", exc_info=True)
        raise
    
    # Data validation: Handle missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        logging.warning('Data contains missing values. Proceeding to interpolate missing values.')
        # Forward-fill and backward-fill missing values
        df[numeric_columns] = df[numeric_columns].ffill().bfill()
        logging.info("Missing values interpolated and filled.")
    else:
        logging.info('No missing values detected.')
    
    # Ensure no missing values remain
    if df[numeric_columns].isnull().values.any():
        logging.error('Missing values remain after interpolation and filling. Exiting.')
        raise ValueError('Missing values remain in data.')
    
    # Enforce 'E_ac' column as float and fill missing values with 0
    if 'E_ac' not in df.columns:
        logging.warning("'E_ac' column is missing in the data file. Creating 'E_ac' with default value 0.")
        df['E_ac'] = 0.0
    else:
        df['E_ac'] = pd.to_numeric(df['E_ac'], errors='coerce').fillna(0.0).astype(float)
        logging.info("'E_ac' column enforced as float and missing values filled with 0.")
    
    return df

def calculate_solar_position(df, latitude, longitude):
    """
    Calculate solar position (zenith and azimuth angles) and add them to the DataFrame.

    Parameters:
    - df (DataFrame): DataFrame with datetime index.
    - latitude (float): Latitude of the location in decimal degrees.
    - longitude (float): Longitude of the location in decimal degrees.

    Returns:
    - df (DataFrame): DataFrame with 'zenith' and 'azimuth' columns added.
    """
    try:
        solar_position = pvlib.solarposition.get_solarposition(df.index, latitude, longitude)
        df['zenith'] = solar_position['apparent_zenith']
        df['azimuth'] = solar_position['azimuth']
        logging.info("Solar position (zenith and azimuth) calculated and added to DataFrame.")
    except Exception as e:
        logging.error(f"Error calculating solar position: {e}", exc_info=True)
        raise
    
    return df

def calculate_dni(df):
    """
    Calculate DNI using pvlib's disc() function and add it to the DataFrame.

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
        # Correctly pass 'solar_zenith' and 'datetime_or_doy'
        dni = pvlib.irradiance.disc(
            ghi=df['SolRad_Hor'],
            solar_zenith=df['zenith'],
            datetime_or_doy=df.index  # Provide datetime index
        )['dni']
        df['DNI'] = dni
        logging.info("DNI calculated and added to DataFrame.")
    except Exception as e:
        logging.error(f"Error calculating DNI: {e}", exc_info=True)
        raise

    return df

def calculate_weighting_factors(df, strategy='adaptive_improved'):
    """
    IMPROVED: Better weighting factors to address mismatch issues.
    
    Your current mismatch improvement is -6.27%, indicating SE orientation 
    isn't optimal for load matching. This improved version provides better 
    strategies for aligning production with consumption patterns.
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

    if strategy == 'adaptive_improved':
        # IMPROVED: Less time bias, more load-shape focus
        if len(daylight_load) > 1:
            min_load = daylight_load.min()
            max_load = daylight_load.max()

            if max_load != min_load:
                normalized_load = (daylight_load - min_load) / (max_load - min_load)
            else:
                normalized_load = pd.Series(0.5, index=daylight_load.index)

            # REDUCED time bias - less preference for midday
            time_factor = np.exp(-0.5 * ((hours - 13) / 4) ** 2)  # Wider, less aggressive
            time_factor = pd.Series(time_factor, index=daylight_load.index)
            time_factor = (time_factor - time_factor.min()) / (time_factor.max() - time_factor.min())

            # MORE emphasis on load matching (80% vs previous 70%)
            combined_weight = 0.8 * normalized_load + 0.2 * time_factor
            
            if combined_weight.max() != combined_weight.min():
                combined_weight = (combined_weight - combined_weight.min()) / (combined_weight.max() - combined_weight.min())
            
            # WIDER range for more differentiation
            final_weights = 0.1 + 0.9 * combined_weight
            df.loc[daylight_mask, 'weighting_factor'] = final_weights
    
    elif strategy == 'pure_load_matching':
        # PURE load shape matching - no time bias at all
        if daylight_load.max() != daylight_load.min():
            weights = (daylight_load - daylight_load.min()) / (daylight_load.max() - daylight_load.min())
            weights = 0.2 + 0.8 * weights
        else:
            weights = pd.Series(0.5, index=daylight_load.index)
        df.loc[daylight_mask, 'weighting_factor'] = weights
    
    elif strategy == 'peak_focused':
        # FOCUS on peak demand periods
        base_weight = 0.2
        if daylight_load.max() != daylight_load.min():
            load_norm = (daylight_load - daylight_load.min()) / (daylight_load.max() - daylight_load.min())
        else:
            load_norm = pd.Series(0.5, index=daylight_load.index)
        
        # Identify peak hours (top 20% of load)
        peak_threshold = daylight_load.quantile(0.8)
        peak_multiplier = pd.Series(1.0, index=daylight_load.index)
        peak_multiplier[daylight_load >= peak_threshold] = 2.0
        
        weights = base_weight + 0.8 * load_norm * peak_multiplier
        df.loc[daylight_mask, 'weighting_factor'] = weights

    avg_weight = df.loc[daylight_mask, 'weighting_factor'].mean()
    logging.info(f"Weighting strategy '{strategy}': Average weight = {avg_weight:.3f}")
    
    return df['weighting_factor']

def calculate_weighted_energy(df):
    # Parameters:
    # - df (DataFrame): DataFrame containing the following columns:
    #     - 'E_ac': AC energy output in watts.
    #- 'weighting_factor': Pre-calculated weighting factors.

    # Returns:
    # - total_weighted_energy (float): Total weighted energy production in kilowatt-hours (kWh).

    # Ensure necessary columns are present
    for col in ['E_ac', 'weighting_factor']:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the DataFrame.")
            raise ValueError(f"'{col}' column is missing in the DataFrame.")

    # Calculate weighted energy production
    weighted_energy = (df['E_ac'] * df['weighting_factor']).sum() / 1000  # Convert Wh to kWh

    return weighted_energy

def calculate_total_irradiance(df, tilt_angle, azimuth_angle, dni_extra):
    """
    Calculate total irradiance on the tilted surface.

    Parameters:
    - df (DataFrame): DataFrame with necessary solar data.
    - tilt_angle (float): Tilt angle in degrees.
    - azimuth_angle (float): Azimuth angle in degrees.
    - dni_extra (Series): Extraterrestrial DNI values.

    Returns:
    - df (DataFrame): DataFrame with 'total_irradiance' column added.
    """
    required_columns = ['zenith', 'azimuth', 'DNI', 'SolRad_Hor', 'SolRad_Dif']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the DataFrame.")
            raise ValueError(f"'{col}' column is missing in the DataFrame.")

    try:
        irradiance_data = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt_angle,
            surface_azimuth=azimuth_angle,
            solar_zenith=df['zenith'],
            solar_azimuth=df['azimuth'],
            dni=df['DNI'],
            ghi=df['SolRad_Hor'],
            dhi=df['SolRad_Dif'],
            dni_extra=dni_extra,
            model='haydavies'
        )
        df['total_irradiance'] = irradiance_data['poa_global']
        logging.info(f"Total irradiance calculated with tilt {tilt_angle}° and azimuth {azimuth_angle}°.")
    except Exception as e:
        logging.error(f"Error calculating total irradiance: {e}", exc_info=True)
        raise

    return df

def calculate_energy_production(df, number_of_panels, inverter_params):
    """
    FIXED: Calculate energy production with proper PR calculation order.
    This version calculates PR before attempting to validate it.
    """
    required_columns = ['total_irradiance', 'Air Temp']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the DataFrame.")
            raise ValueError(f"'{col}' column is missing in the DataFrame.")

    try:
        # Panel parameters from Sharp ND-R240A5 datasheet
        panel_area = 1.642  # m²
        panel_efficiency_stc = 0.146  # 14.6% at STC
        panel_power_stc = 240  # Wp at STC
        NOCT = 47.5  # °C
        TEMP_COEFF_PMAX = -0.00440  # -0.440% / °C
        
        # System parameters
        total_panel_area = panel_area * number_of_panels
        total_system_power_stc = panel_power_stc * number_of_panels  # Wp
        
        # Loss factors (as fractions, not percentages)
        soiling_factor = 0.98      # 2% soiling loss
        shading_factor = 0.97      # 3% shading loss  
        reflection_factor = 0.97   # 3% reflection loss
        mismatch_factor = 0.98     # 2% mismatch loss
        dc_wiring_factor = 0.98    # 2% DC wiring loss
        
        # Combined pre-temperature losses
        pre_temp_efficiency = soiling_factor * shading_factor * reflection_factor * mismatch_factor * dc_wiring_factor
        
        # Time interval
        TIME_INTERVAL_HOURS = 1
        
        # Step 1: Calculate incident solar energy on panel plane
        df['incident_irradiance'] = df['total_irradiance']  # W/m²
        df['incident_energy'] = df['incident_irradiance'] * total_panel_area * TIME_INTERVAL_HOURS  # Wh
        
        # Step 2: Calculate ideal DC output at STC (no temperature effects)
        # This is what panels would produce at 25°C with the given irradiance
        df['dc_power_ideal_stc'] = df['incident_irradiance'] * total_panel_area * panel_efficiency_stc * pre_temp_efficiency  # W
        
        # Step 3: Calculate cell temperature and temperature losses
        df['cell_temperature'] = df['Air Temp'] + ((NOCT - 20) / 800) * df['incident_irradiance']
        df['temperature_factor'] = 1 + TEMP_COEFF_PMAX * (df['cell_temperature'] - 25)
        df['temperature_factor'] = df['temperature_factor'].clip(lower=0)  # Can't be negative
        
        # Step 4: Apply temperature derating to get actual DC output
        df['dc_power_actual'] = df['dc_power_ideal_stc'] * df['temperature_factor']  # W
        
        # Ensure we don't exceed rated capacity
        df['dc_power_actual'] = df['dc_power_actual'].clip(upper=total_system_power_stc)
        
        # Step 5: Apply inverter efficiency
        inverter_efficiency = inverter_params['eta_inv_nom'] / 100
        df['ac_power_output'] = df['dc_power_actual'] * inverter_efficiency  # W
        
        # Apply inverter clipping
        inverter_max_ac = inverter_params['pdc0'] * inverter_efficiency
        df['ac_power_output'] = df['ac_power_output'].clip(upper=inverter_max_ac)
        
        # Calculate energies
        df['E_incident'] = df['incident_energy']  # Already calculated above
        df['E_dc_ideal'] = df['dc_power_ideal_stc'] * TIME_INTERVAL_HOURS  # Wh
        df['E_dc_actual'] = df['dc_power_actual'] * TIME_INTERVAL_HOURS  # Wh
        df['E_ac'] = df['ac_power_output'] * TIME_INTERVAL_HOURS  # Wh
        
        # CORRECTED LOSS CALCULATIONS
        # These represent energy NOT produced due to each factor
        
        # Pre-temperature losses (soiling, shading, reflection, mismatch, DC wiring)
        # This is the difference between what could be produced with clean panels and actual
        df['E_loss_pre_temperature'] = df['incident_energy'] * panel_efficiency_stc - df['E_dc_ideal']
        
        # Temperature losses
        df['E_loss_temperature'] = df['E_dc_ideal'] - df['E_dc_actual']
        
        # Inverter losses
        df['E_loss_inverter'] = df['E_dc_actual'] - df['E_ac']
        
        # Total losses (should equal incident energy × panel efficiency at STC - actual AC output)
        df['E_loss_total'] = (df['incident_energy'] * panel_efficiency_stc) - df['E_ac']
        
        # FIXED: Calculate PR BEFORE the validation check
        # Performance Ratio calculation
        # PR = Actual Energy / (Incident Irradiation × Rated Power)
        # Where incident irradiation is in kWh/m² (hours of equivalent 1000 W/m² sun)
        df['reference_yield'] = df['incident_irradiance'] * TIME_INTERVAL_HOURS / 1000  # hours

        df['PR'] = np.where(
            df['reference_yield'] > 0,
            df['E_ac'] / (df['reference_yield'] * total_system_power_stc),
            0
        )
        df['PR'] = df['PR'].clip(0, 1)

        # NOW it's safe to check PR (this was the problematic line 502)
        if 'PR' in df.columns and len(df['PR']) > 0:
            avg_pr = df['PR'].mean()
            if avg_pr < 0.1:
                logging.error("=== EMERGENCY PR DIAGNOSTIC ===")
                sample_idx = df.index[100] if len(df) > 100 else df.index[0]
                logging.error(f"Sample calculation:")
                logging.error(f"  E_ac: {df.loc[sample_idx, 'E_ac']:.1f} Wh")
                logging.error(f"  Incident irradiance: {df.loc[sample_idx, 'incident_irradiance']:.1f} W/m²")
                logging.error(f"  Reference yield: {df.loc[sample_idx, 'reference_yield']:.3f} hours")
                logging.error(f"  Total system power: {total_system_power_stc:.0f} W")
                logging.error(f"  Expected PR: {df.loc[sample_idx, 'E_ac'] / (df.loc[sample_idx, 'reference_yield'] * total_system_power_stc):.6f}")
                logging.error("Check if total_system_power_stc is calculated correctly!")
        
        # CRITICAL DEBUG: Log PR calculation components
        logging.debug(f"PR Calculation Debug:")
        logging.debug(f"  Total system power STC: {total_system_power_stc} W")
        logging.debug(f"  Number of panels: {number_of_panels}")
        logging.debug(f"  Panel power STC: {panel_power_stc} W")
        logging.debug(f"  Reference yield range: {df['reference_yield'].min():.3f} - {df['reference_yield'].max():.3f} hours")
        logging.debug(f"  E_ac range: {df['E_ac'].min():.1f} - {df['E_ac'].max():.1f} Wh")

        # CRITICAL DEBUG: Check PR results
        pr_stats = {
            'min': df['PR'].min(),
            'max': df['PR'].max(),
            'mean': df['PR'].mean(),
            'zero_count': (df['PR'] == 0).sum(),
            'valid_count': (df['PR'] > 0).sum()
        }
        logging.debug(f"PR Results: {pr_stats}")

        if pr_stats['mean'] < 0.1:  # Less than 10% is definitely wrong
            logging.error(f"CRITICAL PR ISSUE: Mean PR is only {pr_stats['mean']:.3f} ({pr_stats['mean']*100:.1f}%)")
            logging.error("This indicates a serious calculation error in energy production or PR formula")
        
        # Add per-panel metrics for reference
        df['dc_power_output_per_panel'] = df['dc_power_actual'] / number_of_panels
        df['ac_power_output'] = df['ac_power_output']  # Keep for compatibility
        df['dc_power_output'] = df['dc_power_actual']  # Keep for compatibility
        
        logging.info(f"Energy calculations completed. Average PR: {df['PR'].mean():.3f}")
        
    except Exception as e:
        logging.error(f"Error calculating energy production: {e}", exc_info=True)
        raise
    
    return df

def summarize_energy(df):
    """
    REWRITTEN: Summarize energy flows with correct loss accounting.
    """
    try:
        # Sum up energies (convert to kWh)
        total_incident = df['E_incident'].sum() / 1000  # kWh
        total_dc_ideal = df['E_dc_ideal'].sum() / 1000  # kWh  
        total_dc_actual = df['E_dc_actual'].sum() / 1000  # kWh
        total_ac = df['E_ac'].sum() / 1000  # kWh
        
        # Calculate system efficiency metrics
        # Overall system efficiency = AC out / Solar in
        system_efficiency = (total_ac / total_incident) * 100 if total_incident > 0 else 0
        
        # Calculate losses (in kWh)
        pre_temp_losses = df['E_loss_pre_temperature'].sum() / 1000
        temp_losses = df['E_loss_temperature'].sum() / 1000
        inverter_losses = df['E_loss_inverter'].sum() / 1000
        total_losses = df['E_loss_total'].sum() / 1000
        
        # Calculate average Performance Ratio
        avg_pr = df[df['PR'] > 0]['PR'].mean() * 100  # Convert to percentage
        
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
                f"{total_incident * 0.146:,.0f}",  # Panel efficiency at STC
                f"{total_dc_ideal:,.0f}",
                f"{total_dc_actual:,.0f}",
                f"{total_ac:,.0f}"
            ],
            'Efficiency (%)': [
                '100.0',
                '14.6',
                f"{(total_dc_ideal/total_incident)*100:.1f}",
                f"{(total_dc_actual/total_incident)*100:.1f}",
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
            'Energy Lost (kWh)': [  # Store raw numeric values here
                pre_temp_losses,    # Should be a float or int
                temp_losses,        # Should be a float or int
                inverter_losses,    # Should be a float or int
                total_losses        # Should be a float or int
            ],
            'Percentage of Input': [ # This column is for display, f-strings are fine
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
        raise

def objective_function_multi(angles, df_subset, dni_extra, number_of_panels, inverter_params, 
                            weighting_strategy='adaptive_improved', azimuth_preference_strength='medium'):
    """
    FIXED: Multi-objective function that never returns infinite values.
    Uses penalty approach instead of infinite returns for bounds violations.
    """
    try:
        tilt_angle, azimuth_angle = angles
        
        # FIXED: Use penalty approach instead of returning infinite values
        penalty = 0.0
        
        # Apply bounds penalties but don't return inf
        if tilt_angle < 0:
            penalty += abs(tilt_angle) * 10000
            tilt_angle = 0
        elif tilt_angle > 90:
            penalty += (tilt_angle - 90) * 10000
            tilt_angle = 90
            
        if azimuth_angle < 90:
            penalty += (90 - azimuth_angle) * 1000
            azimuth_angle = 90
        elif azimuth_angle > 270:
            penalty += (azimuth_angle - 270) * 1000
            azimuth_angle = 270

        # Calculate performance with corrected angles
        df_temp = df_subset.copy()
        df_temp = calculate_total_irradiance(df_temp, tilt_angle, azimuth_angle, dni_extra)
        df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)
        
        total_energy_production = df_temp['E_ac'].sum() / 1000  # kWh
        
        # IMPROVED: Use better weighting strategy
        df_temp['weighting_factor'] = calculate_weighting_factors(df_temp, strategy=weighting_strategy)
        df_temp['load_wh'] = df_temp['Load (kW)'] * 1000
        df_temp['hourly_mismatch'] = df_temp['E_ac'] - df_temp['load_wh']
        df_temp['weighted_mismatch'] = df_temp['weighting_factor'] * np.abs(df_temp['hourly_mismatch'] / 1000)
        
        total_weighted_mismatch = df_temp['weighted_mismatch'].sum()
        
        # Add penalties to mismatch
        total_weighted_mismatch += penalty
        
        # IMPROVED: Dynamic azimuth penalty scaling
        azimuth_deviation = abs(azimuth_angle - 180)
        
        if azimuth_preference_strength == 'weak':
            penalty_factor = 0.01
        elif azimuth_preference_strength == 'medium':
            penalty_factor = 0.05  # 5x stronger than your current 0.01
        elif azimuth_preference_strength == 'strong':
            penalty_factor = 0.1
        else:
            penalty_factor = 0.05
        
        # SCALE penalty based on mismatch magnitude
        base_mismatch_scale = total_weighted_mismatch / 100000  # Normalize to your data scale
        scaled_penalty_factor = penalty_factor * min(base_mismatch_scale, 2.0)  # Cap the scaling
        
        azimuth_penalty = scaled_penalty_factor * (azimuth_deviation ** 2)
        
        adjusted_mismatch = total_weighted_mismatch + azimuth_penalty
        
        # CRITICAL: Ensure finite values
        if not np.isfinite(adjusted_mismatch):
            adjusted_mismatch = 1e6  # Large penalty instead of inf
        if not np.isfinite(total_energy_production):
            total_energy_production = 0.0  # Zero production instead of -inf
            
        return (adjusted_mismatch, total_energy_production)

    except Exception as e:
        logging.error(f"Error in objective_function_multi: {e}", exc_info=True)
        # FIXED: Return penalty values instead of inf/-inf
        return (1e6, 0.0)

def run_deap_multi_objective_optimization(
    df_subset,
    dni_extra,
    number_of_panels,
    inverter_params,
    output_dir,
    pop_size=None,    # Allow dynamic population size
    max_gen=None      # Allow dynamic number of generations
):
    """
    FIXED: Run multi-objective optimization with proper constraint handling.
    Uses bounded operators and penalty methods instead of infinite values.
    """

    # ----------------------------------------------------------------
    # 1. Dynamic defaults for pop_size / max_gen
    if pop_size is None:
        pop_size = 50
    if max_gen is None:
        max_gen = 30

    # ----------------------------------------------------------------
    # 2. Create an optimization context object
    context = OptimizationContext(
        df_subset=df_subset,
        dni_extra=dni_extra,
        number_of_panels=number_of_panels,
        inverter_params=inverter_params
    )

    # ----------------------------------------------------------------
    # 3. Set up DEAP 'creator' for 2-objective optimization (if not already created)
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    # ----------------------------------------------------------------
    # 4. FIXED: Toolbox with bounded operators
    toolbox = base.Toolbox()

    # FIXED: Bounded individual creation
    def create_bounded_individual():
        """Create individual within valid bounds"""
        tilt = random.uniform(0, 90)      # Valid tilt range
        azimuth = random.uniform(90, 270) # Valid azimuth range  
        return creator.Individual([tilt, azimuth])

    # Register functions with bounds
    toolbox.register("individual", create_bounded_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    
    # FIXED: Use bounded operators
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                     eta=20.0, low=[0, 90], up=[90, 270])
    toolbox.register("mutate", tools.mutPolynomialBounded, 
                     eta=20.0, low=[0, 90], up=[90, 270], indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    
    # FIXED: Add bounds checking decorator as safety net
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
    
    toolbox.decorate("mate", checkBounds([0, 90], [90, 270]))
    toolbox.decorate("mutate", checkBounds([0, 90], [90, 270]))

    # Optional: Parallelize
    pool = multiprocessing.Pool(initializer=init_worker, initargs=(context,))
    toolbox.register("map", pool.map)

    # ----------------------------------------------------------------
    # 5. Initialize population
    population = toolbox.population(n=pop_size)

    # ----------------------------------------------------------------
    # 6. Prepare statistics and HallOfFame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    hof = tools.ParetoFront()

    # ----------------------------------------------------------------
    # 7. Run the global custom NSGA-II evolution loop
    final_pop, logbook = custom_nsga2_evolution(
        population,
        toolbox,
        cxpb=0.7,               # Crossover probability
        mutpb_start=0.2,        # Starting mutation probability
        mutpb_end=0.05,         # Ending mutation probability
        ngen=max_gen,
        stats=stats,
        halloffame=hof
    )

    # Close and join the multiprocessing pool
    pool.close()
    pool.join()

    # ----------------------------------------------------------------
    # 8. Extract final Pareto front and perform post-processing
    pareto_front = list(hof)

    # Filter solutions based on a production threshold (e.g., >= 2000 kWh)
    production_threshold = 2000.0  # kWh; adjust as needed
    filtered_front = [ind for ind in pareto_front if ind.fitness.values[1] >= production_threshold]
    logging.info(f"Filtered front: {len(filtered_front)} of {len(pareto_front)} pass production >= {production_threshold}")

    # Select the "most balanced" solution (minimizing the difference between normalized objectives)
    if filtered_front:
        mismatch_vals = np.array([ind.fitness.values[0] for ind in filtered_front])
        prod_vals     = np.array([ind.fitness.values[1] for ind in filtered_front])
        # Normalize both objectives
        mismatch_norm = (mismatch_vals - mismatch_vals.min()) / (mismatch_vals.ptp() + 1e-9)
        prod_norm     = (prod_vals - prod_vals.min()) / (prod_vals.ptp() + 1e-9)
        diff = np.abs(mismatch_norm - prod_norm)
        best_idx = np.argmin(diff)
        best_balanced = filtered_front[best_idx]
    else:
        best_balanced = None

    # ----------------------------------------------------------------
    # 9. Save the Pareto front to CSV
    rows = []
    for ind in pareto_front:
        mismatch, production = ind.fitness.values
        rows.append({
            'tilt_angle': ind[0],
            'azimuth_angle': ind[1],
            'weighted_mismatch_kWh': mismatch,
            'total_energy_production_kWh': production
        })
    pareto_df = pd.DataFrame(rows)
    outfile = os.path.join(output_dir, 'pareto_front_results.csv')
    pareto_df.to_csv(outfile, index=False)
    logging.info(f"Pareto front results saved to {outfile}")

    # ----------------------------------------------------------------
    # 10. Return results
    return pareto_front, filtered_front, best_balanced

def custom_nsga2_evolution(pop, toolbox, cxpb, mutpb_start, mutpb_end, ngen, stats, halloffame):
    """
    Custom NSGA-II evolution loop with adaptive mutation probability.

    Parameters:
      - pop: The initial population.
      - toolbox: DEAP toolbox with evaluation, mating, mutation, and selection operators.
      - cxpb: Crossover probability.
      - mutpb_start: Starting mutation probability.
      - mutpb_end: Ending mutation probability.
      - ngen: Number of generations.
      - stats: DEAP statistics object.
      - halloffame: DEAP HallOfFame object.
    
    Returns:
      - pop: The final evolved population.
      - logbook: Logbook of the evolution process.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Evaluate the initial population.
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))  # Assign crowding distance.
    if halloffame is not None:
        halloffame.update(pop)
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    
    # FIXED: Log meaningful values instead of inf/-inf
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

def run_constrained_optimization(df_subset, dni_extra, number_of_panels, inverter_params, 
                               output_dir, max_mismatch=120000):
    """
    FIXED: Constrained optimization approach that doesn't use objective_function_multi.
    Creates its own safe objective function to avoid the PR KeyError.
    """
    logging.info(f"Running constrained optimization with max mismatch = {max_mismatch} kWh")
    
    def safe_constrained_objective(angles):
        tilt_angle, azimuth_angle = angles
        
        # Apply bounds with penalties
        penalty = 0.0
        if tilt_angle < 0:
            penalty += abs(tilt_angle) * 10000
            tilt_angle = 0
        elif tilt_angle > 90:
            penalty += (tilt_angle - 90) * 10000
            tilt_angle = 90
            
        if azimuth_angle < 90:
            penalty += (90 - azimuth_angle) * 1000
            azimuth_angle = 90
        elif azimuth_angle > 270:
            penalty += (azimuth_angle - 270) * 1000
            azimuth_angle = 270
        
        try:
            # Calculate performance - use safe copy to avoid modifying original
            df_temp = df_subset.copy()
            df_temp = calculate_total_irradiance(df_temp, tilt_angle, azimuth_angle, dni_extra)
            df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)
            
            total_energy_production = df_temp['E_ac'].sum() / 1000
            
            # Calculate mismatch with pure load matching
            df_temp['weighting_factor'] = calculate_weighting_factors(df_temp, strategy='pure_load_matching')
            df_temp['load_wh'] = df_temp['Load (kW)'] * 1000
            df_temp['hourly_mismatch'] = df_temp['E_ac'] - df_temp['load_wh']
            df_temp['weighted_mismatch'] = df_temp['weighting_factor'] * np.abs(df_temp['hourly_mismatch'] / 1000)
            
            total_weighted_mismatch = df_temp['weighted_mismatch'].sum() + penalty
            
            # PENALTY for constraint violation
            if total_weighted_mismatch > max_mismatch:
                penalty_extra = (total_weighted_mismatch - max_mismatch) * 10  # Heavy penalty
                return -total_energy_production + penalty_extra  # Minimize this
            else:
                return -total_energy_production  # Maximize production
                
        except Exception as e:
            logging.error(f"Error in constrained objective: {e}")
            return 1e6  # Large penalty instead of crashing
    
    # Use scipy optimization for constrained case
    from scipy.optimize import differential_evolution
    
    bounds = [(0, 90), (90, 270)]  # [tilt, azimuth]
    
    result = differential_evolution(
        safe_constrained_objective,
        bounds,
        maxiter=100,
        popsize=15,
        seed=42
    )
    
    optimal_angles = result.x
    optimal_value = result.fun
    
    logging.info(f"Constrained optimization result:")
    logging.info(f"  Optimal angles: Tilt={optimal_angles[0]:.2f}°, Azimuth={optimal_angles[1]:.2f}°")
    logging.info(f"  Estimated production: {-optimal_value:.0f} kWh")
    
    return optimal_angles

def calculate_actual_power_output(df, system_params):
    """Calculate actual power output with proper validation"""
    # Apply system losses (inverter, wiring, soiling, etc.)
    system_losses = system_params.get('system_losses', 0.2)  # 20% typical losses
    
    # Calculate DC power first
    dc_power = df['irradiance'] * df['panel_area'] * system_params['panel_efficiency'] / 1000
    
    # Apply system losses to get AC power
    ac_power = dc_power * (1 - system_losses)
    
    return ac_power

def create_robust_objective_function():
    """Create objective function that never returns infinite values"""
    
    def safe_objective_function_multi(individual):
        """Robust multi-objective function for PV optimization"""
        tilt, azimuth = individual[0], individual[1]
        
        try:
            # Validate bounds first - use penalties instead of infinite values
            bounds_penalty = 0
            
            # Tilt bounds checking (0-90 degrees)
            if tilt < 0:
                bounds_penalty += abs(tilt) * 1000
                tilt = 0
            elif tilt > 90:
                bounds_penalty += (tilt - 90) * 1000
                tilt = 90
            
            # Azimuth bounds checking (90-270 degrees)  
            if azimuth < 90:
                bounds_penalty += (90 - azimuth) * 100
                azimuth = 90
            elif azimuth > 270:
                bounds_penalty += (azimuth - 270) * 100
                azimuth = 270
            
            # Calculate PV system performance with valid parameters
            weather_data = load_weather_data()  # Your weather data source
            system_params = {
                'tilt': tilt,
                'azimuth': azimuth,
                'panel_efficiency': 0.2,
                'panel_area': 100,
                'system_losses': 0.2,
                'temp_coefficient': -0.004
            }
            
            # Use fixed calculate_energy_production function
            total_production, avg_pr = calculate_energy_production(weather_data, system_params)
            
            # Calculate mismatch (deviation from target production)
            target_production = system_params.get('target_production', 50000)  # kWh/year
            mismatch = abs(total_production - target_production) + bounds_penalty
            
            # Ensure finite values
            if not np.isfinite(mismatch):
                mismatch = 1e6  # Large penalty instead of inf
            if not np.isfinite(total_production):
                total_production = 0.0  # Zero production instead of -inf
            
            return mismatch, total_production
            
        except Exception as e:
            logging.error(f"Objective function error with tilt={tilt}, azimuth={azimuth}: {e}")
            return 1e6, 0.0  # Return penalty values instead of crashing
    
    return safe_objective_function_multi

def plot_pareto_front(pareto_front, filtered_front, best_balanced, output_dir):
    """
    Plot the Pareto front showing the trade-off between the two objectives.
    """
    mismatch = [ind.fitness.values[0] for ind in pareto_front]
    production = [ind.fitness.values[1] for ind in pareto_front]

    plt.figure(figsize=(10, 6))
    plt.scatter(mismatch, production, c='blue', alpha=0.7, label='All Pareto Solutions')

    # Highlight the filtered subset, if any
    if filtered_front:
        fmismatch = [ind.fitness.values[0] for ind in filtered_front]
        fproduction = [ind.fitness.values[1] for ind in filtered_front]
        plt.scatter(fmismatch, fproduction, c='orange', alpha=0.9, edgecolors='k', label='Filtered Solutions')

    # Highlight the "best balanced" solution, if available
    if best_balanced is not None:
        plt.scatter(
            best_balanced.fitness.values[0],
            best_balanced.fitness.values[1],
            c='red', s=100, marker='*', label='Most Balanced'
        )

    plt.xlabel('Weighted Energy Mismatch (kWh) [Lower is Better]')
    plt.ylabel('Total Energy Production (kWh) [Higher is Better]')
    plt.title('Pareto Front (Mismatch vs. Production)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_front.png'))
    plt.close()
    logging.info("Pareto front plot saved to pareto_front.png")

def save_summary_results(pareto_front, output_dir):

    """
    Save a summary of Pareto front results to a CSV file.
    """
    pareto_data = []
    for ind in pareto_front:
        tilt, azimuth = ind
        mismatch, production = ind.fitness.values
        pareto_data.append({
            'tilt_angle': tilt,
            'azimuth_angle': azimuth,
            'weighted_energy_mismatch_kWh': mismatch,
            'total_energy_production_kWh': production
        })
    
    pareto_df = pd.DataFrame(pareto_data)
    output_file = os.path.join(output_dir, 'pareto_front_summary.csv')
    pareto_df.to_csv(output_file, index=False, sep=';', decimal='.', float_format='%.2f')
    logging.info(f"Pareto front summary saved to {output_file}")

def perform_sensitivity_analysis(param_name, param_values, fixed_params, df_subset, dni_extra,
                                 number_of_panels, inverter_params, output_dir):
    """
    Perform sensitivity analysis by varying a specific parameter and observing its impact on optimization results.
    """
    results = []
    for value in param_values:
        try:
            if param_name == 'tilt_angle':
                fixed_tilt = value
                # Define a new evaluation function that fixes the tilt angle
                def evaluate_fixed_tilt(individual):
                    azimuth = individual[0]
                    return objective_function_multi(
                        [fixed_tilt, azimuth],
                        df_subset,
                        dni_extra,
                        number_of_panels,
                        inverter_params
                    )
                # Create unique class names for this iteration
                fitness_name = f"FitnessMulti_{param_name}_{value:.1f}"
                individual_name = f"Individual_{param_name}_{value:.1f}"
                
                # Set up DEAP for this fixed tilt
                try:
                    # Check if the classes already exist and only create if they don't
                    if not hasattr(creator, fitness_name):
                        creator.create(fitness_name, base.Fitness, weights=(-1.0, 1.0))
                    if not hasattr(creator, individual_name):
                        creator.create(individual_name, list, fitness=getattr(creator, fitness_name))
                except Exception as e:
                    logging.warning(f"Error creating DEAP classes: {e}. Continuing with analysis.")
                
                toolbox_sens = base.Toolbox()
                toolbox_sens.register("attr_azimuth", np.random.uniform, 90, 270)
                toolbox_sens.register("individual", tools.initCycle,
                                      getattr(creator, individual_name),
                                      (toolbox_sens.attr_azimuth,), n=1)
                toolbox_sens.register("population", tools.initRepeat, list, toolbox_sens.individual)
                toolbox_sens.register("evaluate", evaluate_fixed_tilt)
                toolbox_sens.register("mate", tools.cxBlend, alpha=0.5)
                toolbox_sens.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
                toolbox_sens.register("select", tools.selNSGA2)

                population = toolbox_sens.population(n=50)
                hof = tools.ParetoFront()
                algorithms.eaMuPlusLambda(population, toolbox_sens,
                                          mu=50,
                                          lambda_=100,
                                          cxpb=0.7,
                                          mutpb=0.2,
                                          ngen=30,
                                          halloffame=hof,
                                          verbose=False)
                for ind in hof:
                    azimuth = ind[0]
                    mismatch, production = ind.fitness.values
                    results.append({
                        param_name: value,
                        'azimuth_angle': azimuth,
                        'weighted_energy_mismatch_kWh': mismatch,
                        'total_energy_production_kWh': production
                    })
                # Instead of trying to delete the classes, we simply don't delete them
                # They'll remain in the creator module, but with properly formatted unique names

            elif param_name == 'azimuth_angle':
                fixed_azimuth = value
                # Define a new evaluation function that fixes the azimuth angle
                def evaluate_fixed_azimuth(individual):
                    tilt = individual[0]
                    return objective_function_multi(
                        [tilt, fixed_azimuth],
                        df_subset,
                        dni_extra,
                        number_of_panels,
                        inverter_params
                    )
                
                # Create unique class names for this iteration
                fitness_name = f"FitnessMulti_{param_name}_{value:.1f}"
                individual_name = f"Individual_{param_name}_{value:.1f}"
                
                # Set up DEAP for this fixed azimuth
                try:
                    # Check if the classes already exist and only create if they don't
                    if not hasattr(creator, fitness_name):
                        creator.create(fitness_name, base.Fitness, weights=(-1.0, 1.0))
                    if not hasattr(creator, individual_name):
                        creator.create(individual_name, list, fitness=getattr(creator, fitness_name))
                except Exception as e:
                    logging.warning(f"Error creating DEAP classes: {e}. Continuing with analysis.")
                
                toolbox_sens = base.Toolbox()
                toolbox_sens.register("attr_tilt", np.random.uniform, 0, 90)
                toolbox_sens.register("individual", tools.initCycle,
                                      getattr(creator, individual_name),
                                      (toolbox_sens.attr_tilt,), n=1)
                toolbox_sens.register("population", tools.initRepeat, list, toolbox_sens.individual)
                toolbox_sens.register("evaluate", evaluate_fixed_azimuth)
                toolbox_sens.register("mate", tools.cxBlend, alpha=0.5)
                toolbox_sens.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
                toolbox_sens.register("select", tools.selNSGA2)

                population = toolbox_sens.population(n=50)
                hof = tools.ParetoFront()
                algorithms.eaMuPlusLambda(population, toolbox_sens,
                                          mu=50,
                                          lambda_=100,
                                          cxpb=0.7,
                                          mutpb=0.2,
                                          ngen=30,
                                          halloffame=hof,
                                          verbose=False)
                for ind in hof:
                    tilt = ind[0]
                    mismatch, production = ind.fitness.values
                    results.append({
                        param_name: value,
                        'tilt_angle': tilt,
                        'weighted_energy_mismatch_kWh': mismatch,
                        'total_energy_production_kWh': production
                    })
                # Instead of trying to delete the classes, we simply don't delete them

            else:
                logging.error(f"Sensitivity analysis for parameter '{param_name}' is not supported.")
                continue

        except Exception as e:
            logging.error(f"Error during sensitivity analysis for {param_name}={value}: {e}", exc_info=True)
            continue

    sensitivity_df = pd.DataFrame(results)
    sensitivity_df.to_csv(os.path.join(output_dir, f'sensitivity_{param_name}.csv'),
                          index=False, sep=';', decimal='.', float_format='%.2f')
    logging.info(f"Sensitivity analysis results saved to sensitivity_{param_name}.csv")

    plt.figure(figsize=(10, 6))
    if param_name == 'tilt_angle':
        plt.plot(sensitivity_df['tilt_angle'], sensitivity_df['weighted_energy_mismatch_kWh'], label='Mismatch', marker='o')
        plt.plot(sensitivity_df['tilt_angle'], sensitivity_df['total_energy_production_kWh'], label='Production', marker='o')
        plt.xlabel('Tilt Angle (°)')
    elif param_name == 'azimuth_angle':
        plt.plot(sensitivity_df['azimuth_angle'], sensitivity_df['weighted_energy_mismatch_kWh'], label='Mismatch', marker='o')
        plt.plot(sensitivity_df['azimuth_angle'], sensitivity_df['total_energy_production_kWh'], label='Production', marker='o')
        plt.xlabel('Azimuth Angle (°)')
    plt.ylabel('Performance Metrics')
    plt.title(f'Sensitivity Analysis: {param_name.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sensitivity_{param_name}.png'), dpi=300)
    plt.close()
    logging.info(f"Sensitivity analysis plot saved to sensitivity_{param_name}.png")

def plot_consumption_profile(df, output_dir):
    """
    Plot the consumption profile to identify peak hours and understand how it aligns with solar production.

    Parameters:
    - df (DataFrame): DataFrame containing 'Load (kW)' and 'E_ac'.
    - output_dir (str): Directory to save the plot.
    """
    # Resample data to daily sums for visualization
    df_daily = df.resample('D').sum()
    
    # Check if df_daily is empty
    if df_daily.empty:
        logging.warning("No data available to plot consumption profile.")
        return
    
    # Plot daily consumption and solar production
    plt.figure(figsize=(12, 6))
    plt.plot(df_daily.index, df_daily['Load (kW)'], label='Daily Consumption (kWh)', color='red')
    plt.plot(df_daily.index, df_daily['E_ac'] / 1000, label='Daily Solar Production (kWh)', color='green')  # Convert Wh to kWh
    plt.title('Daily Consumption and Solar Production')
    plt.xlabel('Date')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    consumption_plot_file = os.path.join(output_dir, 'daily_consumption_and_production.png')
    plt.savefig(consumption_plot_file, dpi=300)
    plt.close()
    logging.info(f"Consumption profile plot saved to {consumption_plot_file}")

def plot_energy_losses(energy_losses, output_dir):
    """
    Plot energy losses.

    Parameters:
    - energy_losses (DataFrame): DataFrame with energy losses.
    - output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    loss_labels = energy_losses['Loss Type']
    losses = energy_losses['Energy Lost (kWh)']
    plt.bar(loss_labels, losses, color=['orange', 'red', 'purple'])
    plt.title('Annual Energy Losses by Type')
    plt.ylabel('Energy Lost (kWh)')
    plt.tight_layout()
    losses_plot_file = os.path.join(output_dir, 'annual_energy_losses.png')
    plt.savefig(losses_plot_file, dpi=300)
    plt.close()
    logging.info(f"Energy losses plot saved to {losses_plot_file}")

def plot_daily_irradiance_and_energy(df, output_dir):
    """
    Plot daily total irradiance and energy production over the year.

    Parameters:
    - df (DataFrame): DataFrame with datetime index and necessary columns.
    - output_dir (str): Directory to save the plot.
    """
    # Resample data to daily sums for better visualization
    df_daily = df.resample('D').sum()
    
    # Check if df_daily is empty
    if df_daily.empty:
        logging.warning("No data available to plot daily irradiance and energy production.")
        return
    
    # Plot daily total irradiance and daily energy production
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:orange'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Irradiance (kWh/m²/day)', color=color)
    ax1.plot(df_daily.index, df_daily['total_irradiance'] / 1000, color=color, label='Total Irradiance')  # Convert Wh/m² to kWh/m²
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('AC Energy Production (kWh/day)', color=color)
    ax2.plot(df_daily.index, df_daily['ac_power_output'] / 1000, color=color, label='Energy Production')  # Convert Wh to kWh
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')
    
    plt.title('Daily Total Irradiance and Energy Production Over the Year')
    plt.tight_layout()
    
    # Save the plot
    daily_plot_file = os.path.join(output_dir, 'daily_irradiance_and_energy_production.png')
    plt.savefig(daily_plot_file, dpi=300)
    plt.close()
    logging.info(f"Daily irradiance and energy production plot saved to {daily_plot_file}")

def plot_average_daily_consumption(df, output_dir):
    """
    Plot average energy consumption over an average day.

    Parameters:
    - df (DataFrame): DataFrame containing 'Load (kW)'.
    - output_dir (str): Directory to save the plot.
    """
    # Ensure 'Load (kW)' is in the DataFrame
    if 'Load (kW)' not in df.columns:
        logging.error("'Load (kW)' column is missing in the DataFrame.")
        raise ValueError("'Load (kW)' column is missing in the DataFrame.")

    # Extract hour from the index without modifying the DataFrame
    hours = df.index.hour

    # Calculate average consumption per hour
    hourly_consumption = df.groupby(hours)['Load (kW)'].mean()

    # Plot the average daily consumption
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_consumption.index, hourly_consumption.values, marker='o', color='blue')
    plt.title('Average Energy Consumption Over an Average Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Consumption (kW)')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, 'average_daily_consumption.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()
    logging.info(f"Average daily consumption plot saved to {plot_file}")

def plot_average_hourly_consumption_vs_production(df, output_dir):
    """
    Plot average hourly consumption vs average hourly production.

    Parameters:
    - df (DataFrame): DataFrame containing 'Load (kW)' and 'ac_power_output'.
    - output_dir (str): Directory to save the plot.
    """
    # Ensure necessary columns are present
    if 'Load (kW)' not in df.columns or 'ac_power_output' not in df.columns:
        logging.error("DataFrame must contain 'Load (kW)' and 'ac_power_output' columns.")
        raise ValueError("DataFrame must contain 'Load (kW)' and 'ac_power_output' columns.")

    # Extract hour from the index without modifying the DataFrame
    hours = df.index.hour

    # Calculate average consumption and production per hour
    hourly_consumption = df.groupby(hours)['Load (kW)'].mean()
    hourly_production = df.groupby(hours)['ac_power_output'].mean() / 1000  # Convert W to kW

    # Plot the average hourly consumption and production
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_consumption.index, hourly_consumption.values, label='Average Consumption (kW)', marker='o')
    plt.plot(hourly_production.index, hourly_production.values, label='Average Production (kW)', marker='o')
    plt.title('Average Hourly Consumption vs. Average Hourly Production')
    plt.xlabel('Hour of Day')
    plt.ylabel('Power (kW)')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, 'average_hourly_consumption_vs_production.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()
    logging.info(f"Average hourly consumption vs production plot saved to {plot_file}")

def plot_average_hourly_weighting_factors(df, output_dir):
    """
    Plot average hourly weighting factors.

    Parameters:
    - df (DataFrame): DataFrame containing 'weighting_factor'.
    - output_dir (str): Directory to save the plot.
    """
    # Ensure 'weighting_factor' is in the DataFrame
    if 'weighting_factor' not in df.columns:
        logging.error("'weighting_factor' column is missing in the DataFrame.")
        raise ValueError("'weighting_factor' column is missing in the DataFrame.")

    # Extract hour from the index without modifying the DataFrame
    hours = df.index.hour

    # Calculate average weighting factor per hour
    hourly_weighting = df.groupby(hours)['weighting_factor'].mean()

    # Plot the average hourly weighting factors
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_weighting.index, hourly_weighting.values, label='Average Weighting Factor', marker='o', color='purple')
    plt.title('Average Hourly Weighting Factors')
    plt.xlabel('Hour of Day')
    plt.ylabel('Weighting Factor')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, 'average_hourly_weighting_factors.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()
    logging.info(f"Average hourly weighting factors plot saved to {plot_file}")

def plot_combined_hourly_data(df, output_dir):
    """
    Plot average hourly consumption, production, and weighting factors.

    Parameters:
    - df (DataFrame): DataFrame containing 'Load (kW)', 'ac_power_output', and 'weighting_factor'.
    - output_dir (str): Directory to save the plot.
    """
    # Ensure necessary columns are present
    required_columns = ['Load (kW)', 'ac_power_output', 'weighting_factor']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"DataFrame must contain '{col}' column.")
            raise ValueError(f"DataFrame must contain '{col}' column.")

    # Extract hour from the index without modifying the DataFrame
    hours = df.index.hour

    # Calculate average values per hour
    hourly_consumption = df.groupby(hours)['Load (kW)'].mean()
    hourly_production = df.groupby(hours)['ac_power_output'].mean() / 1000  # Convert W to kW
    hourly_weighting = df.groupby(hours)['weighting_factor'].mean()

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot consumption and production
    ax1.plot(hourly_consumption.index, hourly_consumption.values, label='Average Consumption (kW)', marker='o', color='red')
    ax1.plot(hourly_production.index, hourly_production.values, label='Average Production (kW)', marker='o', color='green')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Power (kW)')
    ax1.set_xticks(range(0, 24))
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Create a twin axis for weighting factors
    ax2 = ax1.twinx()
    ax2.plot(hourly_weighting.index, hourly_weighting.values, label='Average Weighting Factor', marker='o', color='purple')
    ax2.set_ylabel('Weighting Factor')
    ax2.legend(loc='upper right')

    plt.title('Average Hourly Consumption, Production, and Weighting Factors')
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, 'combined_hourly_data.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()
    logging.info(f"Combined hourly data plot saved to {plot_file}")

def plot_representative_day_profiles(df, output_dir, date_str):
    """
    Plot consumption and production profiles for a representative day.

    Parameters:
    - df (DataFrame): DataFrame containing 'Load (kW)' and 'ac_power_output'.
    - output_dir (str): Directory to save the plot.
    - date_str (str): Date string in 'YYYY-MM-DD' format representing the day to plot.

    Raises:
    - ValueError: If the specified date is not in the DataFrame.
    """
    # Ensure necessary columns are present
    if 'Load (kW)' not in df.columns or 'ac_power_output' not in df.columns:
        logging.error("DataFrame must contain 'Load (kW)' and 'ac_power_output' columns.")
        raise ValueError("DataFrame must contain 'Load (kW)' and 'ac_power_output' columns.")

    try:
        # Convert date string to pandas Timestamp with the same timezone as the DataFrame
        date = pd.to_datetime(date_str).tz_localize('Europe/Athens')
        # Extract the day's data using string-based slicing
        df_day = df.loc[date_str]
        if df_day.empty:
            logging.error(f"No data available for the specified date: {date_str}")
            raise ValueError(f"No data available for the specified date: {date_str}")
    except KeyError:
        logging.error(f"No data available for the specified date: {date_str}")
        raise ValueError(f"No data available for the specified date: {date_str}")
    except Exception as e:
        logging.error(f"Error extracting data for date {date_str}: {e}", exc_info=True)
        raise

    # Convert 'ac_power_output' from W to kW
    df_day = df_day.copy()  # Avoid SettingWithCopyWarning
    df_day['ac_power_output_kW'] = df_day['ac_power_output'] / 1000

    # Plot consumption and production profiles
    plt.figure(figsize=(10, 6))
    plt.plot(df_day.index.hour, df_day['Load (kW)'], label='Consumption (kW)', marker='o', color='red')
    plt.plot(df_day.index.hour, df_day['ac_power_output_kW'], label='Production (kW)', marker='o', color='green')
    plt.title(f'Consumption and Production Profiles on {date_str}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Power (kW)')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, f'consumption_production_{date_str}.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()
    logging.info(f"Representative day profiles plot saved to {plot_file}")

def plot_hourly_heatmaps(df, output_dir):
    """
    Create heatmaps of hourly consumption and production over the year.

    Parameters:
    - df (DataFrame): DataFrame containing 'Load (kW)' and 'ac_power_output'.
    - output_dir (str): Directory to save the plots.
    """
    # Ensure necessary columns are present
    if 'Load (kW)' not in df.columns or 'ac_power_output' not in df.columns:
        logging.error("DataFrame must contain 'Load (kW)' and 'ac_power_output' columns.")
        raise ValueError("DataFrame must contain 'Load (kW)' and 'ac_power_output' columns.")

    # Prepare data for heatmaps
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['ac_power_output_kW'] = df['ac_power_output'] / 1000
    df['day_of_year'] = df.index.dayofyear
    df['hour'] = df.index.hour

    # Pivot tables for consumption and production
    try:
        consumption_pivot = df.pivot_table(index='hour', columns='day_of_year', values='Load (kW)')
        production_pivot = df.pivot_table(index='hour', columns='day_of_year', values='ac_power_output_kW')
        logging.info("Pivot tables for heatmaps created.")
    except Exception as e:
        logging.error(f"Error creating pivot tables for heatmaps: {e}", exc_info=True)
        raise

    # Create heatmap for consumption
    try:
        plt.figure(figsize=(15, 6))
        sns.heatmap(consumption_pivot, cmap='Reds')
        plt.title('Hourly Consumption Heatmap')
        plt.xlabel('Day of Year')
        plt.ylabel('Hour of Day')
        plt.tight_layout()
        consumption_heatmap_file = os.path.join(output_dir, 'hourly_consumption_heatmap.png')
        plt.savefig(consumption_heatmap_file, dpi=300)
        plt.close()
        logging.info(f"Hourly consumption heatmap saved to {consumption_heatmap_file}")
    except Exception as e:
        logging.error(f"Error creating hourly consumption heatmap: {e}", exc_info=True)
        raise

    # Create heatmap for production
    try:
        plt.figure(figsize=(15, 6))
        sns.heatmap(production_pivot, cmap='Greens')
        plt.title('Hourly Production Heatmap')
        plt.xlabel('Day of Year')
        plt.ylabel('Hour of Day')
        plt.tight_layout()
        production_heatmap_file = os.path.join(output_dir, 'hourly_production_heatmap.png')
        plt.savefig(production_heatmap_file, dpi=300)
        plt.close()
        logging.info(f"Hourly production heatmap saved to {production_heatmap_file}")
    except Exception as e:
        logging.error(f"Error creating hourly production heatmap: {e}", exc_info=True)
        raise

def plot_tilt_vs_energy(tilt_results_df, output_dir):
    """
    Plot Annual Energy Production vs. Tilt Angle.

    Parameters:
    - tilt_results_df (DataFrame): DataFrame with tilt angles and corresponding annual energy production.
    - output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(tilt_results_df['tilt_angle'], tilt_results_df['annual_energy'], marker='o', color='blue')
    plt.title('Annual Energy Production vs. Tilt Angle')
    plt.xlabel('Tilt Angle (°)')
    plt.ylabel('Annual Energy Production (kWh)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    tilt_plot_file = os.path.join(output_dir, 'annual_energy_vs_tilt_angle.png')
    plt.savefig(tilt_plot_file, dpi=300)
    plt.close()
    logging.info(f"Annual energy vs. tilt angle plot saved to {tilt_plot_file}")

def create_heatmap(angle_results_df, output_dir):
    """
    Create a heatmap to visualize how weighted energy production varies with tilt and azimuth angles.

    Parameters:
    - angle_results_df (DataFrame): DataFrame containing 'tilt_angle', 'azimuth_angle', and 'weighted_energy'.
    - output_dir (str): Directory to save the heatmap.
    """
    try:
        # Pivot the DataFrame to create a matrix suitable for a heatmap
        heatmap_data = angle_results_df.pivot(index='tilt_angle', columns='azimuth_angle', values='weighted_energy')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='viridis')
        plt.title('Weighted Energy Production Heatmap')
        plt.xlabel('Azimuth Angle (°)')
        plt.ylabel('Tilt Angle (°)')
        plt.tight_layout()
    
        # Save the heatmap
        heatmap_file = os.path.join(output_dir, 'weighted_energy_heatmap.png')
        plt.savefig(heatmap_file, dpi=300)
        plt.close()
        logging.info(f"Weighted energy production heatmap saved to {heatmap_file}")
    except Exception as e:
        logging.error(f"Error creating heatmap: {e}", exc_info=True)
        raise

def plot_pareto_front_with_efficiency(pareto_front, output_dir):
    # For demonstration, we assign a placeholder random efficiency value
    mismatch_vals = []
    production_vals = []
    efficiency_vals = []
    for ind in pareto_front:
        mismatch, production = ind.fitness.values[:2]
        system_eff = random.uniform(10, 20)  # placeholder efficiency
        mismatch_vals.append(mismatch)
        production_vals.append(production)
        efficiency_vals.append(system_eff)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        mismatch_vals, production_vals, c=efficiency_vals, cmap='viridis',
        s=80, alpha=0.8, edgecolors='k'
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('System Efficiency (%)')
    ax.set_xlabel('Weighted Energy Mismatch (kWh)')
    ax.set_ylabel('Total Energy Production (kWh)')
    ax.set_title('Pareto Front with Efficiency Heatmap')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_front_with_efficiency.png'))
    plt.show()

def plot_detailed_pareto_front(pareto_front, output_dir):
    """
    A sample detailed Pareto front plot.
    This function could, for example, annotate each point with its index or additional metrics.
    """
    mismatch = [ind.fitness.values[0] for ind in pareto_front]
    production = [ind.fitness.values[1] for ind in pareto_front]
    
    plt.figure(figsize=(12, 8))
    plt.scatter(mismatch, production, c='green', alpha=0.8, label='Pareto Solutions')
    for i, ind in enumerate(pareto_front):
        plt.annotate(str(i), (mismatch[i], production[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Weighted Energy Mismatch (kWh)')
    plt.ylabel('Total Energy Production (kWh)')
    plt.title('Detailed Pareto Front Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    detailed_plot_file = os.path.join(output_dir, 'detailed_pareto_front.png')
    plt.savefig(detailed_plot_file)
    plt.close()
    logging.info(f"Detailed Pareto front plot saved to {detailed_plot_file}")

def plot_pareto_front_comparison(pareto_front, grid_search_results_path, output_dir):
    """
    Compare the Pareto front obtained from the DEAP optimization with grid search results.
    
    This function expects the grid search CSV to contain columns for both the weighted mismatch 
    and the total energy production. It first attempts to read the CSV using a semicolon delimiter.
    """
    # Extract mismatch and production from the Pareto front
    pareto_mismatch = [ind.fitness.values[0] for ind in pareto_front]
    pareto_production = [ind.fitness.values[1] for ind in pareto_front]

    # Read the grid search results from CSV using semicolon as delimiter
    try:
        grid_df = pd.read_csv(grid_search_results_path, sep=';')
        logging.info(f"Grid search CSV columns: {grid_df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Error reading grid search results from {grid_search_results_path}: {e}", exc_info=True)
        return

    # Build a mapping from lower-case, stripped header names to their original names
    col_mapping = {col.strip().lower(): col for col in grid_df.columns}

    # Look for the mismatch column under the two possible names
    expected_mismatch = None
    if 'weighted_energy_mismatch_kwh' in col_mapping:
        expected_mismatch = col_mapping['weighted_energy_mismatch_kwh']
    elif 'weighted_mismatch_kwh' in col_mapping:
        expected_mismatch = col_mapping['weighted_mismatch_kwh']

    # Look for the production column
    expected_production = col_mapping.get('total_energy_production_kwh', None)

    if expected_mismatch is None or expected_production is None:
        logging.error("Grid search CSV is missing required columns.")
        return

    grid_mismatch = grid_df[expected_mismatch].tolist()
    grid_production = grid_df[expected_production].tolist()

    # Create a plot comparing the two sets of solutions
    plt.figure(figsize=(10, 6))
    plt.scatter(pareto_mismatch, pareto_production, c='blue', alpha=0.7, label='Pareto Front (DEAP)')
    plt.scatter(grid_mismatch, grid_production, c='red', alpha=0.7, label='Grid Search Results')
    plt.xlabel('Weighted Energy Mismatch (kWh)')
    plt.ylabel('Total Energy Production (kWh)')
    plt.title('Pareto Front Comparison: DEAP vs. Grid Search')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    comparison_file = os.path.join(output_dir, 'pareto_front_comparison.png')
    plt.savefig(comparison_file, dpi=300)
    plt.close()
    logging.info(f"Pareto front comparison plot saved to {comparison_file}")

def run_grid_search(df_subset, dni_extra, number_of_panels, inverter_params, output_dir):
    import numpy as np
    import pandas as pd
    import logging

    # Define your grid. Adjust the range and number of points as needed.
    tilt_values = np.linspace(0, 90, 10)         # 10 tilt angles from 0° to 90°
    azimuth_values = np.linspace(90, 270, 10)      # 10 azimuth angles from 90° to 270°
    
    results = []
    for tilt in tilt_values:
        for azimuth in azimuth_values:
            try:
                # Calculate the objectives using your objective_function_multi
                mismatch, production = objective_function_multi(
                    [tilt, azimuth],
                    df_subset,
                    dni_extra,
                    number_of_panels,
                    inverter_params
                )
                results.append({
                    'tilt_angle': tilt,
                    'azimuth_angle': azimuth,
                    'weighted_mismatch_kWh': mismatch,
                    'total_energy_production_kWh': production
                })
            except Exception as e:
                logging.error(f"Error processing tilt={tilt}, azimuth={azimuth}: {e}", exc_info=True)

    grid_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "angle_results.csv")
    grid_df.to_csv(output_file, index=False, sep=';')
    logging.info(f"Grid search results saved to {output_file}")
    return output_file

def select_balanced_solution(pareto_front):
    """
    Selects the solution from the Pareto front that clearly balances both objectives.
    """
    if not pareto_front:
        logging.error("Pareto front is empty. Cannot select a balanced solution.")
        return None

    mismatch_vals = np.array([ind.fitness.values[0] for ind in pareto_front])
    production_vals = np.array([ind.fitness.values[1] for ind in pareto_front])

    # Normalize mismatch and production clearly
    mismatch_norm = (mismatch_vals - mismatch_vals.min()) / (mismatch_vals.ptp() + 1e-9)
    production_norm = (production_vals - production_vals.min()) / (production_vals.ptp() + 1e-9)

    # Here we explicitly adjust emphasis on production
    balance_score = mismatch_norm * 0.3 + (1 - production_norm) * 0.7  # 70% emphasis on maximizing production

    best_idx = np.argmin(balance_score)
    best_balanced_solution = pareto_front[best_idx]

    logging.info(f"Selected clearly balanced solution with tilt={best_balanced_solution[0]:.2f}, azimuth={best_balanced_solution[1]:.2f}, "
                 f"mismatch={mismatch_vals[best_idx]:.2f}, production={production_vals[best_idx]:.2f}")

    return best_balanced_solution

def analyze_seasonal_performance(df):
    """
    Analyze seasonal variations in production, consumption, and matching.
    
    Parameters:
    - df (DataFrame): DataFrame with datetime index and required columns:
                     'E_ac', 'Load (kW)', 'ac_power_output', etc.
                     
    Returns:
    - seasonal_stats (DataFrame): DataFrame with seasonal performance metrics
    - daily_seasonal (DataFrame): DataFrame with daily averages by season
    """
    # Make a copy to avoid modifying the original dataframe
    df_season = df.copy()
    
    # Add time-based columns
    df_season['month'] = df_season.index.month
    df_season['day_of_year'] = df_season.index.dayofyear
    
    # Define seasons (Northern Hemisphere)
    # Winter: Dec(12), Jan(1), Feb(2)
    # Spring: Mar(3), Apr(4), May(5)
    # Summer: Jun(6), Jul(7), Aug(8)
    # Fall: Sep(9), Oct(10), Nov(11)
    season_mapping = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall',
        11: 'Fall', 12: 'Winter'
    }
    df_season['season'] = df_season['month'].map(season_mapping)
    
    # Calculate mismatch between production and consumption
    df_season['load_wh'] = df_season['Load (kW)'] * 1000  # Convert kW to W for comparison
    df_season['mismatch'] = df_season['ac_power_output'] - df_season['load_wh']
    df_season['surplus'] = df_season['mismatch'].clip(lower=0)
    df_season['deficit'] = (-df_season['mismatch']).clip(lower=0)
    
    # Calculate self-consumption and self-sufficiency
    df_season['consumed_solar'] = np.minimum(df_season['ac_power_output'], df_season['load_wh'])
    
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
    
    # Calculate daily averages by season - Fixed version to avoid datetime issues
    # Instead of grouping by date, we'll use a more direct approach:
    # Add a date column as string (not datetime.date object)
    df_season['date_str'] = df_season.index.strftime('%Y-%m-%d')
    
    # Now group by season and date string, then calculate daily sums
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

def plot_seasonal_production_consumption(seasonal_stats, output_dir):
    """
    Plot seasonal production vs consumption.
    
    Parameters:
    - seasonal_stats (DataFrame): DataFrame with seasonal stats
    - output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar chart
    seasons = seasonal_stats.index
    x = np.arange(len(seasons))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, seasonal_stats['E_ac_kwh'], width, label='Production (kWh)', color='green')
    plt.bar(x + width/2, seasonal_stats['load_wh_kwh'], width, label='Consumption (kWh)', color='red')
    
    # Customize plot
    plt.xlabel('Season')
    plt.ylabel('Energy (kWh)')
    plt.title('Seasonal Energy Production vs Consumption')
    plt.xticks(x, seasons)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(seasonal_stats['E_ac_kwh']):
        plt.text(i - width/2, v + 50, f'{v:.0f}', ha='center', va='bottom')
    
    for i, v in enumerate(seasonal_stats['load_wh_kwh']):
        plt.text(i + width/2, v + 50, f'{v:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_production_consumption.png'), dpi=300)
    plt.close()
    logging.info("Seasonal production vs consumption plot saved")

def plot_seasonal_self_consumption(seasonal_stats, output_dir):
    """
    Plot seasonal self-consumption and self-sufficiency ratios.
    
    Parameters:
    - seasonal_stats (DataFrame): DataFrame with seasonal stats
    - output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create line plot with markers
    plt.plot(seasonal_stats.index, seasonal_stats['self_consumption_ratio'], 
             'o-', label='Self-Consumption Ratio (%)', color='blue', linewidth=2, markersize=10)
    plt.plot(seasonal_stats.index, seasonal_stats['self_sufficiency_ratio'], 
             's-', label='Self-Sufficiency Ratio (%)', color='orange', linewidth=2, markersize=10)
    
    # Customize plot
    plt.xlabel('Season')
    plt.ylabel('Ratio (%)')
    plt.title('Seasonal Self-Consumption and Self-Sufficiency')
    plt.ylim(0, 110)  # Give some space above 100%
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add value labels for each point
    for i, season in enumerate(seasonal_stats.index):
        plt.text(i, seasonal_stats.loc[season, 'self_consumption_ratio'] + 3, 
                 f'{seasonal_stats.loc[season, "self_consumption_ratio"]:.1f}%', 
                 ha='center', va='bottom')
        plt.text(i, seasonal_stats.loc[season, 'self_sufficiency_ratio'] - 3, 
                 f'{seasonal_stats.loc[season, "self_sufficiency_ratio"]:.1f}%', 
                 ha='center', va='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_self_consumption.png'), dpi=300)
    plt.close()
    logging.info("Seasonal self-consumption and self-sufficiency plot saved")

def plot_seasonal_energy_balance(seasonal_stats, output_dir):
    """
    Plot seasonal energy balance (surplus/deficit).
    
    Parameters:
    - seasonal_stats (DataFrame): DataFrame with seasonal stats
    - output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create bar chart for net balance
    plt.bar(seasonal_stats.index, seasonal_stats['net_balance_wh_kwh'], 
            color=['green' if x >= 0 else 'red' for x in seasonal_stats['net_balance_wh_kwh']])
    
    # Customize plot
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Season')
    plt.ylabel('Net Energy Balance (kWh)')
    plt.title('Seasonal Energy Balance (Surplus/Deficit)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(seasonal_stats['net_balance_wh_kwh']):
        color = 'green' if v >= 0 else 'red'
        label = f'+{v:.0f}' if v >= 0 else f'{v:.0f}'
        va = 'bottom' if v >= 0 else 'top'
        offset = 50 if v >= 0 else -50
        plt.text(i, v + offset, label, ha='center', va=va, color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_energy_balance.png'), dpi=300)
    plt.close()
    logging.info("Seasonal energy balance plot saved")

def plot_seasonal_hourly_profiles(df, output_dir):
    """
    Plot average hourly production and consumption profiles by season.
    
    Parameters:
    - df (DataFrame): DataFrame with hourly data
    - output_dir (str): Directory to save the plot
    """
    # Make a copy and add season column
    df_copy = df.copy()
    df_copy['month'] = df_copy.index.month
    season_mapping = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall',
        11: 'Fall', 12: 'Winter'
    }
    df_copy['season'] = df_copy['month'].map(season_mapping)
    df_copy['hour'] = df_copy.index.hour
    
    # Create a figure with subplots for each season
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, season in enumerate(seasons):
        # Filter data for this season
        season_data = df_copy[df_copy['season'] == season]
        
        # Calculate average hourly values
        hourly_avg = season_data.groupby('hour').agg({
            'ac_power_output': 'mean',
            'Load (kW)': 'mean'
        })
        
        # Convert ac_power_output from W to kW for comparison
        hourly_avg['ac_power_output_kw'] = hourly_avg['ac_power_output'] / 1000
        
        # Plot on the appropriate subplot
        ax = axes[i]
        ax.plot(hourly_avg.index, hourly_avg['ac_power_output_kw'], 
                label='Production (kW)', color='green', linewidth=2)
        ax.plot(hourly_avg.index, hourly_avg['Load (kW)'], 
                label='Consumption (kW)', color='red', linewidth=2)
        
        # Fill the area between curves
        ax.fill_between(hourly_avg.index, hourly_avg['ac_power_output_kw'], hourly_avg['Load (kW)'],
                       where=hourly_avg['ac_power_output_kw'] >= hourly_avg['Load (kW)'],
                       color='lightgreen', alpha=0.5, label='Surplus')
        ax.fill_between(hourly_avg.index, hourly_avg['ac_power_output_kw'], hourly_avg['Load (kW)'],
                       where=hourly_avg['ac_power_output_kw'] <= hourly_avg['Load (kW)'],
                       color='lightcoral', alpha=0.5, label='Deficit')
        
        # Set title and grid
        ax.set_title(f'{season}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(0, 23)
        
        # Only set labels on edges
        if i >= 2:
            ax.set_xlabel('Hour of Day', fontsize=12)
        if i % 2 == 0:
            ax.set_ylabel('Power (kW)', fontsize=12)
        
        # Custom legend for the first subplot only
        if i == 0:
            ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_hourly_profiles.png'), dpi=300)
    plt.close()
    logging.info("Seasonal hourly profiles plot saved")

def calculate_seasonal_battery_needs(df, output_dir):
    """
    Calculate and visualize battery storage needs for each season.
    
    Parameters:
    - df (DataFrame): DataFrame with hourly data
    - output_dir (str): Directory to save the results
    
    Returns:
    - battery_requirements (DataFrame): Seasonal battery requirements
    """
    # Make a copy and add season column
    df_copy = df.copy()
    df_copy['month'] = df_copy.index.month
    season_mapping = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall',
        11: 'Fall', 12: 'Winter'
    }
    df_copy['season'] = df_copy['month'].map(season_mapping)
    
    # Calculate hourly energy balance
    df_copy['load_wh'] = df_copy['Load (kW)'] * 1000  # Convert kW to Wh
    df_copy['balance'] = df_copy['E_ac'] - df_copy['load_wh']
    
    # Initialize results dictionary
    battery_results = []
    
    # Process each season
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = df_copy[df_copy['season'] == season]
        
        # Calculate battery needs for typical days in this season
        # Use date string instead of datetime.date
        season_data = season_data.copy(deep=True)  # Make an explicit copy first
        season_data['date_str'] = season_data.index.strftime('%Y-%m-%d')
        
        # Group by day and calculate each day's requirements
        daily_requirements = []
        
        for day_str, day_data in season_data.groupby('date_str'):
            # Simulate battery state of charge throughout the day
            day_data = day_data.sort_index()  # Ensure chronological order
            soc = 0  # State of charge (Wh)
            min_soc = 0
            max_soc = 0
            
            for i, row in day_data.iterrows():
                # Positive balance: charging battery
                if row['balance'] > 0:
                    soc += row['balance'] * 0.9  # Charging efficiency
                # Negative balance: discharging battery
                else:
                    soc += row['balance'] / 0.9  # Discharging efficiency
                
                min_soc = min(min_soc, soc)
                max_soc = max(max_soc, soc)
            
            # Calculate required capacity: must cover the lowest SOC point
            capacity_needed = abs(min_soc) if min_soc < 0 else 0
            
            # Calculate max power
            max_charging = day_data[day_data['balance'] > 0]['balance'].max() if not day_data[day_data['balance'] > 0].empty else 0
            max_discharging = abs(day_data[day_data['balance'] < 0]['balance'].min()) if not day_data[day_data['balance'] < 0].empty else 0
            
            daily_requirements.append({
                'date': day_str,  # Use string instead of datetime.date
                'capacity_needed_wh': capacity_needed,
                'max_charging_w': max_charging,
                'max_discharging_w': max_discharging
            })
        
        # Calculate statistics for this season
        daily_req_df = pd.DataFrame(daily_requirements)
        if not daily_req_df.empty:
            p95_capacity = daily_req_df['capacity_needed_wh'].quantile(0.95)
            max_capacity = daily_req_df['capacity_needed_wh'].max()
            avg_capacity = daily_req_df['capacity_needed_wh'].mean()
            
            p95_charging = daily_req_df['max_charging_w'].quantile(0.95)
            p95_discharging = daily_req_df['max_discharging_w'].quantile(0.95)
            
            battery_results.append({
                'season': season,
                'avg_capacity_needed_wh': avg_capacity,
                'p95_capacity_needed_wh': p95_capacity,
                'max_capacity_needed_wh': max_capacity,
                'p95_charging_w': p95_charging,
                'p95_discharging_w': p95_discharging
            })
        else:
            # Handle empty dataframe case
            battery_results.append({
                'season': season,
                'avg_capacity_needed_wh': 0,
                'p95_capacity_needed_wh': 0,
                'max_capacity_needed_wh': 0,
                'p95_charging_w': 0,
                'p95_discharging_w': 0
            })
    
    # Convert to DataFrame
    battery_df = pd.DataFrame(battery_results)
    
    # Convert Wh to kWh for better readability
    battery_df['avg_capacity_needed_kwh'] = battery_df['avg_capacity_needed_wh'] / 1000
    battery_df['p95_capacity_needed_kwh'] = battery_df['p95_capacity_needed_wh'] / 1000
    battery_df['max_capacity_needed_kwh'] = battery_df['max_capacity_needed_wh'] / 1000
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar chart
    seasons = battery_df['season']
    x = np.arange(len(seasons))
    width = 0.25
    
    # Plot bars
    plt.bar(x - width, battery_df['avg_capacity_needed_kwh'], width, label='Average Daily Need', color='green')
    plt.bar(x, battery_df['p95_capacity_needed_kwh'], width, label='95th Percentile', color='orange')
    plt.bar(x + width, battery_df['max_capacity_needed_kwh'], width, label='Maximum Need', color='red')
    
    # Customize plot
    plt.xlabel('Season')
    plt.ylabel('Battery Capacity Needed (kWh)')
    plt.title('Seasonal Battery Storage Requirements')
    plt.xticks(x, seasons)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_battery_requirements.png'), dpi=300)
    plt.close()
    logging.info("Seasonal battery requirements plot saved")
    
    # Save to CSV
    battery_df.to_csv(os.path.join(output_dir, 'seasonal_battery_requirements.csv'), index=False)
    logging.info("Seasonal battery requirements saved to CSV")
    
    return battery_df

def calculate_optimal_battery_capacity(df, output_dir, 
                                      min_capacity=1,  # kWh
                                      max_capacity=100,  # kWh
                                      capacity_step=1,  # kWh
                                      battery_efficiency=0.9,
                                      depth_of_discharge=0.8,
                                      battery_cost_per_kwh=500,  # $/kWh - realistic value
                                      electricity_buy_price=0.20,  # $/kWh
                                      electricity_sell_price=0.10,  # $/kWh
                                      battery_lifetime_years=10):  # Battery expected lifetime
    """
    Calculate optimal battery capacity by simulating different battery sizes
    and analyzing their economic and technical performance.
    
    Parameters:
    - df (DataFrame): DataFrame with energy production and consumption data
    - output_dir (str): Directory to save results
    - min_capacity (float): Minimum battery capacity to consider (kWh)
    - max_capacity (float): Maximum battery capacity to consider (kWh)
    - capacity_step (float): Step size for capacity increments (kWh)
    - battery_efficiency (float): Round-trip efficiency of the battery (0-1)
    - depth_of_discharge (float): Maximum depth of discharge allowed (0-1)
    - battery_cost_per_kwh (float): Cost of battery per kWh capacity ($)
    - electricity_buy_price (float): Price to buy electricity from grid ($/kWh)
    - electricity_sell_price (float): Price to sell electricity to grid ($/kWh)
    
    Returns:
    - optimal_capacity (float): Optimal battery capacity (kWh)
    - battery_results (DataFrame): Results for all simulated capacities
    """
    logging.info("Starting battery capacity optimization analysis...")
    
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
        battery_charged_wh = 0  # Energy used to charge battery
        battery_discharged_wh = 0  # Energy discharged from battery
        battery_losses_wh = 0  # Energy lost in battery charging/discharging
        
        # Simulate a full year with hourly timesteps
        soc_wh = capacity_wh * 0.5  # Start with half-full battery
        soc_percent = []  # Track state of charge for histogram
        
        for i, row in df_battery.iterrows():
            # Direct self-consumption
            direct_consumption = min(row['E_ac'], row['load_wh'])
            self_consumed_wh += direct_consumption
            
            # Surplus solar energy available after direct consumption
            remaining_surplus = row['E_ac'] - direct_consumption
            
            # Remaining load to be satisfied
            remaining_load = row['load_wh'] - direct_consumption
            
            # Handle energy surplus (charge battery or export)
            if remaining_surplus > 0:
                # Calculate how much can be stored in battery
                space_in_battery = capacity_wh - soc_wh
                to_battery = min(remaining_surplus, space_in_battery)
                
                # Account for charging efficiency
                effective_charge = to_battery * battery_efficiency
                battery_charged_wh += to_battery
                battery_losses_wh += to_battery - effective_charge
                
                # Update battery state of charge
                soc_wh += effective_charge
                
                # Export any remaining surplus
                grid_export_wh += remaining_surplus - to_battery
            
            # Handle energy deficit (discharge battery or import)
            elif remaining_load > 0:
                # Calculate how much can be drawn from battery
                available_energy = max(0, soc_wh - (capacity_wh * (1 - depth_of_discharge)))
                from_battery = min(remaining_load, available_energy)
                
                # Account for discharging efficiency
                effective_discharge = from_battery
                battery_discharged_wh += effective_discharge
                
                # Update battery state of charge
                soc_wh -= effective_discharge
                
                # Import any remaining deficit
                grid_import_wh += remaining_load - from_battery
            
            # Record state of charge percentage for histogram
            soc_percent.append((soc_wh / capacity_wh) * 100)
        
        # Calculate key metrics
        total_consumption_wh = df_battery['load_wh'].sum()
        total_production_wh = df_battery['E_ac'].sum()
        
        # Self-consumption and self-sufficiency rates
        battery_contribution = battery_discharged_wh
        self_consumption_rate = (self_consumed_wh + battery_contribution) / total_production_wh if total_production_wh > 0 else 0
        self_sufficiency_rate = (self_consumed_wh + battery_contribution) / total_consumption_wh if total_consumption_wh > 0 else 0
        
        # Economic calculations
        battery_investment = (capacity_wh / 1000) * battery_cost_per_kwh  # Battery cost
        
        # Calculate avoided grid import costs (this is the main benefit)
        grid_import_without_battery = df_battery['deficit'].sum() / 1000  # kWh
        grid_import_with_battery = grid_import_wh / 1000  # kWh
        avoided_grid_import = grid_import_without_battery - grid_import_with_battery
        
        # Annual savings from avoided grid imports
        annual_savings = avoided_grid_import * electricity_buy_price
        
        # Revenue from increased grid exports (if any)
        grid_export_without_battery = df_battery['surplus'].sum() / 1000  # kWh
        grid_export_with_battery = grid_export_wh / 1000  # kWh
        additional_export = max(0, grid_export_with_battery - grid_export_without_battery)
        annual_revenue = additional_export * electricity_sell_price
        
        # Total annual benefit
        total_annual_benefit = annual_savings + annual_revenue
        
        # CORRECTED Payback calculation
        # Must account for battery degradation and replacement
        if total_annual_benefit > 0:
            simple_payback = battery_investment / total_annual_benefit
            
            # Adjust for battery lifetime
            if simple_payback > battery_lifetime_years:
                # Battery needs replacement before payback
                effective_payback = float('inf')
            else:
                effective_payback = simple_payback
        else:
            effective_payback = float('inf')
        # Simplified payback (years)
        annual_benefit = annual_savings + annual_revenue
        simple_payback = battery_investment / annual_benefit if annual_benefit > 0 else float('inf')
        
        # Cycle counting (simplified)
        total_energy_cycled = battery_discharged_wh
        equivalent_full_cycles = total_energy_cycled / usable_capacity_wh if usable_capacity_wh > 0 else 0
        
        # Calculate histogram data for state of charge
        hist, bins = np.histogram(soc_percent, bins=10, range=(0, 100))
        soc_histogram = {f"{bins[i]:.0f}-{bins[i+1]:.0f}%": hist[i] for i in range(len(hist))}
        
        # Save results for this capacity
        results.append({
            'capacity_kwh': capacity_wh / 1000,
            'usable_capacity_kwh': usable_capacity_wh / 1000,
            'self_consumption_rate': self_consumption_rate * 100,
            'self_sufficiency_rate': self_sufficiency_rate * 100,
            'grid_import_kwh': grid_import_wh / 1000,
            'grid_export_kwh': grid_export_wh / 1000,
            'avoided_grid_import_kwh': avoided_grid_import,
            'battery_charged_kwh': battery_charged_wh / 1000,
            'battery_discharged_kwh': battery_discharged_wh / 1000,
            'battery_losses_kwh': battery_losses_wh / 1000,
            'equivalent_full_cycles': equivalent_full_cycles,
            'battery_investment': battery_investment,
            'annual_savings': annual_savings,
            'annual_revenue': annual_revenue,
            'total_annual_benefit': total_annual_benefit,
            'simple_payback_years': effective_payback,
            'soc_histogram': soc_histogram,
            'avg_soc_percent': np.mean(soc_percent)
        })
    
    # Convert results to DataFrame
    battery_results = pd.DataFrame(results)
    
    # Determine optimal capacity - we'll define it as best economic value
    # Find capacity with shortest payback period that meets minimum self-sufficiency
    valid_options = battery_results[battery_results['self_sufficiency_rate'] >= 50]  # Minimum 50% self-sufficiency
    if not valid_options.empty:
        best_economic = valid_options.loc[valid_options['simple_payback_years'].idxmin()]
        optimal_capacity = best_economic['capacity_kwh']
    else:
        # If no options meet our criteria, choose the one with highest self-sufficiency
        optimal_capacity = battery_results.loc[battery_results['self_sufficiency_rate'].idxmax()]['capacity_kwh']
    
    # Save results to CSV
    battery_results.drop('soc_histogram', axis=1).to_csv(os.path.join(output_dir, 'battery_capacity_analysis.csv'), index=False)
    logging.info(f"Battery capacity analysis saved to {os.path.join(output_dir, 'battery_capacity_analysis.csv')}")
    
    # Create plots
    create_battery_capacity_plots(battery_results, optimal_capacity, output_dir)
    
    # Create detailed analysis for the optimal capacity
    create_optimal_battery_analysis(df_battery, optimal_capacity, depth_of_discharge, battery_efficiency, output_dir)
    
    logging.info(f"Optimal battery capacity determined: {optimal_capacity:.1f} kWh")
    return optimal_capacity, battery_results

def create_battery_capacity_plots(battery_results, optimal_capacity, output_dir):
    """
    Create plots to visualize battery capacity analysis results.
    
    Parameters:
    - battery_results (DataFrame): Results from capacity analysis
    - optimal_capacity (float): Determined optimal capacity
    - output_dir (str): Directory to save plots
    """
    # Plot 1: Self-consumption and self-sufficiency vs capacity
    plt.figure(figsize=(12, 6))
    plt.plot(battery_results['capacity_kwh'], battery_results['self_consumption_rate'], 
             'o-', color='blue', label='Self-Consumption Rate (%)')
    plt.plot(battery_results['capacity_kwh'], battery_results['self_sufficiency_rate'], 
             'o-', color='green', label='Self-Sufficiency Rate (%)')
    
    # Highlight optimal capacity
    plt.axvline(x=optimal_capacity, color='red', linestyle='--', alpha=0.7, 
                label=f'Optimal Capacity: {optimal_capacity:.1f} kWh')
    
    # Mark the optimal capacity on both curves
    optimal_self_consumption = battery_results.loc[battery_results['capacity_kwh'] == optimal_capacity, 'self_consumption_rate'].values[0] \
        if optimal_capacity in battery_results['capacity_kwh'].values else \
        np.interp(optimal_capacity, battery_results['capacity_kwh'], battery_results['self_consumption_rate'])
    
    optimal_self_sufficiency = battery_results.loc[battery_results['capacity_kwh'] == optimal_capacity, 'self_sufficiency_rate'].values[0] \
        if optimal_capacity in battery_results['capacity_kwh'].values else \
        np.interp(optimal_capacity, battery_results['capacity_kwh'], battery_results['self_sufficiency_rate'])
    
    plt.plot(optimal_capacity, optimal_self_consumption, 'o', color='blue', markersize=10)
    plt.plot(optimal_capacity, optimal_self_sufficiency, 'o', color='green', markersize=10)
    
    plt.title('Self-Consumption and Self-Sufficiency vs Battery Capacity')
    plt.xlabel('Battery Capacity (kWh)')
    plt.ylabel('Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'battery_self_consumption_analysis.png'), dpi=300)
    plt.close()
    
    # Plot 2: Payback period vs capacity
    plt.figure(figsize=(12, 6))
    
    # Filter out infinite payback periods for better visualization
    payback_data = battery_results[battery_results['simple_payback_years'] < 100]  # Filter extreme values
    
    if not payback_data.empty:
        plt.plot(payback_data['capacity_kwh'], payback_data['simple_payback_years'], 'o-', color='purple')
        
        # Highlight optimal capacity
        if optimal_capacity in payback_data['capacity_kwh'].values:
            optimal_payback = payback_data.loc[payback_data['capacity_kwh'] == optimal_capacity, 'simple_payback_years'].values[0]
            plt.axvline(x=optimal_capacity, color='red', linestyle='--', alpha=0.7, 
                        label=f'Optimal Capacity: {optimal_capacity:.1f} kWh\nPayback: {optimal_payback:.1f} years')
            plt.plot(optimal_capacity, optimal_payback, 'o', color='red', markersize=10)
    
    plt.title('Payback Period vs Battery Capacity')
    plt.xlabel('Battery Capacity (kWh)')
    plt.ylabel('Simple Payback Period (years)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'battery_payback_analysis.png'), dpi=300)
    plt.close()
    
    # Plot 3: Grid interaction vs capacity
    plt.figure(figsize=(12, 6))
    plt.plot(battery_results['capacity_kwh'], battery_results['grid_import_kwh'], 
             'o-', color='red', label='Grid Import (kWh)')
    plt.plot(battery_results['capacity_kwh'], battery_results['grid_export_kwh'], 
             'o-', color='green', label='Grid Export (kWh)')
    
    # Highlight optimal capacity
    plt.axvline(x=optimal_capacity, color='blue', linestyle='--', alpha=0.7, 
                label=f'Optimal Capacity: {optimal_capacity:.1f} kWh')
    
    plt.title('Grid Interaction vs Battery Capacity')
    plt.xlabel('Battery Capacity (kWh)')
    plt.ylabel('Energy (kWh)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'battery_grid_interaction.png'), dpi=300)
    plt.close()
    
    # Plot 4: Battery utilization vs capacity
    plt.figure(figsize=(12, 6))
    plt.plot(battery_results['capacity_kwh'], battery_results['equivalent_full_cycles'], 
             'o-', color='orange', label='Equivalent Full Cycles')
    
    # Highlight optimal capacity
    optimal_cycles = battery_results.loc[battery_results['capacity_kwh'] == optimal_capacity, 'equivalent_full_cycles'].values[0] \
        if optimal_capacity in battery_results['capacity_kwh'].values else \
        np.interp(optimal_capacity, battery_results['capacity_kwh'], battery_results['equivalent_full_cycles'])
    
    plt.axvline(x=optimal_capacity, color='blue', linestyle='--', alpha=0.7, 
                label=f'Optimal Capacity: {optimal_capacity:.1f} kWh\nCycles: {optimal_cycles:.1f}')
    plt.plot(optimal_capacity, optimal_cycles, 'o', color='orange', markersize=10)
    
    plt.title('Battery Utilization vs Capacity')
    plt.xlabel('Battery Capacity (kWh)')
    plt.ylabel('Equivalent Full Cycles per Year')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'battery_utilization.png'), dpi=300)
    plt.close()
    
    logging.info("Battery capacity analysis plots created")

def create_optimal_battery_analysis(df, optimal_capacity, depth_of_discharge, battery_efficiency, output_dir):
    """
    Create detailed analysis for the optimal battery capacity.
    
    Parameters:
    - df (DataFrame): DataFrame with energy data
    - optimal_capacity (float): Optimal battery capacity in kWh
    - depth_of_discharge (float): Maximum depth of discharge
    - battery_efficiency (float): Battery round-trip efficiency
    - output_dir (str): Directory to save results
    """
    optimal_capacity_wh = optimal_capacity * 1000  # Convert to Wh
    usable_capacity_wh = optimal_capacity_wh * depth_of_discharge
    
    # Make a copy of the dataframe for simulation
    df_optimal = df.copy()
    
    # Simulate a full year with the optimal battery
    soc_wh = optimal_capacity_wh * 0.5  # Start with half-full battery
    df_optimal['battery_soc_wh'] = 0.0
    df_optimal['battery_soc_percent'] = 0.0
    df_optimal['grid_import'] = 0.0
    df_optimal['grid_export'] = 0.0
    df_optimal['battery_charge'] = 0.0
    df_optimal['battery_discharge'] = 0.0
    df_optimal['direct_consumption'] = 0.0
    
    for i, row in df_optimal.iterrows():
        # Direct self-consumption
        direct = min(row['E_ac'], row['load_wh'])
        df_optimal.at[i, 'direct_consumption'] = direct
        
        # Surplus solar energy available after direct consumption
        remaining_surplus = row['E_ac'] - direct
        
        # Remaining load to be satisfied
        remaining_load = row['load_wh'] - direct
        
        # Handle energy surplus (charge battery or export)
        if remaining_surplus > 0:
            # Calculate how much can be stored in battery
            space_in_battery = optimal_capacity_wh - soc_wh
            to_battery = min(remaining_surplus, space_in_battery)
            
            # Account for charging efficiency
            effective_charge = to_battery * battery_efficiency
            df_optimal.at[i, 'battery_charge'] = to_battery
            
            # Update battery state of charge
            soc_wh += effective_charge
            
            # Export any remaining surplus
            df_optimal.at[i, 'grid_export'] = remaining_surplus - to_battery
        
        # Handle energy deficit (discharge battery or import)
        elif remaining_load > 0:
            # Calculate how much can be drawn from battery
            available_energy = max(0, soc_wh - (optimal_capacity_wh * (1 - depth_of_discharge)))
            from_battery = min(remaining_load, available_energy)
            
            # Account for discharging efficiency
            df_optimal.at[i, 'battery_discharge'] = from_battery
            
            # Update battery state of charge
            soc_wh -= from_battery
            
            # Import any remaining deficit
            df_optimal.at[i, 'grid_import'] = remaining_load - from_battery
        
        # Record battery state of charge
        df_optimal.at[i, 'battery_soc_wh'] = soc_wh
        df_optimal.at[i, 'battery_soc_percent'] = (soc_wh / optimal_capacity_wh) * 100
    
    # Add month and hour for analysis
    df_optimal['month'] = df_optimal.index.month
    df_optimal['hour'] = df_optimal.index.hour
    
    # Monthly analysis
    monthly_data = df_optimal.groupby('month').agg({
        'E_ac': 'sum',
        'load_wh': 'sum',
        'direct_consumption': 'sum',
        'battery_discharge': 'sum',
        'grid_import': 'sum',
        'grid_export': 'sum',
        'battery_soc_percent': 'mean'
    })
    
    # Convert to kWh for better readability
    for col in ['E_ac', 'load_wh', 'direct_consumption', 'battery_discharge', 'grid_import', 'grid_export']:
        monthly_data[f'{col}_kwh'] = monthly_data[col] / 1000
    
    # Calculate monthly self-consumption and self-sufficiency
    monthly_data['self_consumption_rate'] = ((monthly_data['direct_consumption'] + monthly_data['battery_discharge']) / 
                                             monthly_data['E_ac']) * 100
    monthly_data['self_sufficiency_rate'] = ((monthly_data['direct_consumption'] + monthly_data['battery_discharge']) / 
                                             monthly_data['load_wh']) * 100
    
    # Save monthly data
    monthly_data.to_csv(os.path.join(output_dir, 'optimal_battery_monthly_analysis.csv'))
    
    # Create monthly plots
    plt.figure(figsize=(14, 7))
    
    # Prepare data for stacked bar chart
    months = monthly_data.index
    direct_use = monthly_data['direct_consumption_kwh']
    from_battery = monthly_data['battery_discharge_kwh']
    from_grid = monthly_data['grid_import_kwh']
    
    # Plot stacked bars
    plt.bar(months, direct_use, label='Direct Solar Consumption', color='green')
    plt.bar(months, from_battery, bottom=direct_use, label='From Battery', color='orange')
    plt.bar(months, from_grid, bottom=direct_use+from_battery, label='From Grid', color='red')
    
    # Add total consumption line
    plt.plot(months, monthly_data['load_wh_kwh'], 'o-', color='blue', label='Total Consumption')
    
    # Add self-sufficiency percentage text
    for i, month in enumerate(months):
        plt.text(month, monthly_data.loc[month, 'load_wh_kwh'] + 100, 
                f"{monthly_data.loc[month, 'self_sufficiency_rate']:.1f}%", 
                ha='center', va='bottom', color='blue')
    
    plt.title(f'Monthly Energy Flow with {optimal_capacity:.1f} kWh Battery')
    plt.xlabel('Month')
    plt.ylabel('Energy (kWh)')
    plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_battery_monthly_energy.png'), dpi=300)
    plt.close()
    
    # Create a week-long detailed analysis for a representative period
    # Find a week with typical production and consumption
    # For simplicity, let's use a week in May (spring)
    may_data = df_optimal[df_optimal.index.month == 5]
    if not may_data.empty:
        # Get the first full week of May
        may_1st = may_data.index[0]
        start_of_week = may_1st + pd.Timedelta(days=7 - may_1st.dayofweek)
        end_of_week = start_of_week + pd.Timedelta(days=7)
        week_data = df_optimal[(df_optimal.index >= start_of_week) & (df_optimal.index < end_of_week)]
        
        if not week_data.empty:
            plt.figure(figsize=(15, 10))
            
            # Plot production, consumption, battery SOC, and grid interaction
            ax1 = plt.subplot(211)
            ax1.plot(week_data.index, week_data['E_ac'] / 1000, label='Solar Production (kW)', color='green')
            ax1.plot(week_data.index, week_data['load_wh'] / 1000, label='Consumption (kW)', color='red')
            ax1.set_ylabel('Power (kW)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            ax2 = plt.subplot(212, sharex=ax1)
            ax2.plot(week_data.index, week_data['battery_soc_percent'], label='Battery SOC (%)', color='blue')
            ax2.set_ylabel('Battery State of Charge (%)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis to show days
            import matplotlib.dates as mdates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
            ax2.xaxis.set_major_locator(mdates.DayLocator())
            
            plt.title(f'Detailed Analysis with {optimal_capacity:.1f} kWh Battery (Sample Week)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'optimal_battery_detailed_week.png'), dpi=300)
            plt.close()
    
    logging.info("Optimal battery detailed analysis completed")

def calculate_initial_investment(number_of_panels, battery_capacity_kwh=None, panel_cost=250, installation_cost_per_panel=150, 
                                inverter_cost_per_kw=120, battery_cost_per_kwh=500, bos_cost_per_panel=50):
    """Calculate the total initial investment costs."""
    panel_cost_total = number_of_panels * panel_cost
    installation_cost = number_of_panels * installation_cost_per_panel
    inverter_cost = (number_of_panels * 0.24) * inverter_cost_per_kw  # Assuming 240W panels
    battery_cost = battery_capacity_kwh * battery_cost_per_kwh if battery_capacity_kwh else 0
    bos_cost = number_of_panels * bos_cost_per_panel  # Balance of system costs
    
    total_investment = panel_cost_total + installation_cost + inverter_cost + battery_cost + bos_cost
    
    return {
        'panel_cost': panel_cost_total,
        'installation_cost': installation_cost,
        'inverter_cost': inverter_cost,
        'battery_cost': battery_cost,
        'bos_cost': bos_cost,
        'total_investment': total_investment
    }
    
def calculate_annual_cashflow_improved(df, battery_capacity_kwh, electricity_price=0.20, 
                                     feed_in_tariff=0.10, annual_maintenance_percent=0.5,
                                     inflation_rate=2.0, electricity_price_increase=3.0, 
                                     system_lifetime=25, initial_investment=None,
                                     battery_cost_per_kwh=500, battery_lifetime=10):
    """
    IMPROVED: Include battery degradation and replacement costs.
    
    Your current model applies general panel degradation but doesn't account 
    for battery capacity fade and replacement needs.
    """
    df_calc = df.copy()
    df_calc['load_wh'] = df_calc['Load (kW)'] * 1000
    
    # Base energy flows
    direct_consumption = df_calc.apply(lambda x: min(x['E_ac'], x['load_wh']), axis=1).sum()
    grid_exports = df_calc.apply(lambda x: max(0, x['E_ac'] - x['load_wh']), axis=1).sum()
    grid_imports = df_calc.apply(lambda x: max(0, x['load_wh'] - x['E_ac']), axis=1).sum()
    
    # Convert to kWh
    direct_consumption_kwh = direct_consumption / 1000
    grid_exports_kwh = grid_exports / 1000
    grid_imports_kwh = grid_imports / 1000
    
    cashflows = []
    
    for year in range(1, system_lifetime + 1):
        # BATTERY DEGRADATION MODEL
        # Linear degradation: 2.5% per year
        battery_degradation_factor = max(0.8, 1 - 0.025 * year)  # Don't go below 80%
        
        # Check if battery replacement is needed
        replacement_cost = 0
        if year > battery_lifetime:
            # Calculate how many replacements have occurred
            replacements = (year - 1) // battery_lifetime
            years_since_last_replacement = year - (replacements * battery_lifetime)
            battery_degradation_factor = max(0.8, 1 - 0.025 * years_since_last_replacement)
            
            # Add replacement cost in replacement years
            if year % battery_lifetime == 1 and year > battery_lifetime:
                replacement_cost = battery_capacity_kwh * battery_cost_per_kwh * ((1 + inflation_rate/100) ** (year-1))
                logging.info(f"Battery replacement in year {year}: ${replacement_cost:,.0f}")
        
        # ADJUST benefits based on battery performance
        effective_battery_benefit = battery_degradation_factor
        
        # Apply price escalation
        current_electricity_price = electricity_price * ((1 + electricity_price_increase/100) ** (year-1))
        current_feed_in_tariff = feed_in_tariff * ((1 + inflation_rate/100) ** (year-1))
        current_maintenance = (initial_investment['total_investment'] * 
                             (annual_maintenance_percent / 100) * 
                             ((1 + inflation_rate/100) ** (year-1)))
        
        # Apply system degradation (panels: 0.5% per year)
        system_degradation_factor = (1 - 0.005) ** (year-1)
        
        # Calculate benefits
        # Battery helps with self-consumption (reduced grid imports)
        battery_savings = (grid_imports_kwh * 0.3 * effective_battery_benefit * 
                          system_degradation_factor * current_electricity_price)
        
        year_savings = (direct_consumption_kwh * system_degradation_factor * 
                       current_electricity_price) + battery_savings
        year_income = (grid_exports_kwh * system_degradation_factor * 
                      current_feed_in_tariff)
        
        year_cashflow = year_savings + year_income - current_maintenance - replacement_cost
        
        cashflows.append({
            'year': year,
            'savings': year_savings,
            'income': year_income,
            'maintenance': current_maintenance,
            'battery_replacement': replacement_cost,
            'battery_degradation_factor': battery_degradation_factor,
            'net_cashflow': year_cashflow
        })
    
    return pd.DataFrame(cashflows)

def npv_to_irr(cashflows, iterations=1000, error=0.0001):
    """Calculate IRR from a series of cash flows using an iterative approach."""
    rate = 0.1  # Initial guess
    step = 0.1
    
    # Simple iterative approach to find IRR
    for i in range(iterations):
        npv = 0
        for t, cf in enumerate(cashflows):
            npv += cf / (1 + rate) ** t
            
        if abs(npv) < error:
            return rate
        
        if npv > 0:
            rate += step
        else:
            rate -= step
            step /= 2
    
    # If we didn't converge, use numpy's IRR function as fallback
    try:
        from numpy import irr
        return irr(cashflows)
    except:
        # If all else fails, return None
        return None

def calculate_financial_metrics(initial_investment, cashflows, discount_rate=5.0, 
                              electricity_price=0.20, electricity_price_increase=3.0,
                              feed_in_tariff=0.10, inflation_rate=2.0):
    """
    Calculate key financial metrics: NPV, IRR, ROI, and payback period.
    """
    # Extract total investment and annual net cash flows
    investment = initial_investment['total_investment']
    annual_cashflows = cashflows['net_cashflow'].tolist()
    
    # Calculate Net Present Value (NPV)
    npv = -investment
    for i, cf in enumerate(annual_cashflows):
        npv += cf / ((1 + discount_rate/100) ** (i+1))
    
    # Calculate Internal Rate of Return (IRR)
    try:
        irr = npv_to_irr([-investment] + annual_cashflows)
    except:
        irr = None  # In case IRR calculation fails
    
    # Calculate ROI
    total_returns = sum(annual_cashflows)
    roi = (total_returns - investment) / investment * 100
    
    # Calculate simple payback period
    cumulative_cashflow = -investment
    payback_period = None
    for i, cf in enumerate(annual_cashflows):
        cumulative_cashflow += cf
        if cumulative_cashflow >= 0 and payback_period is None:
            # Interpolate for more accurate payback period
            if i > 0:
                prev_cf = cumulative_cashflow - cf
                fraction = -prev_cf / cf
                payback_period = i + fraction
            else:
                payback_period = i + 1
    
    # Calculate discounted payback period
    disc_cumulative_cashflow = -investment
    disc_payback_period = None
    for i, cf in enumerate(annual_cashflows):
        disc_cumulative_cashflow += cf / ((1 + discount_rate/100) ** (i+1))
        if disc_cumulative_cashflow >= 0 and disc_payback_period is None:
            disc_payback_period = i + 1
    
    # Calculate Levelized Cost of Electricity (LCOE)
    total_production_kwh = sum([
        cashflows.iloc[i]['savings'] / (electricity_price * ((1 + electricity_price_increase/100) ** i)) + 
        cashflows.iloc[i]['income'] / (feed_in_tariff * ((1 + inflation_rate/100) ** i))
        for i in range(len(cashflows))
    ])
    
    # Apply degradation
    total_production_kwh = total_production_kwh * sum([(1 - 0.005) ** i for i in range(len(cashflows))]) / len(cashflows)
    
    lcoe = investment / total_production_kwh if total_production_kwh > 0 else float('inf')
    
    return {
        'NPV': npv,
        'IRR': irr * 100 if irr is not None else None,  # Convert to percentage
        'ROI': roi,
        'Payback_Period_Years': payback_period,
        'Discounted_Payback_Period_Years': disc_payback_period,
        'LCOE': lcoe
    }

def calculate_efficiency_metrics(df, number_of_panels, panel_area, total_panel_area, panel_nominal_power):
    """
    Calculate comprehensive system efficiency metrics.
    """
    # Make a copy of the dataframe and add the load_wh column
    df_copy = df.copy()
    df_copy['load_wh'] = df_copy['Load (kW)'] * 1000  # Convert kW to Wh
    
    # Calculate total incident solar energy
    total_incident_energy_kwh = df_copy['E_incident'].sum() / 1000
    
    # Calculate total produced energy
    total_produced_ac_kwh = df_copy['E_ac'].sum() / 1000
    
    # Calculate consumption metrics
    total_consumption_kwh = df_copy['load_wh'].sum() / 1000
    direct_consumption_kwh = df_copy.apply(lambda x: min(x['E_ac'], x['load_wh']), axis=1).sum() / 1000
    
    # Basic efficiency metrics
    panel_efficiency = panel_nominal_power / (panel_area * 1000)  # STC irradiance is 1000 W/m²
    system_efficiency = total_produced_ac_kwh / total_incident_energy_kwh
    system_yield = total_produced_ac_kwh / (number_of_panels * panel_nominal_power / 1000)  # kWh/kWp
    specific_yield = total_produced_ac_kwh / total_panel_area  # kWh/m²
    
    # Corrected Performance Ratio calculation
    # PR = Actual Output / Theoretical Output
    # Theoretical Output = Total Irradiance (kWh/m²) * Panel Efficiency * Total Panel Area (m²)
    # But the E_incident (total_incident_energy_kwh) is already multiplied by area, so:
    total_irradiance_per_sqm = total_incident_energy_kwh / total_panel_area  # kWh/m²
    theoretical_max_energy = total_irradiance_per_sqm * panel_efficiency * total_panel_area  # kWh
    
    # Alternative calculation using rated DC power
    peak_dc_capacity = number_of_panels * panel_nominal_power / 1000  # kWp
    reference_yield = total_irradiance_per_sqm  # kWh/kWp (irradiance divided by 1 kW/m²)
    performance_ratio = total_produced_ac_kwh / (reference_yield * peak_dc_capacity)
    
    # Consumption matching metrics
    self_consumption_ratio = direct_consumption_kwh / total_produced_ac_kwh if total_produced_ac_kwh > 0 else 0
    self_sufficiency_ratio = direct_consumption_kwh / total_consumption_kwh if total_consumption_kwh > 0 else 0
    
    # Grid interaction metrics
    grid_imports_kwh = max(0, total_consumption_kwh - direct_consumption_kwh)
    grid_exports_kwh = max(0, total_produced_ac_kwh - direct_consumption_kwh)
    grid_dependency = grid_imports_kwh / total_consumption_kwh
    
    # Calculate capacity factor
    hours_per_year = 8760
    capacity_factor = total_produced_ac_kwh / (number_of_panels * panel_nominal_power / 1000 * hours_per_year)
    
    # Land use efficiency
    land_use_efficiency = total_produced_ac_kwh / total_panel_area  # kWh/m²/year
    
    return {
        'panel_efficiency': panel_efficiency * 100,  # In percentage
        'system_efficiency': system_efficiency * 100,  # In percentage
        'performance_ratio': performance_ratio * 100,  # In percentage
        'system_yield': system_yield,  # kWh/kWp
        'specific_yield': specific_yield,  # kWh/m²
        'capacity_factor': capacity_factor * 100,  # In percentage
        'self_consumption_ratio': self_consumption_ratio * 100,  # In percentage
        'self_sufficiency_ratio': self_sufficiency_ratio * 100,  # In percentage
        'grid_dependency': grid_dependency * 100,  # In percentage
        'land_use_efficiency': land_use_efficiency  # kWh/m²/year
    }

def plot_economic_analysis(initial_investment, cashflows, financial_metrics, output_dir):
    """
    Create visualizations for economic analysis.
    """
    # Create investment breakdown pie chart
    plt.figure(figsize=(10, 6))
    labels = ['Panels', 'Installation', 'Inverter', 'Battery', 'BOS']
    values = [
        initial_investment['panel_cost'],
        initial_investment['installation_cost'],
        initial_investment['inverter_cost'],
        initial_investment['battery_cost'],
        initial_investment['bos_cost']
    ]
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Initial Investment Breakdown')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'investment_breakdown.png'), dpi=300)
    plt.close()
    
    # Create cumulative cash flow chart
    plt.figure(figsize=(12, 6))
    years = cashflows['year'].tolist()
    cumulative_cashflow = [-initial_investment['total_investment']]
    for cf in cashflows['net_cashflow']:
        cumulative_cashflow.append(cumulative_cashflow[-1] + cf)
    cumulative_cashflow = cumulative_cashflow[1:]  # Remove the initial investment point
    
    plt.plot(years, cumulative_cashflow, 'b-', linewidth=2, marker='o')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Cumulative Cash Flow')
    plt.title('Cumulative Cash Flow Over System Lifetime')
    
    # Mark payback point
    if financial_metrics['Payback_Period_Years'] is not None:
        payback_year = financial_metrics['Payback_Period_Years']
        payback_cf = np.interp(payback_year, years, cumulative_cashflow)
        plt.scatter([payback_year], [payback_cf], s=100, c='red', zorder=5)
        plt.annotate(f'Payback: {payback_year:.1f} years', 
                    xy=(payback_year, payback_cf), xytext=(payback_year+1, payback_cf+10000),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_cashflow.png'), dpi=300)
    plt.close()
    
    # Create annual cash flow breakdown
    plt.figure(figsize=(14, 7))
    bar_width = 0.35
    years = cashflows['year'].tolist()
    
    plt.bar(years, cashflows['savings'], bar_width, label='Savings from Direct Use', color='green')
    plt.bar(years, cashflows['income'], bar_width, bottom=cashflows['savings'], 
           label='Income from Exports', color='blue')
    plt.bar(years, -cashflows['maintenance'], bar_width, 
           label='Maintenance Costs', color='red')
    
    plt.xlabel('Year')
    plt.ylabel('Cash Flow')
    plt.title('Annual Cash Flow Breakdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'annual_cashflow_breakdown.png'), dpi=300)
    plt.close()
    
    # Create financial metrics summary
    plt.figure(figsize=(10, 6))
    metrics = ['NPV', 'IRR', 'ROI', 'Payback_Period_Years', 'LCOE']
    values = [financial_metrics[m] for m in metrics]
    
    for i, (metric, value) in enumerate(zip(metrics, values)):
        plt.text(0.5, 1 - (i+1)*0.15, f"{metric}: {value:,.2f}", 
                ha='center', va='center', fontsize=14)
    
    plt.axis('off')
    plt.title('Financial Metrics Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'financial_metrics_summary.png'), dpi=300)
    plt.close()

def export_data_for_forecasting(df, optimal_tilt, optimal_azimuth, optimal_battery_capacity, 
                             output_file, system_params=None):
    """
    Export all data required for energy forecasting to a CSV file.
    
    Parameters:
    - df: DataFrame containing simulation results
    - optimal_tilt: Optimal panel tilt angle
    - optimal_azimuth: Optimal panel azimuth angle
    - optimal_battery_capacity: Optimal battery capacity (kWh)
    - output_file: Path to save the output CSV
    - system_params: Dictionary of system parameters (panels, efficiency, etc.)
    
    Returns:
    - Path to the created file
    """
    # Create a copy to avoid modifying the original
    forecast_df = df.copy()
    
    # Ensure all required columns are present
    required_columns = [
        # Weather/Environmental data
        'SolRad_Hor', 'SolRad_Dif', 'Air Temp', 'WS_10m', 
        
        # Solar position
        'zenith', 'azimuth', 
        
        # Load data
        'Load (kW)',
        
        # Calculated irradiance
        'DNI', 'total_irradiance',
        
        # Panel performance
        'cell_temperature', 'dc_power_output_per_panel', 'ac_power_output',
        
        # Energy metrics
        'E_incident', 'E_effective', 'E_ac',
        
        # Efficiency and performance
        'temperature_factor', 'PR',
        
        # Energy balance
        'mismatch', 'weighting_factor'
    ]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in forecast_df.columns]
    
    if missing_columns:
        logging.warning(f"Missing columns for forecasting: {missing_columns}")
        # Create placeholder columns for missing data
        for col in missing_columns:
            forecast_df[col] = np.nan
    
    # Select only the required columns
    export_df = forecast_df[required_columns].copy()
    
    # Add system configuration data as new columns
    export_df['optimal_tilt'] = optimal_tilt
    export_df['optimal_azimuth'] = optimal_azimuth
    export_df['optimal_battery_capacity'] = optimal_battery_capacity
    
    # Add time features
    export_df['hour'] = export_df.index.hour
    export_df['day'] = export_df.index.day
    export_df['month'] = export_df.index.month
    export_df['dayofweek'] = export_df.index.dayofweek
    export_df['is_weekend'] = export_df.index.dayofweek >= 5
    
    # Add system parameters if provided
    if system_params:
        for key, value in system_params.items():
            # Only add scalar values, not complex objects
            if np.isscalar(value):
                export_df[f'param_{key}'] = value
    
    # Save to CSV, preserving the datetime index
    export_df.to_csv(output_file)
    logging.info(f"Data for forecasting exported to {output_file}")
    
    return output_file

def analyze_neighborhood_solutions(optimal_tilt, optimal_azimuth, df_subset, dni_extra, 
                                  number_of_panels, inverter_params, output_dir, 
                                  tilt_range=15, azimuth_range=30, grid_points=7):
    """
    Analyze solutions in the neighborhood of the optimal angles to show why 
    the optimal solution is indeed optimal.
    
    Parameters:
    - optimal_tilt, optimal_azimuth: Optimal angles found by optimization
    - tilt_range, azimuth_range: Range around optimal to explore (degrees)
    - grid_points: Number of points to sample in each direction
    """
    logging.info("Analyzing neighborhood solutions around optimal angles...")
    
    # Create grid of angles around optimal
    tilt_values = np.linspace(
        max(0, optimal_tilt - tilt_range), 
        min(90, optimal_tilt + tilt_range), 
        grid_points
    )
    azimuth_values = np.linspace(
        max(90, optimal_azimuth - azimuth_range), 
        min(270, optimal_azimuth + azimuth_range), 
        grid_points
    )
    
    # Calculate performance for each combination
    results = []
    for tilt in tilt_values:
        for azimuth in azimuth_values:
            try:
                mismatch, production = objective_function_multi(
                    [tilt, azimuth],
                    df_subset,
                    dni_extra,
                    number_of_panels,
                    inverter_params
                )
                
                # Calculate distance from optimal
                distance = np.sqrt((tilt - optimal_tilt)**2 + (azimuth - optimal_azimuth)**2)
                
                results.append({
                    'tilt': tilt,
                    'azimuth': azimuth,
                    'mismatch': mismatch,
                    'production': production,
                    'distance_from_optimal': distance,
                    'is_optimal': (abs(tilt - optimal_tilt) < 0.1 and abs(azimuth - optimal_azimuth) < 0.1)
                })
            except Exception as e:
                logging.error(f"Error calculating for tilt={tilt}, azimuth={azimuth}: {e}")
                continue
    
    neighborhood_df = pd.DataFrame(results)
    
    # Save neighborhood analysis results
    neighborhood_df.to_csv(os.path.join(output_dir, 'neighborhood_analysis.csv'), index=False)
    
    # Create visualizations
    create_neighborhood_plots(neighborhood_df, optimal_tilt, optimal_azimuth, output_dir)
    
    return neighborhood_df

def create_neighborhood_plots(neighborhood_df, optimal_tilt, optimal_azimuth, output_dir):
    """Create various plots to visualize neighborhood analysis."""
    
    # 1. 3D Surface Plot of Production
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh grid
    tilts = neighborhood_df['tilt'].unique()
    azimuths = neighborhood_df['azimuth'].unique()
    X, Y = np.meshgrid(tilts, azimuths)
    
    # Reshape production values
    Z_production = neighborhood_df.pivot(index='azimuth', columns='tilt', values='production').values
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z_production, cmap='viridis', alpha=0.8)
    
    # Mark optimal point
    ax.scatter([optimal_tilt], [optimal_azimuth], 
              [neighborhood_df[neighborhood_df['is_optimal']]['production'].values[0]], 
              color='red', s=100, marker='*', label='Optimal')
    
    ax.set_xlabel('Tilt Angle (°)')
    ax.set_ylabel('Azimuth Angle (°)')
    ax.set_zlabel('Total Production (kWh)')
    ax.set_title('Energy Production Landscape Around Optimal Solution')
    plt.colorbar(surf)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neighborhood_3d_production.png'), dpi=300)
    plt.close()
    
    # 2. Contour Plot showing both objectives
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Production contour
    Z_production_2d = neighborhood_df.pivot(index='azimuth', columns='tilt', values='production')
    cs1 = ax1.contour(Z_production_2d.columns, Z_production_2d.index, Z_production_2d.values, levels=15)
    ax1.clabel(cs1, inline=True, fontsize=8)
    ax1.scatter(optimal_tilt, optimal_azimuth, color='red', s=100, marker='*', zorder=5)
    ax1.set_xlabel('Tilt Angle (°)')
    ax1.set_ylabel('Azimuth Angle (°)')
    ax1.set_title('Total Energy Production (kWh)')
    ax1.grid(True, alpha=0.3)
    
    # Mismatch contour
    Z_mismatch_2d = neighborhood_df.pivot(index='azimuth', columns='tilt', values='mismatch')
    cs2 = ax2.contour(Z_mismatch_2d.columns, Z_mismatch_2d.index, Z_mismatch_2d.values, levels=15)
    ax2.clabel(cs2, inline=True, fontsize=8)
    ax2.scatter(optimal_tilt, optimal_azimuth, color='red', s=100, marker='*', zorder=5)
    ax2.set_xlabel('Tilt Angle (°)')
    ax2.set_ylabel('Azimuth Angle (°)')
    ax2.set_title('Weighted Energy Mismatch (kWh)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neighborhood_contours.png'), dpi=300)
    plt.close()
    
    # 3. Performance comparison bar chart
    # Select a subset of neighbors for clarity
    neighbors_sample = neighborhood_df.nsmallest(10, 'distance_from_optimal')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(neighbors_sample))
    width = 0.35
    
    # Normalize values for comparison
    prod_norm = neighbors_sample['production'] / neighbors_sample['production'].max()
    mismatch_norm = 1 - (neighbors_sample['mismatch'] / neighbors_sample['mismatch'].max())
    
    bars1 = ax.bar([i - width/2 for i in x], prod_norm, width, label='Normalized Production', color='green')
    bars2 = ax.bar([i + width/2 for i in x], mismatch_norm, width, label='Normalized Match Quality', color='blue')
    
    # Labels
    labels = [f"T:{row['tilt']:.0f}°\nA:{row['azimuth']:.0f}°" for _, row in neighbors_sample.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Mark optimal
    for i, (_, row) in enumerate(neighbors_sample.iterrows()):
        if row['is_optimal']:
            ax.axvspan(i-0.5, i+0.5, alpha=0.3, color='red', label='Optimal Solution')
            break
    
    ax.set_ylabel('Normalized Performance')
    ax.set_title('Performance Comparison: Optimal vs Neighboring Solutions')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neighborhood_comparison.png'), dpi=300)
    plt.close()
    
    # 4. Sensitivity heatmap
    # Calculate percentage change from optimal
    optimal_prod = neighborhood_df[neighborhood_df['is_optimal']]['production'].values[0]
    optimal_mismatch = neighborhood_df[neighborhood_df['is_optimal']]['mismatch'].values[0]
    
    neighborhood_df['prod_change_pct'] = ((neighborhood_df['production'] - optimal_prod) / optimal_prod) * 100
    neighborhood_df['mismatch_change_pct'] = ((neighborhood_df['mismatch'] - optimal_mismatch) / optimal_mismatch) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Production change heatmap
    prod_change_pivot = neighborhood_df.pivot(index='azimuth', columns='tilt', values='prod_change_pct')
    sns.heatmap(prod_change_pivot, annot=True, fmt='.1f', cmap='RdBu_r', center=0, ax=ax1)
    ax1.set_title('Production Change from Optimal (%)')
    ax1.set_xlabel('Tilt Angle (°)')
    ax1.set_ylabel('Azimuth Angle (°)')
    
    # Mismatch change heatmap
    mismatch_change_pivot = neighborhood_df.pivot(index='azimuth', columns='tilt', values='mismatch_change_pct')
    sns.heatmap(mismatch_change_pivot, annot=True, fmt='.1f', cmap='RdBu', center=0, ax=ax2)
    ax2.set_title('Mismatch Change from Optimal (%)')
    ax2.set_xlabel('Tilt Angle (°)')
    ax2.set_ylabel('Azimuth Angle (°)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'neighborhood_sensitivity.png'), dpi=300)
    plt.close()
    
    logging.info("Neighborhood analysis plots created")

def create_case_study_report(df, config, args, optimal_tilt, optimal_azimuth, 
                           optimal_battery_capacity, results_summary, output_dir):
    """
    Create a comprehensive case study report showing the complete input->process->output flow.
    """
    logging.info("Creating case study report...")
    
    # Create case study directory
    case_study_dir = os.path.join(output_dir, 'case_study')
    os.makedirs(case_study_dir, exist_ok=True)
    
    # 1. Document Inputs
    inputs = {
        'Location': {
            'Latitude': args.latitude,
            'Longitude': args.longitude,
            'Timezone': 'Europe/Athens'
        },
        'System Configuration': {
            'Panel Model': config['solar_panel']['name'],
            'Panel Power': f"{config['solar_panel']['power_rating']} W",
            'Number of Panels': results_summary.get('Number of Panels Installed', 'N/A'),
            'Available Area': f"{config.get('available_area', 1500)} m²",
            'Inverter Efficiency': f"{config['inverter']['eta_inv_nom']}%"
        },
        'Economic Parameters': {
            'Electricity Price': '$0.20/kWh',
            'Feed-in Tariff': '$0.10/kWh',
            'System Lifetime': '25 years',
            'Discount Rate': '5%'
        },
        'Data Period': {
            'Start Date': df.index.min().strftime('%Y-%m-%d'),
            'End Date': df.index.max().strftime('%Y-%m-%d'),
            'Data Points': len(df)
        }
    }
    
    # 2. Document Process
    process_steps = {
        'Step 1: Data Preprocessing': [
            'Load hourly weather and consumption data',
            'Clean and validate data',
            'Calculate solar position for each hour',
            'Compute Direct Normal Irradiance (DNI)'
        ],
        'Step 2: Multi-Objective Optimization': [
            'Define objectives: minimize mismatch, maximize production',
            'Apply genetic algorithm (DEAP) optimization',
            'Generate Pareto front of solutions',
            'Select balanced optimal solution'
        ],
        'Step 3: Energy Simulation': [
            'Calculate irradiance on tilted surface',
            'Apply temperature and loss models',
            'Compute DC and AC power output',
            'Calculate energy flows and efficiency'
        ],
        'Step 4: Battery Optimization': [
            'Simulate various battery capacities',
            'Calculate self-sufficiency improvements',
            'Determine optimal capacity based on economics'
        ],
        'Step 5: Economic Analysis': [
            'Calculate initial investment',
            'Project 25-year cash flows',
            'Compute NPV, IRR, and payback period'
        ]
    }
    
    # 3. Document Outputs
    outputs = {
        'Optimal Configuration': {
            'Tilt Angle': f"{optimal_tilt:.2f}°",
            'Azimuth Angle': f"{optimal_azimuth:.2f}°",
            'Battery Capacity': f"{optimal_battery_capacity:.2f} kWh"
        },
        'Energy Performance': {
            'Annual Production': results_summary.get('Balanced Total Energy Produced (kWh)', 'N/A'),
            'System Efficiency': results_summary.get('System Efficiency (%)', 'N/A'),
            'Performance Ratio': results_summary.get('Performance Ratio (%)', 'N/A'),
            'Self-Sufficiency': results_summary.get('Battery Self-Sufficiency Rate (%)', 'N/A')
        },
        'Economic Results': {
            'Total Investment': results_summary.get('Total Investment ($)', 'N/A'),
            'Payback Period': results_summary.get('Payback Period (years)', 'N/A'),
            'Net Present Value': results_summary.get('Net Present Value ($)', 'N/A'),
            'Internal Rate of Return': results_summary.get('Internal Rate of Return (%)', 'N/A')
        }
    }
    
    # Create visualizations for case study
    create_case_study_visualizations(df, optimal_tilt, optimal_azimuth, case_study_dir)
    
    # Generate HTML report
    html_content = generate_case_study_html(inputs, process_steps, outputs, case_study_dir)
    
    # Save HTML report
    with open(os.path.join(case_study_dir, 'case_study_report.html'), 'w') as f:
        f.write(html_content)
    
    # Also save as structured JSON
    case_study_data = {
        'inputs': inputs,
        'process': process_steps,
        'outputs': outputs,
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    with open(os.path.join(case_study_dir, 'case_study_data.json'), 'w') as f:
        json.dump(case_study_data, f, indent=2)
    
    logging.info(f"Case study report created in {case_study_dir}")
    
def create_case_study_visualizations(df, optimal_tilt, optimal_azimuth, output_dir):
    """Create specific visualizations for the case study."""
    
    # 1. System overview diagram
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Monthly production bar chart
    monthly_prod = df.resample('M')['E_ac'].sum() / 1000
    monthly_cons = df.resample('M')['Load (kW)'].sum()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = range(12)
    
    ax1.bar(x, monthly_prod.values, alpha=0.7, label='Production', color='green')
    ax1.bar(x, monthly_cons.values, alpha=0.7, label='Consumption', color='red')
    ax1.set_xticks(x)
    ax1.set_xticklabels(months)
    ax1.set_ylabel('Energy (kWh)')
    ax1.set_title('Monthly Energy Production vs Consumption')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Daily profile (average)
    hourly_avg_prod = df.groupby(df.index.hour)['ac_power_output'].mean() / 1000
    hourly_avg_cons = df.groupby(df.index.hour)['Load (kW)'].mean()
    
    ax2.plot(hourly_avg_prod.index, hourly_avg_prod.values, 'g-', linewidth=2, label='Avg Production')
    ax2.plot(hourly_avg_cons.index, hourly_avg_cons.values, 'r-', linewidth=2, label='Avg Consumption')
    ax2.fill_between(hourly_avg_prod.index, 0, hourly_avg_prod.values, alpha=0.3, color='green')
    ax2.fill_between(hourly_avg_cons.index, 0, hourly_avg_cons.values, alpha=0.3, color='red')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('Average Daily Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # System configuration visualization
    ax3.text(0.5, 0.9, 'SYSTEM CONFIGURATION', ha='center', va='center', fontsize=16, weight='bold')
    ax3.text(0.5, 0.7, f'Panel Tilt: {optimal_tilt:.1f}°', ha='center', va='center', fontsize=14)
    ax3.text(0.5, 0.5, f'Panel Azimuth: {optimal_azimuth:.1f}°', ha='center', va='center', fontsize=14)
    ax3.text(0.5, 0.3, f'Number of Panels: {df.attrs.get("number_of_panels", "N/A")}', ha='center', va='center', fontsize=14)
    ax3.text(0.5, 0.1, f'Total Capacity: {df.attrs.get("total_capacity", "N/A")} kWp', ha='center', va='center', fontsize=14)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Key metrics
    total_prod = df['E_ac'].sum() / 1000
    total_cons = df['Load (kW)'].sum()
    self_consumption = min(total_prod, total_cons)
    
    ax4.text(0.5, 0.9, 'KEY METRICS', ha='center', va='center', fontsize=16, weight='bold')
    ax4.text(0.5, 0.7, f'Annual Production: {total_prod:,.0f} kWh', ha='center', va='center', fontsize=12)
    ax4.text(0.5, 0.5, f'Annual Consumption: {total_cons:,.0f} kWh', ha='center', va='center', fontsize=12)
    ax4.text(0.5, 0.3, f'Self-Consumption: {(self_consumption/total_prod)*100:.1f}%', ha='center', va='center', fontsize=12)
    ax4.text(0.5, 0.1, f'Energy Independence: {(self_consumption/total_cons)*100:.1f}%', ha='center', va='center', fontsize=12)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'case_study_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
def generate_case_study_html(inputs, process_steps, outputs, output_dir):
    """Generate HTML content for the case study report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PV System Optimization Case Study</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .section {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-size: 18px; font-weight: bold; color: #27ae60; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>PV System Optimization Case Study</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>1. INPUTS</h2>
            <h3>Location Information</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                {''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in inputs['Location'].items()])}
            </table>
            
            <h3>System Configuration</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                {''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in inputs['System Configuration'].items()])}
            </table>
        </div>
        
        <div class="section">
            <h2>2. OPTIMIZATION PROCESS</h2>
            {''.join([f"<h3>{step}</h3><ul>{''.join([f'<li>{item}</li>' for item in items])}</ul>" 
                      for step, items in process_steps.items()])}
        </div>
        
        <div class="section">
            <h2>3. OUTPUTS</h2>
            <h3>Optimal Configuration</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                {''.join([f"<tr><td>{k}</td><td class='metric'>{v}</td></tr>" 
                         for k, v in outputs['Optimal Configuration'].items()])}
            </table>
            
            <h3>System Performance</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" 
                         for k, v in outputs['Energy Performance'].items()])}
            </table>
            
            <h3>Economic Analysis</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" 
                         for k, v in outputs['Economic Results'].items()])}
            </table>
        </div>
        
        <div class="section">
            <h2>4. VISUAL RESULTS</h2>
            <img src="case_study_overview.png" alt="System Overview">
            <p><em>Figure 1: Comprehensive system overview showing monthly patterns, daily profiles, 
            configuration, and key metrics</em></p>
        </div>
    </body>
    </html>
    """
    return html

def calculate_baseline_performance(df_subset, dni_extra, number_of_panels, inverter_params, latitude):
    """
    FIXED: Calculate baseline performance with consistent mismatch calculation.
    """
    logging.info("Calculating baseline performance with standard angles...")
    
    baseline_tilt = latitude
    baseline_azimuth = 180.0
    
    logging.info(f"Baseline configuration: Tilt={baseline_tilt:.1f}°, Azimuth={baseline_azimuth:.1f}°")
    
    # Calculate with baseline angles
    df_baseline = df_subset.copy()
    df_baseline = calculate_total_irradiance(df_baseline, baseline_tilt, baseline_azimuth, dni_extra)
    df_baseline = calculate_energy_production(df_baseline, number_of_panels, inverter_params)
    
    # CRITICAL: Debug PR calculation for baseline
    baseline_pr_debug = debug_pr_calculation_detailed(df_baseline, "BASELINE")
    
    # Calculate total production
    baseline_production_kwh = df_baseline['E_ac'].sum() / 1000
    
    # FIXED: Calculate mismatch using the SAME method as optimization
    df_baseline['weighting_factor'] = calculate_weighting_factors(df_baseline)
    df_baseline['load_wh'] = df_baseline['Load (kW)'] * 1000  # Convert to Wh
    df_baseline['hourly_mismatch'] = df_baseline['E_ac'] - df_baseline['load_wh']
    df_baseline['weighted_mismatch'] = df_baseline['weighting_factor'] * np.abs(df_baseline['hourly_mismatch'] / 1000)  # kWh
    baseline_mismatch = df_baseline['weighted_mismatch'].sum()
    
    # Log intermediate values for debugging
    total_irradiance = df_baseline['total_irradiance'].sum()
    avg_pr = df_baseline['PR'].mean()
    
    logging.info(f"Baseline - Total irradiance on panels: {total_irradiance:.2f} Wh/m²")
    logging.info(f"Baseline - Average PR: {avg_pr:.4f}")
    logging.info(f"Baseline - Total production: {baseline_production_kwh:.2f} kWh")
    logging.info(f"Baseline - Weighted mismatch: {baseline_mismatch:.2f} kWh")
    
    return {
        'tilt': baseline_tilt,
        'azimuth': baseline_azimuth,
        'production_kwh': baseline_production_kwh,
        'weighted_mismatch': baseline_mismatch,
        'avg_pr': avg_pr,
        'total_irradiance': total_irradiance
    }
    
def create_comprehensive_battery_analysis_plot(battery_results, optimal_capacity, df, output_dir):
    """
    Create a comprehensive visualization showing why the selected battery capacity is optimal.
    """
    logging.info("Creating comprehensive battery optimization visualization...")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Define grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :2])  # Self-sufficiency and consumption
    ax2 = fig.add_subplot(gs[0, 2])   # Key metrics for optimal
    ax3 = fig.add_subplot(gs[1, :2])  # Economic analysis
    ax4 = fig.add_subplot(gs[1, 2])   # Why 51kWh explanation
    ax5 = fig.add_subplot(gs[2, :])   # Combined view
    
    # 1. Self-sufficiency and Self-consumption curves
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(battery_results['capacity_kwh'], battery_results['self_sufficiency_rate'], 
                     'g-', linewidth=3, label='Self-Sufficiency Rate', marker='o')
    line2 = ax1_twin.plot(battery_results['capacity_kwh'], battery_results['self_consumption_rate'], 
                          'b-', linewidth=3, label='Self-Consumption Rate', marker='s')
    
    # Mark optimal point
    optimal_idx = battery_results['capacity_kwh'] == optimal_capacity
    if optimal_idx.any():
        opt_self_suff = battery_results.loc[optimal_idx, 'self_sufficiency_rate'].values[0]
        opt_self_cons = battery_results.loc[optimal_idx, 'self_consumption_rate'].values[0]
        
        ax1.scatter(optimal_capacity, opt_self_suff, s=200, c='red', marker='*', zorder=5)
        ax1_twin.scatter(optimal_capacity, opt_self_cons, s=200, c='red', marker='*', zorder=5)
        
        # Add vertical line at optimal capacity
        ax1.axvline(x=optimal_capacity, color='red', linestyle='--', alpha=0.5, label=f'Optimal: {optimal_capacity} kWh')
    
    ax1.set_xlabel('Battery Capacity (kWh)', fontsize=12)
    ax1.set_ylabel('Self-Sufficiency Rate (%)', color='g', fontsize=12)
    ax1_twin.set_ylabel('Self-Consumption Rate (%)', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='g')
    ax1_twin.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Energy Independence vs Battery Capacity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    # 2. Key metrics box for optimal battery
    if optimal_idx.any():
        opt_data = battery_results[optimal_idx].iloc[0]
        
        metrics_text = f"""OPTIMAL BATTERY: {optimal_capacity} kWh
        
Self-Sufficiency: {opt_data['self_sufficiency_rate']:.1f}%
Self-Consumption: {opt_data['self_consumption_rate']:.1f}%
Grid Import Reduction: {(1 - opt_data['grid_import_kwh']/battery_results.iloc[0]['grid_import_kwh'])*100:.1f}%
Equivalent Cycles/Year: {opt_data['equivalent_full_cycles']:.1f}
Payback Period: {opt_data['simple_payback_years']:.1f} years"""
        
        ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes, 
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    ax2.axis('off')
    
    # 3. Economic analysis
    # Filter out unrealistic payback periods for better visualization
    econ_data = battery_results[battery_results['simple_payback_years'] < 50].copy()
    
    ax3_twin = ax3.twinx()
    
    ax3.bar(econ_data['capacity_kwh'], econ_data['annual_savings'] + econ_data['annual_revenue'], 
            alpha=0.7, color='green', label='Annual Benefits')
    line3 = ax3_twin.plot(econ_data['capacity_kwh'], econ_data['simple_payback_years'], 
                          'r-', linewidth=2, label='Payback Period', marker='o')
    
    # Mark optimal
    if optimal_capacity in econ_data['capacity_kwh'].values:
        opt_idx_econ = econ_data['capacity_kwh'] == optimal_capacity
        opt_benefits = (econ_data.loc[opt_idx_econ, 'annual_savings'] + 
                       econ_data.loc[opt_idx_econ, 'annual_revenue']).values[0]
        opt_payback = econ_data.loc[opt_idx_econ, 'simple_payback_years'].values[0]
        
        ax3_twin.scatter(optimal_capacity, opt_payback, s=200, c='red', marker='*', zorder=5)
    
    ax3.set_xlabel('Battery Capacity (kWh)', fontsize=12)
    ax3.set_ylabel('Annual Benefits ($)', color='green', fontsize=12)
    ax3_twin.set_ylabel('Payback Period (years)', color='red', fontsize=12)
    ax3.set_title('Economic Analysis', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Why this capacity explanation
    explanation_text = f"""WHY {optimal_capacity} kWh?

1. BALANCE POINT:
   • Maximizes self-sufficiency
   • Maintains high utilization
   • Avoids oversizing

2. ECONOMIC OPTIMUM:
   • Best payback period
   • Maximum ROI
   • Reasonable investment

3. TECHNICAL FIT:
   • Covers typical daily cycles
   • Handles seasonal variations
   • Efficient charge/discharge"""
    
    ax4.text(0.1, 0.5, explanation_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    ax4.axis('off')
    
    # 5. Combined optimization view
    # Normalize metrics for comparison
    norm_self_suff = battery_results['self_sufficiency_rate'] / 100
    norm_grid_reduction = 1 - (battery_results['grid_import_kwh'] / battery_results['grid_import_kwh'].max())
    norm_cycles = 1 - (battery_results['equivalent_full_cycles'] / battery_results['equivalent_full_cycles'].max())
    
    # Calculate combined score (you can adjust weights)
    combined_score = (0.4 * norm_self_suff + 0.3 * norm_grid_reduction + 0.3 * norm_cycles)
    
    ax5.plot(battery_results['capacity_kwh'], combined_score, 'purple', linewidth=3, 
             label='Combined Optimization Score', marker='o')
    
    # Mark optimal
    if optimal_capacity in battery_results['capacity_kwh'].values:
        opt_score = combined_score[battery_results['capacity_kwh'] == optimal_capacity].values[0]
        ax5.scatter(optimal_capacity, opt_score, s=300, c='red', marker='*', zorder=5,
                   label=f'Optimal: {optimal_capacity} kWh')
        ax5.axvline(x=optimal_capacity, color='red', linestyle='--', alpha=0.5)
        
        # Add annotation
        ax5.annotate(f'Optimal Battery\n{optimal_capacity} kWh', 
                    xy=(optimal_capacity, opt_score), 
                    xytext=(optimal_capacity + 5, opt_score + 0.05),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red'))
    
    ax5.set_xlabel('Battery Capacity (kWh)', fontsize=12)
    ax5.set_ylabel('Combined Optimization Score', fontsize=12)
    ax5.set_title('Overall Battery Optimization (Why 51 kWh is Optimal)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    plt.suptitle('Comprehensive Battery Capacity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'battery_optimization_comprehensive.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Comprehensive battery analysis plot created")
    
    # Create additional detailed comparison plot
    create_battery_comparison_table(battery_results, optimal_capacity, output_dir)

def create_battery_comparison_table(battery_results, optimal_capacity, output_dir):
    """Create a visual comparison table for different battery capacities."""
    
    # Select key capacities for comparison
    comparison_capacities = [0, 10, 25, optimal_capacity, 75, 100]
    comparison_capacities = [c for c in comparison_capacities if c in battery_results['capacity_kwh'].values]
    
    comparison_data = battery_results[battery_results['capacity_kwh'].isin(comparison_capacities)].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    headers = ['Capacity\n(kWh)', 'Self-Sufficiency\n(%)', 'Grid Import\nReduction (%)', 
               'Cycles/Year', 'Payback\n(years)', 'Status']
    
    for _, row in comparison_data.iterrows():
        grid_reduction = (1 - row['grid_import_kwh']/battery_results.iloc[0]['grid_import_kwh']) * 100
        status = '★ OPTIMAL ★' if row['capacity_kwh'] == optimal_capacity else ''
        
        table_data.append([
            f"{row['capacity_kwh']:.0f}",
            f"{row['self_sufficiency_rate']:.1f}",
            f"{grid_reduction:.1f}",
            f"{row['equivalent_full_cycles']:.1f}",
            f"{row['simple_payback_years']:.1f}" if row['simple_payback_years'] < 100 else "N/A",
            status
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the table

def create_comprehensive_summary(
    df, df_original, optimal_tilt, optimal_azimuth, optimal_capacity,
    balanced_weighted_mismatch, balanced_production, baseline_results,
    seasonal_stats, battery_results, 
    initial_investment,  # <--- ADD initial_investment HERE
    financial_metrics, 
    efficiency_metrics, number_of_panels, total_panel_area, 
    panel_efficiency, system_efficiency, energy_losses):
    # If you also need battery_requirements for seasonal battery needs in this summary, pass it too.
    
    """Create a comprehensive summary with all metrics."""
    
    # Calculate total annual consumption
    total_annual_consumption = df_original['Load (kW)'].sum()
    
    # Calculate production/consumption ratio
    production_consumption_ratio = (balanced_production / total_annual_consumption) * 100
    
    # Get battery metrics for optimal capacity
    if optimal_capacity in battery_results['capacity_kwh'].values:
        battery_metrics = battery_results[battery_results['capacity_kwh'] == optimal_capacity].iloc[0]
    else:
        battery_metrics = pd.Series({'self_sufficiency_rate': 'N/A', 'self_consumption_rate': 'N/A',
                                   'equivalent_full_cycles': 'N/A', 'simple_payback_years': 'N/A'})
    
    # Calculate improvements from baseline
    production_improvement = ((balanced_production - baseline_results['production_kwh']) / 
                            baseline_results['production_kwh'] * 100)
    mismatch_improvement = ((baseline_results['weighted_mismatch'] - balanced_weighted_mismatch) / 
                          baseline_results['weighted_mismatch'] * 100)
    
    # Create comprehensive summary
    summary = {
        # System Configuration
        'Number of Panels Installed': number_of_panels,
        'Total Panel Area (m²)': f"{total_panel_area:.2f}",
        'Panel Efficiency (%)': f"{panel_efficiency * 100:.2f}",
        
        # Optimal Configuration
        'Optimal Tilt Angle (°)': f"{optimal_tilt:.2f}",
        'Optimal Azimuth Angle (°)': f"{optimal_azimuth:.2f}",
        'Optimal Battery Capacity (kWh)': f"{optimal_capacity:.2f}",
        
        # Energy Production and Consumption
        'Total Annual Consumption (kWh)': f"{total_annual_consumption:.2f}",
        'Total Energy Produced - Optimal Angles (kWh)': f"{balanced_production:.2f}",
        'Production/Consumption Ratio (%)': f"{production_consumption_ratio:.2f}",
        'Energy Coverage Ratio (%)': f"{production_consumption_ratio:.2f}",
        
        # Baseline Comparison
        'Baseline Production (kWh)': f"{baseline_results['production_kwh']:.2f}",
        'Baseline Configuration': f"Tilt: {baseline_results['tilt']:.1f}°, Azimuth: {baseline_results['azimuth']:.1f}°",
        'Production Improvement from Optimization (%)': f"{production_improvement:.2f}",
        'Mismatch Improvement from Optimization (%)': f"{mismatch_improvement:.2f}",
        
        # System Performance
        'Balanced Weighted Energy Mismatch (kWh)': f"{balanced_weighted_mismatch:.2f}",
        'System Efficiency (%)': f"{system_efficiency:.2f}",
        'Performance Ratio (%)': f"{efficiency_metrics['performance_ratio']:.2f}",
        'System Yield (kWh/kWp)': f"{efficiency_metrics['system_yield']:.2f}",
        'Capacity Factor (%)': f"{efficiency_metrics['capacity_factor']:.2f}",
        
        # Battery Performance
        'Battery Self-Sufficiency Rate (%)': f"{battery_metrics['self_sufficiency_rate']:.2f}",
        'Battery Self-Consumption Rate (%)': f"{battery_metrics['self_consumption_rate']:.2f}",
        'Battery Equivalent Full Cycles (per year)': f"{battery_metrics['equivalent_full_cycles']:.2f}",
        'Battery Simple Payback (years)': f"{battery_metrics['simple_payback_years']:.2f}",
        
        # Economic Analysis
        'Total Investment ($)': f"{initial_investment['total_investment']:,.2f}",
        'Net Present Value ($)': f"{financial_metrics['NPV']:,.2f}",
        'Internal Rate of Return (%)': f"{financial_metrics['IRR']:.2f}" if financial_metrics['IRR'] is not None else "N/A",
        'Payback Period (years)': f"{financial_metrics['Payback_Period_Years']:.2f}",
        'Levelized Cost of Electricity ($/kWh)': f"{financial_metrics['LCOE']:.4f}",
    }
    
    # Add seasonal information
    for season in seasonal_stats.index:
        summary[f'Production {season} (kWh)'] = f"{seasonal_stats.loc[season, 'E_ac_kwh']:.2f}"
        summary[f'Consumption {season} (kWh)'] = f"{seasonal_stats.loc[season, 'load_wh_kwh']:.2f}"
        summary[f'Self-Consumption Ratio {season} (%)'] = f"{seasonal_stats.loc[season, 'self_consumption_ratio']:.2f}"
        summary[f'Self-Sufficiency Ratio {season} (%)'] = f"{seasonal_stats.loc[season, 'self_sufficiency_ratio']:.2f}"
        
    # Add energy losses
    
    for _, row in energy_losses.iterrows():
        try:
            # Now, row['Energy Lost (kWh)'] will be numeric from summarize_energy
            energy_loss_kwh = float(row['Energy Lost (kWh)']) 
            summary[f'Energy Loss - {row["Loss Type"]} (kWh)'] = f"{energy_loss_kwh:.2f}" # This line should now work
        except (ValueError, TypeError):
            # This except block is now a fallback for any unexpected non-numeric data
            summary[f'Energy Loss - {row["Loss Type"]} (kWh)'] = "N/A"

    
    return summary

def create_enhanced_html_summary(summary_df, output_dir):
    """Create an enhanced HTML summary with proper formatting and all data."""
    
    # Group metrics by category
    categories = {
        'System Configuration': ['Number of Panels', 'Total Panel Area', 'Panel Efficiency'],
        'Optimal Configuration': ['Optimal Tilt', 'Optimal Azimuth', 'Optimal Battery'],
        'Energy Overview': ['Total Annual Consumption', 'Total Energy Produced', 'Production/Consumption Ratio', 'Energy Coverage'],
        'Performance Metrics': ['System Efficiency', 'Performance Ratio', 'System Yield', 'Capacity Factor'],
        'Battery Performance': ['Battery Self-Sufficiency', 'Battery Self-Consumption', 'Battery Equivalent', 'Battery Simple'],
        'Economic Results': ['Total Investment', 'Net Present Value', 'Internal Rate', 'Payback Period', 'Levelized Cost'],
        'Seasonal Performance': ['Production Winter', 'Production Spring', 'Production Summer', 'Production Fall',
                               'Self-Sufficiency.*Winter', 'Self-Sufficiency.*Spring', 'Self-Sufficiency.*Summer', 'Self-Sufficiency.*Fall'],
        'Energy Losses': ['Energy Loss']
    }
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PV System Optimization - Complete Summary</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            h2 {
                color: #34495e;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-top: 30px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th {
                background-color: #3498db;
                color: white;
                padding: 12px;
                text-align: left;
            }
            td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .highlight {
                background-color: #e8f5e9;
                font-weight: bold;
            }
            .metric-value {
                text-align: right;
                font-weight: bold;
            }
            .category-section {
                margin-bottom: 30px;
                background-color: #fafafa;
                padding: 15px;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PV System Optimization - Complete Summary Report</h1>
            <p style="text-align: center; color: #666;">Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    """
    
    # Add each category
    for category, keywords in categories.items():
        html_content += f"""
            <div class="category-section">
                <h2>{category}</h2>
                <table>
                    <tr>
                        <th style="width: 70%;">Metric</th>
                        <th style="width: 30%;">Value</th>
                    </tr>
        """
        
        # Find matching metrics
        for _, row in summary_df.iterrows():
            metric = row['Metric']
            if any(keyword in metric for keyword in keywords):
                # Highlight important metrics
                row_class = 'highlight' if any(key in metric for key in ['Total Energy Produced', 'Optimal', 'Total Annual Consumption']) else ''
                html_content += f"""
                    <tr class="{row_class}">
                        <td>{metric}</td>
                        <td class="metric-value">{row['Value']}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    html_path = os.path.join(output_dir, 'summary_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"Enhanced HTML summary saved to {html_path}")

def validate_results(summary_dict, df):
    """
    Validate results and flag any unrealistic values.
    """
    warnings = []
    
    # Check panel efficiency
    panel_eff = float(summary_dict.get('Panel Efficiency (%)', '0').replace('%', ''))
    if panel_eff < 15:
        warnings.append(f"Panel efficiency ({panel_eff}%) is low for modern panels")
    
    # Check azimuth angle
    azimuth = float(summary_dict.get('Optimal Azimuth Angle (°)', '180'))
    if abs(azimuth - 180) > 45:
        warnings.append(f"Azimuth angle ({azimuth}°) deviates significantly from south")
    
    # Check production improvement
    prod_improvement = float(summary_dict.get('Production Improvement from Optimization (%)', '0').replace('%', ''))
    if abs(prod_improvement) > 15:
        warnings.append(f"Production improvement ({prod_improvement}%) seems unrealistic")
    
    # Check battery payback
    battery_payback = float(summary_dict.get('Battery Simple Payback (years)', '999'))
    if battery_payback < 3:
        warnings.append(f"Battery payback ({battery_payback} years) is unrealistically short")
    
    # Check losses
    total_production = float(summary_dict.get('Total Energy Produced - Optimal Angles (kWh)', '1'))
    for key, value in summary_dict.items():
        if 'Energy Loss' in key:
            loss_val = float(value)
            if loss_val > total_production:
                warnings.append(f"{key} ({loss_val} kWh) exceeds total production")
    
    # Check PR
    pr = float(summary_dict.get('Performance Ratio (%)', '0').replace('%', ''))
    if pr < 75:
        warnings.append(f"Performance Ratio ({pr}%) is low - check system losses")
    
    if warnings:
        logging.warning("VALIDATION WARNINGS:")
        for warning in warnings:
            logging.warning(f"  - {warning}")
    else:
        logging.info("All results passed validation checks")
    
    return warnings

def compare_configurations(df_subset, dni_extra, number_of_panels, inverter_params, 
                          config1_angles, config2_angles, output_dir):
    """
    Compare two configurations in detail to understand differences.
    """
    configs = [
        {"name": "Baseline", "tilt": config1_angles[0], "azimuth": config1_angles[1]},
        {"name": "Optimal", "tilt": config2_angles[0], "azimuth": config2_angles[1]}
    ]
    
    results = []
    
    for config in configs:
        df_temp = df_subset.copy()
        df_temp = calculate_total_irradiance(df_temp, config["tilt"], config["azimuth"], dni_extra)
        df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)
        
        # Calculate key metrics
        total_irradiance_on_panel = df_temp['total_irradiance'].sum()
        total_production = df_temp['E_ac'].sum() / 1000
        avg_pr = df_temp[df_temp['PR'] > 0]['PR'].mean()
        
        # Monthly breakdown
        df_temp['month'] = df_temp.index.month
        monthly_prod = df_temp.groupby('month')['E_ac'].sum() / 1000
        
        results.append({
            'config': config['name'],
            'tilt': config['tilt'],
            'azimuth': config['azimuth'],
            'total_irradiance_on_panel': total_irradiance_on_panel,
            'total_production': total_production,
            'avg_pr': avg_pr,
            'monthly_production': monthly_prod
        })
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Monthly production comparison
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = np.arange(12)
    width = 0.35
    
    ax1.bar(x - width/2, results[0]['monthly_production'].values, width, 
            label=f"{results[0]['config']} ({results[0]['tilt']:.1f}°, {results[0]['azimuth']:.1f}°)")
    ax1.bar(x + width/2, results[1]['monthly_production'].values, width,
            label=f"{results[1]['config']} ({results[1]['tilt']:.1f}°, {results[1]['azimuth']:.1f}°)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(months)
    ax1.set_ylabel('Production (kWh)')
    ax1.set_title('Monthly Production Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add percentage differences
    for i in range(12):
        baseline = results[0]['monthly_production'].iloc[i]
        optimal = results[1]['monthly_production'].iloc[i]
        if baseline > 0:
            pct_diff = ((optimal - baseline) / baseline) * 100
            ax1.text(i, max(baseline, optimal) + 500, f'{pct_diff:+.1f}%', 
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Summary metrics
    metrics_text = f"""Configuration Comparison:
    
BASELINE:
  Tilt: {results[0]['tilt']:.1f}°, Azimuth: {results[0]['azimuth']:.1f}°
  Total Irradiance on Panel: {results[0]['total_irradiance_on_panel']/1e6:.2f} MWh/m²
  Total Production: {results[0]['total_production']:,.0f} kWh
  Average PR: {results[0]['avg_pr']:.3f}
  
OPTIMAL:
  Tilt: {results[1]['tilt']:.1f}°, Azimuth: {results[1]['azimuth']:.1f}°
  Total Irradiance on Panel: {results[1]['total_irradiance_on_panel']/1e6:.2f} MWh/m²
  Total Production: {results[1]['total_production']:,.0f} kWh
  Average PR: {results[1]['avg_pr']:.3f}
  
IMPROVEMENT:
  Production: {((results[1]['total_production']-results[0]['total_production'])/results[0]['total_production']*100):.2f}%
  Irradiance: {((results[1]['total_irradiance_on_panel']-results[0]['total_irradiance_on_panel'])/results[0]['total_irradiance_on_panel']*100):.2f}%"""
    
    ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'configuration_comparison.png'), dpi=300)
    plt.close()
    
    return results

def debug_mismatch_calculation(df, angles, label=""):
    """
    Debug helper function to analyze mismatch calculation components.
    """
    logging.info(f"=== MISMATCH DEBUG: {label} ===")
    
    # Calculate components
    df['load_wh'] = df['Load (kW)'] * 1000
    df['hourly_mismatch'] = df['E_ac'] - df['load_wh'] 
    df['abs_mismatch'] = np.abs(df['hourly_mismatch'])
    
    # Separate deficit and surplus
    df['deficit'] = (-df['hourly_mismatch']).clip(lower=0)  # When load > production
    df['surplus'] = df['hourly_mismatch'].clip(lower=0)     # When production > load
    
    # Calculate weighting factors
    df['weighting_factor'] = calculate_weighting_factors(df)
    df['weighted_abs_mismatch'] = df['weighting_factor'] * df['abs_mismatch'] / 1000  # kWh
    
    # Summary statistics
    total_production = df['E_ac'].sum() / 1000  # kWh
    total_consumption = df['load_wh'].sum() / 1000  # kWh
    total_deficit = df['deficit'].sum() / 1000  # kWh
    total_surplus = df['surplus'].sum() / 1000  # kWh
    total_weighted_mismatch = df['weighted_abs_mismatch'].sum()  # kWh
    avg_weighting_factor = df['weighting_factor'].mean()
    
    logging.info(f"Angles: Tilt={angles[0]:.2f}°, Azimuth={angles[1]:.2f}°")
    logging.info(f"Total Production: {total_production:,.2f} kWh")
    logging.info(f"Total Consumption: {total_consumption:,.2f} kWh")
    logging.info(f"Total Deficit (Load > Prod): {total_deficit:,.2f} kWh")
    logging.info(f"Total Surplus (Prod > Load): {total_surplus:,.2f} kWh")
    logging.info(f"Average Weighting Factor: {avg_weighting_factor:.3f}")
    logging.info(f"Total Weighted Mismatch: {total_weighted_mismatch:,.2f} kWh")
    logging.info(f"Mismatch as % of Production: {(total_weighted_mismatch/total_production*100):.2f}%")
    
    return total_weighted_mismatch

def validate_mismatch_calculation(df_subset, dni_extra, number_of_panels, inverter_params, 
                                  baseline_angles, optimal_angles, output_dir):
    """
    Comprehensive validation of mismatch calculations between baseline and optimal.
    """
    logging.info("=== VALIDATING MISMATCH CALCULATIONS ===")
    
    # Test both configurations
    configs = [
        {"name": "Baseline", "angles": baseline_angles},
        {"name": "Optimal", "angles": optimal_angles}
    ]
    
    results = {}
    
    for config in configs:
        name = config["name"]
        angles = config["angles"]
        tilt, azimuth = angles
        
        logging.info(f"\n--- Analyzing {name} Configuration ---")
        
        # Calculate energy production
        df_temp = df_subset.copy()
        df_temp = calculate_total_irradiance(df_temp, tilt, azimuth, dni_extra)
        df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)
        
        # Debug the mismatch calculation
        mismatch = debug_mismatch_calculation(df_temp, angles, name)
        
        # Store results
        results[name] = {
            'angles': angles,
            'production_kwh': df_temp['E_ac'].sum() / 1000,
            'weighted_mismatch_kwh': mismatch,
            'dataframe': df_temp
        }
    
    # Compare results
    baseline_prod = results['Baseline']['production_kwh']
    optimal_prod = results['Optimal']['production_kwh']
    baseline_mismatch = results['Baseline']['weighted_mismatch_kwh']
    optimal_mismatch = results['Optimal']['weighted_mismatch_kwh']
    
    prod_improvement = ((optimal_prod - baseline_prod) / baseline_prod) * 100
    mismatch_improvement = ((baseline_mismatch - optimal_mismatch) / baseline_mismatch) * 100
    
    logging.info(f"\n=== COMPARISON RESULTS ===")
    logging.info(f"Production Improvement: {prod_improvement:.2f}%")
    logging.info(f"Mismatch Improvement: {mismatch_improvement:.2f}%")
    
    # Validate reasonable ranges
    validation_passed = True
    
    if abs(prod_improvement) > 20:
        logging.warning(f"WARNING: Production improvement of {prod_improvement:.2f}% seems high")
        validation_passed = False
    
    if abs(mismatch_improvement) > 50:
        logging.warning(f"WARNING: Mismatch improvement of {mismatch_improvement:.2f}% seems extreme")
        validation_passed = False
    
    if optimal_mismatch > 100000:  # 100 MWh threshold
        logging.warning(f"WARNING: Optimal mismatch of {optimal_mismatch:.2f} kWh seems unreasonably high")
        validation_passed = False
    
    if baseline_mismatch > 100000:  # 100 MWh threshold
        logging.warning(f"WARNING: Baseline mismatch of {baseline_mismatch:.2f} kWh seems unreasonably high")
        validation_passed = False
    
    # Create validation plots
    create_mismatch_validation_plots(results, output_dir)
    
    if validation_passed:
        logging.info("Validation PASSED - Results appear reasonable")
    else:
        logging.error("Validation FAILED - Check calculations")
    
    return results, validation_passed

def create_mismatch_validation_plots(results, output_dir):
    """
    Create plots to visualize and validate mismatch calculations.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Production comparison
    configs = list(results.keys())
    productions = [results[config]['production_kwh'] for config in configs]
    
    ax1.bar(configs, productions, color=['blue', 'green'])
    ax1.set_ylabel('Production (kWh)')
    ax1.set_title('Total Energy Production Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(productions):
        ax1.text(i, v + 1000, f'{v:,.0f}', ha='center', va='bottom')
    
    # Plot 2: Mismatch comparison
    mismatches = [results[config]['weighted_mismatch_kwh'] for config in configs]
    
    ax2.bar(configs, mismatches, color=['red', 'orange'])
    ax2.set_ylabel('Weighted Mismatch (kWh)')
    ax2.set_title('Weighted Energy Mismatch Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(mismatches):
        ax2.text(i, v + max(mismatches)*0.02, f'{v:,.0f}', ha='center', va='bottom')
    
    # Plot 3: Hourly profile comparison (sample week)
    baseline_df = results['Baseline']['dataframe']
    optimal_df = results['Optimal']['dataframe']
    
    # Get first week of data for visualization
    sample_data = baseline_df.head(168)  # First week (168 hours)
    sample_optimal = optimal_df.head(168)
    
    hours = range(len(sample_data))
    
    ax3.plot(hours, sample_data['E_ac']/1000, label='Baseline Production', alpha=0.7)
    ax3.plot(hours, sample_optimal['E_ac']/1000, label='Optimal Production', alpha=0.7)
    ax3.plot(hours, sample_data['Load (kW)'], label='Load', color='red', alpha=0.7)
    
    ax3.set_xlabel('Hour of Week')
    ax3.set_ylabel('Power (kW)')
    ax3.set_title('Sample Week: Production vs Load')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Weighting factors distribution
    baseline_weights = baseline_df['weighting_factor']
    optimal_weights = optimal_df['weighting_factor']
    
    ax4.hist(baseline_weights, bins=20, alpha=0.7, label='Baseline', density=True)
    ax4.hist(optimal_weights, bins=20, alpha=0.7, label='Optimal', density=True)
    ax4.set_xlabel('Weighting Factor')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution of Weighting Factors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mismatch_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Mismatch validation plots saved")

def quick_sanity_check(df):
    """
    Quick sanity check of key data ranges and values.
    """
    logging.info("=== QUICK SANITY CHECK ===")
    
    # Check data ranges
    production_range = (df['E_ac'].min(), df['E_ac'].max())
    load_range = (df['Load (kW)'].min(), df['Load (kW)'].max())
    irradiance_range = (df['total_irradiance'].min(), df['total_irradiance'].max())
    
    logging.info(f"E_ac range: {production_range[0]:.1f} - {production_range[1]:.1f} Wh")
    logging.info(f"Load range: {load_range[0]:.1f} - {load_range[1]:.1f} kW")
    logging.info(f"Irradiance range: {irradiance_range[0]:.1f} - {irradiance_range[1]:.1f} W/m²")
    
    # Check for unreasonable values - FIXED: Removed emoji
    if production_range[1] > 50000:  # 50 kWh per hour seems high for residential
        logging.warning(f"WARNING: Maximum hourly production of {production_range[1]/1000:.1f} kWh seems high")
    
    if load_range[1] > 20:  # 20 kW peak load
        logging.warning(f"WARNING: Maximum load of {load_range[1]:.1f} kW seems high for residential")
    
    if irradiance_range[1] > 1500:  # 1500 W/m² is very high
        logging.warning(f"WARNING: Maximum irradiance of {irradiance_range[1]:.1f} W/m² seems very high")
    
    # Check totals
    total_production = df['E_ac'].sum() / 1000  # kWh
    total_consumption = (df['Load (kW)'] * 1000).sum() / 1000  # kWh
    
    logging.info(f"Annual production: {total_production:,.0f} kWh")
    logging.info(f"Annual consumption: {total_consumption:,.0f} kWh")
    logging.info(f"Production/Consumption ratio: {total_production/total_consumption:.2f}")
    
    return {
        'production_range': production_range,
        'load_range': load_range,
        'irradiance_range': irradiance_range,
        'total_production': total_production,
        'total_consumption': total_consumption
    }

def validate_production_gain(df_subset, dni_extra, number_of_panels, inverter_params, 
                           baseline_angles, optimal_angles, output_dir):
    """
    Comprehensive validation of production gain to ensure fair comparison.
    """
    logging.info("=== COMPREHENSIVE PRODUCTION GAIN VALIDATION ===")
    
    configs = [
        {"name": "Baseline", "tilt": baseline_angles[0], "azimuth": baseline_angles[1]},
        {"name": "Optimal", "tilt": optimal_angles[0], "azimuth": optimal_angles[1]}
    ]
    
    detailed_results = {}
    
    for config in configs:
        name = config["name"]
        tilt = config["tilt"]
        azimuth = config["azimuth"]
        
        logging.info(f"\n--- DETAILED ANALYSIS: {name} ({tilt:.1f}°, {azimuth:.1f}°) ---")
        
        # Step-by-step calculation with logging
        df_temp = df_subset.copy()
        
        # Step 1: Calculate POA irradiance
        df_temp = calculate_total_irradiance(df_temp, tilt, azimuth, dni_extra)
        
        # Step 2: Calculate energy production with detailed logging
        df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)
        
        # Extract key metrics at each stage
        results = {
            'config_name': name,
            'tilt': tilt,
            'azimuth': azimuth,
            
            # Irradiance metrics
            'total_horizontal_irradiance_kwh_m2': df_temp['SolRad_Hor'].sum() / 1000,
            'total_poa_irradiance_kwh_m2': df_temp['total_irradiance'].sum() / 1000,
            'dni_total_kwh_m2': df_temp['DNI'].sum() / 1000,
            'dhi_total_kwh_m2': df_temp['SolRad_Dif'].sum() / 1000,
            
            # Energy flow at each stage (in kWh)
            'incident_energy_total_kwh': df_temp['E_incident'].sum() / 1000,
            'dc_ideal_energy_kwh': df_temp['E_dc_ideal'].sum() / 1000,
            'dc_actual_energy_kwh': df_temp['E_dc_actual'].sum() / 1000,
            'ac_final_energy_kwh': df_temp['E_ac'].sum() / 1000,
            
            # Loss breakdown (in kWh)
            'pre_temp_losses_kwh': df_temp['E_loss_pre_temperature'].sum() / 1000,
            'temperature_losses_kwh': df_temp['E_loss_temperature'].sum() / 1000,
            'inverter_losses_kwh': df_temp['E_loss_inverter'].sum() / 1000,
            'total_losses_kwh': df_temp['E_loss_total'].sum() / 1000,
            
            # Performance metrics
            'average_pr': df_temp[df_temp['PR'] > 0]['PR'].mean(),
            'average_cell_temp': df_temp['cell_temperature'].mean(),
            'hours_of_production': len(df_temp[df_temp['E_ac'] > 0]),
            
            # System parameters (to verify consistency)
            'number_of_panels': number_of_panels,
            'inverter_efficiency': inverter_params['eta_inv_nom'],
            'panel_area_m2': 1.642 * number_of_panels,  # From Sharp ND-R240A5
        }
        
        detailed_results[name] = results
        
        # Log key findings
        logging.info(f"POA Irradiance: {results['total_poa_irradiance_kwh_m2']:,.0f} kWh/m²")
        logging.info(f"Final AC Production: {results['ac_final_energy_kwh']:,.0f} kWh")
        logging.info(f"Total System Losses: {results['total_losses_kwh']:,.0f} kWh")
        logging.info(f"Average PR: {results['average_pr']:.4f}")
        logging.info(f"Hours of Production: {results['hours_of_production']:,}")
    
    # Compare the two configurations
    logging.info(f"\n=== COMPARISON ANALYSIS ===")
    
    baseline = detailed_results['Baseline']
    optimal = detailed_results['Optimal']
    
    # Calculate percentage differences
    def pct_diff(optimal_val, baseline_val):
        return ((optimal_val - baseline_val) / baseline_val) * 100 if baseline_val != 0 else 0
    
    # Irradiance comparison
    poa_improvement = pct_diff(optimal['total_poa_irradiance_kwh_m2'], baseline['total_poa_irradiance_kwh_m2'])
    production_improvement = pct_diff(optimal['ac_final_energy_kwh'], baseline['ac_final_energy_kwh'])
    
    logging.info(f"POA Irradiance Improvement: {poa_improvement:.2f}%")
    logging.info(f"Production Improvement: {production_improvement:.2f}%")
    logging.info(f"Ratio (Production/POA): {production_improvement/poa_improvement:.2f}" if poa_improvement != 0 else "N/A")
    
    # Loss comparison
    pre_temp_diff = pct_diff(optimal['pre_temp_losses_kwh'], baseline['pre_temp_losses_kwh'])
    temp_diff = pct_diff(optimal['temperature_losses_kwh'], baseline['temperature_losses_kwh'])
    inverter_diff = pct_diff(optimal['inverter_losses_kwh'], baseline['inverter_losses_kwh'])
    
    logging.info(f"Pre-Temperature Loss Difference: {pre_temp_diff:.2f}%")
    logging.info(f"Temperature Loss Difference: {temp_diff:.2f}%")
    logging.info(f"Inverter Loss Difference: {inverter_diff:.2f}%")
    
    # Performance comparison
    pr_diff = pct_diff(optimal['average_pr'], baseline['average_pr'])
    temp_diff_avg = optimal['average_cell_temp'] - baseline['average_cell_temp']
    
    logging.info(f"Average PR Difference: {pr_diff:.2f}%")
    logging.info(f"Average Cell Temperature Difference: {temp_diff_avg:.2f}°C")
    
    # Validation checks
    validation_issues = []
    
    # Check 1: Production improvement should roughly match POA improvement
    if abs(production_improvement - poa_improvement) > 5:
        validation_issues.append(f"Production improvement ({production_improvement:.1f}%) doesn't match POA improvement ({poa_improvement:.1f}%)")
    
    # Check 2: System parameters should be identical
    if baseline['number_of_panels'] != optimal['number_of_panels']:
        validation_issues.append("Number of panels differs between configurations")
    
    if baseline['inverter_efficiency'] != optimal['inverter_efficiency']:
        validation_issues.append("Inverter efficiency differs between configurations")
    
    # Check 3: Extreme production gains - FIXED: Removed emoji
    if production_improvement > 20:
        validation_issues.append(f"Production improvement of {production_improvement:.1f}% is very high for angle optimization")
    
    # Check 4: Loss model consistency - compare loss RATIOS instead of absolute values
    baseline_pre_temp_ratio = baseline['pre_temp_losses_kwh'] / baseline['incident_energy_total_kwh'] * 100
    optimal_pre_temp_ratio = optimal['pre_temp_losses_kwh'] / optimal['incident_energy_total_kwh'] * 100
    pre_temp_ratio_diff = abs(optimal_pre_temp_ratio - baseline_pre_temp_ratio)

    if pre_temp_ratio_diff > 0.5:  # Loss percentage should be nearly identical
        validation_issues.append(f"Pre-temperature loss ratios differ by {pre_temp_ratio_diff:.2f}% - should be nearly identical")
        validation_issues.append(f"  Baseline: {baseline_pre_temp_ratio:.2f}%, Optimal: {optimal_pre_temp_ratio:.2f}%")
    else:
        logging.info(f"Pre-temperature loss ratios consistent: Baseline {baseline_pre_temp_ratio:.2f}%, Optimal {optimal_pre_temp_ratio:.2f}%")
    
    # Report validation results - FIXED: Removed emoji
    if validation_issues:
        logging.warning("WARNING: VALIDATION ISSUES FOUND:")
        for issue in validation_issues:
            logging.warning(f"  - {issue}")
    else:
        logging.info("SUCCESS: Production gain validation passed")
    
    # Create detailed comparison plots
    create_production_validation_plots(detailed_results, output_dir)
    
    # Save detailed results to CSV
    comparison_df = pd.DataFrame(detailed_results).T
    comparison_df.to_csv(os.path.join(output_dir, 'production_gain_validation.csv'))
    
    return detailed_results, validation_issues

def create_production_validation_plots(detailed_results, output_dir):
    """
    Create detailed plots to visualize production gain validation.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    baseline = detailed_results['Baseline']
    optimal = detailed_results['Optimal']
    
    # Plot 1: Irradiance comparison
    irradiance_types = ['Horizontal', 'POA', 'DNI', 'DHI']
    baseline_irr = [
        baseline['total_horizontal_irradiance_kwh_m2'],
        baseline['total_poa_irradiance_kwh_m2'],
        baseline['dni_total_kwh_m2'],
        baseline['dhi_total_kwh_m2']
    ]
    optimal_irr = [
        optimal['total_horizontal_irradiance_kwh_m2'],
        optimal['total_poa_irradiance_kwh_m2'],
        optimal['dni_total_kwh_m2'],
        optimal['dhi_total_kwh_m2']
    ]
    
    x = np.arange(len(irradiance_types))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_irr, width, label='Baseline', alpha=0.8)
    ax1.bar(x + width/2, optimal_irr, width, label='Optimal', alpha=0.8)
    ax1.set_xlabel('Irradiance Type')
    ax1.set_ylabel('Annual Irradiance (kWh/m²)')
    ax1.set_title('Irradiance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(irradiance_types)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add percentage differences
    for i, (b, o) in enumerate(zip(baseline_irr, optimal_irr)):
        pct_diff = ((o - b) / b) * 100 if b != 0 else 0
        ax1.text(i, max(b, o) + 50, f'{pct_diff:+.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Energy flow waterfall
    energy_stages = ['Incident', 'DC Ideal', 'DC Actual', 'AC Final']
    baseline_energy = [
        baseline['incident_energy_total_kwh'],
        baseline['dc_ideal_energy_kwh'],
        baseline['dc_actual_energy_kwh'],
        baseline['ac_final_energy_kwh']
    ]
    optimal_energy = [
        optimal['incident_energy_total_kwh'],
        optimal['dc_ideal_energy_kwh'],
        optimal['dc_actual_energy_kwh'],
        optimal['ac_final_energy_kwh']
    ]
    
    ax2.plot(energy_stages, baseline_energy, 'o-', label='Baseline', linewidth=2, markersize=8)
    ax2.plot(energy_stages, optimal_energy, 's-', label='Optimal', linewidth=2, markersize=8)
    ax2.set_ylabel('Energy (kWh)')
    ax2.set_title('Energy Flow Through System')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: Loss breakdown
    loss_types = ['Pre-Temp', 'Temperature', 'Inverter', 'Total']
    baseline_losses = [
        baseline['pre_temp_losses_kwh'],
        baseline['temperature_losses_kwh'],
        baseline['inverter_losses_kwh'],
        baseline['total_losses_kwh']
    ]
    optimal_losses = [
        optimal['pre_temp_losses_kwh'],
        optimal['temperature_losses_kwh'],
        optimal['inverter_losses_kwh'],
        optimal['total_losses_kwh']
    ]
    
    x = np.arange(len(loss_types))
    ax3.bar(x - width/2, baseline_losses, width, label='Baseline', alpha=0.8, color='red')
    ax3.bar(x + width/2, optimal_losses, width, label='Optimal', alpha=0.8, color='orange')
    ax3.set_xlabel('Loss Type')
    ax3.set_ylabel('Energy Lost (kWh)')
    ax3.set_title('System Loss Breakdown')
    ax3.set_xticks(x)
    ax3.set_xticklabels(loss_types)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics
    metrics = ['PR', 'Cell Temp (°C)', 'Production Hours']
    baseline_metrics = [
        baseline['average_pr'] * 100,  # Convert to percentage
        baseline['average_cell_temp'],
        baseline['hours_of_production']
    ]
    optimal_metrics = [
        optimal['average_pr'] * 100,
        optimal['average_cell_temp'],
        optimal['hours_of_production']
    ]
    
    # Normalize for comparison (different scales)
    baseline_norm = [
        baseline_metrics[0],  # PR already in %
        baseline_metrics[1],  # Temperature in °C
        baseline_metrics[2] / 100  # Hours in hundreds
    ]
    optimal_norm = [
        optimal_metrics[0],
        optimal_metrics[1],
        optimal_metrics[2] / 100
    ]
    
    x = np.arange(len(metrics))
    ax4.bar(x - width/2, baseline_norm, width, label='Baseline', alpha=0.8)
    ax4.bar(x + width/2, optimal_norm, width, label='Optimal', alpha=0.8)
    ax4.set_xlabel('Performance Metric')
    ax4.set_ylabel('Value (normalized)')
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'production_gain_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Production gain validation plots saved")

def analyze_azimuth_preference(df_subset, dni_extra, number_of_panels, inverter_params, optimal_tilt, output_dir):
    """
    Analyze why the optimization prefers the SE azimuth over due south.
    """
    logging.info("=== ANALYZING AZIMUTH PREFERENCE ===")
    
    # Test multiple azimuth angles with the optimal tilt
    azimuth_range = np.arange(90, 271, 15)  # 90° to 270° in 15° steps
    azimuth_results = []
    
    for azimuth in azimuth_range:
        # Calculate production for this azimuth
        df_temp = df_subset.copy()
        df_temp = calculate_total_irradiance(df_temp, optimal_tilt, azimuth, dni_extra)
        df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)
        
        # Calculate mismatch
        df_temp['weighting_factor'] = calculate_weighting_factors(df_temp)
        df_temp['load_wh'] = df_temp['Load (kW)'] * 1000
        df_temp['hourly_mismatch'] = df_temp['E_ac'] - df_temp['load_wh']
        df_temp['weighted_mismatch'] = df_temp['weighting_factor'] * np.abs(df_temp['hourly_mismatch'] / 1000)
        
        # Calculate timing analysis
        df_temp['hour'] = df_temp.index.hour
        morning_prod = df_temp[(df_temp['hour'] >= 6) & (df_temp['hour'] < 12)]['E_ac'].sum() / 1000
        afternoon_prod = df_temp[(df_temp['hour'] >= 12) & (df_temp['hour'] < 18)]['E_ac'].sum() / 1000
        
        azimuth_results.append({
            'azimuth': azimuth,
            'total_production_kwh': df_temp['E_ac'].sum() / 1000,
            'weighted_mismatch_kwh': df_temp['weighted_mismatch'].sum(),
            'morning_production_kwh': morning_prod,
            'afternoon_production_kwh': afternoon_prod,
            'morning_afternoon_ratio': morning_prod / afternoon_prod if afternoon_prod > 0 else 0,
            'total_poa_irradiance': df_temp['total_irradiance'].sum() / 1000
        })
    
    azimuth_df = pd.DataFrame(azimuth_results)
    
    # Find optimal azimuth for different criteria
    max_production_az = azimuth_df.loc[azimuth_df['total_production_kwh'].idxmax(), 'azimuth']
    min_mismatch_az = azimuth_df.loc[azimuth_df['weighted_mismatch_kwh'].idxmin(), 'azimuth']
    south_facing = 180
    
    logging.info(f"Azimuth for maximum production: {max_production_az:.0f}°")
    logging.info(f"Azimuth for minimum mismatch: {min_mismatch_az:.0f}°")
    logging.info(f"Due south production: {azimuth_df.loc[azimuth_df['azimuth'] == south_facing, 'total_production_kwh'].values[0]:,.0f} kWh")
    logging.info(f"Optimal azimuth production: {azimuth_df.loc[azimuth_df['azimuth'] == max_production_az, 'total_production_kwh'].values[0]:,.0f} kWh")
    
    # Save results
    azimuth_df.to_csv(os.path.join(output_dir, 'azimuth_analysis.csv'), index=False)
    
    # Create azimuth analysis plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Production vs azimuth
    ax1.plot(azimuth_df['azimuth'], azimuth_df['total_production_kwh'], 'b-', linewidth=2)
    ax1.axvline(x=180, color='gray', linestyle='--', alpha=0.7, label='Due South')
    ax1.axvline(x=max_production_az, color='red', linestyle='--', alpha=0.7, label=f'Max Production ({max_production_az:.0f}°)')
    ax1.set_xlabel('Azimuth Angle (°)')
    ax1.set_ylabel('Total Production (kWh)')
    ax1.set_title('Production vs Azimuth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mismatch vs azimuth
    ax2.plot(azimuth_df['azimuth'], azimuth_df['weighted_mismatch_kwh'], 'r-', linewidth=2)
    ax2.axvline(x=180, color='gray', linestyle='--', alpha=0.7, label='Due South')
    ax2.axvline(x=min_mismatch_az, color='blue', linestyle='--', alpha=0.7, label=f'Min Mismatch ({min_mismatch_az:.0f}°)')
    ax2.set_xlabel('Azimuth Angle (°)')
    ax2.set_ylabel('Weighted Mismatch (kWh)')
    ax2.set_title('Weighted Mismatch vs Azimuth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Morning vs afternoon production
    ax3.plot(azimuth_df['azimuth'], azimuth_df['morning_production_kwh'], 'g-', linewidth=2, label='Morning (6-12h)')
    ax3.plot(azimuth_df['azimuth'], azimuth_df['afternoon_production_kwh'], 'orange', linewidth=2, label='Afternoon (12-18h)')
    ax3.axvline(x=180, color='gray', linestyle='--', alpha=0.7, label='Due South')
    ax3.set_xlabel('Azimuth Angle (°)')
    ax3.set_ylabel('Production (kWh)')
    ax3.set_title('Morning vs Afternoon Production')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Combined objective (normalized)
    # Normalize both objectives to 0-1 scale for visualization
    prod_range = azimuth_df['total_production_kwh'].max() - azimuth_df['total_production_kwh'].min()
    mismatch_range = azimuth_df['weighted_mismatch_kwh'].max() - azimuth_df['weighted_mismatch_kwh'].min()
    
    if prod_range > 0:
        prod_norm = (azimuth_df['total_production_kwh'] - azimuth_df['total_production_kwh'].min()) / prod_range
    else:
        prod_norm = pd.Series(0.5, index=azimuth_df.index)
        
    if mismatch_range > 0:
        mismatch_norm = 1 - ((azimuth_df['weighted_mismatch_kwh'] - azimuth_df['weighted_mismatch_kwh'].min()) / mismatch_range)
    else:
        mismatch_norm = pd.Series(0.5, index=azimuth_df.index)
    
    ax4.plot(azimuth_df['azimuth'], prod_norm, 'b-', linewidth=2, label='Production (normalized)')
    ax4.plot(azimuth_df['azimuth'], mismatch_norm, 'r-', linewidth=2, label='Mismatch Quality (normalized)')
    ax4.axvline(x=180, color='gray', linestyle='--', alpha=0.7, label='Due South')
    ax4.set_xlabel('Azimuth Angle (°)')
    ax4.set_ylabel('Normalized Score (0-1)')
    ax4.set_title('Multi-Objective Trade-off')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'azimuth_preference_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return azimuth_df

def diagnose_irradiance_calculation(df_subset, dni_extra, number_of_panels, inverter_params, 
                                   baseline_angles, optimal_angles, output_dir):
    """
    Deep diagnostic of irradiance calculation to find the root cause of extreme gains.
    """
    logging.info("=== ROOT CAUSE DIAGNOSTIC ===")
    
    configs = [
        {"name": "Baseline", "tilt": baseline_angles[0], "azimuth": baseline_angles[1]},
        {"name": "Optimal", "tilt": optimal_angles[0], "azimuth": optimal_angles[1]}
    ]
    
    diagnostic_results = {}
    
    for config in configs:
        name = config["name"]
        tilt = config["tilt"]
        azimuth = config["azimuth"]
        
        logging.info(f"\n--- DIAGNOSING {name} ({tilt:.1f}°, {azimuth:.1f}°) ---")
        
        # Start with fresh copy
        df_temp = df_subset.copy()
        
        # Step 1: Check input irradiance data
        ghi_total = df_temp['SolRad_Hor'].sum()
        dhi_total = df_temp['SolRad_Dif'].sum()
        dni_total = df_temp['DNI'].sum()
        
        logging.info(f"Input - GHI Total: {ghi_total:,.0f} Wh/m²")
        logging.info(f"Input - DHI Total: {dhi_total:,.0f} Wh/m²")
        logging.info(f"Input - DNI Total: {dni_total:,.0f} Wh/m²")
        
        # Step 2: Calculate POA irradiance components
        df_temp = calculate_total_irradiance(df_temp, tilt, azimuth, dni_extra)
        
        # Step 3: Check irradiance calculation using pvlib directly for verification
        # This will help us understand if the issue is in our irradiance calculation
        solar_position = pvlib.solarposition.get_solarposition(df_temp.index, 37.98983, 23.74328)
        
        # Calculate irradiance components separately for diagnosis
        irradiance_poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solar_position['apparent_zenith'],
            solar_azimuth=solar_position['azimuth'],
            dni=df_temp['DNI'],
            ghi=df_temp['SolRad_Hor'],
            dhi=df_temp['SolRad_Dif'],
            dni_extra=dni_extra,
            model='haydavies'
        )
        
        # Compare our calculation with direct pvlib calculation
        our_poa_total = df_temp['total_irradiance'].sum()
        pvlib_poa_total = irradiance_poa['poa_global'].sum()
        
        logging.info(f"Our POA Total: {our_poa_total:,.0f} Wh/m²")
        logging.info(f"PVLib POA Total: {pvlib_poa_total:,.0f} Wh/m²")
        logging.info(f"POA Calculation Difference: {abs(our_poa_total - pvlib_poa_total):,.0f} Wh/m²")
        
        # Check individual POA components
        beam_poa = irradiance_poa['poa_direct'].sum()
        diffuse_poa = irradiance_poa['poa_diffuse'].sum()
        reflected_poa = irradiance_poa['poa_ground_diffuse'].sum()
        
        logging.info(f"POA Components - Direct: {beam_poa:,.0f}, Diffuse: {diffuse_poa:,.0f}, Reflected: {reflected_poa:,.0f} Wh/m²")
        
        # Step 4: Check if the issue is in energy conversion
        # Calculate theoretical energy at each step manually
        panel_area = 1.642  # m² per panel
        total_panel_area = panel_area * number_of_panels
        panel_efficiency_stc = 0.146  # 14.6%
        
        theoretical_energy_incident = our_poa_total * total_panel_area  # Wh
        theoretical_energy_dc_stc = theoretical_energy_incident * panel_efficiency_stc  # Wh at STC
        
        logging.info(f"Theoretical - Incident Energy: {theoretical_energy_incident/1000:,.0f} kWh")
        logging.info(f"Theoretical - DC Energy (STC): {theoretical_energy_dc_stc/1000:,.0f} kWh")
        
        # Now run through our energy calculation
        df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)
        
        actual_incident = df_temp['E_incident'].sum()
        actual_dc_ideal = df_temp['E_dc_ideal'].sum()
        actual_dc_actual = df_temp['E_dc_actual'].sum()
        actual_ac = df_temp['E_ac'].sum()
        
        logging.info(f"Actual - Incident Energy: {actual_incident/1000:,.0f} kWh")
        logging.info(f"Actual - DC Ideal: {actual_dc_ideal/1000:,.0f} kWh")
        logging.info(f"Actual - DC Actual: {actual_dc_actual/1000:,.0f} kWh")
        logging.info(f"Actual - AC Final: {actual_ac/1000:,.0f} kWh")
        
        # Check consistency
        incident_diff = abs(theoretical_energy_incident - actual_incident) / theoretical_energy_incident * 100
        dc_diff = abs(theoretical_energy_dc_stc - actual_dc_ideal) / theoretical_energy_dc_stc * 100
        
        logging.info(f"Incident Energy Difference: {incident_diff:.2f}%")
        logging.info(f"DC Ideal Difference: {dc_diff:.2f}%")
        
        # Store results
        diagnostic_results[name] = {
            'config': config,
            'ghi_total': ghi_total,
            'dni_total': dni_total,
            'dhi_total': dhi_total,
            'our_poa_total': our_poa_total,
            'pvlib_poa_total': pvlib_poa_total,
            'beam_poa': beam_poa,
            'diffuse_poa': diffuse_poa,
            'reflected_poa': reflected_poa,
            'theoretical_incident': theoretical_energy_incident,
            'theoretical_dc_stc': theoretical_energy_dc_stc,
            'actual_incident': actual_incident,
            'actual_dc_ideal': actual_dc_ideal,
            'actual_ac': actual_ac,
            'incident_diff_pct': incident_diff,
            'dc_diff_pct': dc_diff
        }
    
    # Compare the two configurations
    logging.info(f"\n=== COMPARATIVE ANALYSIS ===")
    
    baseline = diagnostic_results['Baseline']
    optimal = diagnostic_results['Optimal']
    
    # Check if the issue is in POA calculation
    poa_ratio = optimal['our_poa_total'] / baseline['our_poa_total']
    energy_ratio = optimal['actual_ac'] / baseline['actual_ac']
    
    logging.info(f"POA Irradiance Ratio (Optimal/Baseline): {poa_ratio:.3f}")
    logging.info(f"AC Energy Ratio (Optimal/Baseline): {energy_ratio:.3f}")
    
    # Check individual irradiance components
    beam_ratio = optimal['beam_poa'] / baseline['beam_poa'] if baseline['beam_poa'] > 0 else 0
    diffuse_ratio = optimal['diffuse_poa'] / baseline['diffuse_poa'] if baseline['diffuse_poa'] > 0 else 0
    reflected_ratio = optimal['reflected_poa'] / baseline['reflected_poa'] if baseline['reflected_poa'] > 0 else 0
    
    logging.info(f"Beam Component Ratio: {beam_ratio:.3f}")
    logging.info(f"Diffuse Component Ratio: {diffuse_ratio:.3f}")
    logging.info(f"Reflected Component Ratio: {reflected_ratio:.3f}")
    
    # Identify the primary driver
    max_component_ratio = max(beam_ratio, diffuse_ratio, reflected_ratio)
    if max_component_ratio == beam_ratio:
        primary_driver = "DIRECT BEAM"
    elif max_component_ratio == diffuse_ratio:
        primary_driver = "DIFFUSE"
    else:
        primary_driver = "REFLECTED"
    
    logging.info(f"PRIMARY DRIVER of POA gain: {primary_driver} (ratio: {max_component_ratio:.3f})")
    
    # Check for calculation errors
    issues_found = []
    
    if poa_ratio > 1.5:  # 50% increase is very high
        issues_found.append(f"POA irradiance increase of {(poa_ratio-1)*100:.1f}% is extremely high")
    
    if beam_ratio > 2.0:  # 100% increase in beam is suspicious
        issues_found.append(f"Direct beam component increased by {(beam_ratio-1)*100:.1f}% - check angle calculations")
    
    if abs(baseline['incident_diff_pct']) > 1 or abs(optimal['incident_diff_pct']) > 1:
        issues_found.append("Incident energy calculation inconsistency detected")
    
    if issues_found:
        logging.error("DIAGNOSTIC ISSUES FOUND:")
        for issue in issues_found:
            logging.error(f"  - {issue}")
    else:
        logging.info("Diagnostic found no calculation errors - high gains may be legitimate")
    
    # Save diagnostic results
    diagnostic_df = pd.DataFrame(diagnostic_results).T
    diagnostic_df.to_csv(os.path.join(output_dir, 'irradiance_diagnostic.csv'))
    
    return diagnostic_results, issues_found

def run_comprehensive_validation(df_subset, dni_extra, number_of_panels, inverter_params, 
                                baseline_angles, optimal_angles, args, output_dir):
    """
    Run comprehensive validation with root cause analysis.
    """
    logging.info("=== COMPREHENSIVE VALIDATION STARTING ===")
    
    # 1. Root cause diagnostic
    diagnostic_results, diagnostic_issues = diagnose_irradiance_calculation(
        df_subset, dni_extra, number_of_panels, inverter_params,
        baseline_angles, optimal_angles, output_dir
    )
    
    # 2. Standard production validation
    production_details, production_issues = validate_production_gain(
        df_subset, dni_extra, number_of_panels, inverter_params,
        baseline_angles, optimal_angles, output_dir
    )
    
    # 3. Mismatch validation
    validation_results, validation_passed = validate_mismatch_calculation(
        df_subset, dni_extra, number_of_panels, inverter_params,
        baseline_angles, optimal_angles, output_dir
    )
    
    # Compile all issues
    all_issues = diagnostic_issues + production_issues
    if not validation_passed:
        all_issues.append("Mismatch calculation validation failed")
    
    return diagnostic_results, production_details, validation_results, all_issues

def debug_pr_calculation_detailed(df, context_name=""):
    """
    CRITICAL: Debug PR calculation in detail to find the 0.0% issue.
    """
    logging.info(f"=== DETAILED PR DEBUG: {context_name} ===")
    
    # Check if required columns exist
    required_cols = ['incident_irradiance', 'E_ac', 'PR']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns for PR calculation: {missing_cols}")
        return
    
    # Extract PR calculation components
    incident_irradiance = df['incident_irradiance']  # W/m²
    e_ac = df['E_ac']  # Wh
    pr_values = df['PR']
    
    # Calculate reference yield manually
    reference_yield = incident_irradiance * TIME_INTERVAL_HOURS / 1000  # hours (equivalent sun hours)
    
    # System parameters (from Sharp ND-R240A5)
    panel_power_stc = 240  # W per panel
    total_system_power_stc = panel_power_stc * len(df)  # This might be wrong - need actual number_of_panels
    
    # Log detailed statistics
    logging.info(f"Data points: {len(df)}")
    logging.info(f"Incident irradiance - Min: {incident_irradiance.min():.1f}, Max: {incident_irradiance.max():.1f}, Mean: {incident_irradiance.mean():.1f} W/m²")
    logging.info(f"E_ac - Min: {e_ac.min():.1f}, Max: {e_ac.max():.1f}, Sum: {e_ac.sum():.0f} Wh")
    logging.info(f"Reference yield - Min: {reference_yield.min():.3f}, Max: {reference_yield.max():.3f}, Sum: {reference_yield.sum():.1f} hours")
    logging.info(f"PR values - Min: {pr_values.min():.3f}, Max: {pr_values.max():.3f}, Mean: {pr_values.mean():.3f}")
    
    # Check for zero/negative values that could cause issues
    zero_incident = (incident_irradiance == 0).sum()
    zero_eac = (e_ac == 0).sum()
    zero_ref_yield = (reference_yield == 0).sum()
    zero_pr = (pr_values == 0).sum()
    
    logging.info(f"Zero values - Incident: {zero_incident}, E_ac: {zero_eac}, Ref_yield: {zero_ref_yield}, PR: {zero_pr}")
    
    # Manual PR calculation for verification
    valid_hours = (reference_yield > 0) & (e_ac >= 0)
    if valid_hours.any():
        manual_pr = e_ac[valid_hours] / (reference_yield[valid_hours] * total_system_power_stc)
        manual_pr = manual_pr.clip(0, 1)  # Limit to reasonable range
        
        logging.info(f"Manual PR calculation - Min: {manual_pr.min():.3f}, Max: {manual_pr.max():.3f}, Mean: {manual_pr.mean():.3f}")
        logging.info(f"Total system power used in calculation: {total_system_power_stc} W")
        
        # Check if our calculation matches the stored PR
        pr_diff = abs(manual_pr.mean() - pr_values[valid_hours].mean())
        logging.info(f"Difference between manual and stored PR: {pr_diff:.6f}")
        
        if pr_diff > 0.001:
            logging.error("CRITICAL: Manual PR calculation doesn't match stored values!")
            # Sample comparison
            sample_idx = valid_hours.idxmax()  # First valid index
            logging.info(f"Sample calculation at index {sample_idx}:")
            logging.info(f"  E_ac: {e_ac[sample_idx]:.1f} Wh")
            logging.info(f"  Reference yield: {reference_yield[sample_idx]:.3f} hours")
            logging.info(f"  Expected PR: {e_ac[sample_idx] / (reference_yield[sample_idx] * total_system_power_stc):.3f}")
            logging.info(f"  Stored PR: {pr_values[sample_idx]:.3f}")
    else:
        logging.error("No valid hours found for PR calculation!")
    
    return {
        'context': context_name,
        'total_incident_hours': reference_yield.sum(),
        'total_energy_wh': e_ac.sum(),
        'mean_pr': pr_values.mean(),
        'system_power_w': total_system_power_stc,
        'valid_hours_count': valid_hours.sum()
    }


def setup_deap_optimization():
    """Proper DEAP setup with constraint handling"""
    
    # Create fitness and individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))  # minimize mismatch, maximize production
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Bounded individual creation
    def create_bounded_individual():
        """Create individual within valid bounds"""
        tilt = random.uniform(0, 90)      # Valid tilt range
        azimuth = random.uniform(90, 270) # Valid azimuth range  
        return creator.Individual([tilt, azimuth])
    
    # Register functions
    toolbox.register("individual", create_bounded_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", create_robust_objective_function())
    
    # Use bounded operators to maintain constraints
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                     eta=20.0, low=[0, 90], up=[90, 270])
    toolbox.register("mutate", tools.mutPolynomialBounded, 
                     eta=20.0, low=[0, 90], up=[90, 270], indpb=0.1)
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
    
    toolbox.decorate("mate", checkBounds([0, 90], [90, 270]))
    toolbox.decorate("mutate", checkBounds([0, 90], [90, 270]))
    
    return toolbox

def run_optimization_with_debugging():
    """Run optimization with comprehensive debugging"""
    
    toolbox = setup_deap_optimization()
    
    # Create initial population
    population = toolbox.population(n=100)
    
    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "min", "max"
    
    # Evaluate initial population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # Debug initial population
    debug_population_health(population, 0)
    
    # Evolution loop with debugging
    for gen in range(1000):
        # Selection and variation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:  # crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.1:  # mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update population
        population = offspring
        
        # Record statistics and debug
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        
        if gen % 50 == 0:  # Debug every 50 generations
            debug_population_health(population, gen)
            print(f"Generation {gen}: Best fitness = {record['min']}")
    
    return population, logbook

def debug_population_health(population, generation):
    """Debug population for infinite values and convergence issues"""
    infinite_count = 0
    valid_fitness_count = 0
    
    for ind in population:
        if ind.fitness.valid:
            valid_fitness_count += 1
            if any(not np.isfinite(val) for val in ind.fitness.values):
                infinite_count += 1
    
    print(f"Generation {generation} health check:")
    print(f"  Valid fitness individuals: {valid_fitness_count}/{len(population)}")
    print(f"  Infinite fitness individuals: {infinite_count}")
    
    if infinite_count > 0:
        print(f"  WARNING: {infinite_count} individuals have infinite fitness")
    
    if valid_fitness_count > 0:
        best_individual = min(population, key=lambda x: x.fitness.values[0] if x.fitness.valid else float('inf'))
        print(f"  Best individual: tilt={best_individual[0]:.2f}°, azimuth={best_individual[1]:.2f}°")
        print(f"  Best fitness: {best_individual.fitness.values}")



def validate_solar_system_params(system_params):
    """Validate solar system parameters"""
    required_params = ['panel_efficiency', 'panel_area', 'system_losses']
    missing_params = [param for param in required_params if param not in system_params]
    
    if missing_params:
        raise ValueError(f"Missing required system parameters: {missing_params}")
    
    # Validate parameter ranges
    if not (0.1 <= system_params['panel_efficiency'] <= 0.3):
        raise ValueError(f"Panel efficiency {system_params['panel_efficiency']} outside valid range (0.1-0.3)")
    
    if not (0.05 <= system_params['system_losses'] <= 0.4):
        raise ValueError(f"System losses {system_params['system_losses']} outside valid range (0.05-0.4)")
    
    return True

def robust_calculation_wrapper(func):
    """Decorator for robust numerical calculations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # Validate results
            if isinstance(result, tuple):
                for i, val in enumerate(result):
                    if not np.isfinite(val):
                        logging.error(f"Function {func.__name__} returned non-finite value: {val} at position {i}")
                        return (1e6, 0.0)  # Return penalty values
            else:
                if not np.isfinite(result):
                    logging.error(f"Function {func.__name__} returned non-finite value: {result}")
                    return 1e6
            
            return result
            
        except ZeroDivisionError as e:
            logging.error(f"Zero division in {func.__name__}: {e}")
            return (1e6, 0.0) if isinstance(result, tuple) else 1e6
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            return (1e6, 0.0) if isinstance(result, tuple) else 1e6
    
    return wrapper

class TestSolarPVOptimization:
    
    def test_pr_calculation_order(self):
        """Test that PR is calculated before validation"""
        test_data = pd.DataFrame({
            'irradiance': [800, 900, 1000],
            'temperature': [20, 25, 30],
            'panel_area': [10, 10, 10]
        })
        
        system_params = {
            'panel_efficiency': 0.2,
            'system_losses': 0.2,
            'temp_coefficient': -0.004
        }
        
        # Should not raise KeyError
        total_prod, avg_pr = calculate_energy_production(test_data, system_params)
        
        assert np.isfinite(total_prod), "Total production should be finite"
        assert 0 <= avg_pr <= 1.2, f"Average PR {avg_pr} outside expected range"
    
    def test_bounds_handling(self):
        """Test genetic algorithm bounds handling"""
        toolbox = setup_deap_optimization()
        
        # Test extreme values
        extreme_individual = creator.Individual([-10, 350])  # Outside bounds
        
        # Should not return infinite values
        fitness = toolbox.evaluate(extreme_individual)
        assert np.isfinite(fitness[0]), "Mismatch should be finite"
        assert np.isfinite(fitness[1]), "Production should be finite"
    
    def test_optimization_convergence(self):
        """Test that optimization can find feasible solutions"""
        population, logbook = run_optimization_with_debugging()
        
        # Check that best individual improved over generations
        initial_fitness = logbook[0]['min'][0]
        final_fitness = logbook[-1]['min'][0]
        
        assert final_fitness < initial_fitness, "Algorithm should improve over generations"
        assert np.isfinite(final_fitness), "Final fitness should be finite"

def main():
    try:
        # -------------------- Step 1: Parse Command-Line Arguments --------------------
        parser = argparse.ArgumentParser(description='Solar Energy Analysis Tool')
        parser.add_argument('--data_file', type=str, required=True, help='Path to the input CSV data file')
        parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output results and plots')
        parser.add_argument('--config_file', type=str, required=True, help='Path to the YAML configuration file')
        parser.add_argument('--latitude', type=float, default=37.98983, help='Latitude of the location (default: 37.98983)')
        parser.add_argument('--longitude', type=float, default=23.74328, help='Longitude of the location (default: 23.74328)')
        parser.add_argument('--representative_date', type=str, default='2023-06-15',
                            help='Date for representative day profiles in YYYY-MM-DD format (default: 2023-06-15)')
        parser.add_argument('--tilt_bounds', type=float, nargs=2, default=[0, 90],
                            help='Bounds for tilt angle as two floats: min max (default: 0 90)')
        parser.add_argument('--azimuth_bounds', type=float, nargs=2, default=[90, 270],
                            help='Bounds for azimuth angle as two floats: min max (default: 90 270)')
        parser.add_argument('--maxiter', type=int, default=1000, help='Maximum number of iterations for optimization (default: 1000)')
        parser.add_argument('--popsize', type=int, default=15, help='Population size for optimization (default: 15)')
        parser.add_argument('--locale', type=str, default='en_US.UTF-8', help='Locale setting for number formatting (default: en_US.UTF-8)')

        args = parser.parse_args()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # -------------------- Step 2: Set Up Logging and Locale --------------------
        setup_logging(args.output_dir)
        logging.info("Logging is set up.")

        # Load configuration
        config = load_config(args.config_file)

        # Define panel and inverter parameters
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

        try:
            locale.setlocale(locale.LC_ALL, args.locale)
            logging.info(f"Locale set to {args.locale}")
        except locale.Error as e:
            logging.error(f"Invalid locale '{args.locale}': {e}. Falling back to default locale.", exc_info=True)
            locale.setlocale(locale.LC_ALL, '')
        pd.options.display.float_format = lambda x: locale.format_string('%.2f', x, grouping=True)

        # -------------------- Step 3: Load and Preprocess Data --------------------
        df_original = load_and_preprocess_data(args.data_file)
        df_original = calculate_solar_position(df_original, args.latitude, args.longitude)
        df_original = calculate_dni(df_original)
        dni_extra = pvlib.irradiance.get_extra_radiation(df_original.index, method='nrel')

        # Check for required columns and create subset
        columns_needed = ['SolRad_Hor', 'SolRad_Dif', 'Air Temp', 'zenith', 'azimuth', 'DNI', 'Load (kW)']
        for col in columns_needed:
            if col not in df_original.columns:
                logging.error(f"'{col}' column is missing in the DataFrame.")
                raise ValueError(f"'{col}' column is missing in the DataFrame.")
        df_subset = df_original[columns_needed].copy()

        # -------------------- Step 4: Define Bounds and Panel Layout Parameters --------------------
        bounds = [args.tilt_bounds, args.azimuth_bounds]

        available_area = config.get('available_area', 1500)  # in m²
        panel_length = config['solar_panel']['length']
        panel_width = config['solar_panel']['width']
        spacing_length = config['solar_panel']['spacing_length']
        spacing_width = config['solar_panel']['spacing_width']
        panel_power_rating = config['solar_panel']['power_rating']
        area_per_panel = (panel_length + spacing_length) * (panel_width + spacing_width)
        number_of_panels = int(available_area // area_per_panel)
        logging.info(f"Maximum number of panels that can be installed: {number_of_panels}")

        panel_area = panel_length * panel_width  # in m²
        total_panel_area = panel_area * number_of_panels
        logging.info(f"Total panel area (excluding spacing): {total_panel_area:.2f} m²")

        STC_irradiance = 1000  # W/m²
        panel_efficiency = panel_power_rating / (panel_area * STC_irradiance)
        logging.info(f"Panel efficiency: {panel_efficiency * 100:.2f}%")

        inverter_params['pdc0'] = panel_params['Pmp'] * number_of_panels
        logging.info(f"Inverter pdc0 updated to: {inverter_params['pdc0']} W")

        # -------------------- Step 5: Multi-Objective + Constrained Optimization --------------------
        # Run standard multi-objective optimization
        pareto_front, filtered_front, best_balanced = run_deap_multi_objective_optimization(
            df_subset, dni_extra, number_of_panels, inverter_params, args.output_dir,
            pop_size=args.popsize, max_gen=args.maxiter
        )

        # Compare with constrained optimization
        logging.info("Running constrained optimization for comparison...")
        optimal_angles_constrained = run_constrained_optimization(
            df_subset, dni_extra, number_of_panels, inverter_params,
            args.output_dir, max_mismatch=120000
        )

        # Choose best approach based on results
        if best_balanced is not None:
            logging.info("Using DEAP multi-objective solution")
            optimal_tilt, optimal_azimuth = best_balanced[0], best_balanced[1]
            balanced_weighted_mismatch, balanced_production = best_balanced.fitness.values
        else:
            logging.info("Using constrained optimization solution as fallback")
            optimal_tilt, optimal_azimuth = optimal_angles_constrained
            # Calculate the metrics for the constrained solution
            df_temp = df_subset.copy()
            df_temp = calculate_total_irradiance(df_temp, optimal_tilt, optimal_azimuth, dni_extra)
            df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)
            
            df_temp['weighting_factor'] = calculate_weighting_factors(df_temp)
            df_temp['load_wh'] = df_temp['Load (kW)'] * 1000
            df_temp['hourly_mismatch'] = df_temp['E_ac'] - df_temp['load_wh']
            df_temp['weighted_mismatch'] = df_temp['weighting_factor'] * np.abs(df_temp['hourly_mismatch'] / 1000)
            
            balanced_weighted_mismatch = df_temp['weighted_mismatch'].sum()
            balanced_production = df_temp['E_ac'].sum() / 1000

        logging.info(f"Final Optimal Tilt Angle: {optimal_tilt:.2f}°")
        logging.info(f"Final Optimal Azimuth Angle: {optimal_azimuth:.2f}°")
        logging.info(f"Final Weighted Energy Mismatch: {balanced_weighted_mismatch:.2f} kWh")
        logging.info(f"Final Total Energy Production: {balanced_production:.2f} kWh")

        # -------------------- Step 6: Visualize and Save Pareto Front Results --------------------
        plot_pareto_front(pareto_front, filtered_front, best_balanced, args.output_dir)
        plot_detailed_pareto_front(pareto_front, args.output_dir)
        save_summary_results(pareto_front, args.output_dir)
        
        # Updated parameter list for the grid search
        grid_search_results_path = run_grid_search(
            df_subset, 
            dni_extra, 
            number_of_panels, 
            inverter_params, 
            args.output_dir
        )
        
        plot_pareto_front_comparison(pareto_front, grid_search_results_path, args.output_dir)
        
        
        plot_pareto_front_with_efficiency(pareto_front, args.output_dir)

        
        
        # -------------------- Step 7: Detailed Analysis with Balanced Optimal Angles --------------------
        if pareto_front:
            best_balanced = select_balanced_solution(pareto_front)
            if best_balanced is None:
                logging.error("No balanced solution found. Exiting.")
                sys.exit(1)

            # Initialize summary dictionary and summary_file path HERE
            summary = {}
            summary_file = os.path.join(args.output_dir, 'summary_results.csv')
            
            optimal_tilt = best_balanced[0]
            optimal_azimuth = best_balanced[1]
            balanced_weighted_mismatch, balanced_production = best_balanced.fitness.values

            logging.info(f"Selected Optimal Tilt Angle: {optimal_tilt:.2f}°")
            logging.info(f"Selected Optimal Azimuth Angle: {optimal_azimuth:.2f}°")
            logging.info(f"Balanced Weighted Energy Mismatch: {balanced_weighted_mismatch:.2f} kWh")
            logging.info(f"Balanced Total Energy Production: {balanced_production:.2f} kWh")

            # Proceed with calculations using optimal_tilt, optimal_azimuth
            df = df_subset.copy()
            df = calculate_total_irradiance(df, optimal_tilt, optimal_azimuth, dni_extra)
            df = calculate_energy_production(df, number_of_panels, inverter_params)
            df['weighting_factor'] = calculate_weighting_factors(df)
            df['E_ac_Wh'] = df['E_ac'] * TIME_INTERVAL_HOURS

            # Verify representative date is in the dataset
            representative_date = args.representative_date
            available_dates = df.index.normalize().unique().strftime('%Y-%m-%d')
            if representative_date not in available_dates:
                logging.error(f"The representative date {representative_date} is not present in the data.")
                raise ValueError(f"The representative date {representative_date} is not present in the data.")

            # -------------------- Step 7.1: COMPREHENSIVE ROOT CAUSE ANALYSIS --------------------
            logging.info("Performing comprehensive root cause analysis...")

            # Quick sanity check
            sanity_results = quick_sanity_check(df)

            # ROOT CAUSE DIAGNOSTIC
            diagnostic_results, production_details, validation_results, all_issues = run_comprehensive_validation(
                df_subset, dni_extra, number_of_panels, inverter_params,
                baseline_angles=[args.latitude, 180.0],
                optimal_angles=[optimal_tilt, optimal_azimuth],
                args=args,
                output_dir=args.output_dir
            )

            # REPORT FINDINGS
            if all_issues:
                logging.error("CRITICAL ISSUES FOUND:")
                for issue in all_issues:
                    logging.error(f"  ERROR: {issue}")
                logging.error("These issues indicate problems with the calculation methodology")
                
                # Log the diagnostic insights
                baseline_diag = diagnostic_results['Baseline']
                optimal_diag = diagnostic_results['Optimal']
                
                poa_ratio = optimal_diag['our_poa_total'] / baseline_diag['our_poa_total']
                beam_ratio = optimal_diag['beam_poa'] / baseline_diag['beam_poa']
                
                logging.info(f"DIAGNOSTIC INSIGHTS:")
                logging.info(f"  POA Irradiance Ratio: {poa_ratio:.3f} ({(poa_ratio-1)*100:.1f}% increase)")
                logging.info(f"  Direct Beam Ratio: {beam_ratio:.3f} ({(beam_ratio-1)*100:.1f}% increase)")
                
            else:
                logging.info("SUCCESS: All validation checks passed")

            # FIXED: Extract results for summary - properly define baseline_results
            baseline_results = {
                'tilt': args.latitude,
                'azimuth': 180.0,
                'production_kwh': production_details['Baseline']['ac_final_energy_kwh'],
                'weighted_mismatch': validation_results['Baseline']['weighted_mismatch_kwh'],
                'avg_pr': production_details['Baseline']['average_pr'],
                'total_irradiance': production_details['Baseline']['total_poa_irradiance_kwh_m2'] * 1000
            }

            # Use validated optimal results
            balanced_weighted_mismatch = validation_results['Optimal']['weighted_mismatch_kwh']
            balanced_production = production_details['Optimal']['ac_final_energy_kwh']

            # Calculate production_gain properly
            production_gain = ((balanced_production - baseline_results['production_kwh']) / 
                              baseline_results['production_kwh'] * 100)

            # Calculate improvement percentages
            production_improvement = production_gain
            mismatch_improvement = ((baseline_results['weighted_mismatch'] - balanced_weighted_mismatch) / 
                                   baseline_results['weighted_mismatch'] * 100)

            logging.info(f"\n=== FINAL VALIDATED RESULTS ===")
            logging.info(f"Baseline Production: {baseline_results['production_kwh']:,.2f} kWh")
            logging.info(f"Optimal Production: {balanced_production:,.2f} kWh")
            logging.info(f"Production Improvement: {production_improvement:.2f}%")
            logging.info(f"Baseline Weighted Mismatch: {baseline_results['weighted_mismatch']:,.2f} kWh")
            logging.info(f"Optimal Weighted Mismatch: {balanced_weighted_mismatch:,.2f} kWh")
            logging.info(f"Mismatch Improvement: {mismatch_improvement:.2f}%")

            # -------------------- NOW baseline_results is properly defined --------------------
            # Perform detailed configuration comparison
            logging.info("Performing detailed configuration comparison...")
            comparison_results = compare_configurations(
                df_subset, dni_extra, number_of_panels, inverter_params,
                [baseline_results['tilt'], baseline_results['azimuth']],  # Now this works!
                [optimal_tilt, optimal_azimuth],
                args.output_dir
            )

            # Validate the improvement
            baseline_prod = comparison_results[0]['total_production']
            optimal_prod = comparison_results[1]['total_production']
            actual_improvement = ((optimal_prod - baseline_prod) / baseline_prod) * 100

            if actual_improvement > 15:
                logging.warning(f"WARNING: Production improvement of {actual_improvement:.1f}% seems high. Checking calculations...")
                logging.info(f"Baseline irradiance on panel: {comparison_results[0]['total_irradiance_on_panel']/1e6:.2f} MWh/m²")
                logging.info(f"Optimal irradiance on panel: {comparison_results[1]['total_irradiance_on_panel']/1e6:.2f} MWh/m²")    

            # -------------------- Step 7.2: Neighborhood Analysis --------------------
            neighborhood_df = analyze_neighborhood_solutions(
                optimal_tilt, optimal_azimuth, df_subset, dni_extra, 
                number_of_panels, inverter_params, args.output_dir,
                tilt_range=15, azimuth_range=30, grid_points=7
            )

            
            # -------------------- Step 7.3: CRITICAL PR DEBUGGING --------------------
            logging.info("Performing critical PR debugging...")

            # Debug PR for the optimal configuration
            optimal_pr_debug = debug_pr_calculation_detailed(df, "OPTIMAL_MAIN")

            # Compare with any validation calculations
            if 'baseline_results' in locals():
                logging.info(f"PR Comparison:")
                logging.info(f"  Main optimal PR: {optimal_pr_debug['mean_pr']:.3f}")
                logging.info(f"  Baseline PR: {baseline_results.get('avg_pr', 'N/A'):.3f}")
                
                # Check for system power consistency
                if 'baseline_pr_debug' in locals():
                    power_consistent = optimal_pr_debug['system_power_w'] == baseline_pr_debug['system_power_w']
                    logging.info(f"  System power consistent: {power_consistent}")
                    if not power_consistent:
                        logging.error(f"CRITICAL: System power mismatch - Optimal: {optimal_pr_debug['system_power_w']} W, Baseline: {baseline_pr_debug['system_power_w']} W")
            
            # -------------------- Step 7.4: Calculate Total Annual Consumption --------------------
            total_annual_consumption = df_original['Load (kW)'].sum()
            logging.info(f"Total Annual Consumption: {total_annual_consumption:.2f} kWh")

            # Update summary with new metrics
            summary['Total Annual Consumption (kWh)'] = f"{total_annual_consumption:.2f}"
            summary['Baseline Production (kWh)'] = f"{baseline_results['production_kwh']:.2f}"
            summary['Baseline Configuration'] = f"Tilt: {baseline_results['tilt']:.1f}°, Azimuth: {baseline_results['azimuth']:.1f}°"
            summary['Production Improvement from Optimization (%)'] = f"{production_improvement:.2f}"
            summary['Mismatch Improvement from Optimization (%)'] = f"{mismatch_improvement:.2f}"
            summary['Balanced Weighted Energy Mismatch (kWh)'] = f"{balanced_weighted_mismatch:.2f}"
            summary['Total Energy Produced - Optimal Angles (kWh)'] = f"{balanced_production:.2f}"
            summary['Energy Coverage Ratio (%)'] = f"{(balanced_production / total_annual_consumption * 100):.2f}"

            
            
            # -------------------- Step 8: Seasonal Analysis --------------------
            logging.info("Performing seasonal analysis...")
            seasonal_stats, daily_seasonal = analyze_seasonal_performance(df)

            # Save seasonal statistics to CSV
            seasonal_stats.to_csv(os.path.join(args.output_dir, 'seasonal_statistics.csv'))
            daily_seasonal.to_csv(os.path.join(args.output_dir, 'daily_seasonal_averages.csv'))
            logging.info("Seasonal statistics saved to CSV files")

            # Create seasonal plots
            plot_seasonal_production_consumption(seasonal_stats, args.output_dir)
            plot_seasonal_self_consumption(seasonal_stats, args.output_dir)
            plot_seasonal_energy_balance(seasonal_stats, args.output_dir)
            plot_seasonal_hourly_profiles(df, args.output_dir)

            # Calculate and visualize seasonal battery requirements
            battery_requirements = calculate_seasonal_battery_needs(df, args.output_dir)
            logging.info("Seasonal analysis completed")

            # Add seasonal information to the summary
            for season in seasonal_stats.index:
                summary[f'Production {season} (kWh)'] = f"{seasonal_stats.loc[season, 'E_ac_kwh']:.2f}"
                summary[f'Consumption {season} (kWh)'] = f"{seasonal_stats.loc[season, 'load_wh_kwh']:.2f}"
                summary[f'Self-Consumption Ratio {season} (%)'] = f"{seasonal_stats.loc[season, 'self_consumption_ratio']:.2f}"
                summary[f'Self-Sufficiency Ratio {season} (%)'] = f"{seasonal_stats.loc[season, 'self_sufficiency_ratio']:.2f}"
                summary[f'Battery Need {season} (kWh)'] = f"{battery_requirements.loc[battery_requirements['season'] == season, 'p95_capacity_needed_kwh'].values[0]:.2f}"
                
            # Update summary DataFrame and save again
            summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
            summary_df.to_csv(summary_file, index=False, sep=';', encoding='utf-8')
            logging.info(f"Updated summary results with seasonal data saved to {summary_file}")
                
            # -------------------- Step 9: Battery Sizing Analysis --------------------
            logging.info("Performing battery sizing analysis...")
            optimal_capacity, battery_results = calculate_optimal_battery_capacity(
                df,
                args.output_dir,
                min_capacity=1,
                max_capacity=50,
                capacity_step=2,
                battery_efficiency=0.9,
                depth_of_discharge=0.8,
                battery_cost_per_kwh=500,
                electricity_buy_price=0.20,
                electricity_sell_price=0.10
            )

            # Add battery information to summary
            summary['Optimal Battery Capacity (kWh)'] = f"{optimal_capacity:.2f}"
            summary['Battery Self-Consumption Rate (%)'] = f"{battery_results.loc[battery_results['capacity_kwh'] == optimal_capacity, 'self_consumption_rate'].values[0]:.2f}"
            summary['Battery Self-Sufficiency Rate (%)'] = f"{battery_results.loc[battery_results['capacity_kwh'] == optimal_capacity, 'self_sufficiency_rate'].values[0]:.2f}"
            summary['Battery Equivalent Full Cycles (per year)'] = f"{battery_results.loc[battery_results['capacity_kwh'] == optimal_capacity, 'equivalent_full_cycles'].values[0]:.2f}"
            summary['Battery Simple Payback (years)'] = f"{battery_results.loc[battery_results['capacity_kwh'] == optimal_capacity, 'simple_payback_years'].values[0]:.2f}"
            
            # Validate results
            warnings = validate_results(summary, df)

            # Add warnings to summary if any
            if warnings:
                summary['WARNINGS'] = '; '.join(warnings)
            
            # Update summary DataFrame and save again
            summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
            summary_df.to_csv(summary_file, index=False, sep=';', encoding='utf-8')
            logging.info(f"Updated summary results with battery sizing data saved to {summary_file}")
            
            # -------------------- Step 10: Economic and Efficiency Analysis --------------------
            logging.info("Performing economic and efficiency analysis...")

            # Define economic parameters
            economic_params = {
                'panel_cost': 250,
                'installation_cost_per_panel': 150,
                'inverter_cost_per_kw': 120,
                'battery_cost_per_kwh': 500,
                'bos_cost_per_panel': 50,
                'electricity_price': 0.20,
                'feed_in_tariff': 0.10,
                'annual_maintenance_percent': 0.5,
                'inflation_rate': 2.0,
                'electricity_price_increase': 3.0,
                'discount_rate': 5.0,
                'system_lifetime': 25,
                'battery_lifetime': 10  # ADD THIS LINE
            }

            # Calculate initial investment
            initial_investment = calculate_initial_investment(
                number_of_panels, 
                optimal_capacity,
                panel_cost=economic_params['panel_cost'],
                installation_cost_per_panel=economic_params['installation_cost_per_panel'],
                inverter_cost_per_kw=economic_params['inverter_cost_per_kw'],
                battery_cost_per_kwh=economic_params['battery_cost_per_kwh'],
                bos_cost_per_panel=economic_params['bos_cost_per_panel']
            )

            # Calculate annual cash flows with improved battery model
            cashflows = calculate_annual_cashflow_improved(
                df,
                optimal_capacity,  # Add the missing battery capacity parameter
                electricity_price=economic_params['electricity_price'],
                feed_in_tariff=economic_params['feed_in_tariff'],
                annual_maintenance_percent=economic_params['annual_maintenance_percent'],
                inflation_rate=economic_params['inflation_rate'],
                electricity_price_increase=economic_params['electricity_price_increase'],
                system_lifetime=economic_params['system_lifetime'],
                initial_investment=initial_investment,
                battery_cost_per_kwh=economic_params['battery_cost_per_kwh'],
                battery_lifetime=10
            )
           

            # Calculate financial metrics
            financial_metrics = calculate_financial_metrics(
                initial_investment, 
                cashflows, 
                discount_rate=economic_params['discount_rate'],
                electricity_price=economic_params['electricity_price'],
                electricity_price_increase=economic_params['electricity_price_increase'],
                feed_in_tariff=economic_params['feed_in_tariff'],
                inflation_rate=economic_params['inflation_rate']
            )

            # Calculate efficiency metrics
            efficiency_metrics = calculate_efficiency_metrics(
                df, 
                number_of_panels, 
                panel_area, 
                total_panel_area, 
                panel_power_rating
            )
        
        
    
        # -------------------- Step 11: Create Case Study Report --------------------
            # Convert summary dict to the format expected by case study function
        results_summary_dict = {item[0]: item[1] for item in summary.items()}

        create_case_study_report(
            df, config, args, optimal_tilt, optimal_azimuth, 
            optimal_capacity, results_summary_dict, args.output_dir
        )
        
        # Generate all plots
        plot_consumption_profile(df, args.output_dir)
        energy_breakdown, energy_losses, system_efficiency = summarize_energy(df)
        logging.info(f"System Efficiency: {system_efficiency:.2f}%")
        plot_energy_losses(energy_losses, args.output_dir)
        total_E_ac_kWh = df['E_ac_Wh'].sum() / 1000

        # Create and save summary data
        if panel_efficiency <= 0 or pd.isna(panel_efficiency):
            logging.warning(f"Panel efficiency calculation issue: {panel_efficiency}")
            panel_power_rating = config['solar_panel']['power_rating']  # Should be 240W
            panel_area_calc = config['solar_panel']['length'] * config['solar_panel']['width']  # m²
            panel_efficiency = panel_power_rating / (panel_area_calc * 1000)  # 1000 W/m² STC
            logging.info(f"Recalculated panel efficiency: {panel_efficiency * 100:.2f}%")
            
        summary = create_comprehensive_summary(
            df, df_original, optimal_tilt, optimal_azimuth, optimal_capacity,
            balanced_weighted_mismatch, balanced_production, baseline_results,
            seasonal_stats, battery_results, 
            initial_investment,  # Ensure initial_investment is correctly calculated and passed
            financial_metrics, 
            efficiency_metrics, number_of_panels, total_panel_area, 
            panel_efficiency, system_efficiency, energy_losses # Pass the energy_losses DataFrame
        )
        

       # Save summary
        summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        # summary_file was defined earlier in the if pareto_front: block, ensure it's accessible here
        # If not, you might need to redefine it or ensure its scope:
        # summary_file = os.path.join(args.output_dir, 'summary_results.csv') 
        summary_df.to_csv(summary_file, index=False, sep=';', encoding='utf-8')
        logging.info(f"Comprehensive summary results saved to {summary_file}")

        create_comprehensive_battery_analysis_plot(battery_results, optimal_capacity, df, args.output_dir)

        # Create enhanced HTML summary
        create_enhanced_html_summary(summary_df, args.output_dir)
        
        # Save detailed results to CSV
        df_file = os.path.join(args.output_dir, 'detailed_results_optimal_angles.csv')
        df.to_csv(df_file)
        logging.info(f"Detailed results saved to {df_file}")

        # Generate all visualization plots
        plot_daily_irradiance_and_energy(df, args.output_dir)
        plot_average_daily_consumption(df, args.output_dir)
        plot_average_hourly_consumption_vs_production(df, args.output_dir)
        plot_average_hourly_weighting_factors(df, args.output_dir)
        plot_combined_hourly_data(df, args.output_dir)
        plot_representative_day_profiles(df, args.output_dir, representative_date)
        plot_hourly_heatmaps(df, args.output_dir)
        
        # Run sensitivity analyses
        tilt_values = np.linspace(0, 90, 10)
        azimuth_values = np.linspace(90, 270, 10)
        perform_sensitivity_analysis('tilt_angle', tilt_values, {}, df_subset, dni_extra,
                                        number_of_panels, inverter_params, args.output_dir)
        perform_sensitivity_analysis('azimuth_angle', azimuth_values, {}, df_subset, dni_extra,
                                        number_of_panels, inverter_params, args.output_dir)
        

        # Create visualizations
        plot_economic_analysis(initial_investment, cashflows, financial_metrics, args.output_dir)

        # Save results to CSV
        pd.DataFrame(list(initial_investment.items()), columns=['Metric', 'Value']).to_csv(
            os.path.join(args.output_dir, 'initial_investment.csv'), index=False)
        cashflows.to_csv(os.path.join(args.output_dir, 'annual_cashflows.csv'), index=False)
        pd.DataFrame(list(financial_metrics.items()), columns=['Metric', 'Value']).to_csv(
            os.path.join(args.output_dir, 'financial_metrics.csv'), index=False)
        pd.DataFrame(list(efficiency_metrics.items()), columns=['Metric', 'Value']).to_csv(
            os.path.join(args.output_dir, 'efficiency_metrics.csv'), index=False)

        # # Add to summary
        # summary['Total Investment ($)'] = f"{initial_investment['total_investment']:,.2f}"
        # summary['Net Present Value ($)'] = f"{financial_metrics['NPV']:,.2f}"
        # summary['Internal Rate of Return (%)'] = f"{financial_metrics['IRR']:.2f}" if financial_metrics['IRR'] is not None else "N/A"
        # summary['Payback Period (years)'] = f"{financial_metrics['Payback_Period_Years']:.2f}"
        # summary['Levelized Cost of Electricity ($/kWh)'] = f"{financial_metrics['LCOE']:.4f}"
        # summary['Performance Ratio (%)'] = f"{efficiency_metrics['performance_ratio']:.2f}"
        # summary['System Yield (kWh/kWp)'] = f"{efficiency_metrics['system_yield']:.2f}"
        # summary['Capacity Factor (%)'] = f"{efficiency_metrics['capacity_factor']:.2f}"

        
    

        # # Update summary DataFrame and save again
        # summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        # summary_df.to_csv(summary_file, index=False, sep=';', encoding='utf-8')
        # logging.info(f"Updated summary results with seasonal data saved to {summary_file}")

    
        system_params = {
        'number_of_panels': number_of_panels,
        'panel_area': panel_area,
        'total_panel_area': total_panel_area,
        'panel_efficiency': panel_efficiency,
        'panel_power': config['solar_panel']['power_rating'],
        'temp_coeff': -0.0044,  # From Sharp ND-R240A5 datasheet
        'battery_efficiency': 0.9,  # Typical value used in simulations
        'depth_of_discharge': 0.8,  # Typical value used in simulations
        }

        forecast_data_path = os.path.join(args.output_dir, 'forecast_data.csv')
        export_data_for_forecasting(
        df, 
        optimal_tilt, 
        optimal_azimuth, 
        optimal_capacity,
        forecast_data_path,
        system_params
        )    
        
    except Exception as e:
        logging.error("An unexpected error occurred:", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

