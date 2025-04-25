#pv_simulation_V13.py

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

def calculate_weighting_factors(df):
    """
    Calculate the weighting factors based on the building's consumption data.
    Weights are assigned only during daylight hours (solar zenith angle < 90°).
    
    Parameters:
    - df (DataFrame): DataFrame containing 'Load (kW)' and 'zenith'.
    
    Returns:
    - Series: The 'weighting_factor' column to be added to df.
    """
    # Ensure necessary columns are present
    for col in ['Load (kW)', 'zenith']:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the DataFrame.")
            raise ValueError(f"'{col}' column is missing in the DataFrame.")

    # Daylight condition: zenith angle less than 90 degrees
    daylight_mask = df['zenith'] < 90

    # Initialize weighting factors to zero
    df['weighting_factor'] = 0.0

    # Extract Load values during daylight only
    daylight_load = df.loc[daylight_mask, 'Load (kW)']

    # Normalize load values during daylight hours (0–1 scale)
    if not daylight_load.empty:
        min_load = daylight_load.min()
        max_load = daylight_load.max()

        if max_load != min_load:
            normalized_load = (daylight_load - min_load) / (max_load - min_load)
        else:
            normalized_load = 1  # If consumption is constant during daylight hours

        # Assign normalized weights during daylight hours
        df.loc[daylight_mask, 'weighting_factor'] = normalized_load
    else:
        logging.warning("No daylight hours detected; weighting factors remain zero.")

    return df['weighting_factor']

def calculate_weighted_energy(df):
    #    Parameters:
    # - df (DataFrame): DataFrame containing the following columns:
    #     - 'E_ac': AC energy output in watts.
    #- 'weighting_factor': Pre-calculated weighting factors.

    #     Returns:
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
    Calculate energy production and losses for Sharp ND-R240A5 panels.

    Parameters:
    - df (DataFrame): DataFrame with solar irradiance and temperature data.
    - number_of_panels (int): Number of panels installed.
    - inverter_params (dict): Dictionary with inverter parameters.

    Returns:
    - df (DataFrame): DataFrame with energy calculations added.
    """
    required_columns = ['total_irradiance', 'Air Temp']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the DataFrame.")
            raise ValueError(f"'{col}' column is missing in the DataFrame.")

    try:
        # Panel parameters from Sharp ND-R240A5 datasheet
        panel_area = 1.642  # m² (1.652m × 0.994m)
        panel_efficiency = 0.146  # 14.6% efficiency
        panel_power = 240  # Wp
        NOCT = 47.5  # °C
        TEMP_COEFF_PMAX = -0.00440  # -0.440% / °C
        TEMP_COEFF_VOC = -0.00329   # -0.329% / °C
        TEMP_COEFF_ISC = 0.00038    # +0.038% / °C
        
        # Calculate total panel area
        total_panel_area = panel_area * number_of_panels
        
        # Calculate total system nominal power
        total_nominal_power = panel_power * number_of_panels  # W
        
        # Apply standard loss factors
        SOILING_LOSS = 0.02  # 2% loss due to soiling
        SHADING_LOSS = 0.03  # 3% loss due to partial shading (adjust based on actual site conditions)
        MISMATCH_LOSS = 0.02  # 2% loss due to mismatch between panels
        WIRING_LOSS = 0.02  # 2% loss in DC wiring
        REFLECTION_LOSS = 0.03  # 3% loss due to reflection (depends on glass and coating)
        
        TOTAL_LOSS_FACTOR = (1 - SOILING_LOSS) * (1 - SHADING_LOSS) * (1 - MISMATCH_LOSS) * (1 - WIRING_LOSS) * (1 - REFLECTION_LOSS)
        
        # Define time interval in hours (adjust based on your data frequency)
        TIME_INTERVAL_HOURS = 1  # Assuming hourly data
        
        # Adjust for shading, soiling, and other losses
        df['effective_irradiance'] = df['total_irradiance'] * TOTAL_LOSS_FACTOR

        # Calculate cell temperature (°C) using NOCT method
        df['cell_temperature'] = df['Air Temp'] + ((NOCT - 20) / 800) * df['total_irradiance']

        # Calculate DC power output per panel before temperature effects (W)
        df['dc_power_raw_per_panel'] = df['effective_irradiance'] * panel_efficiency * panel_area

        # Adjust DC power output for temperature effects
        df['temperature_factor'] = 1 + TEMP_COEFF_PMAX * (df['cell_temperature'] - 25)
        df['dc_power_output_per_panel'] = df['dc_power_raw_per_panel'] * df['temperature_factor']
        
        # Apply maximum module power limit (capped at rated power)
        df['dc_power_output_per_panel'] = df['dc_power_output_per_panel'].clip(upper=panel_power)

        # Multiply by number of panels to get total DC power output (W)
        df['dc_power_output'] = df['dc_power_output_per_panel'] * number_of_panels

        # Calculate AC power output (W)
        inverter_efficiency = inverter_params['eta_inv_nom'] / 100  # Convert to decimal
        df['ac_power_output'] = df['dc_power_output'] * inverter_efficiency

        # Apply inverter clipping if necessary
        inverter_ac_capacity = inverter_params['pdc0']  # Inverter's rated DC input capacity
        df['ac_power_output'] = df['ac_power_output'].clip(upper=inverter_ac_capacity)

        # Energy calculations (Wh)
        df['E_incident'] = df['total_irradiance'] * total_panel_area * TIME_INTERVAL_HOURS
        df['E_effective'] = df['effective_irradiance'] * total_panel_area * TIME_INTERVAL_HOURS
        df['E_dc'] = df['dc_power_output'] * TIME_INTERVAL_HOURS
        df['E_ac'] = df['ac_power_output'] * TIME_INTERVAL_HOURS

        # Loss calculations (Wh)
        df['E_loss_shading_soiling'] = df['E_incident'] - df['E_effective']
        df['E_loss_temperature'] = (df['effective_irradiance'] * panel_efficiency * total_panel_area * 
                                   TIME_INTERVAL_HOURS) - df['E_dc']
        df['E_loss_inverter'] = df['E_dc'] - df['E_ac']
        df['E_loss_total'] = df['E_incident'] - df['E_ac']

        # Performance ratio calculation
        df['PR'] = df['E_ac'] / (df['total_irradiance'] * total_panel_area * panel_efficiency * TIME_INTERVAL_HOURS)
        df['PR'] = df['PR'].fillna(0)  # Replace NaN with 0 for nighttime hours

        # Ensure 'E_ac' is present for further use
        df['E_ac'] = df['E_ac'].fillna(0)

        logging.info("Energy production and loss calculations completed.")

    except Exception as e:
        logging.error(f"Error calculating energy production: {e}", exc_info=True)
        raise

    return df

def summarize_energy(df):
    """
    Summarize energy flows and calculate total energies and efficiencies.

    Parameters:
    - df (DataFrame): DataFrame with energy calculations.

    Returns:
    - energy_breakdown (DataFrame): DataFrame summarizing energy at each stage.
    - energy_losses (DataFrame): DataFrame summarizing energy losses.
    - system_efficiency (float): Overall system efficiency in percent.
    """
    try:
        # Sum up energies
        total_E_incident = df['E_incident'].sum()  # Wh
        total_E_effective = df['E_effective'].sum()  # Wh
        total_E_dc = df['E_dc'].sum()  # Wh
        total_E_ac = df['E_ac'].sum()  # Wh

        # Convert to kWh
        total_E_incident_kWh = total_E_incident / 1000
        total_E_effective_kWh = total_E_effective / 1000
        total_E_dc_kWh = total_E_dc / 1000
        total_E_ac_kWh = total_E_ac / 1000

        # Calculate system efficiency
        system_efficiency = (total_E_ac_kWh / total_E_incident_kWh) * 100 if total_E_incident_kWh != 0 else 0

        # Calculate losses (kWh)
        shading_loss_kWh = total_E_incident_kWh - total_E_effective_kWh
        thermal_loss_kWh = total_E_effective_kWh - total_E_dc_kWh
        inverter_loss_kWh = total_E_dc_kWh - total_E_ac_kWh

        # Create energy breakdown DataFrame
        energy_breakdown = pd.DataFrame({
            'Stage': ['Incident Energy', 'Effective Energy', 'DC Output', 'AC Output', 'System Efficiency'],
            'Energy (kWh)': [total_E_incident_kWh, total_E_effective_kWh, total_E_dc_kWh, total_E_ac_kWh, ''],
            'Efficiency (%)': ['', '', '', '', f"{system_efficiency:.2f}%"]
        })

        # Create energy losses DataFrame
        energy_losses = pd.DataFrame({
            'Loss Type': ['Shading/Soiling/Reflection Losses', 'Thermal Losses', 'Inverter Losses'],
            'Energy Lost (kWh)': [shading_loss_kWh, thermal_loss_kWh, inverter_loss_kWh]
        })

        logging.info("Energy summarization completed.")

    except Exception as e:
        logging.error(f"Error summarizing energy: {e}", exc_info=True)
        raise

    return energy_breakdown, energy_losses, system_efficiency

def objective_function_multi(angles, df_subset, dni_extra, number_of_panels, inverter_params):
    """
    Two-objective function for optimization:
    1. Minimize total weighted energy mismatch.
    2. Maximize total energy production.
    """
    try:
        tilt_angle, azimuth_angle = angles

        # Validate angles
        if not (0 <= tilt_angle <= 90) or not (90 <= azimuth_angle <= 270):
            logging.warning(f"Angles out of bounds: Tilt {tilt_angle}°, Azimuth {azimuth_angle}°")
            # Penalize invalid solutions
            return (np.inf, -np.inf)

        # Calculate total irradiance with the given angles
        df_temp = calculate_total_irradiance(df_subset.copy(), tilt_angle, azimuth_angle, dni_extra)

        # Calculate energy production using the updated function
        df_temp = calculate_energy_production(df_temp, number_of_panels, inverter_params)

        # Calculate weighting factors and add to df_temp
        df_temp['weighting_factor'] = calculate_weighting_factors(df_temp)

        # Calculate mismatch (production - consumption)
        df_temp['mismatch'] = df_temp['ac_power_output'] / 1000 - df_temp['Load (kW)']

        # Calculate weighted absolute mismatch
        df_temp['weighted_mismatch'] = df_temp['weighting_factor'] * np.abs(df_temp['mismatch'])

        # Objective 1: Minimize the total weighted mismatch
        total_weighted_mismatch = df_temp['weighted_mismatch'].sum()

        # Objective 2: Maximize total energy production
        total_energy_production = df_temp['ac_power_output'].sum() / 1000  # Convert Wh to kWh

        # Return only the two objectives for NSGA-II
        return (total_weighted_mismatch, total_energy_production)

    except Exception as e:
        logging.error(f"Error in objective_function_multi with angles {angles}: {e}", exc_info=True)
        # Penalize solutions that cause errors
        return (np.inf, -np.inf)

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
    Run multi-objective optimization using a custom NSGA-II process
    that supports dynamic population size, adaptive mutation rates, and detailed logging.

    Returns:
    - pareto_front (list): List of individuals in the final Pareto front (unfiltered).
    - filtered_front (list): Pareto front after applying any post-processing filters.
    - best_balanced (Individual or None): A 'most balanced' solution, if desired.
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
    # 4. Toolbox + parallelization setup
    toolbox = base.Toolbox()

    # Angle attribute generators
    toolbox.register("attr_tilt", np.random.uniform, 0, 90)         # 0° to 90°
    toolbox.register("attr_azimuth", np.random.uniform, 90, 270)      # 90° to 270°

    # Individual and population definitions
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_tilt, toolbox.attr_azimuth), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

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
    logging.info(f"Generation 0 - best mismatch={record['min'][0]:.2f}, best production={record['max'][1]:.2f}")

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
        best_ind = tools.selBest(pop, 1)[0]
        mismatch, production = best_ind.fitness.values
        logging.info(f"Gen {gen} - best mismatch={mismatch:.2f}, best prod={production:.2f}, mutpb={current_mutpb:.3f}")
    
    return pop, logbook

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
                                      battery_cost_per_kwh=500,  # $/kWh
                                      electricity_buy_price=0.20,  # $/kWh
                                      electricity_sell_price=0.10):  # $/kWh (feed-in tariff)
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
        annual_savings = (grid_import_wh / 1000) * electricity_buy_price  # Value of reduced imports
        annual_revenue = (grid_export_wh / 1000) * electricity_sell_price  # Value of exports
        
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
            'self_consumption_rate': self_consumption_rate * 100,  # Convert to %
            'self_sufficiency_rate': self_sufficiency_rate * 100,  # Convert to %
            'grid_import_kwh': grid_import_wh / 1000,
            'grid_export_kwh': grid_export_wh / 1000,
            'battery_charged_kwh': battery_charged_wh / 1000,
            'battery_discharged_kwh': battery_discharged_wh / 1000,
            'battery_losses_kwh': battery_losses_wh / 1000,
            'equivalent_full_cycles': equivalent_full_cycles,
            'battery_investment': battery_investment,
            'annual_savings': annual_savings,
            'annual_revenue': annual_revenue,
            'simple_payback_years': simple_payback,
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
    
def calculate_annual_cashflow(df, electricity_price=0.20, feed_in_tariff=0.10, 
                             annual_maintenance_percent=0.5, inflation_rate=2.0, 
                             electricity_price_increase=3.0, system_lifetime=25,
                             initial_investment=None):
    """
    Calculate annual cash flows over the system lifetime.
    """
    # Extract relevant annual values
    # Convert 'Load (kW)' to Wh for comparison with 'E_ac'
    df_calc = df.copy()
    df_calc['load_wh'] = df_calc['Load (kW)'] * 1000  # Convert kW to Wh
    
    direct_consumption = df_calc.apply(lambda x: min(x['E_ac'], x['load_wh']), axis=1).sum()
    grid_exports = df_calc.apply(lambda x: max(0, x['E_ac'] - x['load_wh']), axis=1).sum()
    grid_imports = df_calc.apply(lambda x: max(0, x['load_wh'] - x['E_ac']), axis=1).sum()
    
    # Convert to kWh
    direct_consumption_kwh = direct_consumption / 1000
    grid_exports_kwh = grid_exports / 1000
    grid_imports_kwh = grid_imports / 1000
    
    # Calculate first year values
    savings_from_direct_use = direct_consumption_kwh * electricity_price
    income_from_exports = grid_exports_kwh * feed_in_tariff
    annual_maintenance_cost = initial_investment['total_investment'] * (annual_maintenance_percent / 100) if initial_investment else 0
    
    # Calculate cash flows for each year
    cashflows = []
    for year in range(1, system_lifetime + 1):
        # Apply inflation and price increases
        current_electricity_price = electricity_price * ((1 + electricity_price_increase/100) ** (year-1))
        current_feed_in_tariff = feed_in_tariff * ((1 + inflation_rate/100) ** (year-1))
        current_maintenance = annual_maintenance_cost * ((1 + inflation_rate/100) ** (year-1))
        
        # Apply panel degradation (typically 0.5% per year)
        degradation_factor = (1 - 0.005) ** (year-1)
        
        # Calculate year's cash flow
        year_savings = direct_consumption_kwh * degradation_factor * current_electricity_price
        year_income = grid_exports_kwh * degradation_factor * current_feed_in_tariff
        year_cashflow = year_savings + year_income - current_maintenance
        
        cashflows.append({
            'year': year,
            'savings': year_savings,
            'income': year_income,
            'maintenance': current_maintenance,
            'net_cashflow': year_cashflow
        })
    
    return pd.DataFrame(cashflows)
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

        # -------------------- Step 5: Run Multi-Objective Optimization --------------------
        # Updated parameter list for the optimization
        pareto_front, filtered_front, best_balanced = run_deap_multi_objective_optimization(
            df_subset,
            dni_extra,
            number_of_panels,
            inverter_params,
            args.output_dir,
            pop_size=args.popsize,
            max_gen=args.maxiter
        )

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
            
            # Updated calculate_energy_production call using new signature
            df = calculate_energy_production(df, number_of_panels, inverter_params)
            
            df['weighting_factor'] = calculate_weighting_factors(df)
            df['E_ac_Wh'] = df['E_ac'] * TIME_INTERVAL_HOURS

            # Verify representative date is in the dataset
            representative_date = args.representative_date
            available_dates = df.index.normalize().unique().strftime('%Y-%m-%d')
            if representative_date not in available_dates:
                logging.error(f"The representative date {representative_date} is not present in the data.")
                raise ValueError(f"The representative date {representative_date} is not present in the data.")

            # Generate all plots
            plot_consumption_profile(df, args.output_dir)
            energy_breakdown, energy_losses, system_efficiency = summarize_energy(df)
            logging.info(f"System Efficiency: {system_efficiency:.2f}%")
            plot_energy_losses(energy_losses, args.output_dir)
            total_E_ac_kWh = df['E_ac_Wh'].sum() / 1000

            # Create and save summary data
            summary = {
                'Number of Panels Installed': number_of_panels,
                'Total Panel Area (m²)': f"{total_panel_area:.2f}",
                'Panel Efficiency (%)': f"{panel_efficiency * 100:.2f}",
                'Optimal Tilt Angle (°)': f"{optimal_tilt:.2f}",
                'Optimal Azimuth Angle (°)': f"{optimal_azimuth:.2f}",
                'Balanced Weighted Energy Mismatch (kWh)': f"{balanced_weighted_mismatch:.2f}",
                'Balanced Total Energy Produced (kWh)': f"{balanced_production:.2f}",
                'System Efficiency (%)': f"{system_efficiency:.2f}",
            }
            for loss_type, loss_value in energy_losses.set_index('Loss Type')['Energy Lost (kWh)'].items():
                summary[f'Energy Loss - {loss_type} (kWh)'] = f"{loss_value:.2f}"

            summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
            summary_file = os.path.join(args.output_dir, 'summary_results.csv')
            summary_df.to_csv(summary_file, index=False, sep=';', encoding='utf-8')
            logging.info(f"Summary results saved to {summary_file}")

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
        logging.info(f"Updated summary results with economic and efficiency metrics saved to {summary_file}")
            
        
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
        
        # Update summary DataFrame and save again
        summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        summary_df.to_csv(summary_file, index=False, sep=';', encoding='utf-8')
        logging.info(f"Updated summary results with battery sizing data saved to {summary_file}")

        
        # -------------------- Step 10: Economic and Efficiency Analysis --------------------
        logging.info("Performing economic and efficiency analysis...")

        # Define economic parameters (these could also come from config file)
        economic_params = {
            'panel_cost': 250,  # $ per panel
            'installation_cost_per_panel': 150,  # $ per panel
            'inverter_cost_per_kw': 120,  # $ per kW
            'battery_cost_per_kwh': 500,  # $ per kWh
            'bos_cost_per_panel': 50,  # $ balance of system per panel
            'electricity_price': 0.20,  # $ per kWh
            'feed_in_tariff': 0.10,  # $ per kWh
            'annual_maintenance_percent': 0.5,  # % of total investment
            'inflation_rate': 2.0,  # %
            'electricity_price_increase': 3.0,  # %
            'discount_rate': 5.0,  # %
            'system_lifetime': 25  # years
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

        # Calculate annual cash flows
        cashflows = calculate_annual_cashflow(
            df,
            electricity_price=economic_params['electricity_price'],
            feed_in_tariff=economic_params['feed_in_tariff'],
            annual_maintenance_percent=economic_params['annual_maintenance_percent'],
            inflation_rate=economic_params['inflation_rate'],
            electricity_price_increase=economic_params['electricity_price_increase'],
            system_lifetime=economic_params['system_lifetime']
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

        # Add to summary
        summary['Total Investment ($)'] = f"{initial_investment['total_investment']:,.2f}"
        summary['Net Present Value ($)'] = f"{financial_metrics['NPV']:,.2f}"
        summary['Internal Rate of Return (%)'] = f"{financial_metrics['IRR']:.2f}" if financial_metrics['IRR'] is not None else "N/A"
        summary['Payback Period (years)'] = f"{financial_metrics['Payback_Period_Years']:.2f}"
        summary['Levelized Cost of Electricity ($/kWh)'] = f"{financial_metrics['LCOE']:.4f}"
        summary['Performance Ratio (%)'] = f"{efficiency_metrics['performance_ratio']:.2f}"
        summary['System Yield (kWh/kWp)'] = f"{efficiency_metrics['system_yield']:.2f}"
        summary['Capacity Factor (%)'] = f"{efficiency_metrics['capacity_factor']:.2f}"

        
    

        # Update summary DataFrame and save again
        summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        summary_df.to_csv(summary_file, index=False, sep=';', encoding='utf-8')
        logging.info(f"Updated summary results with seasonal data saved to {summary_file}")

    
    except Exception as e:
        logging.error("An unexpected error occurred:", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

