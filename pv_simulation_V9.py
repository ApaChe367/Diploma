#pv_simulation_V9.py

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
    def __init__(self, df_subset, dni_extra, panel_params, inverter_params, number_of_panels, total_panel_area, total_loss_factor, panel_efficiency, panel_area):
        self.df_subset = df_subset
        self.dni_extra = dni_extra
        self.panel_params = panel_params
        self.inverter_params = inverter_params
        self.number_of_panels = number_of_panels
        self.total_panel_area = total_panel_area
        self.total_loss_factor = total_loss_factor
        self.panel_efficiency = panel_efficiency
        self.panel_area = panel_area

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
    - tuple: A tuple containing weighted energy mismatch, total energy production, and system efficiency.
 """
    global optimization_context
    return objective_function_multi(
        individual,
        optimization_context.df_subset,
        optimization_context.dni_extra,
        optimization_context.panel_params,
        optimization_context.inverter_params,
        optimization_context.number_of_panels,
        optimization_context.total_panel_area,
        optimization_context.total_loss_factor,
        optimization_context.panel_efficiency,
        optimization_context.panel_area
    )

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

def calculate_energy_production(df, panel_params, inverter_params, number_of_panels, total_panel_area, panel_efficiency, panel_area):
    """
    Calculate energy production and losses.

    Parameters:
    - df (DataFrame): DataFrame with solar irradiance and temperature data.
    - panel_params (dict): Dictionary with panel parameters.
    - inverter_params (dict): Dictionary with inverter parameters.
    - number_of_panels (int): Number of panels installed.
    - total_panel_area (float): Total area of the panels (m²).
    - panel_efficiency (float): Efficiency of the panels (decimal).
    - panel_area (float): Area of a single panel (m²).

    Returns:
    - df (DataFrame): DataFrame with energy calculations added.
    """
    required_columns = ['total_irradiance', 'Air Temp']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"'{col}' column is missing in the DataFrame.")
            raise ValueError(f"'{col}' column is missing in the DataFrame.")

    try:
        # Adjust for shading, soiling, and reflection losses
        df['effective_irradiance'] = df['total_irradiance'] * TOTAL_LOSS_FACTOR

        # Calculate cell temperature (°C)
        TEMP_COEFF = panel_params['gamma_pdc']  # Temperature coefficient (1/°C)
        df['cell_temperature'] = df['Air Temp'] + ((NOCT - 20) / 800) * df['total_irradiance']

        # Calculate DC power output per panel before temperature effects (W)
        df['dc_power_raw_per_panel'] = df['effective_irradiance'] * panel_efficiency * panel_area

        # Adjust DC power output for temperature effects
        df['temperature_factor'] = 1 + TEMP_COEFF * (df['cell_temperature'] - 25)
        df['dc_power_output_per_panel'] = df['dc_power_raw_per_panel'] * df['temperature_factor']

        # Multiply by number of panels to get total DC power output (W)
        df['dc_power_output'] = df['dc_power_output_per_panel'] * number_of_panels

        # Calculate AC power output (W)
        inverter_efficiency = inverter_params['eta_inv_nom'] / 100  # Convert to decimal
        df['ac_power_output'] = df['dc_power_output'] * inverter_efficiency

        # Apply inverter clipping if necessary
        inverter_ac_capacity = inverter_params['pdc0'] * 0.95  # Assuming inverter size is 95% of total Pmp
        df['ac_power_output'] = df['ac_power_output'].clip(upper=inverter_ac_capacity)

        # **Energy calculations (Wh)**
        df['E_incident'] = df['total_irradiance'] * total_panel_area * TIME_INTERVAL_HOURS
        df['E_effective'] = df['effective_irradiance'] * total_panel_area * TIME_INTERVAL_HOURS
        df['E_dc'] = df['dc_power_output'] * TIME_INTERVAL_HOURS
        df['E_ac'] = df['ac_power_output'] * TIME_INTERVAL_HOURS

        # Loss calculations (Wh)
        df['E_loss_shading_soiling'] = df['E_incident'] - df['E_effective']
        df['E_loss_temperature'] = df['E_effective'] - df['E_dc']
        df['E_loss_inverter'] = df['E_dc'] - df['E_ac']
        df['E_loss_total'] = df['E_incident'] - df['E_ac']

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

def objective_function_multi(
    angles,
    df_subset,
    dni_extra,
    panel_params,
    inverter_params,
    number_of_panels,
    total_panel_area,
    total_loss_factor,
    panel_efficiency,
    panel_area
):
    """
    Two-objective function for optimization:
    1. Minimize total weighted energy mismatch.
    2. Maximize total energy production.

    (System efficiency is no longer a separate objective—it's still computed internally for logs/plots if needed.)
    """
    try:
        tilt_angle, azimuth_angle = angles

        # Validate angles
        if not (0 <= tilt_angle <= 90) or not (90 <= azimuth_angle <= 270):
            logging.warning(f"Angles out of bounds: Tilt {tilt_angle}°, Azimuth {azimuth_angle}°")
            # Penalize invalid solutions
            return (np.inf, -np.inf)

        # Calculate total irradiance with the given angles
        df_temp = calculate_total_irradiance(df_subset, tilt_angle, azimuth_angle, dni_extra)

        # Calculate energy production
        df_temp = calculate_energy_production(
            df_temp,
            panel_params,
            inverter_params,
            number_of_panels,
            total_panel_area,
            panel_efficiency,
            panel_area
        )

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

        # (Optionally compute system efficiency or other metrics for logging/heatmaps)
        _, _, system_efficiency = summarize_energy(df_temp)

        # Return only the two objectives for NSGA-II
        return (total_weighted_mismatch, total_energy_production)

    except Exception as e:
        logging.error(f"Error in objective_function_multi with angles {angles}: {e}", exc_info=True)
        # Penalize solutions that cause errors
        return (np.inf, -np.inf)

def run_deap_multi_objective_optimization(
    df_subset,
    dni_extra,
    panel_params,
    inverter_params,
    number_of_panels,
    total_panel_area,
    total_loss_factor,
    panel_efficiency,
    panel_area,
    output_dir,
    pop_size=None,    # Allow dynamic population size
    max_gen=None      # Allow dynamic number of generations
):
    """
    Run multi-objective optimization using a custom NSGA-II process
    that supports dynamic pop size, adaptive mutation rates, and detailed logging.

    Returns:
    - pareto_front (list): List of individuals in the final Pareto front (unfiltered).
    - filtered_front (list): Pareto front after applying any post-processing filters.
    - best_balanced (Individual or None): A 'most balanced' solution, if desired.
    """

    # ----------------------------------------------------------------
    # 1. Dynamic defaults for pop_size / max_gen
    if pop_size is None:
        # Example: pick a default or scale by problem size
        pop_size = 50
    if max_gen is None:
        # Example: pick a default or scale by problem complexity
        max_gen = 30

    # ----------------------------------------------------------------
    # 2. Create an optimization context object (if needed globally)
    context = OptimizationContext(
        df_subset=df_subset,
        dni_extra=dni_extra,
        panel_params=panel_params,
        inverter_params=inverter_params,
        number_of_panels=number_of_panels,
        total_panel_area=total_panel_area,
        total_loss_factor=total_loss_factor,
        panel_efficiency=panel_efficiency,
        panel_area=panel_area
    )

    # ----------------------------------------------------------------
    # 3. Set up DEAP 'creator' for 2-objective optimization
    #    (If these already exist, make sure not to redefine them)
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    # ----------------------------------------------------------------
    # 4. Toolbox + parallelization setup
    toolbox = base.Toolbox()

    # Angle attribute generators
    toolbox.register("attr_tilt", np.random.uniform, 0, 90)         # 0° to 90°
    toolbox.register("attr_azimuth", np.random.uniform, 90, 270)    # 90° to 270°

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
    # 6. Custom evolution loop with adaptive mutation
    #    (Example: linearly decrease mutation from 0.2 to 0.05)
    def custom_nsga2_evolution(pop, toolbox, cxpb, mutpb_start, mutpb_end, ngen, stats, halloffame):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields if stats else []

        # Evaluate the initial population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop = toolbox.select(pop, len(pop))  # Assign crowding distance
        if halloffame is not None:
            halloffame.update(pop)

        record = stats.compile(pop) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        logging.info(f"Generation 0 - best mismatch={record['min'][0]:.2f}, best production={record['max'][1]:.2f}")

        # Evolve
        for gen in range(1, ngen + 1):
            # Adaptive mutpb
            fraction = gen / float(ngen)
            current_mutpb = mutpb_start + fraction * (mutpb_end - mutpb_start)

            # Variation
            offspring = varOr(pop, toolbox, lambda_=len(pop), cxpb=cxpb, mutpb=current_mutpb)

            # Evaluate invalid offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Combine and select
            pop = toolbox.select(pop + offspring, k=len(pop))
            if halloffame is not None:
                halloffame.update(pop)

            # Gather stats
            record = stats.compile(pop) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # Logging per generation
            best_ind = tools.selBest(pop, 1)[0]
            mismatch, production = best_ind.fitness.values
            logging.info(
                f"Gen {gen} - best mismatch={mismatch:.2f}, best prod={production:.2f}, mutpb={current_mutpb:.3f}"
            )

        return pop, logbook

    # ----------------------------------------------------------------
    # 7. Prepare statistics and HallOfFame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    hof = tools.ParetoFront()

    # ----------------------------------------------------------------
    # 8. Run the custom evolution
    final_pop, logbook = custom_nsga2_evolution(
        population,
        toolbox,
        cxpb=0.7,               # Could also adapt over time if desired
        mutpb_start=0.2,        # Starting mutation probability
        mutpb_end=0.05,         # Ending mutation probability
        ngen=max_gen,
        stats=stats,
        halloffame=hof
    )

    # Close pool
    pool.close()
    pool.join()

    # Extract final Pareto front
    pareto_front = list(hof)

    # ----------------------------------------------------------------
    # 9. (Optional) Post-Processing & Filtering
    # Example: Filter solutions where production >= some threshold
    production_threshold = 2000.0  # kWh, pick your own
    filtered_front = [ind for ind in pareto_front if ind.fitness.values[1] >= production_threshold]
    logging.info(f"Filtered front: {len(filtered_front)} of {len(pareto_front)} pass production >= {production_threshold}")

    # Example: pick "most balanced" (minimizing the difference between normalized mismatch & production)
    if filtered_front:
        mismatch_vals = np.array([ind.fitness.values[0] for ind in filtered_front])
        prod_vals     = np.array([ind.fitness.values[1] for ind in filtered_front])
        # Normalize
        mismatch_norm = (mismatch_vals - mismatch_vals.min()) / (mismatch_vals.ptp() + 1e-9)
        prod_norm     = (prod_vals - prod_vals.min()) / (prod_vals.ptp() + 1e-9)
        diff = np.abs(mismatch_norm - prod_norm)
        best_idx = np.argmin(diff)
        best_balanced = filtered_front[best_idx]
    else:
        best_balanced = None

    # ----------------------------------------------------------------
    # 10. Save the Pareto front to CSV
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

    # Return everything of interest
    return pareto_front, filtered_front, best_balanced

def custom_nsga2_evolution(pop, toolbox, cxpb, mutpb_start, mutpb_end, ngen, stats, halloffame):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))  # Assign crowding distance
    if halloffame is not None:
        halloffame.update(pop)

    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    logging.info(f"Generation 0 - best mismatch={record['min'][0]:.2f}, best production={record['max'][1]:.2f}")

    # Initialize mutation probability for generation 0
    current_mutpb = mutpb_start
    offspring = varOr(pop, toolbox, lambda_=len(pop), cxpb=cxpb, mutpb=current_mutpb)
    
    for gen in range(1, ngen + 1):
        fraction = gen / float(ngen)
        current_mutpb = mutpb_start + fraction * (mutpb_end - mutpb_start)
       
        # Variation
        offspring = varOr(pop, toolbox, lambda_=len(pop), cxpb=cxpb, mutpb=current_mutpb)

        # Evaluate invalid offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Combine and select
        pop = toolbox.select(pop + offspring, k=len(pop))
        if halloffame is not None:
            halloffame.update(pop)

        # Gather stats
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        # Logging per generation
        best_ind = tools.selBest(pop, 1)[0]
        mismatch, production = best_ind.fitness.values
        logging.info(f"Gen {gen} - best mismatch={mismatch:.2f}, best prod={production:.2f}, mutpb={current_mutpb:.3f}")

    return pop, logbook


# #def plot_pareto_front(pareto_front, output_dir):
#     """
#     Plot the Pareto front showing the trade-off between objectives.

#     Parameters:
#     - pareto_front (list): List of individuals in the Pareto front.
#     - output_dir (str): Directory to save the plot.
#     """
#     mismatch = [ind.fitness.values[0] for ind in pareto_front]
#     production = [ind.fitness.values[1] for ind in pareto_front]
#     efficiency = [ind.fitness.values[2] for ind in pareto_front]

#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     color = 'tab:red'
#     ax1.set_xlabel('Weighted Energy Mismatch (kWh)')
#     ax1.set_ylabel('Total Energy Production (kWh)', color=color)
#     ax1.scatter(mismatch, production, color=color, label='Production')
#     ax1.tick_params(axis='y', labelcolor=color)
#     ax1.legend(loc='upper left')

#     ax2 = ax1.twinx()

#     color = 'tab:blue'
#     ax2.set_ylabel('System Efficiency (%)', color=color)
#     ax2.scatter(mismatch, efficiency, color=color, label='Efficiency', alpha=0.6)
#     ax2.tick_params(axis='y', labelcolor=color)
#     ax2.legend(loc='upper right')

#     plt.title('Pareto Front: Weighted Energy Mismatch vs. Production and Efficiency')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'pareto_front.png'))
#     plt.show()
    logging.info("Pareto front plot saved to pareto_front.png")

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
    pareto_df.to_csv(os.path.join(output_dir, 'pareto_front_summary.csv'),
                     index=False, sep=';', decimal='.', float_format='%.2f')
    logging.info("Pareto front summary saved to pareto_front_summary.csv")


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
            'total_energy_production_kWh': production,
        })

    pareto_df = pd.DataFrame(pareto_data)
    pareto_df.to_csv(os.path.join(output_dir, 'pareto_front_summary.csv'),
                     index=False, sep=';', decimal='.', float_format='%.2f')
    logging.info("Pareto front summary saved to pareto_front_summary.csv")

def perform_sensitivity_analysis(param_name, param_values, fixed_params, df_subset, dni_extra,
                                 panel_params, inverter_params, number_of_panels, total_panel_area,
                                 total_loss_factor, panel_efficiency, panel_area, output_dir):
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
                        panel_params,
                        inverter_params,
                        number_of_panels,
                        total_panel_area,
                        total_loss_factor,
                        panel_efficiency,
                        panel_area
                    )
                # Set up DEAP for this fixed tilt
                creator.create(f"FitnessMulti_{param_name}_{value}", base.Fitness, weights=(-1.0, 1.0, 1.0))
                creator.create(f"Individual_{param_name}_{value}", list,
                               fitness=getattr(creator, f"FitnessMulti_{param_name}_{value}"))

                toolbox_sens = base.Toolbox()
                toolbox_sens.register("attr_azimuth", np.random.uniform, 90, 270)
                toolbox_sens.register("individual", tools.initCycle,
                                        getattr(creator, f"Individual_{param_name}_{value}"),
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
                    mismatch, production, efficiency = ind.fitness.values
                    results.append({
                        param_name: value,
                        'azimuth_angle': azimuth,
                        'weighted_energy_mismatch_kWh': mismatch,
                        'total_energy_production_kWh': production,
                        'system_efficiency_percent': efficiency
                    })
                del creator[f"FitnessMulti_{param_name}_{value}"]
                del creator[f"Individual_{param_name}_{value}"]

            elif param_name == 'azimuth_angle':
                fixed_azimuth = value
                # Define a new evaluation function that fixes the azimuth angle
                def evaluate_fixed_azimuth(individual):
                    tilt = individual[0]
                    return objective_function_multi(
                        [tilt, fixed_azimuth],
                        df_subset,
                        dni_extra,
                        panel_params,
                        inverter_params,
                        number_of_panels,
                        total_panel_area,
                        total_loss_factor,
                        panel_efficiency,
                        panel_area
                    )
                creator.create(f"FitnessMulti_{param_name}_{value}", base.Fitness, weights=(-1.0, 1.0, 1.0))
                creator.create(f"Individual_{param_name}_{value}", list,
                               fitness=getattr(creator, f"FitnessMulti_{param_name}_{value}"))

                toolbox_sens = base.Toolbox()
                toolbox_sens.register("attr_tilt", np.random.uniform, 0, 90)
                toolbox_sens.register("individual", tools.initCycle,
                                        getattr(creator, f"Individual_{param_name}_{value}"),
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
                    mismatch, production, efficiency = ind.fitness.values
                    results.append({
                        param_name: value,
                        'tilt_angle': tilt,
                        'weighted_energy_mismatch_kWh': mismatch,
                        'total_energy_production_kWh': production,
                        'system_efficiency_percent': efficiency
                    })
                del creator[f"FitnessMulti_{param_name}_{value}"]
                del creator[f"Individual_{param_name}_{value}"]

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
        plt.plot(sensitivity_df['tilt_angle'], sensitivity_df['system_efficiency_percent'], label='Efficiency', marker='o')
        plt.xlabel('Tilt Angle (°)')
    elif param_name == 'azimuth_angle':
        plt.plot(sensitivity_df['azimuth_angle'], sensitivity_df['weighted_energy_mismatch_kWh'], label='Mismatch', marker='o')
        plt.plot(sensitivity_df['azimuth_angle'], sensitivity_df['total_energy_production_kWh'], label='Production', marker='o')
        plt.plot(sensitivity_df['azimuth_angle'], sensitivity_df['system_efficiency_percent'], label='Efficiency', marker='o')
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

def run_grid_search(df_subset, dni_extra, panel_params, inverter_params,
                    number_of_panels, total_panel_area, total_loss_factor,
                    panel_efficiency, panel_area, output_dir):
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
                    panel_params,
                    inverter_params,
                    number_of_panels,
                    total_panel_area,
                    total_loss_factor,
                    panel_efficiency,
                    panel_area
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
        pareto_front, filtered_front, best_balanced = run_deap_multi_objective_optimization(
            df_subset,
            dni_extra,
            panel_params,
            inverter_params,
            number_of_panels,
            total_panel_area,
            TOTAL_LOSS_FACTOR,
            panel_efficiency,
            panel_area,
            args.output_dir,
            pop_size=args.popsize,
            max_gen=args.maxiter
        )

        # -------------------- Step 6: Visualize and Save Pareto Front Results --------------------
        plot_pareto_front(pareto_front, filtered_front, best_balanced, args.output_dir)
        plot_detailed_pareto_front(pareto_front, args.output_dir)
        save_summary_results(pareto_front, args.output_dir)
        grid_search_results_path = run_grid_search(
            df_subset, dni_extra, panel_params, inverter_params,
            number_of_panels, total_panel_area, TOTAL_LOSS_FACTOR,
            panel_efficiency, panel_area, args.output_dir
        )
        plot_pareto_front_comparison(pareto_front, grid_search_results_path, args.output_dir)

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

            # Proceed with calculations clearly using optimal_tilt, optimal_azimuth
            df = df_subset.copy()
            df = calculate_total_irradiance(df, optimal_tilt, optimal_azimuth, dni_extra)
            df = calculate_energy_production(df, panel_params, inverter_params,
                                            number_of_panels, total_panel_area,
                                            panel_efficiency, panel_area)
            df['weighting_factor'] = calculate_weighting_factors(df)
            df['E_ac_Wh'] = df['E_ac'] * TIME_INTERVAL_HOURS

            representative_date = args.representative_date
            available_dates = df.index.normalize().unique().strftime('%Y-%m-%d')
            if representative_date not in available_dates:
                logging.error(f"The representative date {representative_date} is not present in the data.")
                raise ValueError(f"The representative date {representative_date} is not present in the data.")

            plot_consumption_profile(df, args.output_dir)
            energy_breakdown, energy_losses, system_efficiency = summarize_energy(df)
            logging.info(f"System Efficiency: {system_efficiency:.2f}%")
            plot_energy_losses(energy_losses, args.output_dir)
            total_E_ac_kWh = df['E_ac_Wh'].sum() / 1000

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

            df_file = os.path.join(args.output_dir, 'detailed_results_optimal_angles.csv')
            df.to_csv(df_file)
            logging.info(f"Detailed results saved to {df_file}")

            plot_daily_irradiance_and_energy(df, args.output_dir)
            plot_average_daily_consumption(df, args.output_dir)
            plot_average_hourly_consumption_vs_production(df, args.output_dir)
            plot_average_hourly_weighting_factors(df, args.output_dir)
            plot_combined_hourly_data(df, args.output_dir)
            plot_representative_day_profiles(df, args.output_dir, representative_date)
            plot_hourly_heatmaps(df, args.output_dir)

    except Exception as e:
        logging.error("An unexpected error occurred:", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()


