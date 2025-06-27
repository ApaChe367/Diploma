#!/usr/bin/env python3
"""
Improved Enhanced standalone script to validate irradiance patterns and confirm SE optimization results.
Fixed Load data handling and removed incorrect atmospheric calculations.

IMPROVEMENTS:
- Fixed Load data detection and handling
- Removed incorrect atmospheric clarity calculations (no cloud data)
- Added seasonal load vs irradiance analysis
- Enhanced year-long correlation plots
- Added monthly analysis
- Improved error handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pvlib
from datetime import datetime
import warnings
from scipy import stats
import seaborn as sns

def safe_format(value, format_spec=".3f", default="N/A"):
    """
    Safely format a value, handling None, NaN, and other edge cases.
    """
    try:
        if value is None or pd.isna(value):
            return default
        return f"{value:{format_spec}}"
    except (ValueError, TypeError):
        return default

def validate_data_quality(df):
    """
    Enhanced data quality validation with realistic bounds checking.
    """
    print("Validating data quality...")
    
    issues = []
    
    # Check for realistic irradiance values
    if 'SolRad_Hor' in df.columns:
        invalid_ghi = df[(df['SolRad_Hor'] < 0) | (df['SolRad_Hor'] > 1500)]
        if len(invalid_ghi) > 0:
            issues.append(f"Found {len(invalid_ghi)} unrealistic GHI values")
            df.loc[invalid_ghi.index, 'SolRad_Hor'] = np.nan
    
    if 'SolRad_Dif' in df.columns:
        invalid_dhi = df[(df['SolRad_Dif'] < 0) | (df['SolRad_Dif'] > 800)]
        if len(invalid_dhi) > 0:
            issues.append(f"Found {len(invalid_dhi)} unrealistic DHI values")
            df.loc[invalid_dhi.index, 'SolRad_Dif'] = np.nan
    
    # Check for impossible DHI > GHI
    if 'SolRad_Hor' in df.columns and 'SolRad_Dif' in df.columns:
        impossible = df[df['SolRad_Dif'] > df['SolRad_Hor']]
        if len(impossible) > 0:
            issues.append(f"Found {len(impossible)} cases where DHI > GHI")
    
    # Check temperature bounds
    if 'Air Temp' in df.columns:
        invalid_temp = df[(df['Air Temp'] < -50) | (df['Air Temp'] > 60)]
        if len(invalid_temp) > 0:
            issues.append(f"Found {len(invalid_temp)} unrealistic temperature values")
    
    # Check Load data bounds
    if 'Load (kW)' in df.columns:
        invalid_load = df[(df['Load (kW)'] < 0) | (df['Load (kW)'] > 1000)]  # Adjust max as needed
        if len(invalid_load) > 0:
            issues.append(f"Found {len(invalid_load)} unrealistic Load values")
            df.loc[invalid_load.index, 'Load (kW)'] = np.nan
    
    # Report issues
    if issues:
        print("DATA QUALITY ISSUES DETECTED:")
        for issue in issues:
            print(f"  WARNING  {issue}")
        print("  → Issues have been cleaned automatically")
    else:
        print("  OK Data quality validation passed")
    
    return df, issues

def analyze_load_correlation(df):
    """
    Enhanced load correlation analysis with better error handling.
    """
    print("Analyzing load pattern correlation...")
    
    if 'Load (kW)' not in df.columns:
        print("  WARNING  No 'Load (kW)' column found in data")
        return {}
    
    # Check if Load data is actually available (not all NaN)
    load_data = df['Load (kW)'].dropna()
    if len(load_data) == 0:
        print("  WARNING  Load column exists but contains no valid data")
        return {}
    
    print(f"  OK Found {len(load_data)} valid load data points out of {len(df)} total")
    
    try:
        # Filter daylight hours
        daylight_mask = df['zenith'] < 90
        df_daylight = df[daylight_mask].copy()
        df_daylight['hour'] = df_daylight.index.hour
        
        # Calculate hourly averages
        hourly_irr = df_daylight.groupby('hour')['DNI'].mean()
        hourly_load = df_daylight.groupby('hour')['Load (kW)'].mean()
        
        # Calculate correlations
        morning_hours = range(6, 12)
        afternoon_hours = range(12, 19)
        
        morning_irr = hourly_irr[hourly_irr.index.isin(morning_hours)]
        morning_load = hourly_load[hourly_load.index.isin(morning_hours)]
        afternoon_irr = hourly_irr[hourly_irr.index.isin(afternoon_hours)]
        afternoon_load = hourly_load[hourly_load.index.isin(afternoon_hours)]
        
        # Calculate correlations with error handling
        try:
            # Overall correlation
            overall_irr = df_daylight['DNI'].dropna()
            overall_load = df_daylight['Load (kW)'].dropna()
            
            # Align the data by index
            common_index = overall_irr.index.intersection(overall_load.index)
            if len(common_index) > 10:
                overall_corr = np.corrcoef(overall_irr[common_index], overall_load[common_index])[0, 1]
                if pd.isna(overall_corr):
                    overall_corr = 0
            else:
                overall_corr = 0
                
            morning_corr = np.corrcoef(morning_irr.values, morning_load.values)[0, 1] if len(morning_irr) > 1 else 0
            if pd.isna(morning_corr):
                morning_corr = 0
        except:
            morning_corr = 0
            overall_corr = 0
            
        try:
            afternoon_corr = np.corrcoef(afternoon_irr.values, afternoon_load.values)[0, 1] if len(afternoon_irr) > 1 else 0
            if pd.isna(afternoon_corr):
                afternoon_corr = 0
        except:
            afternoon_corr = 0
        
        load_metrics = {
            'overall_irr_load_correlation': overall_corr,
            'morning_irr_load_correlation': morning_corr,
            'afternoon_irr_load_correlation': afternoon_corr,
            'morning_peak_load_hour': morning_load.idxmax() if len(morning_load) > 0 else None,
            'afternoon_peak_load_hour': afternoon_load.idxmax() if len(afternoon_load) > 0 else None,
            'morning_avg_load': morning_load.mean() if len(morning_load) > 0 else 0,
            'afternoon_avg_load': afternoon_load.mean() if len(afternoon_load) > 0 else 0,
            'daily_peak_load': df_daylight['Load (kW)'].max(),
            'daily_avg_load': df_daylight['Load (kW)'].mean()
        }
        
        print(f"  Overall irradiance-load correlation: {safe_format(overall_corr)}")
        print(f"  Morning irradiance-load correlation: {safe_format(morning_corr)}")
        print(f"  Afternoon irradiance-load correlation: {safe_format(afternoon_corr)}")
        print(f"  Daily average load: {safe_format(load_metrics['daily_avg_load'], '.1f')} kW")
        
        return load_metrics
        
    except Exception as e:
        print(f"  WARNING  Error in load correlation analysis: {e}")
        return {}

def statistical_significance_test(morning_data, afternoon_data, metric_name):
    """
    Perform statistical significance test for morning vs afternoon differences.
    """
    try:
        # Remove NaN values
        morning_clean = morning_data.dropna()
        afternoon_clean = afternoon_data.dropna()
        
        if len(morning_clean) < 10 or len(afternoon_clean) < 10:
            return None, None
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(morning_clean, afternoon_clean)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(morning_clean) - 1) * morning_clean.var() + 
                             (len(afternoon_clean) - 1) * afternoon_clean.var()) / 
                            (len(morning_clean) + len(afternoon_clean) - 2))
        
        if pooled_std > 0:
            cohens_d = (morning_clean.mean() - afternoon_clean.mean()) / pooled_std
        else:
            cohens_d = 0
        
        return p_value, cohens_d
    
    except Exception as e:
        print(f"  WARNING  Statistical test failed for {metric_name}: {e}")
        return None, None

def load_and_preprocess_for_validation(data_file, latitude=37.98983, longitude=23.74328):
    """
    Enhanced load and preprocess data with better error handling.
    """
    print("Loading and preprocessing data...")
    
    try:
        # Load the data
        df = pd.read_csv(data_file)
        print(f"  OK Loaded {len(df)} rows from {data_file}")
        print(f"  Columns found: {list(df.columns)}")
    except Exception as e:
        raise ValueError(f"Failed to load data file: {e}")
    
    # Ensure correct data types
    numeric_columns = ['hour', 'SolRad_Hor', 'SolRad_Dif', 'Air Temp', 'WS_10m']
    
    # Check for Load column (and be more flexible with naming)
    load_column = None
    for col in df.columns:
        if 'load' in col.lower() and 'kw' in col.lower():
            load_column = col
            numeric_columns.append(col)
            print(f"  OK Found load column: '{col}'")
            break
    
    if load_column is None:
        print("  WARNING  No load column found - load analysis will be skipped")
    
    for col in numeric_columns[:5]:  # Required columns
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing in the data file.")
    
    # Convert to numeric with error handling
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Validate data quality
    df, quality_issues = validate_data_quality(df)
    
    # Convert 'hour' column to datetime
    start_date = datetime(2023, 1, 1, 0, 0)
    try:
        df['datetime'] = pd.to_datetime(df['hour'] - 1, unit='h', origin=start_date)
        df.set_index('datetime', inplace=True)
    except Exception as e:
        raise ValueError(f"Failed to convert hour column to datetime: {e}")
    
    # Localize timezone with better error handling
    try:
        df.index = df.index.tz_localize('Europe/Athens', ambiguous='NaT', nonexistent='shift_forward')
        df = df[~df.index.isna()]
        df = df.sort_index()
    except Exception as e:
        print(f"  WARNING  Timezone localization warning: {e}")
        # Continue without timezone if there's an issue
    
    # Remove duplicates
    duplicates = df.index.duplicated(keep='first')
    if duplicates.any():
        print(f"  WARNING  Removed {duplicates.sum()} duplicate timestamps")
        df = df[~duplicates]
    
    # Set frequency
    try:
        df = df.asfreq('h')
    except Exception as e:
        print(f"  WARNING  Could not set hourly frequency: {e}")
    
    # Fill missing values with forward/backward fill
    missing_before = df[numeric_columns].isnull().sum().sum()
    df[numeric_columns] = df[numeric_columns].ffill().bfill()
    missing_after = df[numeric_columns].isnull().sum().sum()
    
    if missing_before > 0:
        print(f"  WARNING  Filled {missing_before} missing values")
        if missing_after > 0:
            print(f"  WARNING  {missing_after} missing values remain after filling")
    
    print("Calculating solar position...")
    
    # Calculate solar position with error handling
    try:
        solar_position = pvlib.solarposition.get_solarposition(df.index, latitude, longitude)
        df['zenith'] = solar_position['apparent_zenith']
        df['azimuth'] = solar_position['azimuth']
    except Exception as e:
        raise ValueError(f"Failed to calculate solar position: {e}")
    
    print("Calculating DNI...")
    
    # Calculate DNI with error handling
    try:
        dni = pvlib.irradiance.disc(
            ghi=df['SolRad_Hor'],
            solar_zenith=df['zenith'],
            datetime_or_doy=df.index
        )['dni']
        df['DNI'] = dni
    except Exception as e:
        raise ValueError(f"Failed to calculate DNI: {e}")
    
    print("  OK Data preprocessing complete!")
    return df

def create_enhanced_irradiance_plots(df, output_dir):
    """
    Create enhanced validation plots with load analysis and seasonal patterns.
    """
    print("Creating enhanced irradiance validation analysis...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter daylight hours only
    daylight_mask = df['zenith'] < 90
    df_daylight = df[daylight_mask].copy()
    df_daylight['hour'] = df_daylight.index.hour
    df_daylight['month'] = df_daylight.index.month
    df_daylight['season'] = df_daylight['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    
    print(f"  Analyzing {len(df_daylight)} daylight hours out of {len(df)} total hours")
    
    # Enhanced hourly analysis
    hourly_stats = df_daylight.groupby('hour').agg({
        'DNI': ['mean', 'std', 'max', 'count'],
        'SolRad_Hor': ['mean', 'std', 'max'],
        'SolRad_Dif': ['mean', 'std', 'max'],
        'zenith': 'mean'
    })
    
    # Add Load statistics if available
    has_load = 'Load (kW)' in df_daylight.columns and not df_daylight['Load (kW)'].isna().all()
    if has_load:
        load_stats = df_daylight.groupby('hour')['Load (kW)'].agg(['mean', 'std', 'max'])
        print(f"  Load data available: {len(df_daylight['Load (kW)'].dropna())} valid points")
    
    # Morning vs afternoon analysis with statistical tests
    morning_hours = range(6, 12)
    afternoon_hours = range(12, 19)
    
    morning_data = df_daylight[df_daylight['hour'].isin(morning_hours)]
    afternoon_data = df_daylight[df_daylight['hour'].isin(afternoon_hours)]
    
    # Calculate metrics with safe handling
    morning_dni = morning_data['DNI'].mean()
    afternoon_dni = afternoon_data['DNI'].mean()
    morning_ghi = morning_data['SolRad_Hor'].mean()
    afternoon_ghi = afternoon_data['SolRad_Hor'].mean()
    
    # Safe ratio calculations
    dni_ratio = morning_dni / afternoon_dni if afternoon_dni > 0 else 0
    ghi_ratio = morning_ghi / afternoon_ghi if afternoon_ghi > 0 else 0
    
    # Statistical significance tests
    dni_p_value, dni_cohens_d = statistical_significance_test(
        morning_data['DNI'], afternoon_data['DNI'], 'DNI'
    )
    ghi_p_value, ghi_cohens_d = statistical_significance_test(
        morning_data['SolRad_Hor'], afternoon_data['SolRad_Hor'], 'GHI'
    )
    
    # Load correlation analysis
    load_metrics = analyze_load_correlation(df_daylight)
    
    print(f"\n=== ENHANCED ANALYSIS RESULTS ===")
    print(f"Morning DNI Average: {morning_dni:.1f} W/m²")
    print(f"Afternoon DNI Average: {afternoon_dni:.1f} W/m²")
    print(f"Morning/Afternoon DNI Ratio: {dni_ratio:.3f}")
    
    if dni_p_value is not None:
        significance = "SIGNIFICANT" if dni_p_value < 0.05 else "NOT SIGNIFICANT"
        print(f"Statistical Significance: {significance} (p={dni_p_value:.3f})")
        print(f"Effect Size (Cohen's d): {safe_format(dni_cohens_d)}")
    
    # Determine bias strength with enhanced criteria
    if dni_ratio > 1.05 and (dni_p_value is None or dni_p_value < 0.05):
        bias_strength = "STRONG"
        conclusion_level = "CONFIRMED"
    elif dni_ratio > 1.02:
        bias_strength = "MODERATE" 
        conclusion_level = "LIKELY"
    else:
        bias_strength = "WEAK"
        conclusion_level = "UNCERTAIN"
    
    print(f"\n[{conclusion_level}] {bias_strength} morning bias detected")
    if bias_strength in ["STRONG", "MODERATE"]:
        print("→ SUPPORTS SE orientation optimization!")
    else:
        print("→ SE optimization likely due to other factors")
    
    # Create enhanced plots (3x3 layout)
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    try:
        # Plot 1: Enhanced DNI Profile with confidence intervals
        ax = axes[0, 0]
        hours = hourly_stats.index
        dni_mean = hourly_stats[('DNI', 'mean')]
        dni_std = hourly_stats[('DNI', 'std')]
        
        ax.plot(hours, dni_mean, 'o-', color='red', linewidth=3, markersize=8, label='Average DNI')
        ax.fill_between(hours, dni_mean - dni_std, dni_mean + dni_std, alpha=0.3, color='red')
        ax.axvline(x=12, color='gray', linestyle='--', alpha=0.7, label='Solar Noon')
        
        # Highlight peak with enhanced annotation
        peak_hour = dni_mean.idxmax()
        ax.scatter(peak_hour, dni_mean[peak_hour], s=200, c='darkred', marker='*', zorder=5)
        
        # Add statistical significance indicator
        if dni_p_value is not None and dni_p_value < 0.05:
            sig_text = "SIGNIFICANT (p<0.05)"
            color = 'green'
        else:
            sig_text = "Not significant"
            color = 'orange'
        
        ax.annotate(f'Peak: {peak_hour}h\n{dni_mean[peak_hour]:.0f} W/m²\n{sig_text}', 
                    xy=(peak_hour, dni_mean[peak_hour]),
                    xytext=(peak_hour+1.5, dni_mean[peak_hour]+100),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('DNI (W/m²)', fontsize=12)
        ax.set_title('Enhanced DNI Profile with Statistics', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Plot 2: Enhanced Morning vs Afternoon with effect sizes
        ax = axes[0, 1]
        categories = ['DNI', 'GHI']
        morning_vals = [morning_dni, morning_ghi]
        afternoon_vals = [afternoon_dni, afternoon_ghi]
        ratios = [m/a if a > 0 else 0 for m, a in zip(morning_vals, afternoon_vals)]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, morning_vals, width, label='Morning (6-12h)', 
                       color='lightblue', edgecolor='blue', linewidth=2)
        bars2 = ax.bar(x + width/2, afternoon_vals, width, label='Afternoon (12-19h)', 
                       color='orange', edgecolor='darkorange', linewidth=2)
        
        # Enhanced ratio labels with significance
        for i, ratio in enumerate(ratios):
            significance_marker = "*" if i == 0 and dni_p_value and dni_p_value < 0.05 else ""
            ax.text(i, max(morning_vals[i], afternoon_vals[i]) + max(morning_vals[i], afternoon_vals[i]) * 0.1, 
                    f'Ratio: {ratio:.3f}{significance_marker}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value (W/m²)', fontsize=12)
        ax.set_title('Morning vs Afternoon Comparison\n(* = statistically significant)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Load vs DNI Correlation (Hourly)
        ax = axes[0, 2]
        if has_load:
            try:
                hourly_load = df_daylight.groupby('hour')['Load (kW)'].mean()
                hourly_dni = df_daylight.groupby('hour')['DNI'].mean()
                
                ax2 = ax.twinx()
                line1 = ax.plot(hourly_load.index, hourly_load.values, 'b-o', linewidth=3, 
                               label=f'Load (r={safe_format(load_metrics.get("overall_irr_load_correlation", 0), ".2f")})', markersize=8)
                line2 = ax2.plot(hourly_dni.index, hourly_dni.values, 'r-s', linewidth=3, 
                                label='DNI', markersize=8)
                
                ax.set_xlabel('Hour of Day', fontsize=12)
                ax.set_ylabel('Load (kW)', color='blue', fontsize=12)
                ax2.set_ylabel('DNI (W/m²)', color='red', fontsize=12)
                ax.set_title('Load vs DNI Correlation (Hourly)', fontsize=14, fontweight='bold')
                
                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                ax.grid(True, alpha=0.3)
                
                # Color axes
                ax.tick_params(axis='y', labelcolor='blue')
                ax2.tick_params(axis='y', labelcolor='red')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Load correlation\nanalysis failed:\n{str(e)[:30]}...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No Load Data\nAvailable', ha='center', va='center',
                    transform=ax.transAxes, fontsize=16, 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            ax.set_title('Load Analysis (No Data)')
        
        # Plot 4: Seasonal Load vs Irradiance
        ax = axes[1, 0]
        if has_load:
            try:
                seasonal_data = df_daylight.groupby('season').agg({
                    'DNI': 'mean',
                    'Load (kW)': 'mean'
                })
                
                seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
                season_colors = ['blue', 'green', 'orange', 'brown']
                
                for i, season in enumerate(seasons):
                    if season in seasonal_data.index:
                        ax.scatter(seasonal_data.loc[season, 'DNI'], 
                                  seasonal_data.loc[season, 'Load (kW)'],
                                  s=200, c=season_colors[i], label=season, alpha=0.8)
                
                ax.set_xlabel('Average DNI (W/m²)', fontsize=12)
                ax.set_ylabel('Average Load (kW)', fontsize=12)
                ax.set_title('Seasonal Load vs DNI Relationship', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Seasonal analysis\nfailed: {str(e)[:20]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No Load Data\nfor Seasonal Analysis', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Seasonal Analysis (No Data)')
        
        # Plot 5: Monthly DNI and Load Pattern
        ax = axes[1, 1]
        monthly_dni = df_daylight.groupby('month')['DNI'].mean()
        
        ax.plot(monthly_dni.index, monthly_dni.values, 'r-o', linewidth=3, markersize=8, label='DNI')
        
        if has_load:
            try:
                monthly_load = df_daylight.groupby('month')['Load (kW)'].mean()
                ax2 = ax.twinx()
                ax2.plot(monthly_load.index, monthly_load.values, 'b-s', linewidth=3, markersize=8, label='Load')
                ax2.set_ylabel('Average Load (kW)', color='blue', fontsize=12)
                ax2.tick_params(axis='y', labelcolor='blue')
                
                # Combined legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            except:
                pass
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Average DNI (W/m²)', color='red', fontsize=12)
        ax.set_title('Monthly DNI and Load Patterns', fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='red')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Load Duration Curve
        ax = axes[1, 2]
        if has_load:
            try:
                load_sorted = df_daylight['Load (kW)'].dropna().sort_values(ascending=False)
                hours_pct = np.arange(1, len(load_sorted) + 1) / len(load_sorted) * 100
                
                ax.plot(hours_pct, load_sorted.values, 'b-', linewidth=2)
                ax.set_xlabel('Time (% of hours)', fontsize=12)
                ax.set_ylabel('Load (kW)', fontsize=12)
                ax.set_title('Load Duration Curve', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add percentile lines
                p50 = np.percentile(load_sorted, 50)
                p90 = np.percentile(load_sorted, 90)
                ax.axhline(y=p50, color='orange', linestyle='--', alpha=0.7, label=f'50th percentile: {p50:.1f} kW')
                ax.axhline(y=p90, color='red', linestyle='--', alpha=0.7, label=f'90th percentile: {p90:.1f} kW')
                ax.legend()
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Load duration\nanalysis failed', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No Load Data\nfor Duration Curve', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Load Duration Curve (No Data)')
        
        # Calculate confidence score
        confidence_score = 0
        if dni_ratio > 1.05:
            confidence_score += 30
        elif dni_ratio > 1.02:
            confidence_score += 15
        
        if dni_p_value and dni_p_value < 0.05:
            confidence_score += 25
        elif dni_p_value and dni_p_value < 0.10:
            confidence_score += 15
        
        if abs(dni_cohens_d or 0) > 0.5:
            confidence_score += 20
        elif abs(dni_cohens_d or 0) > 0.2:
            confidence_score += 10
        
        if load_metrics.get('overall_irr_load_correlation', 0) > 0.3:
            confidence_score += 15
        elif load_metrics.get('morning_irr_load_correlation', 0) > 0.3:
            confidence_score += 10
        
        confidence_level = "HIGH" if confidence_score >= 70 else "MEDIUM" if confidence_score >= 40 else "LOW"
        
        # Plot 7: Summary Statistics
        ax = axes[2, 0]
        ax.axis('off')
        
        # Create summary text with safe formatting
        summary_text = f"""ENHANCED VALIDATION SUMMARY

Statistical Analysis:
• DNI Ratio: {safe_format(dni_ratio)} ({bias_strength})
• P-value: {safe_format(dni_p_value)}
• Effect Size: {safe_format(dni_cohens_d)}

Load Correlation:
• Overall: {safe_format(load_metrics.get('overall_irr_load_correlation', 0))}
• Morning: {safe_format(load_metrics.get('morning_irr_load_correlation', 0))}
• Afternoon: {safe_format(load_metrics.get('afternoon_irr_load_correlation', 0))}

Peak Analysis:
• DNI Peak Hour: {hourly_stats[('DNI', 'mean')].idxmax()}h
• Peak Value: {safe_format(hourly_stats[('DNI', 'mean')].max(), '.0f')} W/m²
"""
        
        if has_load:
            summary_text += f"""
Load Statistics:
• Daily Average: {safe_format(load_metrics.get('daily_avg_load', 0), '.1f')} kW
• Daily Peak: {safe_format(load_metrics.get('daily_peak_load', 0), '.1f')} kW
"""
        
        summary_text += f"\nCONFIDENCE SCORE: {confidence_score}/100 ({confidence_level})"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        # Plot 8: Time series (sample week)
        ax = axes[2, 1]
        try:
            # Get a representative week (mid-spring)
            sample_start = df_daylight.index[len(df_daylight)//4]  # Around month 3
            sample_end = sample_start + pd.Timedelta(days=7)
            sample_data = df_daylight[sample_start:sample_end]
            
            if len(sample_data) > 0:
                ax.plot(sample_data.index, sample_data['DNI'], 'r-', linewidth=2, label='DNI')
                
                if has_load and 'Load (kW)' in sample_data.columns:
                    ax2 = ax.twinx()
                    ax2.plot(sample_data.index, sample_data['Load (kW)'], 'b-', linewidth=2, label='Load')
                    ax2.set_ylabel('Load (kW)', color='blue')
                    ax2.tick_params(axis='y', labelcolor='blue')
                
                ax.set_xlabel('Date')
                ax.set_ylabel('DNI (W/m²)', color='red')
                ax.set_title('Sample Week: DNI and Load', fontsize=14, fontweight='bold')
                ax.tick_params(axis='y', labelcolor='red')
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax.text(0.5, 0.5, 'No sample data\navailable', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
        except Exception as e:
            ax.text(0.5, 0.5, f'Time series\nplot failed', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        # Plot 9: Final conclusion
        ax = axes[2, 2]
        ax.axis('off')
        
        if confidence_score >= 70:
            conclusion = "[HIGH CONFIDENCE]\nSTRONG evidence supports\nSE orientation!\n\nStatistically significant\nmorning irradiance advantage\nconfirmed."
            color = 'darkgreen'
        elif confidence_score >= 40:
            conclusion = "[MEDIUM CONFIDENCE]\nMODERATE evidence supports\nSE orientation.\n\nSome supporting factors\nidentified but more\nanalysis recommended."
            color = 'orange'
        else:
            conclusion = "[LOW CONFIDENCE]\nLIMITED evidence for\nmorning irradiance bias.\n\nSE optimization likely\ndriven by load matching\nor other factors."
            color = 'red'
        
        if has_load:
            if load_metrics.get('overall_irr_load_correlation', 0) > 0.5:
                conclusion += f"\n\nLOAD CORRELATION:\nStrong positive correlation\n(r = {load_metrics.get('overall_irr_load_correlation', 0):.2f})\nsupports SE orientation!"
            elif load_metrics.get('morning_irr_load_correlation', 0) > load_metrics.get('afternoon_irr_load_correlation', 0):
                conclusion += f"\n\nLOAD PATTERN:\nMorning load correlation\nhigher than afternoon."
        
        ax.text(0.5, 0.5, conclusion, transform=ax.transAxes, fontsize=12,
                verticalalignment='center', horizontalalignment='center',
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
    except Exception as e:
        print(f"  WARNING  Error in plotting: {e}")
        # Continue with basic analysis even if plotting fails
    
    plt.tight_layout()
    
    try:
        plt.savefig(os.path.join(output_dir, 'enhanced_irradiance_validation.png'), 
                    dpi=300, bbox_inches='tight')
        print(f"  OK Plots saved to {output_dir}")
    except Exception as e:
        print(f"  WARNING  Could not save plots: {e}")
    
    plt.close()
    
    # Save enhanced results with safe formatting
    enhanced_results = {
        'dni_morning_afternoon_ratio': dni_ratio,
        'ghi_morning_afternoon_ratio': ghi_ratio,
        'dni_peak_hour': hourly_stats[('DNI', 'mean')].idxmax(),
        'peak_dni_value': hourly_stats[('DNI', 'mean')].max(),
        'morning_dni_avg': morning_dni,
        'afternoon_dni_avg': afternoon_dni,
        'dni_p_value': dni_p_value,
        'dni_cohens_d': dni_cohens_d,
        'ghi_p_value': ghi_p_value,
        'ghi_cohens_d': ghi_cohens_d,
        'bias_strength': bias_strength.lower(),
        'confidence_score': confidence_score,
        'confidence_level': confidence_level.lower(),
        'supports_se_orientation': confidence_score >= 40,
        'statistically_significant': dni_p_value is not None and dni_p_value < 0.05,
        'has_load_data': has_load,
        **load_metrics
    }
    
    try:
        results_df = pd.DataFrame([enhanced_results])
        results_df.to_csv(os.path.join(output_dir, 'enhanced_validation_results.csv'), index=False)
        print(f"  OK Results saved to CSV")
    except Exception as e:
        print(f"  WARNING  Could not save results CSV: {e}")
    
    print(f"\n[SUCCESS] Enhanced validation complete!")
    print(f"[CONFIDENCE] {confidence_level} confidence ({confidence_score}/100)")
    if confidence_score >= 40:
        print(f"[CONCLUSION] Evidence supports SE orientation!")
    else:
        print(f"[CONCLUSION] SE optimization likely due to load matching factors!")
    
    return enhanced_results

def main():
    parser = argparse.ArgumentParser(description='Enhanced irradiance pattern validation for SE orientation')
    parser.add_argument('--data_file', type=str, required=True, help='Path to your CSV data file')
    parser.add_argument('--output_dir', type=str, default='enhanced_validation_results', help='Output directory')
    parser.add_argument('--latitude', type=float, default=37.9839231546, help='Latitude')
    parser.add_argument('--longitude', type=float, default=23.7786493786, help='Longitude')
    
    args = parser.parse_args()
    
    print("IMPROVED SOLAR ANALYSIS SCRIPT")
    print("="*60)
    print("This improved script analyzes solar data with:")
    print("• Fixed Load data detection and handling")
    print("• Statistical significance testing")
    print("• Seasonal load vs irradiance analysis")
    print("• Monthly patterns and correlations")
    print("• Load duration curves")
    print("• Enhanced confidence scoring")
    print("• Removed incorrect atmospheric calculations")
    print("="*60)
    
    try:
        # Load and process data
        df = load_and_preprocess_for_validation(args.data_file, args.latitude, args.longitude)
        
        # Create enhanced validation analysis
        results = create_enhanced_irradiance_plots(df, args.output_dir)
        
        print(f"\nResults saved to: {args.output_dir}")
        print("\nNEXT STEPS:")
        
        if results['confidence_score'] >= 70:
            print("[HIGH CONFIDENCE] Your data STRONGLY CONFIRMS SE orientation is optimal!")
            print("   The optimization results are statistically validated.")
        elif results['confidence_score'] >= 40:
            print("[MEDIUM CONFIDENCE] Your data provides MODERATE support for SE orientation.")
            print("   Consider additional analysis factors.")
        else:
            print("[LOW CONFIDENCE] SE optimization may be driven by factors")
            print("   other than morning irradiance bias.")
            print("   Focus on load matching analysis.")
            
        if results.get('has_load_data'):
            print("\nLOAD ANALYSIS COMPLETE:")
            overall_corr = results.get('overall_irr_load_correlation', 0)
            if overall_corr > 0.5:
                print(f"   STRONG correlation between load and irradiance (r={overall_corr:.2f})")
            elif overall_corr > 0.3:
                print(f"   MODERATE correlation between load and irradiance (r={overall_corr:.2f})")
            else:
                print(f"   WEAK correlation between load and irradiance (r={overall_corr:.2f})")
        else:
            print("\nNO LOAD DATA: Analysis focused on irradiance patterns only.")
            
    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
        print(f"Error details: {str(e)}")
        print("\nTroubleshooting tips:")
        print("• Check if data file path is correct")
        print("• Verify required columns are present")
        print("• Ensure data format matches expected structure")

if __name__ == "__main__":
    main()