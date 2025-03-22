# constants.py

# ===============================================
# Solar Energy Analysis Constants
# ===============================================

# Loss Factors
SHADING_LOSS = 0.03       # 3% loss due to shading
SOILING_LOSS = 0.02       # 2% loss due to soiling
REFLECTION_LOSS = 0.02    # 2% loss due to reflection
TOTAL_LOSS_FACTOR = 1 - (SHADING_LOSS + SOILING_LOSS + REFLECTION_LOSS)  # 0.93

# Nominal Operating Cell Temperature
NOCT = 45  # Nominal Operating Cell Temperature in °C

# Time Interval
TIME_INTERVAL_HOURS = 1  # Time interval between data points in hours (e.g., 1 for hourly data)

# Additional Constants (If Needed)
# You can add more constants here as your project evolves
