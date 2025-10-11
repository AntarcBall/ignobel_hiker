"""
Configuration file for velocity analysis parameters
"""
# Stop detection parameters
MIN_STOP_SPEED_THRESHOLD = 1.0  # Minimum speed to consider as stopped (m/s)
MIN_STOP_DURATION = 10  # Minimum duration to consider as a stop (seconds)

# Leader score calculation parameters
LEADER_TIME_INTERVAL = 10  # Time interval for interpolation in seconds

# Filtering parameters
FILTER_START_TIME = "2025-10-07T22:12:12Z"  # Start time for filtering

# Velocity histogram parameters
VELOCITY_HISTOGRAM_BINS = 30

# Table display parameters
SUMMARY_TABLE_FIGURE_WIDTH = 6
SUMMARY_TABLE_FIGURE_HEIGHT = 5

# Subscores for comprehensive score (weights)
TORTUOSITY_WEIGHT = 1
STOP_FRACTION_WEIGHT = 1
LEADER_SCORE_WEIGHT = 1