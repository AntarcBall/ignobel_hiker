# Configuration for GPX Visualization
# Settings for controlling the visualization view and data processing

# View settings
ZOOM_FACTOR = 0.5  # Reduce the axis range to magnify the path (0.25 = 1/4 of original view)
AXIS_BUFFER = 0.001  # Buffer around the path in decimal degrees

# Elevation data settings
GRID_SIZE = 100  # Size of elevation grid (GRID_SIZE x GRID_SIZE)
HALF_SIZE_KM = 3.0  # Half size of area to fetch elevation data (in km)

# Contour visualization settings
CONTOUR_LEVEL_STEP = 5.0  # Elevation step between contour lines (in meters)
CONTOUR_ALPHA = 0.6  # Transparency of contour lines
CONTOUR_LINEWIDTH = 0.8  # Width of contour lines

# Caching settings
ELEVATION_CACHE_ENABLED = True  # Whether to cache elevation data
ELEVATION_CACHE_DIR = "cache"  # Directory for storing cached elevation data