# Configuration for GPX Visualization
# Settings for controlling the visualization view and data processing

# View settings
ZOOM_FACTOR = 1  # Reduce the axis range to magnify the path (0.25 = 1/4 of original view)
ZOOM_FACTOR_X = 3  # Separate zoom factor for x-axis (longitude)
ZOOM_FACTOR_Y = 1  # Separate zoom factor for y-axis (latitude)
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

# Vector visualization settings
VECTOR_SAMPLING_INTERVAL = 3  # Plot vector every N points
VECTOR_SCALE_FACTOR = 5  # Scale factor for vector lengths to make them visible
VECTOR_WIDTH = 0.003  # Width of the arrows
VECTOR_COLOR = 'blue'  # Color of the vectors
VECTOR_ALPHA = 0.7  # Transparency of the vectors
MAX_VECTOR_DISPLAY = 20  # Maximum number of vectors to display to avoid overcrowding