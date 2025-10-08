# Configuration for GPX Visualization
# Settings for controlling the visualization view and data processing

# View settings
ZOOM_FACTOR = 1  # Reduce the axis range to magnify the path (0.25 = 1/4 of original view)
ZOOM_FACTOR_X = 4  # Separate zoom factor for x-axis (longitude)
ZOOM_FACTOR_Y = 1  # Separate zoom factor for y-axis (latitude)
AXIS_BUFFER = 0.001  # Buffer around the path in decimal degrees

# Elevation data settings
GRID_SIZE = 140  # Size of elevation grid (GRID_SIZE x GRID_SIZE)
HALF_SIZE_KM = 3.0  # Half size of area to fetch elevation data (in km)

# Contour visualization settings
CONTOUR_LEVEL_STEP = 4.0  # Elevation step between contour lines (in meters)
CONTOUR_ALPHA = 0.8  # Transparency of contour lines
CONTOUR_LINEWIDTH = 0.15  # Width of contour lines

# Caching settings
ELEVATION_CACHE_ENABLED = True  # Whether to cache elevation data
ELEVATION_CACHE_DIR = "cache"  # Directory for storing cached elevation data

# Vector visualization settings
VECTOR_SAMPLING_INTERVAL = 3  # Plot vector every N points
VECTOR_SCALE_FACTOR = 0.005  # Scale factor for vector lengths to make them visible
VECTOR_WIDTH = 0.001  # Width of the arrows
VECTOR_COLOR = 'black'  # Color of the vectors
VECTOR_ALPHA = 0.5  # Transparency of the vectors
MAX_VECTOR_DISPLAY = 20  # Maximum number of vectors to display to avoid overcrowding

# Multiple hikers visualization settings
HIKER_COLORS = ['yellow', 'red','blue',]  # Colors for different hikers
HIKER_NAMES = ['M', 'H','Z']  # Names for legend

# Visualization enhancement settings
SHOW_ELEVATION_BAR = True  # Whether to show the elevation color bar
ELEVATION_BAR_POSITION = 'left'  # Position of elevation color bar ('left', 'right', 'bottom', 'top')
HIKER_PATH_THICKNESS = 2  # Thickness of hiker path lines
X_AXIS_RANGE_FACTOR = 1  # Factor to control x-axis range (0.3 means 30% of original, with 35% cut from each side)