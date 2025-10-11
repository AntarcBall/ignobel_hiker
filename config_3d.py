# Configuration for 3D GPX Visualization
# Settings specifically for 3D terrain visualization

# Display settings
FIGURE_SIZE_3D = (15, 12)  # Width, height in inches for matplotlib
DPI_3D = 100  # DPI for 3D plots

# Terrain visualization settings
TERRAIN_COLORMAP = 'terrain'  # Matplotlib colormap for terrain
TERRAIN_ALPHA = 0.8  # Transparency of terrain surface
SURFACE_STRIDE = 2  # Stride for surface rendering (detail level)

# Path visualization settings
PATH_3D_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
PATH_3D_LINEWIDTH = 3  # Width of path lines in 3D
PATH_MARKER_SIZE = 4  # Size of path markers in 3D

# View settings
VIEW_ELEVATION_ANGLE = 20  # Elevation angle for 3D view (degrees)
VIEW_AZIMUTH_ANGLE = 45  # Azimuth angle for 3D view (degrees)

# Vertical exaggeration
VERTICAL_EXAGGERATION = 2.0  # Factor to exaggerate elevation differences

# Marker settings
SHOW_START_END_MARKERS = True  # Whether to show start and end markers
SHOW_GROUND_PROJECTION = False  # Whether to show ground projection of paths
PROJECTION_LINESTYLE = '--'  # Style of projection lines
PROJECTION_ALPHA = 0.5  # Transparency of projection lines

# Output format
OUTPUT_FORMAT = 'png'  # Output format: 'png' for static image, 'html' for interactive