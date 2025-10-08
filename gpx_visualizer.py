"""
GPX File Visualizer with Elevation Contours

This application visualizes GPX files with elevation contours,
using techniques inspired by the descend-mountain repository.
"""
import os
import gpxpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dotenv import load_dotenv
from elevation_processor import get_real_elevation_data_around_coords, plot_terrain_with_gpx, integrate_contour_data
import math
from config import (GRID_SIZE, ZOOM_FACTOR, AXIS_BUFFER, 
                   CONTOUR_LEVEL_STEP, CONTOUR_ALPHA, CONTOUR_LINEWIDTH,
                   HALF_SIZE_KM, ELEVATION_CACHE_ENABLED, ELEVATION_CACHE_DIR)  # Import configuration


def load_api_keys():
    """Load API keys from environment variables"""
    load_dotenv()
    google_elevation_api_key = os.getenv('GOOGLE_ELEVATION_API_KEY')
    google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    return google_elevation_api_key, google_maps_api_key


def parse_gpx_file(file_path):
    """
    Parse GPX file and extract track points with coordinates and elevation
    """
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    
    track_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                track_points.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation,
                    'time': point.time
                })
    
    return track_points


def plot_gpx_with_contours(gpx_points, elevation_grid, LAT=None, LON=None, title="GPX Track with Elevation Contours"):
    """
    Plot GPX track overlaid on elevation contours
    """
    from config import ZOOM_FACTOR_X, ZOOM_FACTOR_Y, AXIS_BUFFER  # Import config to use zoom and buffer in this function
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if elevation_grid is not None:
        # Create contour plot
        if LAT is not None and LON is not None:
            # Create contour levels based on elevation range using configuration
            min_elev = np.nanmin(elevation_grid)
            max_elev = np.nanmax(elevation_grid)
            
            # Use CONTOUR_LEVEL_STEP from config for contour intervals
            step = CONTOUR_LEVEL_STEP  # Use the configured step
            contour_levels = np.arange(
                np.floor(min_elev / step) * step,
                np.ceil(max_elev / step) * step + step / 2.0,
                step
            )
            
            # Plot contours using configuration values
            contour = ax.contour(LON, LAT, elevation_grid, levels=contour_levels, 
                                colors='gray', alpha=CONTOUR_ALPHA, linewidths=CONTOUR_LINEWIDTH)
            ax.clabel(contour, inline=True, fontsize=8, fmt='%.0fm')
        else:
            # If no coordinate grids, just plot a basic elevation map
            im = ax.imshow(elevation_grid, extent=(0, 1, 0, 1), origin='lower', cmap='terrain', alpha=0.7)
            plt.colorbar(im, ax=ax, label='Elevation (m)')
    
    # Extract GPX coordinates for plotting
    lats = [p['latitude'] for p in gpx_points]
    lons = [p['longitude'] for p in gpx_points]
    
    # Plot GPX track
    ax.plot(lons, lats, 'r-', linewidth=2, label='GPX Track', zorder=5)
    ax.scatter(lons[0], lats[0], color='green', s=100, zorder=6, label='Start', edgecolors='black')  # Start point
    ax.scatter(lons[-1], lats[-1], color='red', s=100, zorder=6, label='End', edgecolors='black')    # End point
    
    # Calculate axis limits to magnify the view on the GPX path
    if LAT is not None and LON is not None:
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        
        # Add buffer around the path
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        
        # Use separate zoom factors for x and y axes
        lat_range = (lat_max - lat_min) * ZOOM_FACTOR_Y
        lon_range = (lon_max - lon_min) * ZOOM_FACTOR_X
        
        # Set axis limits to create a magnified view
        ax.set_xlim(lon_center - lon_range/2 - AXIS_BUFFER, lon_center + lon_range/2 + AXIS_BUFFER)
        ax.set_ylim(lat_center - lat_range/2 - AXIS_BUFFER, lat_center + lat_range/2 + AXIS_BUFFER)
    
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_elevation_profile(gpx_points, title="Elevation Profile"):
    """
    Plot elevation profile along the GPX track
    """
    # Extract elevation data
    elevations = [p['elevation'] if p['elevation'] is not None else 0 for p in gpx_points]
    distances = [0]  # Starting distance is 0
    
    # Calculate cumulative distances
    for i in range(1, len(gpx_points)):
        lat1, lon1 = gpx_points[i-1]['latitude'], gpx_points[i-1]['longitude']
        lat2, lon2 = gpx_points[i]['latitude'], gpx_points[i]['longitude']
        
        # Calculate distance using haversine formula (simplified)
        R = 6371000  # Earth's radius in meters
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        distances.append(distances[-1] + distance)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(distances, elevations, 'b-', linewidth=2)
    ax.fill_between(distances, elevations, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def calculate_vector_directions(gpx_points):
    """
    Calculate direction vectors between consecutive points in the GPX track
    """
    vectors = []
    
    for i in range(len(gpx_points) - 1):
        p1 = gpx_points[i]
        p2 = gpx_points[i + 1]
        
        # Calculate differences in lat and lon
        delta_lat = p2['latitude'] - p1['latitude']
        delta_lon = p2['longitude'] - p1['longitude']
        
        vectors.append({
            'start_lat': p1['latitude'],
            'start_lon': p1['longitude'],
            'delta_lat': delta_lat,
            'delta_lon': delta_lon
        })
    
    return vectors


def add_vectors_to_plot(ax, gpx_points, sampling_interval=None, scale_factor=None):
    """
    Add arrow vectors to an existing plot showing direction at sampled points
    """
    from config import (VECTOR_SAMPLING_INTERVAL, VECTOR_SCALE_FACTOR, VECTOR_WIDTH, 
                       VECTOR_COLOR, VECTOR_ALPHA, MAX_VECTOR_DISPLAY)
    
    # Use provided values or fallback to config
    sampling_interval = sampling_interval or VECTOR_SAMPLING_INTERVAL
    scale_factor = scale_factor or VECTOR_SCALE_FACTOR
    
    # Extract GPX coordinates for plotting
    lats = [p['latitude'] for p in gpx_points]
    lons = [p['longitude'] for p in gpx_points]
    
    # Calculate direction vectors
    vectors = calculate_vector_directions(gpx_points)
    
    # Determine which vectors to display based on sampling and max display limits
    vector_count = len(vectors)
    if vector_count > MAX_VECTOR_DISPLAY:
        # Adjust sampling interval if there would be too many vectors
        sampling_interval = max(sampling_interval, int(vector_count / MAX_VECTOR_DISPLAY))
    
    # Draw vectors at sampled points
    for i in range(0, len(vectors), sampling_interval):
        if i < len(vectors):  # Make sure we don't go out of bounds
            vector = vectors[i]
            
            # Scaled vector to make it more visible - extend it further than actual
            scaled_delta_lat = vector['delta_lat'] * scale_factor
            scaled_delta_lon = vector['delta_lon'] * scale_factor
            
            # Plot the vector as an arrow using quiver (more appropriate for vector arrows)
            ax.quiver(vector['start_lon'], vector['start_lat'], 
                     vector['delta_lon']*scale_factor, vector['delta_lat']*scale_factor,
                     angles='xy', scale_units='xy', scale=1, 
                     color=VECTOR_COLOR, alpha=VECTOR_ALPHA, width=VECTOR_WIDTH,
                     zorder=4)
    
    return ax


def plot_gpx_with_vectors(gpx_points, title="GPX Track with Direction Vectors", sampling_interval=None, scale_factor=None):
    """
    Plot GPX track with arrow vectors showing direction at sampled points
    """
    from config import (VECTOR_SAMPLING_INTERVAL, VECTOR_SCALE_FACTOR, VECTOR_WIDTH, 
                       VECTOR_COLOR, VECTOR_ALPHA, MAX_VECTOR_DISPLAY, 
                       ZOOM_FACTOR_X, ZOOM_FACTOR_Y, AXIS_BUFFER)
    
    # Use provided values or fallback to config
    sampling_interval = sampling_interval or VECTOR_SAMPLING_INTERVAL
    scale_factor = scale_factor or VECTOR_SCALE_FACTOR
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract GPX coordinates for plotting
    lats = [p['latitude'] for p in gpx_points]
    lons = [p['longitude'] for p in gpx_points]
    
    # Plot the main GPX track
    ax.plot(lons, lats, 'r-', linewidth=2, label='GPX Track', zorder=3)
    ax.scatter(lons[0], lats[0], color='green', s=100, zorder=6, label='Start', edgecolors='black')  # Start point
    ax.scatter(lons[-1], lats[-1], color='red', s=100, zorder=6, label='End', edgecolors='black')    # End point
    
    # Add vectors to the plot
    ax = add_vectors_to_plot(ax, gpx_points, sampling_interval, scale_factor)
    
    # Calculate axis limits to magnify the view on the GPX path
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    # Add buffer around the path
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    
    # Use separate zoom factors for x and y axes
    lat_range = (lat_max - lat_min) * ZOOM_FACTOR_Y
    lon_range = (lon_max - lon_min) * ZOOM_FACTOR_X
    
    # Set axis limits to create a magnified view
    ax.set_xlim(lon_center - lon_range/2 - AXIS_BUFFER, lon_center + lon_range/2 + AXIS_BUFFER)
    ax.set_ylim(lat_center - lat_range/2 - AXIS_BUFFER, lat_center + lat_range/2 + AXIS_BUFFER)
    
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def get_gpx_files():
    """
    Get all GPX files in the main directory and gpx_data subdirectory
    """
    import os
    import glob
    
    # Get GPX files from main directory
    main_gpx_files = glob.glob("*.gpx")
    
    # Get GPX files from gpx_data directory
    data_gpx_files = glob.glob("gpx_data/*.gpx")
    
    # Sort the data files to ensure consistent order
    data_gpx_files.sort()
    
    # Combine all GPX files: main file first, then sorted data files
    all_gpx_files = main_gpx_files + data_gpx_files
    
    return all_gpx_files


def main():
    # Load API keys
    elevation_api_key, maps_api_key = load_api_keys()
    
    # Get all GPX files
    gpx_files = get_gpx_files()
    print(f"Found {len(gpx_files)} GPX files: {gpx_files}")
    
    if not gpx_files:
        print("No GPX files found")
        return
    
    # Load all GPX data
    all_gpx_points = []
    for i, gpx_file in enumerate(gpx_files):
        print(f"Loading GPX data from {gpx_file}...")
        gpx_points = parse_gpx_file(gpx_file)
        print(f"Loaded {len(gpx_points)} track points from {gpx_file}")
        all_gpx_points.append(gpx_points)
    
    # Combine all track points to get bounds for elevation data
    all_lats = []
    all_lons = []
    for gpx_points in all_gpx_points:
        lats = [p['latitude'] for p in gpx_points]
        lons = [p['longitude'] for p in gpx_points]
        all_lats.extend(lats)
        all_lons.extend(lons)
    
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    print("Fetching elevation data...")
    # Use configuration values for size - other parameters are handled within the function
    elevation_grid, LAT, LON = get_real_elevation_data_around_coords(center_lat, center_lon, 
                                                                     elevation_api_key, 
                                                                     size=GRID_SIZE)
    
    print("Creating visualizations...")
    # Plot GPX with contours using the enhanced function from elevation_processor
    fig1, ax1 = plot_terrain_with_gpx(all_gpx_points[0], elevation_grid, LAT, LON, 
                                     title="GPX Track Visualization with Elevation Contours")
    
    # Save the first figure
    fig1.savefig('output/gpx_contour_visualization.png', dpi=300, bbox_inches='tight')
    
    # Plot elevation profile for first track
    fig2, ax2 = plot_elevation_profile(all_gpx_points[0], title="Elevation Profile Along Track")
    
    # Save the second figure
    fig2.savefig('output/elevation_profile.png', dpi=300, bbox_inches='tight')
    
    # Plot GPX with direction vectors (separate visualization)
    fig3, ax3 = plot_gpx_with_vectors(all_gpx_points[0], title="GPX Track with Direction Vectors")
    
    # Save the third figure
    fig3.savefig('output/gpx_vector_visualization.png', dpi=300, bbox_inches='tight')
    
    # Plot GPX with contours and vectors combined
    fig4, ax4 = plot_terrain_with_gpx(all_gpx_points[0], elevation_grid, LAT, LON, 
                                     title="GPX Track Visualization with Elevation Contours and Direction Vectors")
    # Add vectors to this plot too
    ax4 = add_vectors_to_plot(ax4, all_gpx_points[0])
    
    # Save the fourth figure
    fig4.savefig('output/gpx_contour_with_vectors.png', dpi=300, bbox_inches='tight')
    
    # NEW: Plot all hikers together with different colors and legend
    from config import HIKER_COLORS, HIKER_NAMES, SHOW_ELEVATION_BAR, ELEVATION_BAR_POSITION, HIKER_PATH_THICKNESS, X_AXIS_RANGE_FACTOR
    fig5, ax5 = plt.subplots(figsize=(14, 10))
    
    # Create contour plot background
    if elevation_grid is not None and LAT is not None and LON is not None:
        # Create contour levels based on elevation range
        min_elev = np.nanmin(elevation_grid)
        max_elev = np.nanmax(elevation_grid)
        
        # Use CONTOUR_LEVEL_STEP from config for contour intervals
        step = CONTOUR_LEVEL_STEP
        contour_levels = np.arange(
            np.floor(min_elev / step) * step,
            np.ceil(max_elev / step) * step + step / 2.0,
            step
        )
        
        # Plot filled contours for better visualization
        contourf = ax5.contourf(LON, LAT, elevation_grid, levels=contour_levels, cmap='terrain', alpha=CONTOUR_ALPHA)
        
        # Add colorbar based on configuration
        if SHOW_ELEVATION_BAR:
            if ELEVATION_BAR_POSITION == 'left':
                cbar = plt.colorbar(contourf, ax=ax5, label='Elevation (m)', shrink=0.8, pad=0.15, location='left')
            else:
                # Default to right if not specified as left
                cbar = plt.colorbar(contourf, ax=ax5, label='Elevation (m)', shrink=0.8)
        
        # Plot contour lines
        contour = ax5.contour(LON, LAT, elevation_grid, levels=contour_levels, colors='black', alpha=CONTOUR_ALPHA, linewidths=CONTOUR_LINEWIDTH)
        ax5.clabel(contour, inline=True, fontsize=8, fmt='%.0fm')
    
    # Plot each hiker's track with different colors
    for i, gpx_points in enumerate(all_gpx_points):
        lats = [p['latitude'] for p in gpx_points]
        lons = [p['longitude'] for p in gpx_points]
        
        # Get color for this hiker (cycling through available colors if more hikers than colors)
        color_idx = i % len(HIKER_COLORS)
        hiker_color = HIKER_COLORS[color_idx]
        hiker_name = HIKER_NAMES[i] if i < len(HIKER_NAMES) else f'Hiker {i+1}'
        
        # Plot the hiker's track with configurable thickness
        ax5.plot(lons, lats, color=hiker_color, linewidth=HIKER_PATH_THICKNESS, label=hiker_name, zorder=3)
        
        # Plot start and end points
        ax5.scatter(lons[0], lats[0], color=hiker_color, s=100, zorder=6, 
                   label=f'{hiker_name} Start', edgecolors='black', marker='o')
        ax5.scatter(lons[-1], lats[-1], color=hiker_color, s=100, zorder=6, 
                   label=f'{hiker_name} End', edgecolors='black', marker='s')
        
        # Add vectors for each hiker
        from config import VECTOR_SCALE_FACTOR
        ax5 = add_vectors_to_plot(ax5, gpx_points, scale_factor=VECTOR_SCALE_FACTOR*1000)
    
    # Calculate axis limits to magnify the view on all GPX paths
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)
    
    # Add buffer around the path
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    
    # Import required config variables
    from config import ZOOM_FACTOR_X, ZOOM_FACTOR_Y, AXIS_BUFFER, X_AXIS_RANGE_FACTOR
    
    # Use separate zoom factors for x and y axes
    lat_range = (lat_max - lat_min) * ZOOM_FACTOR_Y
    lon_range = (lon_max - lon_min) * ZOOM_FACTOR_X
    
    # Apply X-axis range factor configuration
    lon_range = lon_range * X_AXIS_RANGE_FACTOR
    
    # Set axis limits to create a magnified view
    ax5.set_xlim(lon_center - lon_range/2 - AXIS_BUFFER, lon_center + lon_range/2 + AXIS_BUFFER)
    ax5.set_ylim(lat_center - lat_range/2 - AXIS_BUFFER, lat_center + lat_range/2 + AXIS_BUFFER)
    
    ax5.set_title("Multi-Hiker Visualization with Elevation Contours")
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside plot
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the fifth figure
    fig5.savefig('output/multi_hiker_visualization.png', dpi=300, bbox_inches='tight')
    
    print("Visualizations saved to output/ directory:")
    print("- gpx_contour_visualization.png")
    print("- elevation_profile.png")
    print("- gpx_vector_visualization.png")
    print("- gpx_contour_with_vectors.png")
    print("- multi_hiker_visualization.png")


if __name__ == "__main__":
    main()