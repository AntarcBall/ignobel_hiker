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


def main():
    # Load API keys
    elevation_api_key, maps_api_key = load_api_keys()
    
    # Specify the GPX file to visualize
    gpx_file_path = "Oct_8,_2025_7_12_58_AM_Hiking_Ascent.gpx"
    
    print("Loading GPX data...")
    gpx_points = parse_gpx_file(gpx_file_path)
    print(f"Loaded {len(gpx_points)} track points")
    
    if not gpx_points:
        print("No track points found in GPX file")
        return
    
    # Get bounds for the track
    lats = [p['latitude'] for p in gpx_points]
    lons = [p['longitude'] for p in gpx_points]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    print("Fetching elevation data...")
    # Use configuration values for size - other parameters are handled within the function
    elevation_grid, LAT, LON = get_real_elevation_data_around_coords(center_lat, center_lon, 
                                                                     elevation_api_key, 
                                                                     size=GRID_SIZE)
    
    print("Creating visualizations...")
    # Plot GPX with contours using the enhanced function from elevation_processor
    fig1, ax1 = plot_terrain_with_gpx(gpx_points, elevation_grid, LAT, LON, 
                                     title="GPX Track Visualization with Elevation Contours")
    
    # Save the first figure
    fig1.savefig('output/gpx_contour_visualization.png', dpi=300, bbox_inches='tight')
    
    # Plot elevation profile
    fig2, ax2 = plot_elevation_profile(gpx_points, title="Elevation Profile Along Track")
    
    # Save the second figure
    fig2.savefig('output/elevation_profile.png', dpi=300, bbox_inches='tight')
    
    print("Visualizations saved to output/ directory:")
    print("- gpx_contour_visualization.png")
    print("- elevation_profile.png")


if __name__ == "__main__":
    main()