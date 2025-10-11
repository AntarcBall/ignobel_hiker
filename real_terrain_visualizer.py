#!/usr/bin/env python3
"""
Real Terrain Visualizer
Fetches actual elevation data based on GPX file coordinates and visualizes it in 3D
"""

import os
import sys
import gpxpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dotenv import load_dotenv
import requests
import pickle
from scipy.interpolate import griddata
import math

# Load environment variables
load_dotenv()

def bbox_from_center(lat, lon, half_size_km=3.0):
    """Calculate boundary box (half_size_km from center in each direction)"""
    # Simple spherical approximation (1 degree latitude ≈ 111320 m)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
    dlat = (half_size_km * 1000.0) / meters_per_deg_lat
    dlon = (half_size_km * 1000.0) / meters_per_deg_lon
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)

def make_grid(min_lat, min_lon, max_lat, max_lon, nx=100, ny=100):
    """Generate grid points for the area"""
    lats = np.linspace(min_lat, max_lat, ny)
    lons = np.linspace(min_lon, max_lon, nx)
    LON, LAT = np.meshgrid(lons, lats)
    return LAT, LON

def fetch_elevations(latlon_points, api_key, batch=256, timeout=30):
    """Fetch elevation data from Google Elevation API"""
    elevations = []
    for i in range(0, len(latlon_points), batch):
        chunk = latlon_points[i:i+batch]
        locations = "|".join(f"{lat:.7f},{lon:.7f}" for lat, lon in chunk)
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={api_key}"
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK":
            raise RuntimeError(f"Elevation API error: {data}")
        for res in data["results"]:
            elevations.append(res["elevation"])
    return elevations

def get_cache_filename(lat, lon, size, half_size_km):
    """Generate a cache filename based on coordinates and parameters"""
    return f"cache/elevation_data_{lat:.6f}_{lon:.6f}_{size}x{size}_{half_size_km}km.pkl"

def get_cached_elevation_data(lat, lon, size, half_size_km):
    """Try to load cached elevation data"""
    cache_file = get_cache_filename(lat, lon, size, half_size_km)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded cached elevation data from {cache_file}")
            return data['Z'], data['LAT'], data['LON']
        except Exception as e:
            print(f"Error loading cached data: {e}")
    
    return None

def cache_elevation_data(lat, lon, size, half_size_km, Z, LAT, LON):
    """Cache elevation data to file"""
    cache_file = get_cache_filename(lat, lon, size, half_size_km)
    
    # Create cache directory if it doesn't exist
    os.makedirs('cache', exist_ok=True)
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'Z': Z, 'LAT': LAT, 'LON': LON}, f)
        print(f"Cached elevation data to {cache_file}")
    except Exception as e:
        print(f"Error caching data: {e}")

def get_real_elevation_data_around_coords(lat, lon, api_key, size=100, half_size_km=3.0):
    """
    Fetch real elevation data for the coordinates from Google Elevation API
    """
    # First, try to load from cache
    cached_result = get_cached_elevation_data(lat, lon, size, half_size_km)
    if cached_result is not None:
        Z, LAT, LON = cached_result
        return Z, LAT, LON

    if not api_key or api_key in ["YOUR_API_KEY", "YOUR_API_KEY_HERE", "YOUR_ELEVATION_API_KEY_HERE", ""]:
        print("No valid API key provided.")
        sys.exit(1)

    try:
        # Calculate bounding box around the center point
        min_lat, min_lon, max_lat, max_lon = bbox_from_center(lat, lon, half_size_km=half_size_km)

        # Generate grid
        LAT, LON = make_grid(min_lat, min_lon, max_lat, max_lon, nx=size, ny=size)
        ny, nx = LAT.shape

        # Get elevation points
        points = [(float(LAT.ravel()[i]), float(LON.ravel()[i])) for i in range(LAT.size)]
        Z_list = fetch_elevations(points, api_key, batch=256)

        Z = np.array(Z_list, dtype=float).reshape((ny, nx))

        print(f"Fetched real elevation data for area around {lat}, {lon}")
        print(f"Elevation range: {np.nanmin(Z):.2f}m to {np.nanmax(Z):.2f}m")
        
        # Cache the data
        cache_elevation_data(lat, lon, size, half_size_km, Z, LAT, LON)
        
        return Z, LAT, LON

    except Exception as e:
        print(f"Error fetching real elevation data: {e}")
        sys.exit(1)

def get_gpx_range():
    """Get coordinate ranges from GPX files"""
    gpx_dir = 'gpx_data'
    gpx_files = []
    if os.path.exists(gpx_dir):
        for file in os.listdir(gpx_dir):
            if file.lower().endswith('.gpx'):
                gpx_files.append(os.path.join(gpx_dir, file))

    print(f'Found {len(gpx_files)} GPX files')
    all_lats = []
    all_lons = []

    for gpx_file in gpx_files:
        print(f'Processing {gpx_file}')
        with open(gpx_file, 'r') as file:
            gpx = gpxpy.parse(file)
        
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    all_lats.append(point.latitude)
                    all_lons.append(point.longitude)

    if all_lats and all_lons:
        lat_center = np.mean(all_lats)
        lon_center = np.mean(all_lons)
        print(f'Latitude range: {min(all_lats):.6f} to {max(all_lats):.6f}')
        print(f'Longitude range: {min(all_lons):.6f} to {max(all_lons):.6f}')
        print(f'Center coordinates: {lat_center:.6f}, {lon_center:.6f}')
        return min(all_lats), max(all_lats), min(all_lons), max(all_lons), lat_center, lon_center
    else:
        print('No valid coordinates found in GPX files')
        sys.exit(1)

def plot_3d_terrain_with_contours(elevation_grid, LAT, LON, vertical_exaggeration=2.0):
    """
    Create 3D visualization with terrain surface and contour lines
    """
    # Set backend based on DISPLAY availability
    import matplotlib
    try:
        if os.environ.get('DISPLAY'):
            matplotlib.use('TkAgg')  # Use GUI backend if display is available
        else:
            matplotlib.use('Agg')  # Use non-interactive backend
    except:
        matplotlib.use('Agg')

    # Re-import pyplot after backend change
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Create 3D terrain surface
    X = LON
    Y = LAT
    Z = elevation_grid * vertical_exaggeration

    # Plot terrain surface
    surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8, linewidth=0, antialiased=True)

    # Add contour lines projected at the bottom
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 15)
    for level in levels:
        ax.contour(X, Y, Z, levels=[level], colors='black', alpha=0.4, linewidths=0.5, 
                  offset=np.nanmin(Z)-10)

    # Set labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Visualization of Actual Elevation Data with GPX-Based Coordinates')

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')

    # Adjust view angle
    ax.view_init(elev=20, azim=45)

    # Set pane properties for cleaner background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    plt.tight_layout()
    return fig, ax

def main():
    # Get API key from environment
    elevation_api_key = os.getenv('ELEVATION_API_KEY')
    
    if not elevation_api_key:
        print("Error: ELEVATION_API_KEY not found in .env file")
        sys.exit(1)
    
    # Get coordinate ranges from GPX files
    print("Analyzing GPX files for coordinate ranges...")
    min_lat, max_lat, min_lon, max_lon, center_lat, center_lon = get_gpx_range()
    
    # Fetch elevation data based on GPX coordinates
    print(f"Fetching elevation data for center: {center_lat:.6f}, {center_lon:.6f}")
    elevation_grid, LAT, LON = get_real_elevation_data_around_coords(
        center_lat, center_lon, elevation_api_key, size=100, half_size_km=3.0
    )
    
    # Create 3D visualization
    print("Creating 3D visualization...")
    fig, ax = plot_3d_terrain_with_contours(elevation_grid, LAT, LON)
    
    # Save the visualization
    os.makedirs('output', exist_ok=True)
    output_path = 'output/real_terrain_3d.png'
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Show the window if possible (depends on backend)
    try:
        plt.show()
    except:
        print("Could not display GUI window. Image saved to output directory.")

if __name__ == "__main__":
    main()