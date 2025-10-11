#!/usr/bin/env python3
"""
Script to visualize actual terrain data based on GPX file coordinates
Fetches real elevation data using API keys from .env, caches it, and creates 3D visualization
"""
import os
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

def get_gpx_bounds():
    """Extract coordinate bounds from GPX files"""
    gpx_dir = 'gpx_data'
    gpx_files = []
    
    if os.path.exists(gpx_dir):
        for file in os.listdir(gpx_dir):
            if file.lower().endswith('.gpx'):
                gpx_files.append(os.path.join(gpx_dir, file))
    
    all_lats = []
    all_lons = []
    all_elevs = []
    
    for gpx_file in gpx_files:
        with open(gpx_file, 'r') as file:
            gpx = gpxpy.parse(file)
        
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    all_lats.append(point.latitude)
                    all_lons.append(point.longitude)
                    if point.elevation is not None:
                        all_elevs.append(point.elevation)
    
    if not all_lats:
        raise ValueError("No coordinates found in GPX files")
    
    return {
        'lat_min': min(all_lats),
        'lat_max': max(all_lats),
        'lon_min': min(all_lons),
        'lon_max': max(all_lons),
        'elev_min': min(all_elevs) if all_elevs else None,
        'elev_max': max(all_elevs) if all_elevs else None
    }

def bbox_from_center(lat, lon, half_size_km=3.0):
    """Calculate boundary box from center point"""
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
    dlat = (half_size_km * 1000.0) / meters_per_deg_lat
    dlon = (half_size_km * 1000.0) / meters_per_deg_lon
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)

def make_grid(min_lat, min_lon, max_lat, max_lon, nx=140, ny=140):
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
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    return f"{cache_dir}/elevation_data_{lat:.6f}_{lon:.6f}_{size}x{size}_{half_size_km}km.pkl"

def get_real_elevation_data_around_coords(lat, lon, api_key, size=140, half_size_km=3.0):
    """Fetch or load cached real elevation data for the coordinates"""
    # Try to load from cache first
    cache_file = get_cache_filename(lat, lon, size, half_size_km)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded cached elevation data from {cache_file}")
            return data['Z'], data['LAT'], data['LON']
        except Exception as e:
            print(f"Error loading cached data: {e}")
    
    if not api_key or api_key in ["YOUR_API_KEY", "YOUR_API_KEY_HERE", ""]:
        raise ValueError("No valid API key provided")

    try:
        # Calculate bounding box around the center point
        min_lat, min_lon, max_lat, max_lon = bbox_from_center(lat, lon, half_size_km)

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
        with open(cache_file, 'wb') as f:
            pickle.dump({'Z': Z, 'LAT': LAT, 'LON': LON}, f)
        print(f"Cached elevation data to {cache_file}")
        
        return Z, LAT, LON

    except Exception as e:
        print(f"Error fetching real elevation data: {e}")
        raise

def visualize_terrain_with_contours(Z, LAT, LON, vertical_exaggeration=2.0):
    """Create 3D visualization with terrain surface and contour lines"""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Apply vertical exaggeration
    Z_3d = Z * vertical_exaggeration

    # Plot the terrain surface
    surf = ax.plot_surface(LON, LAT, Z_3d, cmap='terrain', alpha=0.8, 
                          linewidth=0, antialiased=True, shade=True)

    # Add contour lines projected at the bottom
    levels = np.linspace(np.min(Z_3d), np.max(Z_3d), 15)
    for level in levels[::2]:  # Show every other contour for clarity
        ax.contour(LON, LAT, Z_3d, levels=[level], colors='black', 
                  alpha=0.4, linewidths=0.5, offset=np.min(Z_3d)-5)

    # Set labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Visualization of Actual Terrain with Contour Lines')

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
    print("Analyzing GPX files for coordinate ranges...")
    bounds = get_gpx_bounds()
    
    print(f"GPX coordinate ranges:")
    print(f"  Latitude: {bounds['lat_min']:.6f} to {bounds['lat_max']:.6f}")
    print(f"  Longitude: {bounds['lon_min']:.6f} to {bounds['lon_max']:.6f}")
    if bounds['elev_min'] is not None:
        print(f"  Elevation: {bounds['elev_min']:.2f}m to {bounds['elev_max']:.2f}m")
    
    # Calculate center point
    center_lat = (bounds['lat_min'] + bounds['lat_max']) / 2
    center_lon = (bounds['lon_min'] + bounds['lon_max']) / 2
    
    print(f"\nCenter coordinates: {center_lat:.6f}, {center_lon:.6f}")
    
    # Load API key from environment
    api_key = os.getenv('GOOGLE_ELEVATION_API_KEY') or os.getenv('ELEVATION_API_KEY')
    if not api_key or api_key == "YOUR_ELEVATION_API_KEY_HERE":
        raise ValueError("Please set a valid ELEVATION_API_KEY in .env file")
    
    print("Fetching real elevation data...")
    elevation_grid, LAT, LON = get_real_elevation_data_around_coords(
        center_lat, center_lon, api_key, size=140
    )
    
    print("Creating 3D visualization...")
    fig, ax = visualize_terrain_with_contours(elevation_grid, LAT, LON)
    
    # Save the visualization
    os.makedirs('output', exist_ok=True)
    output_path = 'output/actual_terrain_with_contours.png'
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Show the window
    plt.show()

if __name__ == "__main__":
    main()