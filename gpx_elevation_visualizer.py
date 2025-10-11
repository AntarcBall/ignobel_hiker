#!/usr/bin/env python3
"""
GPX Elevation Visualizer
Fetches actual elevation data based on GPX file coordinates and creates 3D visualizations
"""

import os
import sys
import numpy as np
import matplotlib
# Detect if display is available to set appropriate backend
try:
    if os.environ.get('DISPLAY'):
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
except:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gpxpy
import requests
import pickle
from dotenv import load_dotenv
from scipy.interpolate import griddata
import math

# Load environment variables
load_dotenv()

def bbox_from_center(lat, lon, half_size_km=3.0):
    """Calculate boundary box (half_size_km from center in each direction)"""
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
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    return f"{cache_dir}/elevation_data_{lat:.6f}_{lon:.6f}_{size}x{size}_{half_size_km}km.pkl"

def get_cached_elevation_data(lat, lon, size=140, half_size_km=3.0):
    """Try to load cached elevation data"""
    cache_file = get_cache_filename(lat, lon, size, half_size_km)
    
    if os.path.exists(cache_file):
        try:
            # Check if GPX files are newer than cache
            gpx_dir = 'gpx_data'
            if os.path.exists(gpx_dir):
                cache_mtime = os.path.getmtime(cache_file)
                for file in os.listdir(gpx_dir):
                    if file.lower().endswith('.gpx'):
                        gpx_file = os.path.join(gpx_dir, file)
                        if os.path.getmtime(gpx_file) > cache_mtime:
                            print("GPX files are newer than cache, fetching fresh data...")
                            return None  # Force refresh if gpx files are newer
            
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
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'Z': Z, 'LAT': LAT, 'LON': LON}, f)
        print(f"Cached elevation data to {cache_file}")
    except Exception as e:
        print(f"Error caching data: {e}")

def get_real_elevation_data_around_coords(lat, lon, api_key, size=140, half_size_km=3.0):
    """
    Fetch real elevation data for the coordinates from Google Elevation API
    """
    # First, try to load from cache
    cached_result = get_cached_elevation_data(lat, lon, size, half_size_km)
    if cached_result is not None:
        Z, LAT, LON = cached_result
        return Z, LAT, LON

    if not api_key or api_key in ["YOUR_API_KEY", "YOUR_API_KEY_HERE", ""]:
        raise ValueError("No valid API key provided. Please set ELEVATION_API_KEY in your .env file.")
        
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
        cache_elevation_data(lat, lon, size, half_size_km, Z, LAT, LON)
        
        return Z, LAT, LON

    except Exception as e:
        print(f"Error fetching real elevation data: {e}")
        raise

def get_gpx_files():
    """Get all GPX files from the gpx_data directory"""
    gpx_dir = 'gpx_data'
    gpx_files = []
    
    if os.path.exists(gpx_dir):
        for file in os.listdir(gpx_dir):
            if file.lower().endswith('.gpx'):
                gpx_files.append(os.path.join(gpx_dir, file))
    
    return gpx_files

def parse_gpx_file(gpx_file_path):
    """Parse GPX file and extract track points"""
    with open(gpx_file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation,
                    'time': point.time
                })
    
    return points

def plot_3d_contours_with_gpx(elevation_grid, LAT, LON, all_gpx_points, vertical_exaggeration=2.0):
    """
    Create 3D visualization with both contour lines and GPX tracks
    """
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates from all GPX tracks to set view range
    all_lats = [p['latitude'] for gpx in all_gpx_points for p in gpx]
    all_lons = [p['longitude'] for gpx in all_gpx_points for p in gpx]
    
    if all_lats and all_lons:
        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_min, lat_max = min(all_lats), max(all_lats)
        
        # Add padding
        lon_padding = (lon_max - lon_min) * 0.1
        lat_padding = (lat_max - lat_min) * 0.1
    
        ax.set_xlim(lon_min - lon_padding, lon_max + lon_padding)
        ax.set_ylim(lat_min - lat_padding, lat_max + lat_padding)
    
    # Use actual coordinates for the elevation grid
    X = LON
    Y = LAT
    Z = elevation_grid * vertical_exaggeration
    
    # Plot contours
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 15)
    for level in levels:
        ax.contour(X, Y, Z, levels=[level], colors='black', alpha=0.4, linewidths=0.5)
    
    # Plot terrain surface with transparency
    surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6, linewidth=0, antialiased=True)
    
    # Add GPX tracks to the 3D plot
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, gpx_points in enumerate(all_gpx_points):
        lats = [p['latitude'] for p in gpx_points]
        lons = [p['longitude'] for p in gpx_points]
        elevs_gpx = []
        
        # Interpolate GPX elevations to 3D coordinates
        for lat, lon in zip(lats, lons):
            # Find the corresponding elevation from the grid
            points = np.column_stack((LAT.ravel(), LON.ravel()))
            values = elevation_grid.ravel()
            interp_elev = griddata(points, values, (lat, lon), method='linear')
            
            if np.isnan(interp_elev):
                # Use GPX elevation if grid interpolation fails
                gpx_idx = next((j for j, p in enumerate(gpx_points) if p['latitude'] == lat and p['longitude'] == lon), 0)
                interp_elev = gpx_points[gpx_idx]['elevation'] if gpx_points[gpx_idx]['elevation'] else np.nanmin(elevation_grid)
            
            elevs_gpx.append(interp_elev * vertical_exaggeration)
        
        color = colors[i % len(colors)]
        ax.plot(lons, lats, elevs_gpx, color=color, linewidth=2.5, label=f'GPX Track {i+1}')
        
        # Mark start and end points
        if len(lons) > 0:
            ax.scatter([lons[0]], [lats[0]], [elevs_gpx[0]], color=color, s=100, 
                      marker='o', edgecolors='black', linewidth=2, label=f'Start {i+1}')
            ax.scatter([lons[-1]], [lats[-1]], [elevs_gpx[-1]], color=color, s=100, 
                      marker='s', edgecolors='black', linewidth=2, label=f'End {i+1}')
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Elevation Contours with GPX Tracks')
    
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
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')
    
    # Add legend if there are GPX tracks
    if all_gpx_points:
        ax.legend(loc='upper left')
    
    # Adjust view angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig, ax

def main():
    """Main function to create visualization from GPX files"""
    # Load API key from environment
    api_key = os.getenv('GOOGLE_ELEVATION_API_KEY') or os.getenv('ELEVATION_API_KEY')
    
    if not api_key or api_key in ["YOUR_API_KEY", "YOUR_API_KEY_HERE", ""]:
        print("Error: No valid API key found. Please set ELEVATION_API_KEY in your .env file.")
        sys.exit(1)
    
    print("Loading GPX files...")
    gpx_files = get_gpx_files()
    print(f"Found {len(gpx_files)} GPX files")
    
    if not gpx_files:
        print("No GPX files found in gpx_data directory.")
        sys.exit(1)
    
    # Parse all GPX files
    all_gpx_points = []
    all_lats = []
    all_lons = []
    
    for gpx_file in gpx_files:
        print(f"Parsing {gpx_file}...")
        gpx_points = parse_gpx_file(gpx_file)
        all_gpx_points.append(gpx_points)
        
        # Collect all lat/lon values to determine center
        for point in gpx_points:
            all_lats.append(point['latitude'])
            all_lons.append(point['longitude'])
    
    if not all_lats or not all_lons:
        print("No valid coordinates found in GPX files.")
        sys.exit(1)
    
    # Calculate center coordinates
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    print(f"Center coordinates: {center_lat:.6f}, {center_lon:.6f}")
    print(f"Latitude range: {min(all_lats):.6f} to {max(all_lats):.6f}")
    print(f"Longitude range: {min(all_lons):.6f} to {max(all_lons):.6f}")
    
    # Fetch elevation data
    print("Fetching real elevation data...")
    try:
        elevation_grid, LAT, LON = get_real_elevation_data_around_coords(
            center_lat, center_lon, api_key, size=140, half_size_km=3.0
        )
    except Exception as e:
        print(f"Failed to fetch elevation data: {e}")
        sys.exit(1)
    
    # Create visualization
    print("Creating 3D visualization...")
    fig, ax = plot_3d_contours_with_gpx(elevation_grid, LAT, LON, all_gpx_points)
    
    # Save the visualization
    os.makedirs('output', exist_ok=True)
    output_path = 'output/gpx_elevation_3d.png'
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()