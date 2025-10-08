"""
Elevation Processing Module
This module handles real elevation data integration from APIs and contour visualization
based on techniques from the descend-mountain repository.
"""
import numpy as np
import math
import requests
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import pickle
from config import ELEVATION_CACHE_ENABLED, ELEVATION_CACHE_DIR, HALF_SIZE_KM


def bbox_from_center(lat, lon, half_size_km=5.0):
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
    return f"{ELEVATION_CACHE_DIR}/elevation_data_{lat:.6f}_{lon:.6f}_{size}x{size}_{half_size_km}km.pkl"


def get_cached_elevation_data(lat, lon, size, half_size_km):
    """Try to load cached elevation data"""
    if not ELEVATION_CACHE_ENABLED:
        return None
    
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
    if not ELEVATION_CACHE_ENABLED:
        return
    
    cache_file = get_cache_filename(lat, lon, size, half_size_km)
    
    # Create cache directory if it doesn't exist
    os.makedirs(ELEVATION_CACHE_DIR, exist_ok=True)
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'Z': Z, 'LAT': LAT, 'LON': LON}, f)
        print(f"Cached elevation data to {cache_file}")
    except Exception as e:
        print(f"Error caching data: {e}")


def get_real_elevation_data_around_coords(lat, lon, api_key, size=100):
    """
    Fetch real elevation data for the coordinates from Google Elevation API
    """
    # First, try to load from cache
    cached_result = get_cached_elevation_data(lat, lon, size, HALF_SIZE_KM)
    if cached_result is not None:
        Z, LAT, LON = cached_result
        return Z, LAT, LON

    if not api_key or api_key in ["YOUR_API_KEY", "YOUR_API_KEY_HERE", ""]:
        # Return synthetic data if no valid API key
        print("No valid API key provided, using synthetic terrain...")
        Z = generate_realistic_terrain(size, min_elevation=0.0, max_elevation=1000.0)
        cache_elevation_data(lat, lon, size, HALF_SIZE_KM, Z, None, None)
        return Z, None, None

    try:
        # Calculate bounding box around the center point
        min_lat, min_lon, max_lat, max_lon = bbox_from_center(lat, lon, half_size_km=HALF_SIZE_KM)  # Configurable area

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
        cache_elevation_data(lat, lon, size, HALF_SIZE_KM, Z, LAT, LON)
        
        return Z, LAT, LON

    except Exception as e:
        print(f"Error fetching real elevation data: {e}")
        print("Falling back to synthetic terrain...")
        Z = generate_realistic_terrain(size, min_elevation=0.0, max_elevation=1000.0)
        cache_elevation_data(lat, lon, size, HALF_SIZE_KM, Z, None, None)
        return Z, None, None


def generate_realistic_terrain(size: int, min_elevation: float = 0.0, max_elevation: float = 1000.0) -> np.ndarray:
    """
    Generate a more realistic terrain using a combination of different noise functions
    to simulate real-world topography.
    """
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)

    # Create a more complex terrain using multiple noise functions
    Z = np.zeros((size, size))

    # Base terrain with multiple frequency components
    for i in range(1, 5):
        Z += np.sin(X * i / 2) * np.cos(Y * i / 2) / i

    # Add some mountain peaks
    for cx, cy in [(size//3, size//3), (2*size//3, 2*size//3), (size//2, size//4)]:
        peak = np.exp(-((X - cx/size*4*np.pi)**2 + (Y - cy/size*4*np.pi)**2) / 2)
        Z += peak * 0.8

    # Normalize to the desired elevation range
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))  # Normalize to [0, 1]
    Z = Z * (max_elevation - min_elevation) + min_elevation  # Scale to desired range

    return Z


def interpolate_gpx_to_grid(gpx_points, LAT, LON):
    """
    Interpolate GPX track points onto the elevation grid
    """
    if LAT is None or LON is None:
        return None
    
    # Extract coordinates from GPX points
    lats = np.array([p['latitude'] for p in gpx_points])
    lons = np.array([p['longitude'] for p in gpx_points])
    elevs = np.array([p['elevation'] if p['elevation'] is not None else 0 for p in gpx_points])
    
    # Prepare grid points
    grid_points = np.column_stack((LAT.ravel(), LON.ravel()))
    gpx_coords = np.column_stack((lats, lons))
    
    # Interpolate GPX elevations onto grid
    interpolated_elevations = griddata(gpx_coords, elevs, (LAT, LON), method='linear', fill_value=np.nan)
    
    return interpolated_elevations


def plot_terrain_with_gpx(gpx_points, elevation_grid, LAT=None, LON=None, title="GPX Track with Elevation Contours"):
    """
    Plot GPX track overlaid on elevation contours using techniques from the referenced repository
    """
    from config import ZOOM_FACTOR, AXIS_BUFFER, CONTOUR_LEVEL_STEP, CONTOUR_ALPHA, CONTOUR_LINEWIDTH  # Import config at function level to avoid circular imports
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    if elevation_grid is not None and LAT is not None and LON is not None:
        # Create contour levels based on elevation range
        min_elev = np.nanmin(elevation_grid)
        max_elev = np.nanmax(elevation_grid)
        
        # Use the configured contour level step
        step = CONTOUR_LEVEL_STEP
        contour_levels = np.arange(
            np.floor(min_elev / step) * step,
            np.ceil(max_elev / step) * step + step / 2.0,
            step
        )
        
        # Plot filled contours for better visualization
        contourf = ax.contourf(LON, LAT, elevation_grid, levels=contour_levels, cmap='terrain', alpha=CONTOUR_ALPHA)
        cbar = plt.colorbar(contourf, ax=ax, label='Elevation (m)', shrink=0.8)
        
        # Plot contour lines
        contour = ax.contour(LON, LAT, elevation_grid, levels=contour_levels, colors='black', alpha=CONTOUR_ALPHA, linewidths=CONTOUR_LINEWIDTH)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.0fm')
    else:
        # If no coordinate grids, just plot a basic elevation map
        im = ax.imshow(elevation_grid, origin='lower', cmap='terrain', alpha=0.7)
        plt.colorbar(im, ax=ax, label='Elevation (m)')
    
    # Extract GPX coordinates for plotting
    lats = np.array([p['latitude'] for p in gpx_points])
    lons = np.array([p['longitude'] for p in gpx_points])
    
    # Plot GPX track with color mapping based on elevation
    if all(p['elevation'] is not None for p in gpx_points):
        elevations = [p['elevation'] for p in gpx_points]
        scatter = ax.scatter(lons, lats, c=elevations, cmap='viridis', s=8, zorder=5)
        plt.colorbar(scatter, ax=ax, label='GPX Elevation (m)', shrink=0.8)
    else:
        ax.plot(lons, lats, 'r-', linewidth=2, zorder=5)
    
    # Highlight start and end points
    ax.scatter(lons[0], lats[0], color='lime', s=150, zorder=6, label='Start', edgecolors='black', linewidth=1)  # Start point
    ax.scatter(lons[-1], lats[-1], color='red', s=150, zorder=6, label='End', edgecolors='black', linewidth=1)    # End point
    
    # Calculate axis limits to magnify the view on the GPX path
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    
    # Add buffer around the path
    lat_center = (lat_min + lat_max) / 2
    lon_center = (lon_min + lon_max) / 2
    
    lat_range = (lat_max - lat_min) * ZOOM_FACTOR
    lon_range = (lon_max - lon_min) * ZOOM_FACTOR
    
    # Set axis limits to create a magnified view
    ax.set_xlim(lon_center - lon_range/2 - AXIS_BUFFER, lon_center + lon_range/2 + AXIS_BUFFER)
    ax.set_ylim(lat_center - lat_range/2 - AXIS_BUFFER, lat_center + lat_range/2 + AXIS_BUFFER)
    
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def integrate_contour_data(terrain, contour_levels=None):
    """
    Integrate contour data with the terrain for visualization.
    Based on the descend-mountain repository technique.
    """
    if contour_levels is None:
        # Create contour levels from the terrain data
        min_elev = float(np.nanmin(terrain))
        max_elev = float(np.nanmax(terrain))

        # Create levels every 20m (or 10% of elevation range if smaller)
        elev_range = max_elev - min_elev
        step = max(10.0, elev_range / 20.0)  # At most 20 contour lines
        contour_levels = np.arange(
            np.floor(min_elev / step) * step,
            np.ceil(max_elev / step) * step + step / 2.0,
            step
        )

    return {
        'terrain': terrain,
        'contour_levels': contour_levels,
        'min_elevation': float(np.nanmin(terrain)),
        'max_elevation': float(np.nanmax(terrain))
    }