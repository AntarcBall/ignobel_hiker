#!/usr/bin/env python3
"""
Module for elevation data API calls and caching
"""
import os
import pickle
import requests
import numpy as np
from src.utils.coordinate_utils import bbox_from_center, make_grid


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