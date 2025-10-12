#!/usr/bin/env python3
"""
Main module for terrain visualization functionality
"""
import os
import matplotlib
# Try to use TkAgg backend for GUI if available, otherwise Agg for file output
try:
    if os.environ.get('DISPLAY'):
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
except:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from src.data_processing.gpx_processor import get_gpx_bounds, get_gpx_tracks
from src.api.elevation_api import get_real_elevation_data_around_coords
from src.visualization.terrain_visualizer import visualize_terrain_with_contours_and_paths


def main():
    print("Analyzing GPX files for coordinate ranges...")
    # Use filtered data (after 2025-10-07T22:12:12Z) for bounds calculation
    bounds = get_gpx_bounds(apply_time_filter=True)
    
    print(f"GPX coordinate ranges (filtered after 2025-10-07T22:12:12Z):")
    print(f"  Latitude: {bounds['lat_min']:.6f} to {bounds['lat_max']:.6f}")
    print(f"  Longitude: {bounds['lon_min']:.6f} to {bounds['lon_max']:.6f}")
    if bounds['elev_min'] is not None:
        print(f"  Elevation: {bounds['elev_min']:.2f}m to {bounds['elev_max']:.2f}m")
    
    # Calculate center point
    center_lat = (bounds['lat_min'] + bounds['lat_max']) / 2
    center_lon = (bounds['lon_min'] + bounds['lon_max']) / 2
    
    print(f"\nCenter coordinates: {center_lat:.6f}, {center_lon:.6f}")
    
    # Load API key from environment
    load_dotenv()
    api_key = os.getenv('GOOGLE_ELEVATION_API_KEY') or os.getenv('ELEVATION_API_KEY')
    if not api_key or api_key == "YOUR_ELEVATION_API_KEY_HERE":
        raise ValueError("Please set a valid ELEVATION_API_KEY in .env file")
    
    print("Fetching real elevation data...")
    elevation_grid, LAT, LON = get_real_elevation_data_around_coords(
        center_lat, center_lon, api_key, size=140
    )
    
    # Get filtered GPX tracks to overlay on terrain
    gpx_tracks = get_gpx_tracks(apply_time_filter=True)
    print(f"Found {len(gpx_tracks)} GPX tracks to display on terrain (filtered after 2025-10-07T22:12:12Z)")
    
    print("Creating 3D visualization with hiker paths...")
    fig, ax = visualize_terrain_with_contours_and_paths(elevation_grid, LAT, LON, gpx_tracks)
    
    # Save the visualization
    os.makedirs('output', exist_ok=True)
    output_path = 'output/actual_terrain_with_contours_and_paths.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Show the window
    plt.show()


if __name__ == "__main__":
    main()