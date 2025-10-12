#!/usr/bin/env python3
"""
Module for processing GPX files - extracting coordinates, bounds, and track data
"""
import os
import gpxpy
import numpy as np
import pickle
from datetime import datetime, timezone
from src.config.config_velocity import FILTER_START_TIME


def get_cached_gpx_tracks():
    """
    Load GPX tracks from cache files when no GPX files are available
    """
    cache_dir = 'cache'
    cached_tracks = []
    
    if os.path.exists(cache_dir):
        import glob
        cache_files = glob.glob(os.path.join(cache_dir, 'gpx_points_*.pkl'))
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Ensure the cached data is in the correct format
                    if isinstance(cached_data, list) and len(cached_data) > 0:
                        # If it's already in the right format (list of tracks)
                        if isinstance(cached_data[0], list) and len(cached_data[0]) > 0:
                            # If the first element is a dict with required keys, it's good
                            if (isinstance(cached_data[0][0], dict) and 
                                'latitude' in cached_data[0][0] and 
                                'longitude' in cached_data[0][0]):
                                cached_tracks.extend(cached_data)
                        # If it's a flat list of points, wrap it in a track
                        elif isinstance(cached_data[0], dict) and 'latitude' in cached_data[0]:
                            cached_tracks.append(cached_data)
            except Exception as e:
                print(f"Error loading cached GPX data from {cache_file}: {e}")
    
    return cached_tracks


def get_gpx_tracks(apply_time_filter=False):
    """
    Extract GPX tracks from GPX files, or from cached data if no GPX files are available
    
    Parameters:
    apply_time_filter: If True, only include points after FILTER_START_TIME (2025-10-07T22:12:12Z)
    """
    gpx_dir = 'gpx_data'
    gpx_files = []
    
    if os.path.exists(gpx_dir):
        for file in os.listdir(gpx_dir):
            if file.lower().endswith('.gpx'):
                gpx_files.append(os.path.join(gpx_dir, file))
    
    # If no GPX files found, try to load from cache
    if not gpx_files:
        print("No GPX files found in gpx_data directory, attempting to load from cache...")
        return get_cached_gpx_tracks()
    
    all_tracks = []
    
    # Parse the start time from config if filtering is enabled
    start_time = None
    if apply_time_filter:
        filter_time_str = FILTER_START_TIME
        start_time = datetime.strptime(filter_time_str, "%Y-%m-%dT%H:%M:%SZ")
        start_time = start_time.replace(tzinfo=timezone.utc)
    
    for gpx_file in gpx_files:
        with open(gpx_file, 'r') as file:
            gpx = gpxpy.parse(file)
        
        track_points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    # Apply time filter if enabled
                    if apply_time_filter and point.time:
                        # Make point.time timezone-aware if needed
                        point_time = point.time
                        if point_time.tzinfo is None:
                            point_time = point_time.replace(tzinfo=timezone.utc)
                        
                        # Skip points before the start time
                        if point_time < start_time:
                            continue
                    
                    track_points.append({
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation
                    })
        if track_points:
            all_tracks.append(track_points)
    
    return all_tracks


def get_gpx_bounds(apply_time_filter=False):
    """
    Extract coordinate bounds from GPX files, or from cached data if no GPX files are available
    
    Parameters:
    apply_time_filter: If True, only include points after FILTER_START_TIME (2025-10-07T22:12:12Z)
    """
    gpx_tracks = get_gpx_tracks(apply_time_filter=apply_time_filter)
    
    all_lats = []
    all_lons = []
    all_elevs = []
    
    for track in gpx_tracks:
        for point in track:
            all_lats.append(point['latitude'])
            all_lons.append(point['longitude'])
            if point['elevation'] is not None:
                all_elevs.append(point['elevation'])
    
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