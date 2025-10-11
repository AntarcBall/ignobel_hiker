#!/usr/bin/env python3
"""
Module for GPX file parsing and data extraction
"""
import os
import gpxpy
from datetime import datetime
from src.config.config_velocity import FILTER_START_TIME


def parse_gpx_file(file_path):
    """
    Parse GPX file and extract track points with coordinates and elevation
    Filter data points after 2025-10-07T22:12:12Z
    """
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    
    track_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                # Only include points after the specified start time (2025-10-07T22:12:12Z)
                # Handle timezone-aware comparison
                if point.time:
                    # Parse the start time from config
                    filter_time_str = FILTER_START_TIME
                    # Format: YYYY-MM-DDTHH:MM:SSZ
                    start_time = datetime.strptime(filter_time_str, "%Y-%m-%dT%H:%M:%SZ")
                    
                    if point.time.tzinfo is not None:
                        # If point.time is timezone-aware, make start_time timezone-aware too
                        from datetime import timezone
                        start_time = start_time.replace(tzinfo=timezone.utc)
                    
                    if point.time >= start_time:
                        track_points.append({
                            'latitude': point.latitude,
                            'longitude': point.longitude,
                            'elevation': point.elevation,
                            'time': point.time
                        })
    
    return track_points


def get_gpx_files():
    """
    Get all GPX files in the gpx_data subdirectory
    """
    gpx_dir = 'gpx_data'
    gpx_files = []
    
    if os.path.exists(gpx_dir):
        for file in os.listdir(gpx_dir):
            if file.lower().endswith('.gpx'):
                gpx_files.append(os.path.join(gpx_dir, file))
    
    return gpx_files