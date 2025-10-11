#!/usr/bin/env python3
"""
Module for processing GPX files - extracting coordinates, bounds, and track data
"""
import os
import gpxpy
import numpy as np


def get_gpx_tracks():
    """Extract GPX tracks from GPX files"""
    gpx_dir = 'gpx_data'
    gpx_files = []
    
    if os.path.exists(gpx_dir):
        for file in os.listdir(gpx_dir):
            if file.lower().endswith('.gpx'):
                gpx_files.append(os.path.join(gpx_dir, file))
    
    all_tracks = []
    
    for gpx_file in gpx_files:
        with open(gpx_file, 'r') as file:
            gpx = gpxpy.parse(file)
        
        track_points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    track_points.append({
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation
                    })
        if track_points:
            all_tracks.append(track_points)
    
    return all_tracks


def get_gpx_bounds():
    """Extract coordinate bounds from GPX files"""
    gpx_tracks = get_gpx_tracks()
    
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