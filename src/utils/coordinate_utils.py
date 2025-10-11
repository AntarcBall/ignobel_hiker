#!/usr/bin/env python3
"""
Module for coordinate transformation and grid generation
"""
import math
import numpy as np


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