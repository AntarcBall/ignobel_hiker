#!/usr/bin/env python3
"""
Module for coordinate system conversion (LLA to ENU)
"""
import math


def lla_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    """
    Convert Latitude, Longitude, Altitude to East-North-Up (ENU) coordinates.
    
    Parameters:
    lat, lon, alt: Current point coordinates
    lat_ref, lon_ref, alt_ref: Reference point coordinates
    
    Returns:
    x, y, z: ENU coordinates (East, North, Up)
    """
    # Convert degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat_ref_rad = math.radians(lat_ref)
    lon_ref_rad = math.radians(lon_ref)
    
    # Differences in radians
    dlat = lat_rad - lat_ref_rad
    dlon = lon_rad - lon_ref_rad
    dalt = alt - alt_ref
    
    # Earth's radius in meters
    R = 6371000.0
    
    # Calculate ENU coordinates
    x = R * dlon * math.cos(lat_ref_rad)  # East
    y = R * dlat  # North
    z = dalt  # Up
    
    return x, y, z