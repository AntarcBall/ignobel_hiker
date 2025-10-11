#!/usr/bin/env python3
"""
Module for velocity calculations from GPX data
"""
import math
from src.utils.coordinate_converter import lla_to_enu


def calculate_velocity_vectors(gpx_points):
    """
    Calculate velocity vectors for a list of GPX points.
    
    Parameters:
    gpx_points: List of dictionaries with 'latitude', 'longitude', 'elevation', 'time'
    
    Returns:
    velocity_vectors: List of (dx, dy, dz, dt) tuples representing velocity components
    speeds: List of speed magnitudes
    """
    if len(gpx_points) < 2:
        return [], []

    # Use first point as reference for ENU coordinate system
    ref_lat = gpx_points[0]['latitude']
    ref_lon = gpx_points[0]['longitude']
    ref_alt = gpx_points[0]['elevation'] or 0  # Default to 0 if elevation is None

    # Convert all points to ENU coordinates
    enu_coords = []
    for point in gpx_points:
        alt = point['elevation'] or ref_alt  # Default to ref_alt if elevation is None
        x, y, z = lla_to_enu(point['latitude'], point['longitude'], alt,
                            ref_lat, ref_lon, ref_alt)
        enu_coords.append((x, y, z, point['time']))

    # Calculate velocity vectors
    velocity_vectors = []
    speeds = []

    for i in range(1, len(enu_coords)):
        # Calculate displacement
        dx = enu_coords[i][0] - enu_coords[i-1][0]
        dy = enu_coords[i][1] - enu_coords[i-1][1]
        dz = enu_coords[i][2] - enu_coords[i-1][2]

        # Calculate time difference in seconds
        dt = (enu_coords[i][3] - enu_coords[i-1][3]).total_seconds()

        # Avoid division by zero
        if dt <= 0:
            velocity_vectors.append((0, 0, 0, 0))  # No movement
            speeds.append(0)
        else:
            # Calculate velocity components
            vx = dx / dt
            vy = dy / dt
            vz = dz / dt

            # Calculate speed magnitude
            speed = math.sqrt(vx**2 + vy**2 + vz**2)

            velocity_vectors.append((vx, vy, vz, dt))
            speeds.append(speed)

    return velocity_vectors, speeds