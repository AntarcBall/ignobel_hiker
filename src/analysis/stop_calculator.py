#!/usr/bin/env python3
"""
Module for stop metrics calculations
"""
import math
from src.config.config_velocity import MIN_STOP_SPEED_THRESHOLD, MIN_STOP_DURATION


def calculate_stop_metrics(gpx_points, velocity_vectors):
    """
    Calculate stop-related metrics: stop count, total stop time, stop time fraction
    
    Parameters:
    gpx_points: List of dictionaries with 'latitude', 'longitude', 'elevation', 'time'
    velocity_vectors: List of (vx, vy, vz, dt) tuples
    
    Returns:
    stop_count: Number of stops
    total_stop_time: Total time spent stopped (seconds)
    stop_time_fraction: Fraction of total time spent stopped
    """
    if len(velocity_vectors) == 0 or len(gpx_points) < 2:
        return 0, 0, 0

    total_time = (gpx_points[-1]['time'] - gpx_points[0]['time']).total_seconds()
    if total_time <= 0:
        return 0, 0, 0

    stop_count = 0
    total_stop_time = 0

    # Identify stop periods
    current_stop_duration = 0
    in_stop = False

    for i, (vx, vy, vz, dt) in enumerate(velocity_vectors):
        speed = math.sqrt(vx**2 + vy**2 + vz**2)

        if speed < MIN_STOP_SPEED_THRESHOLD:
            if not in_stop:
                # Started a new potential stop
                current_stop_duration = dt
                in_stop = True
            else:
                current_stop_duration += dt
        else:
            # Not moving slowly anymore
            if in_stop and current_stop_duration >= MIN_STOP_DURATION:
                # This was a valid stop
                stop_count += 1
                total_stop_time += current_stop_duration
            in_stop = False
            current_stop_duration = 0

    # Handle case where last segment was a stop
    if in_stop and current_stop_duration >= MIN_STOP_DURATION:
        stop_count += 1
        total_stop_time += current_stop_duration

    stop_time_fraction = total_stop_time / total_time if total_time > 0 else 0

    return stop_count, total_stop_time, stop_time_fraction