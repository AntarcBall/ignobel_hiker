#!/usr/bin/env python3
"""
Module for data synchronization and time calculations
"""
from datetime import timedelta


def synchronize_gpx_data(all_gpx_points):
    """
    Synchronize GPX data for all hikers by finding the common time range
    and ensuring all hikers have data points within the same time window.
    
    Parameters:
    all_gpx_points: List of lists, each containing GPX points for one hiker
    
    Returns:
    synchronized_gpx_points: List of synchronized GPX point lists
    """
    if not all_gpx_points:
        return []

    # Find the latest start time among all hikers
    start_times = []
    end_times = []

    for gpx_points in all_gpx_points:
        if gpx_points:
            start_times.append(gpx_points[0]['time'])
            end_times.append(gpx_points[-1]['time'])

    if not start_times or not end_times:
        return all_gpx_points

    # Use the latest start time and earliest end time as common window
    common_start = max(start_times)
    common_end = min(end_times)

    if common_start >= common_end:
        # No common time window, return empty lists
        return [[] for _ in all_gpx_points]

    # Filter each hiker's data to the common time window
    synchronized_points = []
    for gpx_points in all_gpx_points:
        filtered_points = []
        for point in gpx_points:
            if common_start <= point['time'] <= common_end:
                filtered_points.append(point)
        synchronized_points.append(filtered_points)

    return synchronized_points


def calculate_avg_time_interval(gpx_points):
    """
    Calculate the average time interval between consecutive data points.
    
    Parameters:
    gpx_points: List of dictionaries with 'time' key
    
    Returns:
    avg_interval: Average time interval in seconds
    """
    if len(gpx_points) < 2:
        return 0

    intervals = []
    for i in range(1, len(gpx_points)):
        dt = (gpx_points[i]['time'] - gpx_points[i-1]['time']).total_seconds()
        if dt > 0:  # Only consider positive intervals
            intervals.append(dt)

    if intervals:
        return sum(intervals) / len(intervals)
    else:
        return 0