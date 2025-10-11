#!/usr/bin/env python3
"""
Module for comprehensive score calculations
"""
from src.config.config_velocity import TORTUOSITY_WEIGHT, STOP_FRACTION_WEIGHT, LEADER_SCORE_WEIGHT


def calculate_comprehensive_score(tortuosity, stop_time_fraction, leader_score):
    """
    Calculate a comprehensive weighted sum score using config weights.
    
    Parameters:
    tortuosity: Tortuosity score
    stop_time_fraction: Stop time fraction
    leader_score: Leader score
    
    Returns:
    comprehensive_score: Weighted sum of all components
    """
    # Use weights from config
    w_tort = TORTUOSITY_WEIGHT
    w_stop = STOP_FRACTION_WEIGHT
    w_leader = LEADER_SCORE_WEIGHT

    comprehensive_score = w_tort * tortuosity + w_stop * stop_time_fraction + w_leader * leader_score

    return comprehensive_score


def calculate_total_time_elapsed(gpx_points):
    """
    Calculate total time elapsed from first to last point in the GPX data.
    
    Parameters:
    gpx_points: List of dictionaries with 'latitude', 'longitude', 'elevation', 'time'
    
    Returns:
    total_time: Total time elapsed in seconds
    """
    if not gpx_points or len(gpx_points) < 2:
        return 0

    total_time = (gpx_points[-1]['time'] - gpx_points[0]['time']).total_seconds()
    return total_time if total_time > 0 else 0


def get_common_time_window(all_gpx_points):
    """
    Get the common time window from the synchronized data.
    
    Parameters:
    all_gpx_points: List of lists, each containing GPX points for one hiker
    
    Returns:
    common_start: The common start time
    common_end: The common end time
    """
    if not all_gpx_points:
        return None, None

    start_times = []
    end_times = []

    for gpx_points in all_gpx_points:
        if gpx_points:
            start_times.append(gpx_points[0]['time'])
            end_times.append(gpx_points[-1]['time'])

    if not start_times or not end_times:
        return None, None

    common_start = max(start_times)
    common_end = min(end_times)

    return common_start, common_end