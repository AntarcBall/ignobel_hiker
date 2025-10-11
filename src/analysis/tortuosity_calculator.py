#!/usr/bin/env python3
"""
Module for tortuosity calculations
"""
from src.utils.distance_calculator import haversine_distance


def calculate_tortuosity(gpx_points):
    """
    Calculate tortuosity: total path length / straight-line distance from start to end
    
    Parameters:
    gpx_points: List of dictionaries with 'latitude', 'longitude', 'elevation', 'time'
    
    Returns:
    tortuosity: Total path length / straight-line distance
    total_path_length: Total distance along the path
    straight_line_distance: Straight-line distance from start to end
    """
    if len(gpx_points) < 2:
        return 0, 0, 0

    # Calculate total path length
    total_path_length = 0
    for i in range(1, len(gpx_points)):
        d = haversine_distance(
            gpx_points[i-1]['latitude'], gpx_points[i-1]['longitude'],
            gpx_points[i]['latitude'], gpx_points[i]['longitude']
        )
        total_path_length += d

    # Calculate straight-line distance from start to end
    straight_line_distance = haversine_distance(
        gpx_points[0]['latitude'], gpx_points[0]['longitude'],
        gpx_points[-1]['latitude'], gpx_points[-1]['longitude']
    )

    if straight_line_distance == 0:
        tortuosity = 1  # If no movement, tortuosity is 1
    else:
        tortuosity = total_path_length / straight_line_distance

    return tortuosity, total_path_length, straight_line_distance