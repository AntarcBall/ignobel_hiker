#!/usr/bin/env python3
"""
Module for leader score calculations
"""
from datetime import timedelta
from src.utils.distance_calculator import haversine_distance
from src.config.config_velocity import LEADER_TIME_INTERVAL


def calculate_leader_score(all_gpx_points):
    """
    Calculate leader score for multiple hikers based on who leads at different times.
    Uses linear interpolation to get synchronized positions at regular time intervals.
    The leader is determined by who will be farthest from the group center in the next time step.
    
    Parameters:
    all_gpx_points: List of lists, each containing GPX points for one hiker
    
    Returns:
    leader_scores: List of scores for each hiker
    """
    if len(all_gpx_points) < 2:
        return [0] * len(all_gpx_points)  # No comparison possible with less than 2 hikers

    # Find the overall start and end times
    all_times = []
    for gpx_points in all_gpx_points:
        if gpx_points:
            all_times.extend([point['time'] for point in gpx_points])

    if not all_times:
        return [0] * len(all_gpx_points)

    start_time = min(all_times)
    end_time = max(all_times)

    # Generate time points at regular intervals
    time_points = []
    current_time = start_time
    while current_time <= end_time:
        time_points.append(current_time)
        current_time += timedelta(seconds=LEADER_TIME_INTERVAL)

    # Interpolate positions for each hiker at each time point
    interpolated_positions = []
    for gpx_points in all_gpx_points:
        hiker_positions = []
        for t in time_points:
            # Find the two nearest points to interpolate
            if not gpx_points or t < gpx_points[0]['time']:
                # Use the first point if the time is before the track starts
                if gpx_points:
                    pos = (gpx_points[0]['latitude'], gpx_points[0]['longitude'])
                    hiker_positions.append(pos)
                else:
                    hiker_positions.append((0, 0))
            elif t > gpx_points[-1]['time']:
                # Use the last point if the time is after the track ends
                if gpx_points:
                    pos = (gpx_points[-1]['latitude'], gpx_points[-1]['longitude'])
                    hiker_positions.append(pos)
                else:
                    hiker_positions.append((0, 0))
            else:
                # Find the rightmost point that's before or at the target time
                i = 0
                while i < len(gpx_points) and gpx_points[i]['time'] <= t:
                    i += 1
                
                if i == 0:
                    pos = (gpx_points[0]['latitude'], gpx_points[0]['longitude'])
                elif i == len(gpx_points):
                    pos = (gpx_points[-1]['latitude'], gpx_points[-1]['longitude'])
                else:
                    # Linear interpolation between gpx_points[i-1] and gpx_points[i]
                    t1 = gpx_points[i-1]['time']
                    t2 = gpx_points[i]['time']
                    lat1 = gpx_points[i-1]['latitude']
                    lon1 = gpx_points[i-1]['longitude']
                    lat2 = gpx_points[i]['latitude']
                    lon2 = gpx_points[i]['longitude']
                    
                    if t2 == t1:
                        lat = lat1
                        lon = lon1
                    else:
                        # Linear interpolation
                        ratio = (t - t1).total_seconds() / (t2 - t1).total_seconds()
                        lat = lat1 + ratio * (lat2 - lat1)
                        lon = lon1 + ratio * (lon2 - lon1)
                    
                    pos = (lat, lon)
                
                hiker_positions.append(pos)
        
        interpolated_positions.append(hiker_positions)

    # Calculate leader scores based on who is farthest ahead of group center
    leader_time_counts = [0] * len(all_gpx_points)  # Time spent as leader

    for j in range(len(time_points)):
        if j + 1 < len(time_points):  # Only if we can look ahead to next time step
            # Calculate group center at next time step
            next_lat_sum = sum(interpolated_positions[i][j+1][0] for i in range(len(all_gpx_points)))
            next_lon_sum = sum(interpolated_positions[i][j+1][1] for i in range(len(all_gpx_points)))
            next_center_lat = next_lat_sum / len(all_gpx_points)
            next_center_lon = next_lon_sum / len(all_gpx_points)
            
            # Calculate distances to next center for each hiker at next time step
            next_distances_to_center = []
            for i in range(len(all_gpx_points)):
                hiker_lat_next = interpolated_positions[i][j+1][0]
                hiker_lon_next = interpolated_positions[i][j+1][1]
                dist = haversine_distance(next_center_lat, next_center_lon, hiker_lat_next, hiker_lon_next)
                next_distances_to_center.append(dist)

            # The leader is the one who is farthest from the group center (likely ahead)
            if next_distances_to_center:
                leader_idx = next_distances_to_center.index(max(next_distances_to_center))
                leader_time_counts[leader_idx] += LEADER_TIME_INTERVAL

    # Calculate the fraction of time each hiker was the leader
    total_time = len(time_points) * LEADER_TIME_INTERVAL
    if total_time == 0:
        return [0] * len(all_gpx_points)

    leader_scores = [time_count / total_time if total_time > 0 else 0 for time_count in leader_time_counts]

    return leader_scores