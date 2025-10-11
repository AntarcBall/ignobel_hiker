#!/usr/bin/env python3
"""
Velocity Analysis for Hiking GPX Data

This script:
1. Approximates hiking paths in a local 3D orthogonal coordinate system (ENU)
2. Calculates difference vectors for all hikers
3. Calculates instantaneous velocity vectors by dividing displacement by time
4. Calculates speed from velocity vectors
5. Creates a distribution histogram of velocity vectors with different colors for each hiker
6. Filters data points after 2025-10-07T22:12:12Z
7. Calculates tortuosity (total path length / straight-line distance)
8. Calculates stop-related metrics (stop count, total stop time, stop time fraction)
9. Calculates leader score (for multiple hikers)
10. Computes comprehensive weighted sum score
"""
import os
import gpxpy
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import math
from datetime import datetime, timedelta
import config_velocity

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
                    filter_time_str = config_velocity.FILTER_START_TIME
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

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in meters
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    return c * r

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
    min_speed_threshold = config_velocity.MIN_STOP_SPEED_THRESHOLD
    min_stop_duration = config_velocity.MIN_STOP_DURATION
    
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
        
        if speed < min_speed_threshold:
            if not in_stop:
                # Started a new potential stop
                current_stop_duration = dt
                in_stop = True
            else:
                current_stop_duration += dt
        else:
            # Not moving slowly anymore
            if in_stop and current_stop_duration >= min_stop_duration:
                # This was a valid stop
                stop_count += 1
                total_stop_time += current_stop_duration
            in_stop = False
            current_stop_duration = 0
    
    # Handle case where last segment was a stop
    if in_stop and current_stop_duration >= min_stop_duration:
        stop_count += 1
        total_stop_time += current_stop_duration
    
    stop_time_fraction = total_stop_time / total_time if total_time > 0 else 0
    
    return stop_count, total_stop_time, stop_time_fraction

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
    time_interval = config_velocity.LEADER_TIME_INTERVAL
    
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
        current_time += timedelta(seconds=time_interval)
    
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
                leader_time_counts[leader_idx] += time_interval
        # For the last time point, we can't look ahead, so skip
    
    # Calculate the fraction of time each hiker was the leader
    total_time = len(time_points) * time_interval
    if total_time == 0:
        return [0] * len(all_gpx_points)
    
    leader_scores = [time_count / total_time if total_time > 0 else 0 for time_count in leader_time_counts]
    
    return leader_scores

def calculate_comprehensive_score(tortuosity, stop_time_fraction, leader_score):
    """
    Calculate a comprehensive weighted sum score.
    Currently using equal weights (1:1:1) for all components.
    
    Parameters:
    tortuosity: Tortuosity score
    stop_time_fraction: Stop time fraction
    leader_score: Leader score
    
    Returns:
    comprehensive_score: Weighted sum of all components
    """
    # Normalize the scores (for now, we just sum them as equal weights)
    # Note: This assumes all scores are in similar ranges or already normalized
    
    # Equal weights for now: 1:1:1
    w1, w2, w3 = 1, 1, 1  # Weights for tortuosity, stop fraction, leader score
    
    comprehensive_score = w1 * tortuosity + w2 * stop_time_fraction + w3 * leader_score
    
    return comprehensive_score

def create_velocity_histogram(hiker_velocities, hiker_speeds, hiker_names):
    """
    Create a distribution histogram of speeds for different hikers
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define speed bins
    all_speeds = np.concatenate(hiker_speeds)
    if len(all_speeds) == 0:
        print("No speed data available for plotting")
        return fig, ax
        
    bins = np.linspace(0, max(all_speeds) * 1.1, config_velocity.VELOCITY_HISTOGRAM_BINS)
    
    # Get colors for different hikers
    colors = plt.cm.tab10(np.linspace(0, 1, len(hiker_names)))
    
    # Plot histograms for each hiker
    for i, (speeds, name, color) in enumerate(zip(hiker_speeds, hiker_names, colors)):
        if len(speeds) > 0:
            ax.hist(speeds, bins=bins, alpha=0.6, label=name, color=color, density=True)
    
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Speeds for Different Hikers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def create_3d_velocity_scatter(hiker_velocities, hiker_speeds, hiker_names):
    """
    Create a 3D scatter plot of velocity vectors
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get colors for different hikers
    colors = plt.cm.tab10(np.linspace(0, 1, len(hiker_names)))
    
    for i, (velocities, name, color) in enumerate(zip(hiker_velocities, hiker_names, colors)):
        if len(velocities) > 0:
            vxs = [v[0] for v in velocities if len(v) == 4]  # vx
            vys = [v[1] for v in velocities if len(v) == 4]  # vy
            vzs = [v[2] for v in velocities if len(v) == 4]  # vz
            
            ax.scatter(vxs, vys, vzs, alpha=0.6, label=name, color=color, s=20)
    
    ax.set_xlabel('East Velocity (m/s)')
    ax.set_ylabel('North Velocity (m/s)')
    ax.set_zlabel('Up Velocity (m/s)')
    ax.set_title('3D Distribution of Velocity Vectors for Different Hikers')
    ax.legend()
    
    plt.tight_layout()
    return fig, ax

def create_velocity_projections(hiker_velocities, hiker_speeds, hiker_names):
    """
    Create 2D projections of velocity vectors (XY, XZ, YZ)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Projections of Velocity Vectors for Different Hikers', fontsize=16)
    
    # Get colors for different hikers
    colors = plt.cm.tab10(np.linspace(0, 1, len(hiker_names)))
    
    for i, (velocities, name, color) in enumerate(zip(hiker_velocities, hiker_names, colors)):
        if len(velocities) > 0:
            vxs = [v[0] for v in velocities if len(v) == 4]  # vx
            vys = [v[1] for v in velocities if len(v) == 4]  # vy
            vzs = [v[2] for v in velocities if len(v) == 4]  # vz
            
            # XY projection (East-North)
            axes[0, 0].scatter(vxs, vys, alpha=0.6, label=name, color=color, s=20)
            
            # XZ projection (East-Up)
            axes[0, 1].scatter(vxs, vzs, alpha=0.6, label=name, color=color, s=20)
            
            # YZ projection (North-Up)
            axes[1, 0].scatter(vys, vzs, alpha=0.6, label=name, color=color, s=20)
    
    # Set labels for each subplot
    axes[0, 0].set_xlabel('East Velocity (m/s)')
    axes[0, 0].set_ylabel('North Velocity (m/s)')
    axes[0, 0].set_title('East vs North Velocity')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('East Velocity (m/s)')
    axes[0, 1].set_ylabel('Up Velocity (m/s)')
    axes[0, 1].set_title('East vs Up Velocity')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('North Velocity (m/s)')
    axes[1, 0].set_ylabel('Up Velocity (m/s)')
    axes[1, 0].set_title('North vs Up Velocity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add legend to the empty subplot
    axes[1, 1].legend(hiker_names, title="Hikers", loc='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig, axes

def main():
    print("Analyzing hiking velocity data...")
    
    # Get all GPX files
    gpx_files = get_gpx_files()
    print(f"Found {len(gpx_files)} GPX files: {gpx_files}")
    
    if not gpx_files:
        print("No GPX files found in gpx_data directory")
        return
    
    # Process each GPX file and extract velocity data
    all_gpx_points = []  # Store original points for tortuosity and stop calculations
    all_velocity_vectors = []
    all_speeds = []
    hiker_names = []
    
    for i, gpx_file in enumerate(gpx_files):
        print(f"Processing {gpx_file}...")
        gpx_points = parse_gpx_file(gpx_file)
        print(f"Loaded {len(gpx_points)} track points from {gpx_file} (filtered after 2025-10-07T22:12:12Z)")
        
        if len(gpx_points) > 0:
            velocity_vectors, speeds = calculate_velocity_vectors(gpx_points)
            print(f"Calculated {len(velocity_vectors)} velocity vectors for {gpx_file}")
            
            all_gpx_points.append(gpx_points)  # Store original points
            all_velocity_vectors.append(velocity_vectors)
            all_speeds.append(speeds)
            hiker_names.append(os.path.basename(gpx_file).replace('.gpx', ''))
    
    if not all_gpx_points:
        print("No valid GPX data found")
        return
    
    # Calculate additional metrics
    print("\nCalculating additional metrics...")
    
    # Calculate tortuosity for each hiker
    tortuosities = []
    total_path_lengths = []
    straight_line_distances = []
    
    for i, gpx_points in enumerate(all_gpx_points):
        tortuosity, total_length, straight_dist = calculate_tortuosity(gpx_points)
        tortuosities.append(tortuosity)
        total_path_lengths.append(total_length)
        straight_line_distances.append(straight_dist)
        print(f"{hiker_names[i]} - Tortuosity: {tortuosity:.3f}, Total path: {total_length:.2f}m, Straight-line: {straight_dist:.2f}m")
    
    # Calculate stop metrics for each hiker
    stop_counts = []
    total_stop_times = []
    stop_time_fractions = []
    
    for i, (gpx_points, velocity_vectors) in enumerate(zip(all_gpx_points, all_velocity_vectors)):
        stop_count, total_stop_time, stop_fraction = calculate_stop_metrics(gpx_points, velocity_vectors)
        stop_counts.append(stop_count)
        total_stop_times.append(total_stop_time)
        stop_time_fractions.append(stop_fraction)
        print(f"{hiker_names[i]} - Stops: {stop_count}, Total stop time: {total_stop_time:.2f}s, Stop fraction: {stop_fraction:.3f}")
    
    # Calculate leader scores (only if there are multiple hikers)
    leader_scores = [0] * len(hiker_names)  # Default to 0 if not enough hikers
    if len(all_gpx_points) > 1:
        print("\nCalculating leader scores...")
        leader_scores = calculate_leader_score(all_gpx_points)
        for i, leader_score in enumerate(leader_scores):
            print(f"{hiker_names[i]} - Leader score: {leader_score:.3f}")
    
    # Calculate comprehensive scores
    comprehensive_scores = []
    print("\nCalculating comprehensive scores...")
    for i in range(len(hiker_names)):
        comp_score = calculate_comprehensive_score(tortuosities[i], stop_time_fractions[i], leader_scores[i])
        comprehensive_scores.append(comp_score)
        print(f"{hiker_names[i]} - Comprehensive score: {comp_score:.3f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    print("Creating velocity distribution histogram...")
    fig_hist, ax_hist = create_velocity_histogram(all_velocity_vectors, all_speeds, hiker_names)
    
    # Create 3D velocity scatter plot
    print("Creating 3D velocity scatter plot...")
    fig_3d, ax_3d = create_3d_velocity_scatter(all_velocity_vectors, all_speeds, hiker_names)
    
    # Create velocity projection plots
    print("Creating velocity projection plots...")
    fig_proj, ax_proj = create_velocity_projections(all_velocity_vectors, all_speeds, hiker_names)
    
    # Create additional metric visualizations
    print("Creating additional metric visualizations...")
    # Bar chart for tortuosity
    fig_tort, ax_tort = plt.subplots(figsize=(10, 6))
    bars = ax_tort.bar(hiker_names, tortuosities, color=plt.cm.tab10(np.linspace(0, 1, len(hiker_names))))
    ax_tort.set_xlabel('Hiker')
    ax_tort.set_ylabel('Tortuosity')
    ax_tort.set_title('Tortuosity Comparison')
    ax_tort.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, tortuosities):
        ax_tort.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Bar chart for stop time fractions
    fig_stop, ax_stop = plt.subplots(figsize=(10, 6))
    bars = ax_stop.bar(hiker_names, stop_time_fractions, color=plt.cm.tab10(np.linspace(0, 1, len(hiker_names))))
    ax_stop.set_xlabel('Hiker')
    ax_stop.set_ylabel('Stop Time Fraction')
    ax_stop.set_title('Stop Time Fraction Comparison')
    ax_stop.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, stop_time_fractions):
        ax_stop.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Bar chart for comprehensive scores
    fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
    bars = ax_comp.bar(hiker_names, comprehensive_scores, color=plt.cm.tab10(np.linspace(0, 1, len(hiker_names))))
    ax_comp.set_xlabel('Hiker')
    ax_comp.set_ylabel('Comprehensive Score')
    ax_comp.set_title('Comprehensive Score Comparison')
    ax_comp.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, comprehensive_scores):
        ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save visualizations
    os.makedirs('output', exist_ok=True)
    fig_hist.savefig('output/velocity_histogram.png', dpi=300, bbox_inches='tight')
    fig_3d.savefig('output/velocity_3d_scatter.png', dpi=300, bbox_inches='tight')
    fig_proj.savefig('output/velocity_projections.png', dpi=300, bbox_inches='tight')
    fig_tort.savefig('output/tortuosity_comparison.png', dpi=300, bbox_inches='tight')
    fig_stop.savefig('output/stop_time_comparison.png', dpi=300, bbox_inches='tight')
    fig_comp.savefig('output/comprehensive_scores.png', dpi=300, bbox_inches='tight')
    
    print("\nVisualizations saved to output/ directory:")
    print("- velocity_histogram.png")
    print("- velocity_3d_scatter.png")
    print("- velocity_projections.png")
    print("- tortuosity_comparison.png")
    print("- stop_time_comparison.png")
    print("- comprehensive_scores.png")
    
    # Create a summary report
    print("\n=== SUMMARY REPORT ===")
    for i, name in enumerate(hiker_names):
        print(f"\n{name}:")
        print(f"  - Total points: {len(all_gpx_points[i])}")
        print(f"  - Tortuosity: {tortuosities[i]:.3f}")
        print(f"  - Total path length: {total_path_lengths[i]:.2f}m")
        print(f"  - Straight-line distance: {straight_line_distances[i]:.2f}m")
        print(f"  - Stop count: {stop_counts[i]}")
        print(f"  - Total stop time: {total_stop_times[i]:.2f}s")
        print(f"  - Stop time fraction: {stop_time_fractions[i]:.3f}")
        if len(leader_scores) > 0:
            print(f"  - Leader score: {leader_scores[i]:.3f}")
        print(f"  - Comprehensive score: {comprehensive_scores[i]:.3f}")
    
    # Create summary table as PNG
    print("Creating summary table...")
    # Adjust figure size for transposed table
    fig, ax = plt.subplots(figsize=(config_velocity.SUMMARY_TABLE_FIGURE_WIDTH, config_velocity.SUMMARY_TABLE_FIGURE_HEIGHT))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for the transposed table
    # Rows will be metrics, columns will be hikers
    metrics = ['Total Points', 'Tortuosity', 'Path Length (m)', 'Straight Dist (m)', 
               'Stop Count', 'Stop Time (s)', 'Stop Fraction', 'Leader Score', 'Comprehensive Score']
    
    cell_data = []
    cell_data.append([f"{len(gpx_points)}" for gpx_points in all_gpx_points])  # Total Points
    cell_data.append([f"{t:.3f}" for t in tortuosities])  # Tortuosity
    cell_data.append([f"{l:.2f}" for l in total_path_lengths])  # Path Length
    cell_data.append([f"{d:.2f}" for d in straight_line_distances])  # Straight Dist
    cell_data.append([f"{c}" for c in stop_counts])  # Stop Count
    cell_data.append([f"{t:.2f}" for t in total_stop_times])  # Stop Time
    cell_data.append([f"{f:.3f}" for f in stop_time_fractions])  # Stop Fraction
    cell_data.append([f"{s:.3f}" for s in leader_scores])  # Leader Score
    cell_data.append([f"{c:.3f}" for c in comprehensive_scores])  # Comprehensive Score
    
    # Create transposed table
    table = ax.table(cellText=cell_data,
                     rowLabels=metrics,
                     colLabels=hiker_names,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Format the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.5, 2)  # Increase both width and height scaling
    
    # Color rows for better readability
    for i in range(len(metrics)):
        for j in range(len(hiker_names)):
            if i % 2 == 0:  # Even rows (0-indexed)
                table[(i + 1, j)].set_facecolor('#f8f8f8')  # Light gray for even rows
            else:
                table[(i + 1, j)].set_facecolor('#ffffff')  # White for odd rows
    
    # Bold header row (hiker names)
    for j in range(len(hiker_names)):
        table[(0, j)].set_text_props(weight='bold')
        table[(0, j)].set_facecolor('#d0e0ff')  # Light blue for header
    
    # Bold header column (metrics)
    for i in range(len(metrics)):
        table[(i + 1, -1)].set_text_props(weight='bold')
        table[(i + 1, -1)].set_facecolor('#e0e0f0')  # Light purple for row labels
    
    ax.set_title('Hiking Metrics Summary Table', fontsize=16, pad=20)
    
    # Save the table
    fig.savefig('output/summary_table.png', dpi=300, bbox_inches='tight')
    print("- summary_table.png")
    
    # Close all plots to free memory
    plt.close('all')

if __name__ == "__main__":
    main()