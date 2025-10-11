#!/usr/bin/env python3
"""
Module for velocity data visualizations
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        
    from src.config.config_velocity import VELOCITY_HISTOGRAM_BINS
    bins = np.linspace(0, max(all_speeds) * 1.1, VELOCITY_HISTOGRAM_BINS)
    
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