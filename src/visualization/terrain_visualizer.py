#!/usr/bin/env python3
"""
Module for terrain visualization
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.config.config_3d import (
    VERTICAL_EXAGGERATION, TERRAIN_COLORMAP, TERRAIN_ALPHA,
    PATH_3D_LINEWIDTH, PATH_3D_COLORS, PATH_ELEVATION_BIAS,
    SHOW_START_END_MARKERS, SHOW_GROUND_PROJECTION, PROJECTION_ALPHA,
    PROJECTION_LINESTYLE, VIEW_ELEVATION_ANGLE, VIEW_AZIMUTH_ANGLE,
    FIGURE_SIZE_3D
)


def visualize_terrain_with_contours_and_paths(Z, LAT, LON, gpx_tracks, vertical_exaggeration=VERTICAL_EXAGGERATION):
    """Create 3D visualization with terrain surface, contour lines, and hiker paths"""
    fig = plt.figure(figsize=FIGURE_SIZE_3D)
    ax = fig.add_subplot(111, projection='3d')

    # Apply vertical exaggeration
    Z_3d = Z * vertical_exaggeration

    # Plot the terrain surface
    surf = ax.plot_surface(LON, LAT, Z_3d, cmap=TERRAIN_COLORMAP, alpha=TERRAIN_ALPHA, 
                          linewidth=0, antialiased=True, shade=True)

    # Add contour lines projected at the bottom
    levels = np.linspace(np.min(Z_3d), np.max(Z_3d), 15)
    for level in levels[::2]:  # Show every other contour for clarity
        ax.contour(LON, LAT, Z_3d, levels=[level], colors='black', 
                  alpha=0.4, linewidths=0.5, offset=np.min(Z_3d)-5)

    # Plot hiker paths
    for i, track in enumerate(gpx_tracks):
        if len(track) == 0:
            continue
            
        # Extract coordinates
        lats = np.array([p['latitude'] for p in track])
        lons = np.array([p['longitude'] for p in track])
        elevs = np.array([p['elevation'] if p['elevation'] is not None else np.nanmedian(Z) for p in track])
        
        # Interpolate GPX elevations to terrain grid to get accurate heights at track locations
        # We'll use the terrain elevation at the closest point and apply elevation bias
        elevs_interp = []
        for j, (lat, lon, gpx_elev) in enumerate(zip(lats, lons, elevs)):
            # Find the closest grid point in the terrain
            dist = (LAT - lat)**2 + (LON - lon)**2
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            terrain_elev = Z[idx]  # Get terrain elevation at this location
            
            # Apply elevation bias to lift path above terrain for better visibility
            elev_with_bias = terrain_elev + PATH_ELEVATION_BIAS
            
            elevs_interp.append(elev_with_bias * vertical_exaggeration)
        
        elevs_interp = np.array(elevs_interp)
        
        # Plot the path
        color = PATH_3D_COLORS[i % len(PATH_3D_COLORS)]
        ax.plot(lons, lats, elevs_interp, color=color, linewidth=PATH_3D_LINEWIDTH, 
                label=f'Hiker {i+1}', zorder=10)
        
        # Add start and end markers if enabled
        if SHOW_START_END_MARKERS:
            # Apply bias to markers as well
            start_idx = np.unravel_index(np.argmin((LAT - lats[0])**2 + (LON - lons[0])**2), Z.shape)
            end_idx = np.unravel_index(np.argmin((LAT - lats[-1])**2 + (LON - lons[-1])**2), Z.shape)
            start_elev = (Z[start_idx] + PATH_ELEVATION_BIAS) * vertical_exaggeration
            end_elev = (Z[end_idx] + PATH_ELEVATION_BIAS) * vertical_exaggeration
            
            ax.scatter([lons[0]], [lats[0]], [start_elev], color=color, s=200, 
                      marker='o', edgecolors='black', linewidth=2, zorder=11, label=f'Hiker {i+1} Start')
            ax.scatter([lons[-1]], [lats[-1]], [end_elev], color=color, s=200, 
                      marker='s', edgecolors='black', linewidth=2, zorder=11, label=f'Hiker {i+1} End')
        
        # Add ground projection if enabled
        if SHOW_GROUND_PROJECTION:
            z_ground = np.full_like(elevs_interp, np.min(Z_3d) - 10)
            ax.plot(lons, lats, z_ground, color=color, 
                   linestyle=PROJECTION_LINESTYLE, alpha=PROJECTION_ALPHA, 
                   linewidth=PATH_3D_LINEWIDTH/2, zorder=1)

    # Set labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Visualization of Actual Terrain with Hiker Paths')

    # Set viewing angle from config
    ax.view_init(elev=VIEW_ELEVATION_ANGLE, azim=VIEW_AZIMUTH_ANGLE)

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')

    # Add legend for hiker paths
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

    # Set pane properties for cleaner background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    plt.tight_layout()
    return fig, ax