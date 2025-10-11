#!/usr/bin/env python3
"""
Main module for velocity analysis functionality
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from src.data_processing.gpx_parser import parse_gpx_file, get_gpx_files
from src.data_processing.data_synchronizer import synchronize_gpx_data, calculate_avg_time_interval
from src.analysis.velocity_calculator import calculate_velocity_vectors
from src.analysis.tortuosity_calculator import calculate_tortuosity
from src.analysis.stop_calculator import calculate_stop_metrics
from src.analysis.leader_calculator import calculate_leader_score
from src.analysis.score_calculator import calculate_comprehensive_score, calculate_total_time_elapsed, get_common_time_window
from src.visualization.velocity_visualizer import create_velocity_histogram, create_3d_velocity_scatter, create_velocity_projections


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
    
    # Synchronize the GPX data to common time window
    print("Synchronizing GPX data...")
    synchronized_gpx_points = synchronize_gpx_data(all_gpx_points)
    
    # Recalculate velocity vectors for synchronized data
    all_velocity_vectors = []
    all_speeds = []
    for i, gpx_points in enumerate(synchronized_gpx_points):
        if len(gpx_points) > 0:
            velocity_vectors, speeds = calculate_velocity_vectors(gpx_points)
            all_velocity_vectors.append(velocity_vectors)
            all_speeds.append(speeds)
            print(f"Synchronized data for {hiker_names[i]}: {len(gpx_points)} points, {len(velocity_vectors)} velocity vectors")
        else:
            all_velocity_vectors.append([])
            all_speeds.append([])
            print(f"No synchronized data for {hiker_names[i]}")
    
    # Calculate additional metrics
    print("\nCalculating additional metrics...")
    
    # Calculate tortuosity for each hiker
    tortuosities = []
    total_path_lengths = []
    straight_line_distances = []
    
    for i, gpx_points in enumerate(synchronized_gpx_points):
        tortuosity, total_length, straight_dist = calculate_tortuosity(gpx_points)
        tortuosities.append(tortuosity)
        total_path_lengths.append(total_length)
        straight_line_distances.append(straight_dist)
        print(f"{hiker_names[i]} - Tortuosity: {tortuosity:.3f}, Total path: {total_length:.2f}m, Straight-line: {straight_dist:.2f}m")
    
    # Get common time window for all hikers
    common_start, common_end = get_common_time_window(synchronized_gpx_points)
    
    if common_start and common_end:
        common_time_elapsed = (common_end - common_start).total_seconds()
        print(f"Common time window: {common_start} to {common_end} (Total: {common_time_elapsed:.2f}s)")
        
        # All hikers now have the same total time elapsed based on the common time window
        total_times_elapsed = [common_time_elapsed] * len(hiker_names)
    else:
        # Fallback if no common window exists
        total_times_elapsed = []
        for i, gpx_points in enumerate(synchronized_gpx_points):
            total_time = calculate_total_time_elapsed(gpx_points)
            total_times_elapsed.append(total_time)
        print("Warning: No common time window found, using individual time ranges")
    
    # Calculate average time interval between data points
    avg_time_intervals = []
    for i, gpx_points in enumerate(synchronized_gpx_points):
        avg_interval = calculate_avg_time_interval(gpx_points)
        avg_time_intervals.append(avg_interval)
        print(f"{hiker_names[i]} - Average time interval: {avg_interval:.2f}s")
    
    # Calculate stop metrics for each hiker
    stop_counts = []
    total_stop_times = []
    stop_time_fractions = []
    
    for i, (gpx_points, velocity_vectors) in enumerate(zip(synchronized_gpx_points, all_velocity_vectors)):
        stop_count, total_stop_time, stop_fraction = calculate_stop_metrics(gpx_points, velocity_vectors)
        stop_counts.append(stop_count)
        total_stop_times.append(total_stop_time)
        stop_time_fractions.append(stop_fraction)
        print(f"{hiker_names[i]} - Stops: {stop_count}, Total stop time: {total_stop_time:.2f}s, Stop fraction: {stop_fraction:.3f}")
    
    # Calculate leader scores (only if there are multiple hikers)
    leader_scores = [0] * len(hiker_names)  # Default to 0 if not enough hikers
    if len(synchronized_gpx_points) > 1:
        print("\nCalculating leader scores...")
        leader_scores = calculate_leader_score(synchronized_gpx_points)
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
        print(f"  - Total points: {len(synchronized_gpx_points[i])}")
        print(f"  - Tortuosity: {tortuosities[i]:.3f}")
        print(f"  - Total path length: {total_path_lengths[i]:.2f}m")
        print(f"  - Straight-line distance: {straight_line_distances[i]:.2f}m")
        print(f"  - Total time elapsed: {total_times_elapsed[i]:.2f}s")
        print(f"  - Average time interval: {avg_time_intervals[i]:.2f}s")
        print(f"  - Stop count: {stop_counts[i]}")
        print(f"  - Total stop time: {total_stop_times[i]:.2f}s")
        print(f"  - Stop time fraction: {stop_time_fractions[i]:.3f}")
        if len(leader_scores) > 0:
            print(f"  - Leader score: {leader_scores[i]:.3f}")
        print(f"  - Comprehensive score: {comprehensive_scores[i]:.3f}")
    
    # Create summary table as PNG
    print("Creating summary table...")
    from src.config.config_velocity import SUMMARY_TABLE_FIGURE_WIDTH, SUMMARY_TABLE_FIGURE_HEIGHT
    # Adjust figure size for transposed table
    fig, ax = plt.subplots(figsize=(SUMMARY_TABLE_FIGURE_WIDTH, SUMMARY_TABLE_FIGURE_HEIGHT))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for the transposed table
    # Rows will be metrics, columns will be hikers
    metrics = ['Total Points', 'Tortuosity', 'Path Length (m)', 'Straight Dist (m)', 
               'Total Time Elapsed (s)', 'Avg Time Interval (s)', 'Stop Count', 'Stop Time (s)', 
               'Stop Fraction', 'Leader Score', 'Comprehensive Score']
    
    cell_data = []
    cell_data.append([f"{len(gpx_points)}" for gpx_points in synchronized_gpx_points])  # Total Points
    cell_data.append([f"{t:.3f}" for t in tortuosities])  # Tortuosity
    cell_data.append([f"{l:.2f}" for l in total_path_lengths])  # Path Length
    cell_data.append([f"{d:.2f}" for d in straight_line_distances])  # Straight Dist
    cell_data.append([f"{t:.2f}" for t in total_times_elapsed])  # Total Time Elapsed
    cell_data.append([f"{t:.2f}" for t in avg_time_intervals])  # Avg Time Interval
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
    table.set_fontsize(7)  # Smaller font to fit the additional row
    table.scale(1.5, 1.6)  # Adjust scaling
    
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
    
    # Create a separate sub-scores table to show the components of the comprehensive score
    print("Creating sub-scores table...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax2.axis('tight')
    ax2.axis('off')
    
    # Subscore metrics (components of comprehensive score)
    subscore_metrics = ['Tortuosity', 'Stop Fraction', 'Leader Score', 'Weight']
    subscore_data = []
    
    # Data for each hiker
    from src.config.config_velocity import TORTUOSITY_WEIGHT, STOP_FRACTION_WEIGHT, LEADER_SCORE_WEIGHT
    for i, name in enumerate(hiker_names):
        row = [f"{tortuosities[i]:.3f}", f"{stop_time_fractions[i]:.3f}", f"{leader_scores[i]:.3f}",
               f"T:{TORTUOSITY_WEIGHT},S:{STOP_FRACTION_WEIGHT},L:{LEADER_SCORE_WEIGHT}"]
        subscore_data.append(row)
    
    # Create subscores table
    subscore_table = ax2.table(cellText=subscore_data,
                               rowLabels=hiker_names,
                               colLabels=subscore_metrics,
                               cellLoc='center',
                               loc='center',
                               bbox=[0, 0, 1, 1])
    
    # Format the subscores table
    subscore_table.auto_set_font_size(False)
    subscore_table.set_fontsize(9)
    subscore_table.scale(1.2, 2)
    
    # Color alternate rows
    for i in range(len(hiker_names)):
        for j in range(len(subscore_metrics)):
            if i % 2 == 0:  # Even rows (0-indexed)
                subscore_table[(i + 1, j)].set_facecolor('#f8f8f8')  # Light gray for even rows
            else:
                subscore_table[(i + 1, j)].set_facecolor('#ffffff')  # White for odd rows
    
    # Bold header row (subscore metrics)
    for j in range(len(subscore_metrics)):
        subscore_table[(0, j)].set_text_props(weight='bold')
        subscore_table[(0, j)].set_facecolor('#d0e0ff')  # Light blue for header
    
    # Bold header column (hiker names)
    for i in range(len(hiker_names)):
        subscore_table[(i + 1, -1)].set_text_props(weight='bold')
        subscore_table[(i + 1, -1)].set_facecolor('#e0e0f0')  # Light purple for row labels
    
    ax2.set_title('Sub-scores for Comprehensive Score Calculation', fontsize=14, pad=20)
    
    # Save the subscores table
    fig2.savefig('output/subscores_table.png', dpi=300, bbox_inches='tight')
    print("- subscores_table.png")
    
    # Close all plots to free memory
    plt.close('all')


if __name__ == "__main__":
    main()