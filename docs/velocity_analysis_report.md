# Hiking Velocity Analysis Report

## Overview

This report details the methodology and results of an analysis comparing hiking performance metrics for multiple hikers using GPS trajectory data. The analysis includes advanced techniques for synchronizing data from different sources and calculating comprehensive performance metrics.

## Key Methodology

### 1. Local 3D Coordinate System (ENU)

All GPS coordinates (latitude, longitude, altitude) are converted to a local East-North-Up (ENU) coordinate system using the first recorded point as the reference. This allows for accurate 3D spatial analysis in a locally flat coordinate system:

- X-axis: East displacement
- Y-axis: North displacement  
- Z-axis: Up (altitude) displacement

### 2. Data Synchronization Using Common Time Window

To ensure fair comparison between hikers with potentially different track durations, all data is synchronized to a common time window. The algorithm finds:
- **Common start time**: Latest start time among all hikers
- **Common end time**: Earliest end time among all hikers
- All hikers' data is truncated to this shared time span (793.00 seconds in this analysis)

### 3. Linear Interpolation for Time Alignment

**Critical Component**: To enable precise analysis of leader behavior and synchronized metrics, a linear interpolation method is used to align data points across hikers:

For each time interval (configurable, default 10 seconds):
1. **Temporal Alignment**: GPS coordinates are estimated at identical time points for all hikers
2. **Linear Interpolation**: For time t, if a hiker's GPS points are at (t₁, lat₁, lon₁) and (t₂, lat₂, lon₂), the position at time t is calculated as:
   ```
   lat(t) = lat₁ + ((t - t₁) / (t₂ - t₁)) × (lat₂ - lat₁)
   lon(t) = lon₁ + ((t - t₁) / (t₂ - t₁)) × (lon₂ - lon₁)
   ```
3. **Synchronized Analysis**: This creates a synchronized dataset where all hikers have estimated positions at identical time points, enabling precise leader detection and comparative analysis.

### 4. Velocity and Speed Calculations

- **Displacement Vectors**: Calculated between consecutive ENU coordinate points
- **Velocity Vectors**: Displacement divided by time interval
- **Speed**: Magnitude of velocity vector

### 5. Stop Detection Algorithm

Stops are identified using configurable parameters:
- **Minimum Speed Threshold**: 1.0 m/s (speed below which movement is considered stopped)
- **Minimum Stop Duration**: 10 seconds (minimum duration for a valid stop)

## Performance Metrics

### Primary Metrics

| Metric | Description |
|--------|-------------|
| Total Points | Total number of GPS data points in synchronized time window |
| Tortuosity | Total path length / straight-line distance (higher = more circuitous route) |
| Path Length | Total distance traveled (m) |
| Straight-line Distance | Direct distance from start to end (m) |
| Total Time Elapsed | Synchronized time window (793.00 s) |
| Avg Time Interval | Average time between consecutive GPS points (s) |
| Stop Count | Number of stop events |
| Stop Time | Total stopped time (s) |
| Stop Fraction | Stop time / total time (proportion of time stopped) |

### Advanced Metrics

#### Leader Score
- Calculated using linear interpolation at 10s intervals
- Determines who leads at each time point by calculating distance from group center
- Fraction of total time spent in leading position

#### Comprehensive Score
Weighted sum of key metrics:
```
Comprehensive Score = 1.0 × Tortuosity + 1.0 × Stop Fraction + 1.0 × Leader Score
```

## Results Summary

| Hiker | Total Points | Tortuosity | Path Length (m) | Straight Dist (m) | Time (s) | Avg Interval (s) | Stop Count | Stop Time (s) | Stop Fraction | Leader Score | Comprehensive Score |
|-------|-------------|------------|-----------------|------------------|----------|-----------------|------------|---------------|---------------|--------------|---------------------|
| H | 55 | 1.321 | 574.48 | 434.98 | 793.00 | 14.69 | 8 | 644.00 | 0.812 | 0.543 | 2.676 |
| M | 120 | 1.282 | 1257.41 | 980.44 | 793.00 | 11.57 | 7 | 722.00 | 0.524 | 0.275 | 2.082 |
| Z | 117 | 1.293 | 1261.97 | 975.89 | 793.00 | 11.78 | 16 | 744.00 | 0.545 | 0.174 | 2.012 |

## Key Findings

### Hiker H
- **Highest Tortuosity**: Most circuitous route (1.321)
- **Highest Stop Fraction**: Spent 81.2% of time stopped
- **Highest Leader Score**: Led for 54.3% of the time
- **Highest Comprehensive Score**: Overall highest score (2.676)
- **Lowest Point Density**: 14.69s average interval between points

### Hiker M (Middle performance)
- **Balanced Performance**: Moderate values across metrics
- **High Path Length**: Traveled 1257.41m in the time window
- **Second Highest Stop Time**: 722.00s stopped
- **Second Highest Leader Score**: Led for 27.5% of the time

### Hiker Z
- **Second Highest Path Length**: Traveled 1261.97m
- **Highest Stop Count**: 16 separate stop events
- **Lowest Leader Score**: Led for only 17.4% of the time
- **Lowest Comprehensive Score**: Despite longer path, lowest overall score (2.012)

## Linear Interpolation Significance

The linear interpolation method is crucial for several analyses:

1. **Leader Detection**: By synchronizing positions to identical time points, we can accurately determine who is leading at any moment
2. **Comparative Analysis**: Enables fair comparison of metrics computed over the same time window
3. **Stop Synchronization**: Ensures stop events are compared in the same temporal context
4. **Group Dynamics**: Allows analysis of how hikers' positions relative to each other evolve over time

This interpolation technique addresses the challenge of comparing GPS tracks with potentially different sampling rates and timing, providing a robust foundation for multi-hiker analysis.

## Conclusion

The analysis reveals distinct hiking styles among the three hikers. Hiker H demonstrated a more contemplative approach with frequent stops and complex routing (highest tortuosity), while Hikers M and Z had more direct paths but different stopping patterns. The linear interpolation methodology ensures that all comparisons are made over the same synchronized temporal framework, providing meaningful and fair metric comparisons.