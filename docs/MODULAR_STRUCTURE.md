# Ignobel Hiker - Modular Structure

This repository has been restructured to be more modular and maintainable. Here's the new organization:

## Directory Structure

```
src/
├── analysis/                 # Analysis-specific modules
│   ├── velocity_calculator.py
│   ├── tortuosity_calculator.py
│   ├── stop_calculator.py
│   ├── leader_calculator.py
│   ├── score_calculator.py
│   └── velocity_analysis_main.py
├── api/                      # API and external service modules
│   ├── elevation_api.py
│   └── __init__.py
├── config/                   # Configuration files
│   ├── config_3d.py
│   ├── config_velocity.py
│   └── __init__.py
├── data_processing/          # Data processing modules
│   ├── gpx_processor.py
│   ├── gpx_parser.py
│   ├── data_synchronizer.py
│   └── __init__.py
├── utils/                    # Utility functions
│   ├── coordinate_converter.py
│   ├── coordinate_utils.py
│   ├── distance_calculator.py
│   └── __init__.py
├── visualization/            # Visualization modules
│   ├── terrain_visualizer.py
│   ├── velocity_visualizer.py
│   └── terrain_visualization_main.py
└── cache/                    # Cache files and management
    └── __init__.py
```

## Module Descriptions

### Analysis
- `velocity_calculator.py`: Calculates velocity vectors and speeds from GPX data
- `tortuosity_calculator.py`: Computes tortuosity metrics (path length vs straight-line distance)
- `stop_calculator.py`: Analyzes stopping behavior and duration
- `leader_calculator.py`: Determines leader scores based on position relative to group
- `score_calculator.py`: Computes comprehensive scores and time metrics

### API
- `elevation_api.py`: Handles elevation data fetching from Google Elevation API with caching

### Config
- `config_3d.py`: Configuration for 3D terrain visualization
- `config_velocity.py`: Configuration for velocity analysis

### Data Processing
- `gpx_processor.py`: Processes GPX files to extract tracks and bounds
- `gpx_parser.py`: Parses GPX files and applies time filtering
- `data_synchronizer.py`: Synchronizes GPX data across multiple hikers

### Utils
- `coordinate_converter.py`: Converts coordinates between LLA and ENU systems
- `coordinate_utils.py`: Utility functions for coordinate transformations and grid generation
- `distance_calculator.py`: Calculates distances using Haversine formula

### Visualization
- `terrain_visualizer.py`: Creates 3D terrain visualizations with hiker paths
- `velocity_visualizer.py`: Creates visualizations for velocity analysis results
- `terrain_visualization_main.py`: Main entry point for terrain visualization
- `velocity_analysis_main.py`: Main entry point for velocity analysis

## Entry Points

- `quick_visualize.py`: Entry point for terrain visualization
- `run_velocity_analysis.py`: Entry point for velocity analysis
- `tkinter-run.py`: GUI entry point for both visualization and analysis
- `requirements.txt`: Project dependencies file

## Installation

Korean installation instructions:
```
pip install -r requirements.txt
python tkinter-run.py
```

## Benefits of This Structure

1. **Modularity**: Each module has a single responsibility and can be tested independently
2. **Maintainability**: Clear separation of concerns makes code easier to maintain
3. **Reusability**: Modules can be imported and reused in other contexts
4. **Readability**: Clear naming and organization makes it easy to understand the codebase
5. **Scalability**: New features can be added without disrupting existing functionality