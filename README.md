# GPX File Visualizer with Elevation Contours

This application visualizes GPX files with elevation contours, using techniques inspired by the [descend-mountain](https://github.com/AntarcBall/descend-mountain) repository.

## Features

- Parses GPX files to extract track points with coordinates and elevation data
- Fetches real elevation data from Google Elevation API to create contour maps
- Visualizes GPX tracks overlaid on elevation contour maps
- Displays elevation profiles along the track
- Uses .env pattern for secure API key management

## Requirements

- Python 3.7+
- Google Elevation API key (with billing enabled)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ignobel_hiker
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. Create a `.env` file in the root directory with your Google API keys:
   ```env
   # Google Elevation API Key
   GOOGLE_ELEVATION_API_KEY=your_actual_api_key_here

   # Google Maps API Key (for satellite imagery)
   # Note: You can use the same API key if it has Maps Static API enabled
   GOOGLE_MAPS_API_KEY=your_actual_api_key_here
   ```

2. Make sure you have a GPX file in the directory (the application expects `Oct_8,_2025_7_12_58_AM_Hiking_Ascent.gpx` by default)

## Usage

Run the application:
```bash
python gpx_visualizer.py
```

The application will:
1. Load the GPX file
2. Fetch elevation data for the area
3. Create two visualizations:
   - Map with GPX track overlaid on elevation contours
   - Elevation profile along the track

## How it Works

The application uses techniques from the referenced repository:

1. **GPX Parsing**: Uses the `gpxpy` library to parse GPX files
2. **Elevation Data**: Fetches real elevation data using Google Elevation API
3. **Contour Visualization**: Creates contour maps using matplotlib's contour functions
4. **Elevation Profiles**: Generates elevation vs distance plots

The visualization includes:
- Elevation contours with labels
- GPX track overlaid on the map
- Start (green) and end (red) markers
- Elevation profile chart

## Customization

To visualize a different GPX file, modify the `gpx_file_path` variable in the `main()` function.

## Troubleshooting

- If you get an API error, make sure your Google Elevation API key is valid and has the required permissions
- If the visualization doesn't show contours, make sure you have a valid API key
- For large GPX files, you may need to adjust the grid size parameter in the elevation fetching function