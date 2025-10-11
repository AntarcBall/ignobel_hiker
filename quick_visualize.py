#!/usr/bin/env python3
"""
One-line command script to visualize actual terrain data
"""
import os
import sys
import matplotlib
# Try to use TkAgg backend for GUI if available, otherwise Agg for file output
try:
    if os.environ.get('DISPLAY'):
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
except:
    matplotlib.use('Agg')

from src.visualization.terrain_visualization_main import main

if __name__ == "__main__":
    main()