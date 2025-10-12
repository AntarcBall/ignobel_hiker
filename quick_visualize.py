#!/usr/bin/env python3
"""
One-line command script to visualize actual terrain data
"""
import os
import sys
import platform
import matplotlib

# Use TkAgg backend for interactive plots on non-Linux systems
if platform.system() != 'Linux':
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Agg')
else:
    # On Linux, check if DISPLAY is available for GUI support
    if os.environ.get('DISPLAY'):
        try:
            matplotlib.use('TkAgg')
        except:
            matplotlib.use('Agg')
    else:
        matplotlib.use('Agg')  # Use non-interactive backend when no display is available

from src.visualization.terrain_visualization_main import main

if __name__ == "__main__":
    main()