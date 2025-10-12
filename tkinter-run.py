#!/usr/bin/env python3
"""
Simple GUI with 2 buttons to run the main wrapper scripts
"""
import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import sys
import os


def run_visualization():
    """Run the terrain visualization script"""
    subprocess.Popen([sys.executable, 'quick_visualize.py'])


def run_analysis():
    """Run the velocity analysis script"""
    subprocess.Popen([sys.executable, 'run_velocity_analysis.py'])


def main():
    root = tk.Tk()
    root.title("Hiker Analysis Tool")
    root.geometry("400x150")
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Create a main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Configure grid
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(0, weight=1)
    
    # First button - Run Terrain Visualization
    viz_button = ttk.Button(
        main_frame,
        text="Run Terrain Visualization",
        command=run_visualization
    )
    viz_button.grid(row=0, column=0, padx=10, pady=20, sticky=(tk.W, tk.E))
    
    # Second button - Run Velocity Analysis
    analysis_button = ttk.Button(
        main_frame,
        text="Run Velocity Analysis",
        command=run_analysis
    )
    analysis_button.grid(row=0, column=1, padx=10, pady=20, sticky=(tk.W, tk.E))
    
    root.mainloop()


if __name__ == "__main__":
    main()