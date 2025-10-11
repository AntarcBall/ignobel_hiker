"""
GPX Visualizer with Interactive UI
This script provides a dynamic UI window with interactive 2D and 3D visualizations of GPX data.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend for embedding in PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import plotly.graph_objects as go
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                            QWidget, QPushButton, QHBoxLayout, QComboBox,
                            QLabel, QFileDialog, QTabWidget, QSplitter)
from PyQt5.QtCore import Qt
import gpxpy
import elevation_processor
from config import GRID_SIZE, HIKER_COLORS, HIKER_NAMES
from config_3d import VERTICAL_EXAGGERATION, PATH_3D_COLORS


class GPXVisualizerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive GPX Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data
        self.gpx_files = []
        self.all_gpx_points = []
        self.elevation_grid = None
        self.LAT = None
        self.LON = None
        self.selected_gpx_idx = 0
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs for different visualizations
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create widgets for different views
        self._create_control_panel(main_layout)
        self._create_2d_tab()
        self._create_3d_tab()
        self._create_plotly_3d_tab()
        
        # Load GPX files automatically
        self.load_gpx_files()
        
    def _create_control_panel(self, parent_layout):
        """Create control panel for loading and selecting GPX files"""
        control_layout = QHBoxLayout()
        
        # Load GPX files button
        load_btn = QPushButton("Load GPX Files")
        load_btn.clicked.connect(self.load_gpx_files)
        control_layout.addWidget(load_btn)
        
        # GPX file selection combo
        self.gpx_combo = QComboBox()
        self.gpx_combo.currentIndexChanged.connect(self.on_gpx_selection_changed)
        control_layout.addWidget(QLabel("Select GPX File:"))
        control_layout.addWidget(self.gpx_combo)
        
        parent_layout.addLayout(control_layout)
        
    def _create_2d_tab(self):
        """Create 2D visualization tab"""
        # Create figure for 2D visualization
        self.fig_2d = Figure(figsize=(10, 8))
        self.canvas_2d = FigureCanvas(self.fig_2d)
        
        # Create widget and add to tab
        widget_2d = QWidget()
        layout_2d = QVBoxLayout(widget_2d)
        layout_2d.addWidget(self.canvas_2d)
        
        self.tabs.addTab(widget_2d, "2D Visualization")
        
    def _create_3d_tab(self):
        """Create 3D visualization tab using matplotlib"""
        # Create figure for 3D visualization
        self.fig_3d = Figure(figsize=(10, 8))
        self.canvas_3d = FigureCanvas(self.fig_3d)
        
        # Create widget and add to tab
        widget_3d = QWidget()
        layout_3d = QVBoxLayout(widget_3d)
        layout_3d.addWidget(self.canvas_3d)
        
        self.tabs.addTab(widget_3d, "3D Visualization (Matplotlib)")
        
    def _create_plotly_3d_tab(self):
        """Create 3D visualization tab using Plotly (interactive)"""
        # Create a placeholder for the interactive 3D view
        # For now we'll add a label with instructions, as Plotly integration in PyQt
        # requires more complex implementation
        widget_plotly = QWidget()
        layout_plotly = QVBoxLayout(widget_plotly)
        
        label = QLabel("Interactive 3D Visualization\nThis tab will show an interactive 3D view using Plotly.\nClick 'Generate Interactive 3D' to export an HTML file.")
        label.setAlignment(Qt.AlignCenter)
        
        generate_btn = QPushButton("Generate Interactive 3D HTML")
        generate_btn.clicked.connect(self.export_interactive_3d)
        
        layout_plotly.addWidget(label)
        layout_plotly.addWidget(generate_btn, alignment=Qt.AlignCenter)
        
        self.tabs.addTab(widget_plotly, "3D Interactive (HTML)")
        
    def load_gpx_files(self):
        """Load GPX files from the gpx_data directory"""
        # Get GPX files from gpx_data directory
        gpx_dir = 'gpx_data'
        self.gpx_files = []
        
        if os.path.exists(gpx_dir):
            for file in os.listdir(gpx_dir):
                if file.lower().endswith('.gpx'):
                    self.gpx_files.append(os.path.join(gpx_dir, file))
        
        # Update combo box
        self.gpx_combo.clear()
        for gpx_file in self.gpx_files:
            self.gpx_combo.addItem(os.path.basename(gpx_file))
            
        # Load GPX data
        self.all_gpx_points = []
        for gpx_file in self.gpx_files:
            gpx_points = self._parse_gpx_file(gpx_file)
            self.all_gpx_points.append(gpx_points)
        
        # Update UI if we have data
        if self.all_gpx_points:
            self._update_visualizations()
        
    def _parse_gpx_file(self, file_path):
        """Parse a GPX file and extract track points"""
        with open(file_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
        
        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    points.append({
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation,
                        'time': point.time
                    })
        
        return points
    
    def on_gpx_selection_changed(self):
        """Handle when a different GPX file is selected"""
        self.selected_gpx_idx = self.gpx_combo.currentIndex()
        if self.selected_gpx_idx < len(self.all_gpx_points):
            self._update_visualizations()
    
    def _update_visualizations(self):
        """Update all visualizations with the current GPX data"""
        if not self.all_gpx_points or self.selected_gpx_idx >= len(self.all_gpx_points):
            return
        
        gpx_points = self.all_gpx_points[self.selected_gpx_idx]
        
        # Calculate center coordinates for elevation
        all_lats = [p['latitude'] for p in gpx_points]
        all_lons = [p['longitude'] for p in gpx_points]
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        # Fetch elevation data (this will use cache if available)
        print("Fetching elevation data...")
        self.elevation_grid, self.LAT, self.LON = elevation_processor.get_real_elevation_data_around_coords(
            center_lat, center_lon, None, size=GRID_SIZE
        )
        
        # Update 2D visualization
        self._update_2d_visualization(gpx_points)
        
        # Update 3D visualization
        self._update_3d_visualization(gpx_points)
    
    def _update_2d_visualization(self, gpx_points):
        """Update the 2D visualization"""
        self.fig_2d.clear()
        ax = self.fig_2d.add_subplot(111)
        
        # Plot elevation contours if available
        if self.elevation_grid is not None and self.LAT is not None and self.LON is not None:
            from config import CONTOUR_LEVEL_STEP, CONTOUR_ALPHA, CONTOUR_LINEWIDTH
            
            min_elev = np.nanmin(self.elevation_grid)
            max_elev = np.nanmax(self.elevation_grid)
            
            step = CONTOUR_LEVEL_STEP
            contour_levels = np.arange(
                np.floor(min_elev / step) * step,
                np.ceil(max_elev / step) * step + step / 2.0,
                step
            )
            
            ax.contourf(self.LON, self.LAT, self.elevation_grid, levels=contour_levels, 
                       cmap='terrain', alpha=0.5)
            
            # Plot contour lines
            contour = ax.contour(self.LON, self.LAT, self.elevation_grid, 
                                levels=contour_levels, colors='black', 
                                alpha=CONTOUR_ALPHA, linewidths=CONTOUR_LINEWIDTH)
            ax.clabel(contour, inline=True, fontsize=8, fmt='%.0fm')
        
        # Extract GPX coordinates for plotting
        lats = [p['latitude'] for p in gpx_points]
        lons = [p['longitude'] for p in gpx_points]
        
        # Plot GPX track
        ax.plot(lons, lats, 'r-', linewidth=2, label='GPX Track', zorder=5)
        ax.scatter(lons[0], lats[0], color='green', s=100, zorder=6, 
                  label='Start', edgecolors='black')  # Start point
        ax.scatter(lons[-1], lats[-1], color='red', s=100, zorder=6, 
                  label='End', edgecolors='black')    # End point
        
        # Calculate axis limits to focus on the GPX path
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        
        # Add buffer around the path
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        
        from config import ZOOM_FACTOR_X, ZOOM_FACTOR_Y, AXIS_BUFFER
        lat_range = (lat_max - lat_min) * ZOOM_FACTOR_Y
        lon_range = (lon_max - lon_min) * ZOOM_FACTOR_X
        
        ax.set_xlim(lon_center - lon_range/2 - AXIS_BUFFER, 
                   lon_center + lon_range/2 + AXIS_BUFFER)
        ax.set_ylim(lat_center - lat_range/2 - AXIS_BUFFER, 
                   lat_center + lat_range/2 + AXIS_BUFFER)
        
        ax.set_title('GPX Track with Elevation Contours')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig_2d.tight_layout()
        self.canvas_2d.draw()
    
    def _update_3d_visualization(self, gpx_points):
        """Update the 3D visualization"""
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d')
        
        # Create 3D terrain surface if elevation data is available
        if self.elevation_grid is not None:
            from config_3d import TERRAIN_COLORMAP, TERRAIN_ALPHA, SURFACE_STRIDE
            
            # Create coordinate grids if not available
            if self.LAT is None or self.LON is None:
                from config import HALF_SIZE_KM
                lat_step = (HALF_SIZE_KM * 2) / (111.0 * GRID_SIZE)
                lon_step = (HALF_SIZE_KM * 2) / (111.0 * np.cos(np.radians(np.mean([p['latitude'] for p in gpx_points]))) * GRID_SIZE)
                
                center_lat = np.mean([p['latitude'] for p in gpx_points])
                center_lon = np.mean([p['longitude'] for p in gpx_points])
                
                min_lat = center_lat - (GRID_SIZE // 2) * lat_step
                max_lat = center_lat + (GRID_SIZE // 2) * lat_step
                min_lon = center_lon - (GRID_SIZE // 2) * lon_step
                max_lon = center_lon + (GRID_SIZE // 2) * lon_step
                
                lats = np.linspace(min_lat, max_lat, GRID_SIZE)
                lons = np.linspace(min_lon, max_lon, GRID_SIZE)
                self.LAT, self.LON = np.meshgrid(lons, lats)
            
            # Create 3D terrain
            X = self.LON
            Y = self.LAT
            Z = self.elevation_grid * VERTICAL_EXAGGERATION
            
            # Plot terrain surface
            ax.plot_surface(X, Y, Z, cmap=TERRAIN_COLORMAP, 
                           alpha=TERRAIN_ALPHA, 
                           rstride=SURFACE_STRIDE, 
                           cstride=SURFACE_STRIDE,
                           linewidth=0, antialiased=True)
        
        # Extract GPX coordinates for 3D plotting
        lons = np.array([p['longitude'] for p in gpx_points])
        lats = np.array([p['latitude'] for p in gpx_points])
        
        # Get elevation for GPX path from the grid
        elevs = []
        if self.elevation_grid is not None and self.LAT is not None and self.LON is not None:
            from scipy.interpolate import griddata
            
            for lon, lat in zip(lons, lats):
                points = np.column_stack((self.LAT.ravel(), self.LON.ravel()))
                values = self.elevation_grid.ravel()
                interp_elev = griddata(points, values, (lat, lon), method='linear')
                
                if np.isnan(interp_elev):
                    interp_elev = gpx_points[0]['elevation'] or 0
                
                elevs.append(interp_elev * VERTICAL_EXAGGERATION)
        else:
            # Use GPX elevation if no terrain data
            elevs = np.array([p['elevation'] or 0 for p in gpx_points]) * VERTICAL_EXAGGERATION
        
        elevs = np.array(elevs)
        
        # Plot the GPX path in 3D
        ax.plot(lons, lats, elevs, color='red', linewidth=3, 
                label='GPX Track', zorder=10)
        
        # Add start/end markers
        ax.scatter(lons[0], lats[0], elevs[0], color='green', s=150, 
                  marker='o', edgecolors='black', linewidth=2, zorder=11)
        ax.scatter(lons[-1], lats[-1], elevs[-1], color='red', s=150, 
                  marker='s', edgecolors='black', linewidth=2, zorder=11)
        
        # Set view angle
        ax.view_init(elev=20, azim=45)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Elevation (m)')
        ax.set_title('3D Mountain Terrain with GPX Path')
        
        ax.legend()
        
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
    
    def export_interactive_3d(self):
        """Export interactive 3D visualization as HTML"""
        if not self.all_gpx_points or self.selected_gpx_idx >= len(self.all_gpx_points):
            print("No GPX data to visualize")
            return
        
        gpx_points = self.all_gpx_points[self.selected_gpx_idx]
        
        # Create Plotly 3D visualization
        fig = go.Figure()
        
        # Add terrain surface if available
        if self.elevation_grid is not None:
            # Create 3D terrain
            X = self.LON if self.LON is not None else np.linspace(-1, 1, GRID_SIZE)
            Y = self.LAT if self.LAT is not None else np.linspace(-1, 1, GRID_SIZE)
            Z = self.elevation_grid * VERTICAL_EXAGGERATION if self.elevation_grid is not None else np.random.rand(GRID_SIZE, GRID_SIZE) * 100
            
            fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='earth', opacity=0.8, name='Terrain'))
        
        # Extract GPX coordinates for 3D plotting
        lons = [p['longitude'] for p in gpx_points]
        lats = [p['latitude'] for p in gpx_points]
        
        # Get elevation for GPX path
        elevs = []
        if self.elevation_grid is not None and self.LAT is not None and self.LON is not None:
            from scipy.interpolate import griddata
            
            for lon, lat in zip(lons, lats):
                points = np.column_stack((self.LAT.ravel(), self.LON.ravel()))
                values = self.elevation_grid.ravel()
                interp_elev = griddata(points, values, (lat, lon), method='linear')
                
                if np.isnan(interp_elev):
                    interp_elev = gpx_points[0]['elevation'] or 0
                
                elevs.append(interp_elev * VERTICAL_EXAGGERATION)
        else:
            elevs = [p['elevation'] or 0 for p in gpx_points]
        
        # Plot GPX path
        fig.add_trace(go.Scatter3d(
            x=lons, y=lats, z=elevs,
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(size=3, color='red'),
            name='GPX Track'
        ))
        
        # Add start and end markers
        fig.add_trace(go.Scatter3d(
            x=[lons[0]], y=[lats[0]], z=[elevs[0]],
            mode='markers',
            marker=dict(size=8, color='green', symbol='circle'),
            name='Start'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[lons[-1]], y=[lats[-1]], z=[elevs[-1]],
            mode='markers',
            marker=dict(size=8, color='red', symbol='square'),
            name='End'
        ))
        
        fig.update_layout(
            title='Interactive 3D GPX Visualization',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Elevation (m)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1000,
            height=700
        )
        
        # Export to HTML
        output_path = f"output/interactive_gpx_3d_{os.path.basename(self.gpx_files[self.selected_gpx_idx]).replace('.gpx', '')}.html"
        fig.write_html(output_path)
        print(f"Interactive 3D visualization exported to {output_path}")


def main():
    app = QApplication(sys.argv)
    window = GPXVisualizerUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()