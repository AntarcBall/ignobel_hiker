import os
# Set matplotlib backend before importing pyplot to avoid Qt issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.interpolate import griddata
import gpxpy
import requests
from dotenv import load_dotenv
import math
import pickle

# Load configuration files
import config
from config_3d import *
import elevation_processor

# Load environment variables
load_dotenv()

def load_api_keys():
    """Load API keys from environment variables"""
    elevation_api_key = os.getenv('ELEVATION_API_KEY')
    maps_api_key = os.getenv('MAPS_API_KEY')
    
    if not elevation_api_key:
        print("Warning: ELEVATION_API_KEY not found in environment")
    if not maps_api_key:
        print("Warning: MAPS_API_KEY not found in environment")
        
    return elevation_api_key, maps_api_key

def get_gpx_files():
    """Get all GPX files from the gpx_data directory"""
    gpx_dir = 'gpx_data'
    gpx_files = []
    
    if os.path.exists(gpx_dir):
        for file in os.listdir(gpx_dir):
            if file.lower().endswith('.gpx'):
                gpx_files.append(os.path.join(gpx_dir, file))
    
    return gpx_files

def parse_gpx_file(gpx_file_path):
    """Parse GPX file and extract track points"""
    try:
        with open(gpx_file_path, 'r') as gpx_file:
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
    except Exception as e:
        print(f"Error parsing GPX file {gpx_file_path}: {e}")
        return []

def get_cache_filename_for_gpx(gpx_file_path):
    """Generate a cache filename based on GPX file path"""
    # Create a hash of the file path to use as part of the filename
    import hashlib
    file_hash = hashlib.md5(gpx_file_path.encode()).hexdigest()[:12]
    cache_dir = getattr(config, 'ELEVATION_CACHE_DIR', 'cache')
    return f"{cache_dir}/gpx_points_{file_hash}.pkl"

def get_cached_gpx_data(gpx_file_path):
    """Try to load cached GPX data"""
    cache_file = get_cache_filename_for_gpx(gpx_file_path)
    
    if os.path.exists(cache_file):
        try:
            # Check if the original GPX file is newer than the cache
            gpx_mtime = os.path.getmtime(gpx_file_path)
            cache_mtime = os.path.getmtime(cache_file)
            
            if gpx_mtime <= cache_mtime:
                with open(cache_file, 'rb') as f:
                    points = pickle.load(f)
                print(f"Loaded cached GPX data from {cache_file}")
                return points
        except Exception as e:
            print(f"Error loading cached GPX data: {e}")
    
    return None

def cache_gpx_data(gpx_file_path, points):
    """Cache GPX data to file"""
    cache_file = get_cache_filename_for_gpx(gpx_file_path)
    cache_dir = getattr(config, 'ELEVATION_CACHE_DIR', 'cache')
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(points, f)
        print(f"Cached GPX data to {cache_file}")
    except Exception as e:
        print(f"Error caching GPX data: {e}")

def get_gpx_points_with_cache(gpx_file_path):
    """Get GPX points with caching to avoid repeated parsing"""
    # First, try to load from cache
    cached_points = get_cached_gpx_data(gpx_file_path)
    if cached_points is not None:
        return cached_points

    # If not in cache, parse the file
    points = parse_gpx_file(gpx_file_path)
    
    # Cache the result if parsing was successful
    if points:
        cache_gpx_data(gpx_file_path, points)
    
    return points

def get_real_elevation_data_around_coords(center_lat, center_lon, api_key, size=config.GRID_SIZE):
    """Fetch real elevation data around given coordinates using the elevation_processor module"""
    # Use the elevation_processor module to fetch and cache elevation data
    try:
        elevation_grid, LAT, LON = elevation_processor.get_real_elevation_data_around_coords(
            center_lat, center_lon, api_key, size=size
        )
        
        # If LAT or LON are None (which happens when using synthetic data), create them
        if LAT is None or LON is None:
            # Calculate coordinate steps based on area size and grid dimensions
            # Assuming the area is HALF_SIZE_KM kilometers in each direction from center
            # Approximate conversion: 1 degree of latitude ~ 111 km
            lat_step = (config.HALF_SIZE_KM * 2) / (111.0 * size)
            # For longitude, this varies with latitude, but for small areas we can approximate
            lon_step = (config.HALF_SIZE_KM * 2) / (111.0 * math.cos(math.radians(center_lat)) * size)
            
            min_lat = center_lat - (size // 2) * lat_step
            max_lat = center_lat + (size // 2) * lat_step
            min_lon = center_lon - (size // 2) * lon_step
            max_lon = center_lon + (size // 2) * lon_step
            
            # Create coordinate grids
            lats = np.linspace(min_lat, max_lat, size)
            lons = np.linspace(min_lon, max_lon, size)
            LAT, LON = np.meshgrid(lons, lats)
        
        return elevation_grid, LAT, LON
    except Exception as e:
        print(f"Error fetching real elevation data: {e}")
        # Fallback to simulated data
        return get_simulated_elevation_data_around_coords(center_lat, center_lon, size)


def get_simulated_elevation_data_around_coords(center_lat, center_lon, size=config.GRID_SIZE):
    """Generate simulated elevation data using the elevation_processor module"""
    print("Generating simulated elevation data...")
    
    # Use the elevation_processor module to generate realistic terrain
    elevation_grid = elevation_processor.generate_realistic_terrain(size, min_elevation=0.0, max_elevation=1000.0)
    
    # Calculate coordinate steps based on area size and grid dimensions
    # Assuming the area is HALF_SIZE_KM kilometers in each direction from center
    # Approximate conversion: 1 degree of latitude ~ 111 km
    lat_step = (config.HALF_SIZE_KM * 2) / (111.0 * size)
    # For longitude, this varies with latitude, but for small areas we can approximate
    lon_step = (config.HALF_SIZE_KM * 2) / (111.0 * math.cos(math.radians(center_lat)) * size)
    
    min_lat = center_lat - (size // 2) * lat_step
    max_lat = center_lat + (size // 2) * lat_step
    min_lon = center_lon - (size // 2) * lon_step
    max_lon = center_lon + (size // 2) * lon_step
    
    # Create coordinate grids
    lats = np.linspace(min_lat, max_lat, size)
    lons = np.linspace(min_lon, max_lon, size)
    LAT, LON = np.meshgrid(lons, lats)
    
    return elevation_grid, LAT, LON

def create_3d_terrain_surface(elevation_grid, LAT, LON, vertical_exaggeration=2.0):
    """
    3D 지형 표면 데이터 준비
    
    Parameters:
    -----------
    elevation_grid : np.ndarray
        고도 데이터 (140x140)
    LAT, LON : np.ndarray
        위경도 메쉬그리드
    vertical_exaggeration : float
        고도 과장 비율
    
    Returns:
    --------
    X, Y, Z : 3D 표면 좌표
    """
    X = LON
    Y = LAT
    Z = elevation_grid * vertical_exaggeration
    
    return X, Y, Z

def convert_gpx_to_3d(gpx_points, elevation_grid, LAT, LON, 
                      vertical_exaggeration=2.0, offset=0):
    """
    2D GPX 경로를 3D 공간으로 변환
    
    Parameters:
    -----------
    gpx_points : list
        GPX 포인트 리스트
    elevation_grid : np.ndarray
        지형 고도 데이터
    LAT, LON : np.ndarray
        위경도 그리드
    vertical_exaggeration : float
        고도 과장 비율
    offset : float
        경로를 지형 위로 띄울 오프셋 (미터)
    
    Returns:
    --------
    lons, lats, elevs : 3D 경로 좌표
    """
    if not gpx_points:
        return np.array([]), np.array([]), np.array([])
        
    lons = [p['longitude'] for p in gpx_points]
    lats = [p['latitude'] for p in gpx_points]
    
    # 지형 그리드에서 해당 위치의 고도 보간
    elevs = []
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        # 그리드에서 해당 위치의 고도 찾기
        points = np.column_stack((LAT.ravel(), LON.ravel()))
        values = elevation_grid.ravel()
        interp_elev = griddata(points, values, (lat, lon), method='linear')
        
        if np.isnan(interp_elev) or interp_elev is None:
            # Use the GPX elevation if available, otherwise default to 0
            gpx_elev = gpx_points[i].get('elevation')
            if gpx_elev is not None:
                interp_elev = gpx_elev
            else:
                # If no GPX elevation, use the terrain elevation at this point
                # Find the nearest point in the elevation grid
                min_dist = float('inf')
                closest_elev = 0
                for y in range(elevation_grid.shape[0]):
                    for x in range(elevation_grid.shape[1]):
                        dist = math.sqrt((LAT[y, x] - lat)**2 + (LON[y, x] - lon)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_elev = elevation_grid[y, x]
                interp_elev = closest_elev
        
        elevs.append((interp_elev + offset) * vertical_exaggeration)
    
    return np.array(lons), np.array(lats), np.array(elevs)

def plot_3d_terrain_matplotlib(elevation_grid, LAT, LON, all_gpx_points):
    """
    Matplotlib을 사용한 3D 시각화 (정적 이미지)
    """
    fig = plt.figure(figsize=FIGURE_SIZE_3D)
    ax = fig.add_subplot(111, projection='3d')
    
    # 지형 표면 생성
    X, Y, Z = create_3d_terrain_surface(elevation_grid, LAT, LON, 
                                        VERTICAL_EXAGGERATION)
    
    # 지형 표면 그리기
    surf = ax.plot_surface(X, Y, Z, cmap=TERRAIN_COLORMAP, 
                           alpha=TERRAIN_ALPHA, 
                           rstride=SURFACE_STRIDE, 
                           cstride=SURFACE_STRIDE,
                           linewidth=0, antialiased=True)
    
    # 등산객 경로 그리기
    hiker_names = getattr(config, 'HIKER_NAMES', [f'Hiker {i+1}' for i in range(len(all_gpx_points))])
    hiker_colors = getattr(config, 'HIKER_COLORS', PATH_3D_COLORS)
    
    for i, gpx_points in enumerate(all_gpx_points):
        color = hiker_colors[i % len(hiker_colors)]
        name = hiker_names[i] if i < len(hiker_names) else f'Hiker {i+1}'
        
        lons, lats, elevs = convert_gpx_to_3d(gpx_points, elevation_grid, 
                                               LAT, LON, VERTICAL_EXAGGERATION)
        
        # 3D 경로 선
        ax.plot(lons, lats, elevs, color=color, linewidth=PATH_3D_LINEWIDTH, 
                label=name, zorder=10)
        
        # 시작/종료 마커
        if SHOW_START_END_MARKERS:
            ax.scatter(lons[0], lats[0], elevs[0], color=color, s=150, 
                      marker='o', edgecolors='black', linewidth=2, zorder=11, label=f'{name} Start')
            ax.scatter(lons[-1], lats[-1], elevs[-1], color=color, s=150, 
                      marker='s', edgecolors='black', linewidth=2, zorder=11, label=f'{name} End')
        
        # 지면 투영 (선택사항)
        if SHOW_GROUND_PROJECTION:
            z_ground = np.full_like(elevs, Z.min())
            ax.plot(lons, lats, z_ground, color=color, 
                   linestyle=PROJECTION_LINESTYLE, 
                   alpha=PROJECTION_ALPHA, linewidth=1, zorder=1)
    
    # 동적으로 위도/경도 범위 설정
    all_gpx_lons = [lon for gpx_points in all_gpx_points for lon in [p['longitude'] for p in gpx_points]]
    all_gpx_lats = [lat for gpx_points in all_gpx_points for lat in [p['latitude'] for p in gpx_points]]
    
    if all_gpx_lons and all_gpx_lats:
        lon_min, lon_max = min(all_gpx_lons), max(all_gpx_lons)
        lat_min, lat_max = min(all_gpx_lats), max(all_gpx_lats)
        
        # 위도/경도 범위에 여유 공간 추가
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        lon_padding = lon_range * 0.05  # 5% 여유
        lat_padding = lat_range * 0.05  # 5% 여유
        
        ax.set_xlim(lon_min - lon_padding, lon_max + lon_padding)
        ax.set_ylim(lat_min - lat_padding, lat_max + lat_padding)
    
    # 뷰 각도 설정
    ax.view_init(elev=VIEW_ELEVATION_ANGLE, azim=VIEW_AZIMUTH_ANGLE)
    
    # 축 레이블
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Mountain Terrain with Hiker Paths')
    
    # 범례
    ax.legend(loc='upper left')
    
    # 컬러바
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')
    
    plt.tight_layout()
    return fig, ax

def plot_3d_terrain_plotly(elevation_grid, LAT, LON, all_gpx_points):
    """
    Plotly를 사용한 인터랙티브 3D 시각화 (HTML 출력)
    """
    # 지형 표면 생성
    X, Y, Z = create_3d_terrain_surface(elevation_grid, LAT, LON, 
                                        VERTICAL_EXAGGERATION)
    
    # 동적으로 위도/경도 범위 설정
    all_gpx_lons = []
    all_gpx_lats = []
    for gpx_points in all_gpx_points:
        for p in gpx_points:
            all_gpx_lons.append(p['longitude'])
            all_gpx_lats.append(p['latitude'])
    
    lon_min, lon_max = min(all_gpx_lons), max(all_gpx_lons)
    lat_min, lat_max = min(all_gpx_lats), max(all_gpx_lats)
    
    # 위도/경도 범위에 여유 공간 추가
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    lon_padding = lon_range * 0.05  # 5% 여유
    lat_padding = lat_range * 0.05  # 5% 여유
    
    # Figure 생성
    fig = go.Figure()
    
    # 지형 표면 추가
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='earth',
        opacity=TERRAIN_ALPHA,
        name='Terrain',
        showscale=True,
        colorbar=dict(title='Elevation (m)', x=1.1)
    ))
    
    # 등산객 경로 추가
    hiker_names = getattr(config, 'HIKER_NAMES', [f'Hiker {i+1}' for i in range(len(all_gpx_points))])
    hiker_colors = getattr(config, 'HIKER_COLORS', PATH_3D_COLORS)
    
    for i, gpx_points in enumerate(all_gpx_points):
        color = hiker_colors[i % len(hiker_colors)]
        name = hiker_names[i] if i < len(hiker_names) else f'Hiker {i+1}'
        
        lons, lats, elevs = convert_gpx_to_3d(gpx_points, elevation_grid, 
                                               LAT, LON, VERTICAL_EXAGGERATION, 
                                               offset=5)  # 경로를 약간 띄움
        
        # 3D 경로 선
        fig.add_trace(go.Scatter3d(
            x=lons, y=lats, z=elevs,
            mode='lines+markers',
            line=dict(color=color, width=PATH_3D_LINEWIDTH),
            marker=dict(size=PATH_MARKER_SIZE, color=color),
            name=name
        ))
        
        # 시작/종료 마커
        if SHOW_START_END_MARKERS:
            # 시작점
            fig.add_trace(go.Scatter3d(
                x=[lons[0]], y=[lats[0]], z=[elevs[0]],
                mode='markers',
                marker=dict(size=15, color=color, symbol='circle',
                           line=dict(color='black', width=2)),
                name=f'{name} Start',
                showlegend=True
            ))
            # 종료점
            fig.add_trace(go.Scatter3d(
                x=[lons[-1]], y=[lats[-1]], z=[elevs[-1]],
                mode='markers',
                marker=dict(size=15, color=color, symbol='square',
                           line=dict(color='black', width=2)),
                name=f'{name} End',
                showlegend=True
            ))
    
    # 레이블 이름이 잘못되었던 부분 수정
    fig.update_layout(
        title='3D Interactive Mountain Terrain with Hiker Paths',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Elevation (m)',
            xaxis=dict(range=[lon_min - lon_padding, lon_max + lon_padding]),
            yaxis=dict(range=[lat_min - lat_padding, lat_max + lat_padding]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)  # 초기 카메라 위치
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)  # Z축 비율 조정
        ),
        width=1400,
        height=1000
    )
    
    return fig

def main():
    """3D 시각화 메인 함수"""
    try:
        # API 키 로드
        elevation_api_key, maps_api_key = load_api_keys()
        
        # Output 디렉토리 생성
        os.makedirs('output', exist_ok=True)
        
        # GPX 파일 로드
        gpx_files = get_gpx_files()
        print(f"Found {len(gpx_files)} GPX files")
        
        if len(gpx_files) == 0:
            print("No GPX files found in gpx_data directory. Creating sample GPX file...")
            # Create a sample GPX file for testing
            sample_gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Sample GPX">
  <trk>
    <name>Sample Hike</name>
    <trkseg>
      <trkpt lat="37.7749" lon="-122.4194">
        <ele>50</ele>
      </trkpt>
      <trkpt lat="37.7750" lon="-122.4190">
        <ele>75</ele>
      </trkpt>
      <trkpt lat="37.7752" lon="-122.4185">
        <ele>100</ele>
      </trkpt>
      <trkpt lat="37.7755" lon="-122.4180">
        <ele>125</ele>
      </trkpt>
      <trkpt lat="37.7758" lon="-122.4175">
        <ele>150</ele>
      </trkpt>
    </trkseg>
  </trk>
</gpx>"""
            
            os.makedirs('gpx_data', exist_ok=True)
            with open('gpx_data/sample_hike.gpx', 'w') as f:
                f.write(sample_gpx_content)
            gpx_files = ['gpx_data/sample_hike.gpx']
            print(f"Created sample GPX file: {gpx_files[0]}")
        
        # 모든 GPX 데이터 로드 (캐시 사용)
        all_gpx_points = []
        for gpx_file in gpx_files:
            gpx_points = get_gpx_points_with_cache(gpx_file)
            if gpx_points:  # Only add if parsed successfully
                all_gpx_points.append(gpx_points)
        
        if not all_gpx_points:
            print("No valid GPX data found in any files. Exiting.")
            return
        
        # 중심 좌표 계산
        all_lats = [p['latitude'] for gpx in all_gpx_points for p in gpx]
        all_lons = [p['longitude'] for gpx in all_gpx_points for p in gpx]
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        # 고도 데이터 가져오기
        print("Fetching elevation data...")
        elevation_grid, LAT, LON = get_real_elevation_data_around_coords(
            center_lat, center_lon, elevation_api_key, size=config.GRID_SIZE
        )
        
        # 3D 시각화 생성
        print("Creating 3D visualization...")
        
        if OUTPUT_FORMAT == 'html':
            # Plotly 인터랙티브 버전
            fig = plot_3d_terrain_plotly(elevation_grid, LAT, LON, all_gpx_points)
            output_path = 'output/terrain_3d_interactive.html'
            fig.write_html(output_path)
            print(f"Interactive 3D visualization saved to {output_path}")
            
        else:  # 'png'
            # Matplotlib 정적 버전
            fig, ax = plot_3d_terrain_matplotlib(elevation_grid, LAT, LON, 
                                                 all_gpx_points)
            output_path = 'output/terrain_3d_static.png'
            fig.savefig(output_path, dpi=DPI_3D, bbox_inches='tight')
            print(f"Static 3D visualization saved to {output_path}")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()