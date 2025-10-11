# 3D 시각화 전용 설정
# =====================

# 3D 뷰 설정
VIEW_ELEVATION_ANGLE = 20      # 카메라 상승 각도 (도)
VIEW_AZIMUTH_ANGLE = 45        # 카메라 회전 각도 (도)
VERTICAL_EXAGGERATION = 0.6    # 고도 과장 비율 (지형 높이 강조) - Reduced from 2.0 to 1.0

# 지형 렌더링
TERRAIN_COLORMAP = 'terrain'   # 'terrain', 'gist_earth', 'viridis'
TERRAIN_ALPHA = 0.3            # 지형 투명도 - More transparent so paths are visible
SURFACE_STRIDE = 2             # 표면 메쉬 간격 (작을수록 세밀함)

# 등산객 경로 설정
PATH_3D_LINEWIDTH = 3          # 3D 경로 선 두께 - Increased for better visibility
PATH_3D_COLORS = ['yellow', 'red', 'blue']  # 3명의 색상
PATH_MARKER_SIZE = 5          # 경로 포인트 마커 크기 - Increased
SHOW_START_END_MARKERS = True  # 시작/종료 마커 표시
PATH_ELEVATION_BIAS = 0       # 경로 높이 보정치 (지형 위로 올릴 높이, 실제 미터 단위) - Lift paths above terrain

# 그림자/투영 설정
SHOW_GROUND_PROJECTION = True  # 지면에 경로 투영 표시
PROJECTION_ALPHA = 1         # 투영 투명도
PROJECTION_LINESTYLE = '--'    # 투영 선 스타일

# 애니메이션 설정 (선택사항)
ENABLE_ANIMATION = False       # 시간에 따른 애니메이션
ANIMATION_FPS = 10             # 프레임 레이트
ANIMATION_DURATION = 10        # 애니메이션 길이(초)

# 출력 설정
OUTPUT_FORMAT = 'html'         # 'html' (plotly), 'png' (matplotlib)
FIGURE_SIZE_3D = (10, 8)       # 3D 그래프 크기 - Reduced from (14, 12)
DPI_3D = 300                   # 해상도