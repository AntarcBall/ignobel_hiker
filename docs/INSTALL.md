# 설치 및 실행 안내

## 한글 설치 안내

[정헌규] [오후 12:52] just type 
[정헌규] [오후 12:52] pip install -r requirements.txt
[정헌규] [오후 12:52] and
[정헌규] [오후 12:53] python tkinter-run.py

## 상세 설치 방법

1. **Python 가상 환경 생성**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 또는
   venv\\Scripts\\activate     # Windows
   ```

2. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **GUI 실행**
   ```bash
   python tkinter-run.py
   ```

GUI가 실행되면 "Run Terrain Visualization" 또는 "Run Velocity Analysis" 버튼을 클릭하여 각 기능을 실행할 수 있습니다.