@echo off
REM ===== Crowd Management System - 1회 설치 스크립트 =====
setlocal
cd /d %~dp0

echo [1/4] Python 가상환경 생성...
python -m venv venv
if errorlevel 1 (
  echo [ERROR] Python 3.10+ 필요. python --version 확인.
  pause & exit /b 1
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [2/4] PyTorch 설치 (CUDA 12.6 빌드, GPU 없으면 자동 CPU 동작)...
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
  echo [WARN] CUDA 빌드 실패 - CPU 전용 빌드로 재시도...
  pip install torch==2.12.0 torchvision==0.27.0
)

echo [3/4] 나머지 의존성 설치...
pip install -r requirements.txt

echo [4/4] 완료.
echo  - 영상 파일을 cctv_crowd_dashboard\data\ 에 넣으세요 (예: sibuya_test.mp4)
echo  - ROI 설정: run_setup_roi.bat
echo  - 실행: run.bat
pause
