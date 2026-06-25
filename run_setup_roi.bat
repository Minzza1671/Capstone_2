@echo off
REM ===== ROI(관심영역) 설정 GUI =====
REM 사용법: run_setup_roi.bat --video data\내영상.mp4   (생략시 기본 sibuya_test.mp4)
setlocal
cd /d %~dp0
call venv\Scripts\activate.bat
cd cctv_crowd_dashboard
python -m analyzer.setup_roi %*
pause
