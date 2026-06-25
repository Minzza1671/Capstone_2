@echo off
REM ===== 서버(웹 대시보드) 실행 =====
setlocal
cd /d %~dp0
call venv\Scripts\activate.bat
cd cctv_crowd_dashboard
python -m server.main
pause
