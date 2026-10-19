@echo off
rem One-time setup: Python venv + CUDA PyTorch + DINOv2 weights. Needs Python 3.10-3.12 on PATH.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv || goto :err
)
set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :err
"%PY%" -m pip install -r requirements.txt || goto :err
"%PY%" src\download_weights.py || goto :err
echo.
echo Setup done. Start the app with start.bat
pause
exit /b 0
:err
echo Setup failed.
pause
exit /b 1
