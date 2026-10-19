@echo off
rem Start the local web app on http://127.0.0.1:7860 (opens the browser).
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" app\server.py %*
pause
