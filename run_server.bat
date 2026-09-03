@echo off
setlocal

REM Change to the directory containing this batch file
cd /d "%~dp0"

REM Activate the local virtual environment
call ".venv\Scripts\activate.bat"

REM Start each Python program in its own process

start "Server" cmd /k python server.py

echo All processes started.
endlocal