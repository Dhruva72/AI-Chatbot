@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.conda\python.exe"

if not exist "%PROJECT_PYTHON%" (
    echo Project Python was not found at "%PROJECT_PYTHON%".
    echo Recreate the .conda environment, then run:
    echo   .\.conda\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%PROJECT_PYTHON%" DATA\main.py
pause
