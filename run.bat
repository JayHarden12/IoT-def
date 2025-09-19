@echo off
echo 🛡️ IoT Intrusion Detection System
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import streamlit, pandas, numpy, plotly, sklearn" >nul 2>&1
if errorlevel 1 (
    echo ❌ Required packages are not installed
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install requirements
        pause
        exit /b 1
    )
)

REM Create necessary directories
if not exist "models" mkdir models
if not exist "reports" mkdir reports
if not exist "data" mkdir data

echo ✅ All requirements satisfied
echo.
echo 🚀 Starting the application...
echo The application will open in your default web browser.
echo If it doesn't open automatically, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo ================================================

REM Run the Streamlit app
python -m streamlit run app.py

pause
