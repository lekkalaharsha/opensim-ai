@echo off
REM OpenQSim Setup Script (Windows)
REM Usage: scripts\setup.bat

echo ============================================
echo   OpenQSim Benchmark Suite - Setup
echo ============================================

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    exit /b 1
)

echo Python found.

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
pip install python-dotenv

REM Create .env from example if not exists
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env
        echo Created .env from .env.example. EDIT IT with your API keys.
    ) else (
        echo WARNING: .env.example not found. Create .env manually.
    )
)

echo.
echo ✅ Setup complete.
echo    Activate environment: venv\Scripts\activate
echo    Edit .env with your Kaggle and NVIDIA API keys.
echo    Then run: scripts\run_sweep.bat