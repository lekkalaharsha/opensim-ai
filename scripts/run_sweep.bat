@echo off
REM OpenQSim Sweep Runner (Windows)
REM Usage: scripts\run_sweep.bat [config_file]

set CONFIG_FILE=%1
if "%CONFIG_FILE%"=="" set CONFIG_FILE=benchmark\sweep_config_0a.yaml

echo ============================================
echo   OpenQSim Benchmark Sweep
echo ============================================

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Load .env (simple parsing)
if exist ".env" (
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if not "%%a"=="" if not "%%b"=="" set %%a=%%b
    )
    echo Loaded environment variables from .env
) else (
    echo WARNING: .env not found.
)

REM Check required variables
if "%KAGGLE_USERNAME%"=="" (
    echo ERROR: KAGGLE_USERNAME not set.
    exit /b 1
)
if "%KAGGLE_KEY%"=="" (
    echo ERROR: KAGGLE_KEY not set.
    exit /b 1
)

REM Run sweep
echo Running sweep with config: %CONFIG_FILE%
python scripts\run_sweep.py --config %CONFIG_FILE% ^
    --output-dir data\benchmark_outputs ^
    --checkpoint-interval 10 ^
    --artifact-interval 50 ^
    --kaggle-dataset %KAGGLE_USERNAME%/openqsim-benchmarks

echo.
echo ✅ Sweep completed.