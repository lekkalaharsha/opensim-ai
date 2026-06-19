#!/bin/bash
# OpenQSim Sweep Runner (Linux/Mac)
# Usage: bash scripts/run_sweep.sh [config_file]

set -e

CONFIG_FILE="${1:-benchmark/sweep_config_0a.yaml}"

echo "============================================"
echo "  OpenQSim Benchmark Sweep"
echo "============================================"

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Load .env if exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded environment variables from .env"
else
    echo "WARNING: .env not found. Set KAGGLE_USERNAME, KAGGLE_KEY, NVIDIA_API_KEY manually."
fi

# Check for required environment variables
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "ERROR: Kaggle credentials not set. Edit .env file."
    exit 1
fi

# Run the sweep
echo "Running sweep with config: $CONFIG_FILE"
python scripts/run_sweep.py --config "$CONFIG_FILE" \
    --output-dir data/benchmark_outputs \
    --checkpoint-interval 10 \
    --artifact-interval 50 \
    --kaggle-dataset "$KAGGLE_USERNAME/openqsim-benchmarks"

echo ""
echo "✅ Sweep completed."