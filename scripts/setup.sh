#!/bin/bash
# OpenQSim Setup Script (Linux/Mac)
# Usage: bash scripts/setup.sh

set -e

echo "============================================"
echo "  OpenQSim Benchmark Suite - Setup"
echo "============================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
if [[ $(echo "$python_version 3.10" | awk '{print ($1 < $2)}') -eq 1 ]]; then
    echo "ERROR: Python 3.10+ is required."
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install python-dotenv

# Create .env from example if not exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from .env.example. EDIT IT with your API keys."
    else
        echo "WARNING: .env.example not found. Create .env manually."
    fi
fi

echo ""
echo "✅ Setup complete."
echo "   Activate environment: source venv/bin/activate"
echo "   Edit .env with your Kaggle and NVIDIA API keys."
echo "   Then run: bash scripts/run_sweep.sh"