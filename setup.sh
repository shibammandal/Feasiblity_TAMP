#!/bin/bash
set -e
echo "Creating virtual environment..."
python3 -m venv .venv

echo "Installing dependencies (CPU version for speed)..."
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

echo "Setup complete! Now you can run ./scripts/smoke_test.sh"
