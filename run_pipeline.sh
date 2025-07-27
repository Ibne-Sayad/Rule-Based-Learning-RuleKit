#!/bin/bash

# === CONFIG ===
CSV_PATH="./data/TCGA Pan-Cancer (PANCAN).csv"
OUTPUT_DIR="./data"
N_CLUSTERS=3
VENV_DIR="venv"
PYTHON="$VENV_DIR/Scripts/python.exe"
PIP="$PYTHON -m pip"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
  echo "🔧 Creating virtual environment..."
  py -m venv "$VENV_DIR" || python -m venv "$VENV_DIR"
fi

#NSTALL DEPENDENCIES
echo "Installing dependencies..."
$PYTHON -m ensurepip
$PIP install --upgrade pip
$PIP install -r requirements.txt

#RUN PIPELINE
echo "Running pipeline..."
$PYTHON pipeline.py \
  --csv "$CSV_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --n-clusters "$N_CLUSTERS"

echo "Pipeline finished."

#KEEP WINDOW OPEN
echo ""
read -rp "Press ENTER to close this window..."
