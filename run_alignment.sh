#!/bin/bash
set -e

# --- CONFIGURATION ---
ENV_NAME="mfa_temp"
DATA_DIR="data/LJSpeech-1.1/wavs"
DICT_PATH="data/arpa_dict.txt"     # Changed
OUTPUT_DIR="data/aligned"
MODEL_NAME="english_us_arpa"       # Changed (Available in your list)

# --- LOCATE CONDA (Same as before) ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/opt/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniconda3/etc/profile.d/conda.sh"
fi

echo "=========================================="
echo "   MFA AUTOMATION: ARPA MODE"
echo "=========================================="

# --- CREATE ENV ---
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' exists."
else
    echo "Creating environment..."
    conda create -n $ENV_NAME -c conda-forge montreal-forced-aligner -y
fi

# --- RUN MFA ---
echo "Downloading ARPA Acoustic Model..."
conda run -n $ENV_NAME mfa model download acoustic $MODEL_NAME

echo "Running Alignment..."
conda run -n $ENV_NAME mfa align \
    "$DATA_DIR" \
    "$DICT_PATH" \
    "$MODEL_NAME" \
    "$OUTPUT_DIR" \
    --clean \
    --num_jobs 8

echo ""
echo "SUCCESS! TextGrids saved to: $OUTPUT_DIR"
echo ""

# --- CLEANUP ---
read -p "Delete environment? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    conda env remove -n $ENV_NAME -y
    rm -rf ~/Documents/MFA
    echo "Cleaned up."
fi