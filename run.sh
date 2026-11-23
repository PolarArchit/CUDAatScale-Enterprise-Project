#!/usr/bin/env bash
# ==============================================================================
# Execution script for Enterprise CUDA Image Processing Pipeline
# Designed for Coursera Linux Lab and Peer Grading Evaluation
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BIN_PATH="bin/cuda_enterprise_pipeline"

# 1. Ensure dataset exists
if [ ! -d "data/input" ] || [ -z "$(ls -A data/input 2>/dev/null)" ]; then
    echo "[run.sh] Dataset directory empty or missing. Generating 125 benchmark images..."
    python3 tools/generate_dataset.py
fi

# 2. Compile if executable is missing
if [ ! -f "$BIN_PATH" ]; then
    echo "[run.sh] Binary $BIN_PATH not found. Invoking Makefile..."
    make -j$(nproc 2>/dev/null || echo 4)
fi

mkdir -p output artifacts

# 3. If arguments provided, pass them directly; otherwise run full batch
if [ $# -gt 0 ]; then
    echo "[run.sh] Executing pipeline with custom arguments: $@"
    ./"$BIN_PATH" "$@"
else
    echo "[run.sh] Executing standard enterprise batch workflow across all images..."
    ./"$BIN_PATH" \
        --input_dir data/input \
        --output_dir output \
        --streams 4 \
        --operation all \
        --angle 45.0 \
        | tee artifacts/execution_proof.log

    echo ""
    echo "[run.sh] Executing concurrency and scaling benchmark..."
    ./"$BIN_PATH" \
        --input_dir data/input \
        --output_dir output \
        --benchmark \
        | tee -a artifacts/execution_proof.log
fi

echo ""
echo "[run.sh] Pipeline execution completed successfully."
echo "         Outputs stored in: output/"
echo "         Execution log stored in: artifacts/execution_proof.log"
