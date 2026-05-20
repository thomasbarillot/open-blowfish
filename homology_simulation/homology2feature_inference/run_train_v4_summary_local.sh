#!/usr/bin/env bash
# Summary-target baseline: entropy -> z -> (noise[t], faith[t]) per neighbor.
# No VAE, no labels-as-target, no class weighting — direct MSE regression
# on the (50, 2) summary derived from labels.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

MODEL_DIR="$SCRIPT_DIR/outputs/model_v4_summary"

echo "=== Summary baseline: entropy -> z -> (noise, faith) per neighbor ==="
rm -rf "$MODEL_DIR/runs"
mkdir -p "$MODEL_DIR"

poetry run python -u "$SCRIPT_DIR/train_feature_predictor_v4.py" \
    --phase summary \
    --data-dir "/Users/thomas.barillot/Downloads" \
    --output-dir "$MODEL_DIR" \
    --epochs 200 \
    --batch-size 256 \
    --lr 3e-4 \
    --weight-decay 1e-2 \
    --warmup-steps 500 \
    --patience 30 \
    --early-stop-min-delta 1e-4 \
    --d-z 256 \
    --d-model 256 \
    --d-ff 1024 \
    --heads 4 \
    --enc-layers 3 \
    --dropout 0.1 \
    --num-workers 8 \
    --max-samples 200000

echo ""
echo "=== Done ==="
echo "Checkpoint at: $MODEL_DIR/model_v4_summary.pt"
