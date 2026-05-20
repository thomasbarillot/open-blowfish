#!/usr/bin/env bash
# Run on EC2: generates 150k entropy maps, uploads to S3, then trains the CNN VAE.
#
# Prerequisites:
#   - conda environment 'env' with poetry install done
#   - AWS credentials configured (IAM role or profile)
#
# Usage:
#   chmod +x scripts/ottp_topology_analysis/run_generate_and_train.sh
#   cd /path/to/data_platform
#   ./scripts/ottp_topology_analysis/run_generate_and_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

N_MAPS=1000000
S3_BUCKET="homology-experiment"
S3_PREFIX="simulation_dataset"
OUTPUT_DIR="$SCRIPT_DIR/outputs/simulation_dataset"
MODEL_DIR="$SCRIPT_DIR/outputs/model"

echo "=== Step 1: Generate $N_MAPS entropy maps ==="
poetry run python -u "$SCRIPT_DIR/generate_entropy_maps_fast.py" \
    --n-maps "$N_MAPS" \
    --output-dir "$OUTPUT_DIR" \
    --upload-s3 \
    --s3-bucket "$S3_BUCKET" \
    --s3-prefix "$S3_PREFIX"

# echo ""
# echo "=== Step 2: Train CNN VAE ==="
# # Clean previous TensorBoard runs
# rm -rf "$MODEL_DIR/runs"

# poetry run python -u "$SCRIPT_DIR/train_feature_predictor.py" \
#     --data-dir "$OUTPUT_DIR" \
#     --output-dir "$MODEL_DIR" \
#     --epochs 300 \
#     --alpha 10.0 \
#     --patience 30 \
#     --batch-size 128 \
#     --upload-model-s3

# echo ""
# echo "=== Done ==="
# echo "Model saved at: $MODEL_DIR/model.pt"
# echo "Model uploaded to: s3://$S3_BUCKET/model/model.pt"
# echo "Training data at: s3://$S3_BUCKET/$S3_PREFIX/"
