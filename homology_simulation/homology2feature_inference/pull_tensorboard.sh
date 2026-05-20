#!/usr/bin/env bash
# Pull TensorBoard runs and/or training result plots from S3.
#
# Usage:
#   ./scripts/ottp_topology_analysis/pull_tensorboard.sh              # pull runs + plots, launch tensorboard
#   ./scripts/ottp_topology_analysis/pull_tensorboard.sh --pull       # pull runs + plots only
#   ./scripts/ottp_topology_analysis/pull_tensorboard.sh --launch     # launch tensorboard only (assumes already pulled)
#   ./scripts/ottp_topology_analysis/pull_tensorboard.sh --plots      # pull plots only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

S3_BUCKET="homology-experiment"
RUNS_DIR="$SCRIPT_DIR/outputs/model/runs_v2"
PLOTS_DIR="$SCRIPT_DIR/outputs/model"

DO_PULL_RUNS=true
DO_PULL_PLOTS=true
DO_LAUNCH=true

case "${1:-}" in
    --pull)
        DO_LAUNCH=false
        ;;
    --launch)
        DO_PULL_RUNS=false
        DO_PULL_PLOTS=false
        ;;
    --plots)
        DO_PULL_RUNS=false
        DO_LAUNCH=false
        ;;
esac

if $DO_PULL_RUNS; then
    echo "Pulling TensorBoard runs from s3://$S3_BUCKET/model/runs_v2/ ..."
    mkdir -p "$RUNS_DIR"
    aws s3 sync "s3://$S3_BUCKET/model/runs_v2/" "$RUNS_DIR/" --delete
    echo "Synced to $RUNS_DIR"
fi

if $DO_PULL_PLOTS; then
    echo "Pulling training plots from s3://$S3_BUCKET/model/plots/ ..."
    mkdir -p "$PLOTS_DIR"
    aws s3 sync "s3://$S3_BUCKET/model/plots/" "$PLOTS_DIR/" --exclude "*" --include "*.png"
    echo "Synced plots to $PLOTS_DIR"
    echo "Files:"
    ls -1 "$PLOTS_DIR"/*.png 2>/dev/null || echo "  (no .png files found)"
fi

if $DO_LAUNCH; then
    echo "Launching TensorBoard at http://localhost:6006"
    poetry run tensorboard --logdir "$RUNS_DIR" --port 6006
fi
