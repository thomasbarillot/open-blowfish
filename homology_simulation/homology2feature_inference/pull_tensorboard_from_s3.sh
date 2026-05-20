#!/usr/bin/env bash
# Pull TensorBoard logs from S3 and optionally launch TensorBoard.
#
# Usage:
#   ./pull_tensorboard_from_s3.sh [--profile AWS_PROFILE] [--launch]
#
# Options:
#   --profile  AWS profile name (default: reads from config.py or uses default)
#   --launch   Launch TensorBoard after syncing (default: false)

set -euo pipefail

BUCKET="homology-experiment"
S3_PREFIX="model/runs"
LOCAL_DIR="$(dirname "$0")/outputs/model/runs"
AWS_PROFILE="${AWS_PROFILE:-}"
LAUNCH_TB=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            AWS_PROFILE="$2"
            shift 2
            ;;
        --launch)
            LAUNCH_TB=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$LOCAL_DIR"

PROFILE_ARG=""
if [[ -n "$AWS_PROFILE" ]]; then
    PROFILE_ARG="--profile $AWS_PROFILE"
fi

echo "Syncing s3://${BUCKET}/${S3_PREFIX}/ -> ${LOCAL_DIR}/"
aws s3 sync "s3://${BUCKET}/${S3_PREFIX}/" "$LOCAL_DIR/" $PROFILE_ARG

echo "Sync complete."

if $LAUNCH_TB; then
    echo "Launching TensorBoard at http://localhost:6006 ..."
    poetry run tensorboard --logdir "$LOCAL_DIR"
fi
