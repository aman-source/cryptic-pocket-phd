#!/usr/bin/env bash
# Download PocketMiner model weights from upstream GitHub repo.
# Safe to re-run: exits early if weights already present.
#
# Usage:
#   bash scripts/download_pocketminer_weights.sh
#
# Source: github.com/Mickdub/gvp branch pocket_pred, models/pocketminer.*
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/external/pocketminer/models"
BASE_URL="https://raw.githubusercontent.com/Mickdub/gvp/pocket_pred/models"

DATA_FILE="pocketminer.data-00000-of-00001"
INDEX_FILE="pocketminer.index"

if [ -f "$MODELS_DIR/$INDEX_FILE" ] && [ -f "$MODELS_DIR/$DATA_FILE" ]; then
    echo "PocketMiner weights already present at $MODELS_DIR/"
    exit 0
fi

mkdir -p "$MODELS_DIR"
echo "Downloading PocketMiner weights (~9 MB)..."

curl -fL --progress-bar \
    "$BASE_URL/$DATA_FILE" \
    -o "$MODELS_DIR/$DATA_FILE"

curl -fL --progress-bar \
    "$BASE_URL/$INDEX_FILE" \
    -o "$MODELS_DIR/$INDEX_FILE"

echo "Done. Weights written to $MODELS_DIR/"
ls -lh "$MODELS_DIR/pocketminer."*
