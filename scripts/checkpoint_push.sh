#!/usr/bin/env bash
# Periodic git-push loop for Task A1 noisy frame data.
# Run in a separate tmux pane BEFORE launching generate_boltz_intermediates.py.
#
# Usage (on pod):
#   bash scripts/checkpoint_push.sh
#
# Pushes every 20 min. Ctrl-C to stop.
# Requires: git remote already configured with PAT token in URL.

set -euo pipefail

REPO=/workspace/cryptic-pocket-phd
INTERVAL=1200  # 20 min

echo "[checkpoint] Starting push loop. Interval: ${INTERVAL}s"
echo "[checkpoint] Repo: ${REPO}"
echo ""

while true; do
    sleep "${INTERVAL}"

    echo "[checkpoint] $(date -u +%Y-%m-%dT%H:%M:%SZ) — pushing..."
    cd "${REPO}"

    # Stage noisy frames (the bulk of the data)
    git add data/noisy_frames/ -A 2>/dev/null || true
    # Stage metadata + logs
    git add data/md_frames/ data/md_labels/ data/protein_lists/ -A 2>/dev/null || true
    git add OPERATIONAL_LOG.md 2>/dev/null || true

    # Only commit if there are staged changes
    if git diff --cached --quiet; then
        echo "[checkpoint] Nothing new to commit."
    else
        git commit -m "checkpoint: $(date -u +%Y-%m-%dT%H:%M:%SZ)" 2>/dev/null || true
        git push origin master 2>/dev/null && echo "[checkpoint] Push OK" || echo "[checkpoint] Push FAILED — check PAT"
    fi
    echo ""
done
