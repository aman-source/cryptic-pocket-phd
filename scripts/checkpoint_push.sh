#!/usr/bin/env bash
# Periodic git-push loop for Task A1 noisy frame data.
# Run in a separate tmux pane BEFORE launching generate_boltz_intermediates.py.
#
# Usage (on pod):
#   bash scripts/checkpoint_push.sh
#
# Pushes every 5 min. Handles merge conflicts between concurrent pods via
# pull --rebase with retry loop. Ctrl-C to stop.
# Requires: git remote already configured with PAT token in URL.

set -uo pipefail   # NOT -e: push failures are handled manually below

REPO=/workspace/cryptic-pocket-phd
INTERVAL=300   # 5 min

echo "[checkpoint] Starting push loop. Interval: ${INTERVAL}s  Host: $(hostname)"
echo "[checkpoint] Repo: ${REPO}"
echo ""

while true; do
    sleep "${INTERVAL}"

    echo "[checkpoint] $(date -u +%Y-%m-%dT%H:%M:%SZ) — syncing..."
    cd "${REPO}"

    # Pull latest before staging (avoids stale-ref conflicts)
    git pull --rebase origin master 2>/dev/null || {
        echo "[checkpoint] WARNING: pull --rebase failed, attempting reset to remote..."
        git rebase --abort 2>/dev/null || true
    }

    # Stage everything relevant
    git add data/noisy_frames/ -A 2>/dev/null || true
    git add data/md_frames/ data/md_labels/ data/protein_lists/ data/splits/ -A 2>/dev/null || true
    git add data/noisy_frames/progress.log 2>/dev/null || true
    git add OPERATIONAL_LOG.md 2>/dev/null || true

    # Commit only if staged changes exist
    if git diff --cached --quiet; then
        echo "[checkpoint] Nothing new to commit."
        echo ""
        continue
    fi

    git commit -m "checkpoint: $(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ)" 2>/dev/null || true

    # Push with retry (up to 3 attempts, rebase between each)
    PUSHED=0
    for i in 1 2 3; do
        if git push origin master 2>/dev/null; then
            echo "[checkpoint] Push OK (attempt ${i})"
            PUSHED=1
            break
        fi
        echo "[checkpoint] Push failed (attempt ${i}), pulling --rebase and retrying..."
        git pull --rebase origin master 2>/dev/null || {
            git rebase --abort 2>/dev/null || true
        }
        sleep 5
    done

    if [ "${PUSHED}" -eq 0 ]; then
        echo "[checkpoint] WARNING: all 3 push attempts failed. Data safe locally. Will retry next cycle."
    fi
    echo ""
done
