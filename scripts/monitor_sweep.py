#!/usr/bin/env python
"""Monitor Phase 1 sweep on RunPod. Pings every 10 min, sends mobile notification.

Setup:
  1. Install ntfy app on phone (Android/iOS)
  2. Subscribe to topic: cryptic-pocket-sweep (or change NTFY_TOPIC below)
  3. Run: python scripts/monitor_sweep.py

Notifications include: completed configs, GPU usage, errors.
"""

import subprocess
import time
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime

# ============================================================
# CONFIGURE THESE
# ============================================================
SSH_HOST = "69.30.85.11"
SSH_PORT = "22068"
SSH_KEY = "~/.ssh/id_ed25519"
NTFY_TOPIC = "cryptic-pocket-sweep"  # change if you want private topic
POLL_INTERVAL = 600  # 10 minutes
REPO_PATH = "/workspace/cryptic-pocket-phd"
TOTAL_CONFIGS = 40  # 5 proteins × 8 configs
# ============================================================


def ssh_cmd(cmd, timeout=15):
    """Run command on pod via SSH. Returns (stdout, success)."""
    full_cmd = [
        "ssh", "-i", SSH_KEY, "-p", SSH_PORT,
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={timeout}",
        f"root@{SSH_HOST}",
        cmd,
    ]
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True,
                                timeout=timeout + 10)
        return result.stdout.strip(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return "SSH timeout", False
    except Exception as e:
        return str(e), False


def send_notification(title, message, priority="default"):
    """Send push notification via ntfy.sh."""
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    data = message.encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Title", title)
    req.add_header("Priority", priority)
    req.add_header("Tags", "microscope")
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.URLError as e:
        print(f"  [ntfy error: {e}]")


def check_status():
    """Check sweep progress. Returns status dict."""
    # Count completed configs
    out, ok = ssh_cmd(
        f"find {REPO_PATH}/results/phase1_sweep -name done.json 2>/dev/null | wc -l"
    )
    n_done = int(out) if ok and out.isdigit() else -1

    # GPU usage
    gpu_out, _ = ssh_cmd(
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
    )

    # Last log lines from both GPUs
    log0, _ = ssh_cmd(
        "tail -3 /tmp/sweep_gpu0.log 2>/dev/null | tr '\\r' '\\n' | grep -v '^\\s*$' | tail -2"
    )
    log1, _ = ssh_cmd(
        "tail -3 /tmp/sweep_gpu1.log 2>/dev/null | tr '\\r' '\\n' | grep -v '^\\s*$' | tail -2"
    )

    # Check for errors
    err0, _ = ssh_cmd(
        "grep -c 'Error\\|error\\|OOM\\|Traceback' /tmp/sweep_gpu0.log 2>/dev/null || echo 0"
    )
    err1, _ = ssh_cmd(
        "grep -c 'Error\\|error\\|OOM\\|Traceback' /tmp/sweep_gpu1.log 2>/dev/null || echo 0"
    )

    # Check if processes still alive
    procs, _ = ssh_cmd("pgrep -c -f run_phase1_sweep || echo 0")
    n_procs = int(procs) if procs.isdigit() else 0

    return {
        "n_done": n_done,
        "total": TOTAL_CONFIGS,
        "gpu": gpu_out,
        "log0": log0,
        "log1": log1,
        "errors0": int(err0) if err0.isdigit() else 0,
        "errors1": int(err1) if err1.isdigit() else 0,
        "alive": n_procs,
    }


def main():
    print(f"Monitoring sweep on {SSH_HOST}:{SSH_PORT}")
    print(f"Notifications → ntfy.sh/{NTFY_TOPIC}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Subscribe on phone: open ntfy app → + → topic: {NTFY_TOPIC}")
    print()

    send_notification("Sweep Monitor Started",
                      f"Monitoring {SSH_HOST}. Polling every {POLL_INTERVAL//60} min.",
                      priority="low")

    prev_done = 0
    consecutive_failures = 0

    while True:
        now = datetime.now().strftime("%H:%M")
        status = check_status()

        if status["n_done"] < 0:
            consecutive_failures += 1
            print(f"[{now}] SSH failed ({consecutive_failures}x)")
            if consecutive_failures >= 3:
                send_notification("Sweep: SSH DOWN",
                                  f"Can't reach pod after {consecutive_failures} attempts. Check RunPod.",
                                  priority="high")
            time.sleep(POLL_INTERVAL)
            continue

        consecutive_failures = 0
        pct = status["n_done"] / status["total"] * 100
        new_done = status["n_done"] - prev_done

        line = (f"[{now}] {status['n_done']}/{status['total']} ({pct:.0f}%) "
                f"+{new_done} new | procs={status['alive']} | GPU: {status['gpu']}")
        print(line)

        # Send notification on progress
        if new_done > 0:
            msg = (f"{status['n_done']}/{status['total']} configs done ({pct:.0f}%)\n"
                   f"GPU: {status['gpu']}\n"
                   f"Processes alive: {status['alive']}")
            send_notification(f"Sweep: {pct:.0f}% ({status['n_done']}/{status['total']})", msg)

        # Alert on errors
        total_errors = status["errors0"] + status["errors1"]
        if total_errors > 0 and new_done == 0 and prev_done > 0:
            send_notification("Sweep: Possible Error",
                              f"Errors in logs: GPU0={status['errors0']} GPU1={status['errors1']}\n"
                              f"GPU0: {status['log0']}\nGPU1: {status['log1']}",
                              priority="high")

        # Alert on completion
        if status["n_done"] >= status["total"]:
            send_notification("Sweep: COMPLETE!",
                              f"All {status['total']} configs finished.\n"
                              f"scp results from pod NOW, then terminate.",
                              priority="urgent")
            print("SWEEP COMPLETE. Fetch results and kill pod.")
            break

        # Alert if processes died but not all done
        if status["alive"] == 0 and status["n_done"] < status["total"]:
            send_notification("Sweep: Processes DIED",
                              f"Only {status['n_done']}/{status['total']} done but no processes running.\n"
                              f"GPU0: {status['log0']}\nGPU1: {status['log1']}",
                              priority="urgent")

        prev_done = status["n_done"]
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
