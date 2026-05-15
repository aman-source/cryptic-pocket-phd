# Operational Log — Pod Accounting

Per CLAUDE.md Rule 4: every pod session tracked.

| Pod ID | Start (UTC) | End (UTC) | Hours | Cost | Purpose | Result |
|--------|-------------|-----------|-------|------|---------|--------|
| (old pods from May 11) | ~2026-05-11 | ~2026-05-11 | ~2.5h×2 | ~$2.00 | Task F sweep v1 | ESS valid, RMSD/coverage invalid (nm/Å bug). Coords lost. |
| yammering_fuchsia_felidae (4tamtv6m2fpn1u) | 2026-05-12 | 2026-05-12 | ~0.3h | ~$0.12 | Smoke test (SSH broken, replaced) | SSH port mapping broken. Killed. |
| ec60b070fda1 | 2026-05-12 | 2026-05-12 | ~0.5h | ~$0.20 | Validation smoke: NPC2 β=0.1 ×5 | PASS: RMSD=7.18Å, unit checks green, coords round-trip verified |
| defiant_orange_barnacle (f31f19cghtwogs) | 2026-05-12 | 2026-05-13 | ~3.5h | ~$3.12 | Task F sweep v2 (2×A40) | 55/55 configs complete. Coverage metric later found broken (alignment). |
| cc1f64276c26 | 2026-05-13 | 2026-05-13 | ~0.7h | ~$0.28 | Unguided baseline (5 proteins) | 5/5 complete. Proved guidance ≤ baseline after alignment fix. |

| A40 timing pod (194.68.245.124:22023) | 2026-05-14 | 2026-05-14 | ~0.3h | ~$0.12 | Task A1 timing: 1k5n_A, 5 frames, 200 steps, 5 timesteps | 20.5s/frame. 25 npz written. Extrapolate: 200p×200f×20.5s ≈ 228 GPU-hrs ≈ ~$90 on A40. |

## Task A1 — Spec #2 Noisy Boltz Intermediates

| Pod | Start (UTC) | End (UTC) | Hours | Cost | Purpose | Result |
|-----|-------------|-----------|-------|------|---------|--------|
| Pod 1 (2×A40) | 2026-05-14 ~14:00 | 2026-05-14 ~22:30 | ~8.5h | ~$TBD | Task A1 pod1_gpu0 + pod1_gpu1 (30+30 proteins) | 30,000 npz uploaded to HF |
| Pod 2 (2×A40) | 2026-05-14 ~14:00 | 2026-05-14 ~22:30 | ~8.5h | ~$TBD | Task A1 pod2_gpu0 + pod2_gpu1 (30+30 proteins) | 30,240 npz uploaded to HF |

**Task A1 Total:**
- GPU spend: ~$TBD (fill from RunPod billing — check pod history)
- Output: 60,240 npz files at `aman-gpt/cryptic-pocket-task-a1` (HF dataset)
- 120 proteins × 100 frames × 5 timesteps = 60,000 training examples
- Status: COMPLETE. Pods terminated: TBD

## Phase 1 Total

| Item | Cost |
|------|------|
| GPU pods | ~$5.72 |
| Task G (cancelled) | $0 (saved ~$42) |
| **Total Phase 1** | **~$5.72** |
