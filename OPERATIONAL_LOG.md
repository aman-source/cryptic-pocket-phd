# Operational Log — Pod Accounting

Per CLAUDE.md Rule 4: every pod session tracked.

| Pod ID | Start (UTC) | End (UTC) | Hours | Cost | Purpose | Result |
|--------|-------------|-----------|-------|------|---------|--------|
| (old pods from May 11) | ~2026-05-11 | ~2026-05-11 | ~2.5h×2 | ~$2.00 | Task F sweep v1 | ESS valid, RMSD/coverage invalid (nm/Å bug). Coords lost. |
| yammering_fuchsia_felidae (4tamtv6m2fpn1u) | 2026-05-12 | 2026-05-12 | ~0.3h | ~$0.12 | Smoke test (SSH broken, replaced) | SSH port mapping broken. Killed. |
| ec60b070fda1 | 2026-05-12 | 2026-05-12 | ~0.5h | ~$0.20 | Validation smoke: NPC2 β=0.1 ×5 | PASS: RMSD=7.18Å, unit checks green, coords round-trip verified |
| defiant_orange_barnacle (f31f19cghtwogs) | 2026-05-12 | 2026-05-13 | ~3.5h | ~$3.12 | Task F sweep v2 (2×A40) | 55/55 configs complete. Coverage metric later found broken (alignment). |
| cc1f64276c26 | 2026-05-13 | 2026-05-13 | ~0.7h | ~$0.28 | Unguided baseline (5 proteins) | 5/5 complete. Proved guidance ≤ baseline after alignment fix. |

## Phase 1 Total

| Item | Cost |
|------|------|
| GPU pods | ~$5.72 |
| Task G (cancelled) | $0 (saved ~$42) |
| **Total Phase 1** | **~$5.72** |
