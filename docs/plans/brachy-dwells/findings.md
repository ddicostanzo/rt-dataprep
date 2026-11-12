# Findings — brachy-dwells (#282)

Spec: docs/specs/2026-11-11-brachy-dwells.md (REVIEWED). Plan: docs/plans/brachy-dwells/plan.json.

Summary: convert writes `dwell.parquet` for HDR/PDR brachy RTPLANs (one row per paired-CP dwell, zero-time dwells
kept) through a new `dwell_table` module with the one fixed reason REASON_DWELL, in a fifth catch; preprocess/manifest
carry the new kind `dwell` (position_i/j/k re-indexed on geometric runs). Pattern: ion_plan.parquet (PR #296).

Order: task-01 dwell_table + synthetic writer → task-02 preprocess/manifest → task-03 convert → task-04 docs/config.

Carry-over gotchas:
- Commit gate: ≤ 5 files per commit (findings.md counts); private-name mentions need CLAUDE_ALLOW_PRIVATE_API=1 listed in the body.
- `uv run --group dev ...` only; no bare python. Full suite ~4 min.
- Synthetic DICOM via save_as/dcmread; SYNTHETIC_ identifiers; no UIDs/values in log assertions; DS values ≤ 16 chars.
- Temp files only in your own scratchpad (tmp-guard blocks elsewhere).
