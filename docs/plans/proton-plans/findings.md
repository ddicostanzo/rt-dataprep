# Findings — proton-plans (#281)

Spec: docs/specs/2026-11-11-proton-plans.md (REVIEWED). Plan: docs/plans/proton-plans/plan.json.

Summary: convert writes `ion_plan.parquet` (one row per non-zero PBS spot, MODULATED + MU only) through a new
`ion_plan_table` module with the one fixed reason REASON_ION_PLAN; preprocess/manifest carry the new kind `ion_plan`;
metadata RTPLAN falls back to IonBeamSequence. Photon plan.parquet / beam_mu / fluence are untouched.

Order: task-01 metadata → task-02 ion_plan_table + write_pbs_rtplan → task-03 preprocess/manifest → task-04 convert + docs.

Carry-over gotchas:
- Commit gate: ≤ 5 files per commit; private-name mentions need CLAUDE_ALLOW_PRIVATE_API=1 listed in the body.
- `uv run --group dev ...` only; no bare python. Full suite ~4 min.
- Synthetic DICOM via save_as/dcmread; SYNTHETIC_ identifiers; no UIDs/values in log assertions.
- Temp files only in your own scratchpad (tmp-guard blocks elsewhere).
