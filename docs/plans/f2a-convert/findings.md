# F2a PR 1 findings (shared, append-only)

Spec: docs/specs/2026-11-03-f2a-convert.md (REVIEWED). Plan: docs/plans/f2a-convert/plan.json (6 tasks, sequential).

Summary: organize renames fx_<n> to plan_<n> and always copies the resolved RTSTRUCT (#7). convert reads each
<patient>/plan_<n>/, follows RTPLAN -> RTSTRUCT -> CT by UID, writes planning_ct.nii.gz (float32 HU), dose.nii.gz
(Gy on the CT grid; PLAN > FRACTION > BEAM-sum precedence), {0,1} masks via DicomRTTool 6.2 (verified after the call)
and applicator text files, plus series.csv per plan folder and convert_report.csv. Every third-party call runs in a
wrapper that captures warnings, logs and fd-level stdout/stderr and raises CaseRefused(reason) outside the except
block. Read plan.json metadata.commit_gate_landmines before editing tests.

Verify: uv run --group dev python -m compileall -q src; uv run --group dev pytest -q --tb=short; uv run --group dev ruff check .
