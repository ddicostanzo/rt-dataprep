# Findings: --overwrite for organize, convert and preprocess

Spec: `docs/specs/2026-11-05-overwrite-flag.md`. Plan: `docs/plans/overwrite-flag/plan.json`.

Summary: `--overwrite` on organize/convert/preprocess (API `overwrite=False`) replaces a patient's existing output folder
only after the new build holds at least one successful case; otherwise the old folder is kept. A shared helper in
`dicom_io.py` recovers interrupted swaps (`.<name>.old`, leftover `.<name>.partial`) and publishes partials with rollback.
Convert/preprocess gain a casefolded same-run guard under `--overwrite`. Summary log lines append `; N replaced, N kept`.

Order: task-01 (helper) → task-02/03/04 (stages) → task-05 (CLI + README).

Guardrails: worktree uses its own `.venv` via `uv run --group dev`; at most 5 staged files per commit; no `raise` inside
an except block; logs carry counts and REASON_ constants only; tests use SYNTHETIC_ names.
