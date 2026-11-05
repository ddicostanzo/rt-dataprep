# Findings: --overwrite for organize, convert and preprocess

Spec: `docs/specs/2026-11-05-overwrite-flag.md`. Plan: `docs/plans/overwrite-flag/plan.json`.

Summary: `--overwrite` on organize/convert/preprocess (API `overwrite=False`) replaces a patient's existing output folder
only after the new build holds at least one successful case; otherwise the old folder is kept. A shared helper in
`dicom_io.py` recovers interrupted swaps (`.<name>.old`, leftover `.<name>.partial`) and publishes partials with rollback.
Convert/preprocess gain a casefolded same-run guard under `--overwrite`. Summary log lines append `; N replaced, N kept`.

Order: task-01 (helper) → task-02/03/04 (stages) → task-05 (CLI + README).

Guardrails: worktree uses its own `.venv` via `uv run --group dev`; at most 5 staged files per commit; no `raise` inside
an except block; logs carry counts and REASON_ constants only; tests use SYNTHETIC_ names.

## task-01
- `dicom_io.rename_dir(src, dst)`: the only rename the helper uses; tests monkeypatch it.
- `dicom_io.recover(out_root, name)`: restores `.<name>.old` when `<name>` is missing, deletes it when both exist, always
  deletes a leftover `.<name>.partial`; a failed restore raises OSError.
- `dicom_io.publish(out_root, name, *, overwrite, success)` → 'new' | 'replaced' | 'kept'. Existing target with
  overwrite=False raises ValueError (misuse, not caught by the stages' OSError handler). The re-raised OSError is the
  original (its text carries paths): stages must keep logging only their fixed reason.
- Existence uses `os.path.lexists`: a dangling symlink at `<name>` counts as existing.
- Stages call `recover` first and `publish` only for a built partial; with no target and no partial publish raises.

## review-1
No BLOCKING. Deferred: W1 AC-H7 sibling shares no name prefix (a `.{name}*` glob would pass); N2 no symlink test;
N3 publish assumes recover ran first (document); N4 no target + no partial raises (callers must not call publish then);
N5 callers must never log str() of the re-raised OSError.
