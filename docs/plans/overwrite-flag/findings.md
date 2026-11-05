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

## task-02
- organize(): keyword-only overwrite; `_organize_patient(..., overwrite)` returns `(row, read_s, copy_s, walked, status)`.
- recover runs after `claimed.add` at the old exists check; publish replaces `partial.rename(target)` inside the existing
  `except BaseException` cleanup; success is always True there (a completed organize build is always organized).
- Deviation: the exists check uses `os.path.lexists` so a dangling symlink at `<name>` skips (REASON_EXISTS) instead of
  reaching publish's ValueError misuse guard. Untested.
- Kept only from REASON_NO_SOP_UID / REASON_SPLIT_SERIES with an existing, unclaimed name. Summary ends `; N replaced, N kept`.

## task-03
- convert(): keyword-only overwrite; `_convert_patient(folder, out_root, exclude_models, workers, overwrite, claimed)`
  returns `(rows, reading, files, status)`. Under overwrite: casefold guard (new convert.REASON_NAME_CLAIMED, one
  patient-level row, status None) → recover → `os.path.lexists` check.
- publish replaces `partial.rename(target)` in the `done`-flag finally; success = any converted row. Kept from the
  REASON_NO_PLANS early return or publish('kept'). Summary ends `; N replaced, N kept` (tests/test_series.py updated).
- Failure injection: OSError via `conv.sitk.WriteImage` (build) and `dicom_io.rename_dir` (swap); unreadable via
  `dicom_io.read_header_core`.
- Untested: claimed untouched without overwrite (indirect only); recover raising on a failed restore.

## task-04
- preprocess(..., *, overwrite=False); `preprocess_patient(folder, out_root, steps, overwrite=False, claimed=None)`
  returns `(rows, status)`; same flow as convert. New preprocess.REASON_NAME_CLAIMED (same text as convert's).
- recover runs on every patient (plain runs too) and cleans leftover `.old`/`.partial`. Skip check uses `os.path.lexists`.
- Failure injection: OSError via `conv.write_atomic` (build) and `dicom_io.rename_dir` (swap); tests use `SMALL`.
- Untested: dangling symlink at the target; recover raising on a failed restore.

## review-2
No BLOCKING. Deferred: W1 convert/preprocess case-guard tests would not catch recover running before the guard (add a
`.<lowercase>.old` canary); W2 AC-O7 second half only checked indirectly; N1 `preprocess_patient(overwrite=True)` without
`claimed` raises TypeError (default `frozenset()`); N2 convert failed-build test lacks the caplog name check; N3 untested
dangling-symlink target and recover failing to restore; N4 organize's redundant `rmtree(partial)` after recover.

## task-05
- `--overwrite` (shared OVERWRITE_HELP) on organize/convert/preprocess; `__main__` always passes `overwrite=args.overwrite`.
- Test seams updated: PRE_RUN forwards `**kw`; `lambda *args` stubs take `**kwargs`; exact kwargs assertions include
  `'overwrite': False`.
- Deviation: tests/test_mask_retention.py:97 had a fourth `lambda *args` stub on `cli.preprocess` (not in the plan's file
  list); same one-line fix.
- README: commands table, a "Replacing existing output with `--overwrite`" subsection, troubleshooting row, reworded
  review-before-replacing line.

## review-3 (task-05 wave + integration)
No BLOCKING. Fixed in README: W1 replacement is per patient folder (refused cases are not carried over), W2 a root holding
splits.json needs a fresh preprocess root, N1 IO-error and pre-name organize refusals keep the old folder uncounted.
Deferred: N2 Python API section omits `overwrite=`; N3 organize's REASON_NAME_CLAIMED wording differs from convert/
preprocess; N4 a dangling symlink at `<name>` under --overwrite is moved to `.<name>.old` and never removed (rmtree refuses
links); N5 preprocess_patient now returns a pair.
