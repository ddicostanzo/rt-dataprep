# F2b findings (shared, append-only)

Spec: `docs/specs/2026-11-03-f2b-preprocess.md`. Plan: `docs/plans/f2b-preprocess/plan.json`.

Summary:
- New `rt_dataprep preprocess <converted> <out> --legacy-rnt [--legacy-linear-masks]` reads F2a output through
  `series.csv` and writes a separate tree.
- Steps run on the CT in legacy order: initial resample 1 mm, rescale, final spacing 1.25 mm, crop/pad 701x701x433.
  The grid follows torchio 1.2.1: `floor` size and centre-preserving origin; crop/pad puts `ceil(d/2)` at the start.
- Dose and masks are resampled once onto the final CT image, with nearest neighbour (uint8) for masks or linear
  (float32) under legacy_linear_masks. Names are prefixed with no ahe.
- Failure table follows convert: a partial folder per patient, and REASON_ constants.
- Every row is checked before the first write of a plan. REASON_CROP refuses label truncation.

Guardrails:
- PHI: fixed REASON_ constants. No paths, IDs or UIDs in logs or exceptions. Synthetic IDs have no digits or
  underscores.
- Commit gate: at most 5 staged files, and no dotted name whose last part starts with `_` in added lines (docs
  included). tests/test_organize.py:24 and tests/test_dicom_io.py:85 trip the MRN rule if edited.
- No test pads to 701x701x433: use dataclasses.replace(LEGACY_RNT, new_shape=<small>).
- Third-party calls go through dicom_io.quiet_call. A CaseRefused with REASON_INTERNAL re-raises to an abort.

## task-01

Files: `src/rt_dataprep/preprocess.py`, `tests/test_preprocess.py` (new).

Public API for task-02: `PreprocessSteps`, `LEGACY_RNT`, `read_image(path)`, `process_ct(ct, steps)` -> final CT,
`process_dose(dose, final_ct)`, `process_mask(mask, final_ct, steps)`, `output_name(name, steps, ct=False)`, plus the pure
helpers `reference_grid`, `resample_to`, `rescale`, `crop_or_pad`. `REASON_FILE_UNREADABLE` and `REASON_INTERNAL` are
preprocess's own constants (convert's `REASON_FILE_UNREADABLE` says "DICOM", so it is not reused).

Decisions:
- `reference_grid` returns `(size, origin, spacing, direction)` and is fed to the explicit-grid `sitk.Resample` overload,
  so no scratch volume is allocated for the reference.
- Only the three `process_*` entry points and `read_image` run inside `quiet_call`; the pure helpers are unwrapped.
  Task-02 must call the entry points, not the helpers.
- `process_dose` reuses `convert.resample_dose` (the same Resample call the spec names).
- `process_mask` resamples whenever it is called; task-02 decides that this happens only when a geometric step is on.
- Nothing here writes: task-02 writes through `convert.write_atomic`.
- `new_shape` and `new_spacing` are in SimpleITK (x, y, z) order, so `GetSize() == new_shape`.

Gotchas:
- `image + 1024.0` / `/ 4096.0` on a float32 sitk image returns float64. `rescale` uses `sitk.ShiftScale`, which keeps float32
  and is exact (1/4096 is a power of two).
- ITK treats a point up to half a voxel outside the outermost voxel centres as inside the buffer and clamps to the edge
  value instead of the default 0. Tests that assert 0 outside a grid must sit more than half a native voxel beyond it.
- With the 5^3 fixture at 1 mm, new_spacing 1.5 and final 1.25, a direct 1.25 resample gives n = 4, the two-step gives 3;
  `test_ct_steps_run_in_legacy_order` pins that.

## review-1
- BLOCKING (fixed by the orchestrator): no test pinned nearest-neighbour masks; linear+uint8 passed every test. Added
  test_default_masks_take_the_nearest_native_voxel (independent rounding of native indices); the mutant fails it.
- WARNING (deferred, hardening at PR time): an axis with N >= 2 and N*s < s' gets size 0 and aborts the run
  (REASON_INTERNAL) instead of refusing one plan. Plan gap; unlikely for a CT.
- NOTEs: the float32 rescale test uses float32 input (int16 checked by hand); the array-view test is a regression guard.

## task-02

Files: `src/rt_dataprep/preprocess.py` (walk, report, plan/patient functions), `src/rt_dataprep/convert.py` (`_crash_site` -> `crash_site`, 2 sites,
no behaviour change), `tests/test_preprocess.py` (+42 tests).

Decisions:
- `preprocess_plan` writes into `partial/plan_<n>`; `preprocess_one` removes that folder on any refusal. `check_rows` runs kind, ct/dose counts,
  file presence (membership in the folder's non-symlink file listing) and the grid check from `read_info` (header only) before `mkdir`.
- The crop check (`check_crop`) needs the pixels, so it runs per image at write time. A REASON_CROP refusal after earlier files were written
  still removes the whole plan folder. It uses the nonzero bounding box: with the CT's direction, each final axis follows one native axis, so
  the box corners are the extremes.
- Without a geometric step (rescale only) dose and masks are `shutil.copyfile`d under `output_name` (the native name) and never read.
- `REASON_EXISTS`, `REASON_NO_PLANS` and `REASON_IO_ERROR` are preprocess's own constants (convert's texts name the organized tree).
- The report row class is `PreprocessRow`; outcomes reuse `conv.OUTCOME_SKIPPED/REFUSED`.

Gotchas:
- `preprocess(..., steps=PreprocessSteps())` needs `# noqa: B008` (frozen dataclass default).
- A one-voxel NN-emptied fixture needs native 0.5 mm: final centres land on native indices 1.5, 4, 6.5, 9, ..., so index 10 is never nearest.
- A final-grid boundary voxel (continuous index exactly -0.5) is inside; tests place cut-offs well past it, one at 4.3 on a 4-voxel axis.

Deferred (not fixed here):
- Two series.csv rows naming one file write the same output twice. A series.csv with extra columns fails in `write_series_csv`
  (ValueError -> REASON_INTERNAL abort). Both only arise from a hand-edited series.csv.

## review-2
- No BLOCKING. 19 of 24 mutants killed (scratch copy; run `uv run --group dev python -m pytest` there, since a copied
  .venv/bin/pytest shebang points at the original repo).
- WARNING (deferred): the low-side REASON_CROP bound is barely tested (`< -1.5` survives); only the high side has a
  near-boundary case.
- WARNING (deferred): symlinked patient/plan folders and series.csv are followed (read only).
- WARNING (deferred): `rmtree(out_dir, ignore_errors=True)` failing silently would let a refused plan's partial
  files be renamed into place while the report says refused.
- NOTEs: the CT-first sort is untested (F2a writes the CT first); out_root mkdir/report open OSError names the root.

## task-04

Files: `tests/test_preprocess.py` (+`test_matches_torchio`), `README.md` (never-commit list names `preprocess_report.csv`).

A2.6 torchio run, for the PR body:
- Command: `uv run --with torchio==1.2.1 --group dev pytest -q --tb=short tests/test_preprocess.py` (no 3.12 fallback needed: torch resolved on 3.13)
- Python: 3.13.16
- Result: `59 passed in 5.14s` (the A2.6 test ran, not skipped). Normal suite: `325 passed, 1 skipped`.
- The test runs with the rescale step on: `(x + 1024) / 4096` is applied between the two torchio resamples, so it covers the
  preset's order. Native 11x9x7 at (1.2, 1.0, 2.0) mm; new_spacing 1.5 and final 1.25 mm; new_shape (14, 4, 13) pads x by 3+2,
  crops y by 2+1 and pads z by 2+1, so the ceil/floor split is exercised. Swapping start and end in `crop_or_pad` makes it fail
  (origin x -6.5 vs -7.75), so it does pin the split.
- torchio side: only public API (`Subject`, `ScalarImage`, `LabelMap`, `Resample(..., antialias=False)`, `CropOrPad`, `set_data`, `save`).
