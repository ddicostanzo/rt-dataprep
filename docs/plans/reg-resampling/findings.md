# Findings: registered MR/PT aligned onto the planning CT grid (#1)

Spec: `docs/specs/2026-11-04-reg-resampling.md` (REVIEWED). Plan: `plan.json` beside this file.

Summary: preprocess resamples each registered `kind=series` row of a plan folder once onto the final CT grid, through
the inverse of `reg_matrix` (series frame -> CT frame), linear float32, 0 outside; a CT series is rescaled first. Left
out and counted in a new `skipped_series` report column: unusable matrix, no overlap, unreadable file. A missing file
or bad modality/name refuses the plan. Output rows carry `identity` and the CT frame UID; the manifest gains a `series`
column listing identity + same-grid series. Tasks: task-01 (series parser), task-02 (check_rows), task-03 (alignment),
task-04 (manifest, README).

Append under `## <task-id>`: files changed, decisions, gotchas, and every change to an existing test.

## Carried gotchas (from imaging-cases/findings.md)
- tests/test_plan_regression.py imports BOX, CT_DATE, SHAPE, SPACING, ORIGIN, dose_spec from tests/test_convert.py and VALUES, rigid from tests/test_series.py. Do not change them; the pin is never re-pinned.
- The commit gate flags dunder calls (setitem/delitem) and dotted private names in added lines, also in markdown; use getattr(module, '_name')  # noqa: B009 or plain functions.
- At most 5 staged files per commit: if a task touches more, split the commit; never squeeze.
- test_series_csv_keeps_every_column_renames_files_and_drops_series_rows is edited in task-02 (file added) and again in task-03 (series no longer dropped), by design.
- Deferred from plan review: #121 (all-zero native series kept), #122 (duplicate series ids on the plan path).

## Decisions Log
- max_parallel_worktrees is 1, so tasks run in order directly on feat/reg-resampling (one commit or more per task) instead of per-task worktrees and squash merges; the wave review runs over each task's commits.

## task-01
- Files: src/rt_dataprep/series.py, tests/test_series.py.
- `_is_rigid(matrix)` holds the rigidity test (finite, last row, R^T R = I, det > 0); `_rigid_matrix` calls it. New public `reg_matrix_array(text)`: `identity` -> np.eye(4); else 16 floats that pass `_is_rigid`, or None (empty, wrong count, non-number, nan/inf, non-rigid).
- Decision: tokens parsed with the existing `_or_none(float, token)`, so no try/except was added.
- Gotcha: float() accepts 'nan', 'inf' and '1_0'; the finite check in `_is_rigid` catches nan and inf, and '1_0' is read as 10 (harmless: convert writes only '%.10g').
- Existing tests changed: none (12 tests added at the end of tests/test_series.py).
- Red evidence: with src reverted, the 12 new tests failed (AttributeError on reg_matrix_array / _is_rigid); 167 existing passed.

## task-02
- Files: src/rt_dataprep/preprocess.py, src/rt_dataprep/manifest.py, tests/test_preprocess.py, tests/test_manifest.py.
- `check_rows` returns `(kept, ct, series)`; `series` is the kind=series rows in input order. Checks, in this order after the duplicate-file check: modality in CT/MR/PT (REASON_STUDY_MODALITY), `series_id(file)` not None (REASON_STUDY_NAME), the MOD inside the stripped id equals the modality (REASON_STUDY_MODALITY), then every kept and series file present (REASON_FILE_MISSING). No series header is read.
- `preprocess_plan` and `manifest` unpack `(kept, ct, _)`; series rows are still dropped from the output.
- REASON_STUDY_MODALITY / REASON_STUDY_NAME reworded to "a series.csv series row ..." (names unchanged; no digits, so the reasons test holds).
- Gotcha: a series row named like the CT file (planning_ct.nii.gz) is a duplicate file and gives REASON_SERIES_CSV before the name check; tests use a distinct bad name.
- Gotcha: step-prefix names (reshape_resample_initialresample_series_MR_1.nii.gz) pass because `series_id` strips them; the MOD check runs on the stripped id.
- Existing tests changed: `test_series_csv_keeps_every_column_renames_files_and_drops_series_rows` (tests/test_preprocess.py): it now writes a small series_MR_1.nii.gz first, because its series file was missing and check_rows now refuses that; its assertions are unchanged. No other existing test changed (test_manifest.py's monkeypatched `check_rows` wrapper passes the tuple through, so it needed no edit).
- New tests: 12 in tests/test_preprocess.py (check_rows shape and order, missing file, 7 bad-row cases, step prefixes, no header read, reworded reasons) and 1 in tests/test_manifest.py (missing series file counted refused). Red evidence: with src reverted 13 failed (shape, refusals, wording), 162 passed.
