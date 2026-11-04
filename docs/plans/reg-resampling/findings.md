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
