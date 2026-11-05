# Findings: series FOV mask (#129)

Spec: docs/specs/2026-11-04-series-fov-mask.md. Plan: docs/plans/series-fov-mask/plan.json (read its metadata.guardrails).
Summary: opt-in `PreprocessSteps.series_fov_mask` / `--series-fov-mask` writes `fovmask_<series output name>` (uint8,
final CT grid, the all-ones nearest-neighbour resample covers_ct already builds) per aligned plan series, lists it in the
plan row's files after its series, and the manifest series entry gains `fov_mask` when the file exists.

Decisions log:
- max_parallel_worktrees is 1 and the three tasks are small, so tasks run sequentially on feat/fov-mask itself (no
  per-task worktrees); each task is its own commit and wave review covers each group's diff.

Append below under `## <task-id>` (append only).

## task-01
Files changed: src/rt_dataprep/preprocess.py (series_fov_mask field; covers_ct returns (fov image, has_voxels); preprocess_plan
writes fovmask_<name> after the series and lists it in files), tests/test_preprocess.py (7 test cases + no-step loop entry).
Decisions:
- covers_ct returns `(landed, has_voxels(landed))`; the loop sets `fov = None` right after the check when the option is off
  (M6), and `del native, fov` at the end of each iteration.
- process_series signature untouched (tests monkeypatch it with *args).
Gotchas:
- The exact-equality clause holds as is: constant-v MR through linear resample gives exactly v at every mask==1 voxel.
- Module-level `FOV = replace(SMALL, series_fov_mask=True)` makes the whole test module fail collection until the field
  exists; red run was done by adding only the field first (3 tests failed, rest passed).
- test_preprocess.py imports `dataclasses`, not `replace`.
- Full suite: 783 passed, 4 skipped; ruff clean.

## task-02
Files changed: src/rt_dataprep/__main__.py (--series-fov-mask flag, passed in the same dataclasses.replace as legacy_linear_masks),
tests/test_cli.py (parametrized steps test via monkeypatched cli.preprocess; flag added to the no-steps loop).
Gotchas:
- Red run: 3 failed against unchanged code (argparse "unrecognized arguments"), then green. Full suite 786 passed, 4 skipped; ruff clean.
- No change was needed for the alone-flag case: PreprocessSteps() + series_fov_mask hits REASON_NO_STEPS from task-01.

## task-03
Files changed: src/rt_dataprep/manifest.py (_aligned adds fov_mask when fovmask_<file> is a file in the plan folder; module and
function docstrings updated), tests/test_manifest.py (one test: on/off trees, converted trees, read_info recorded to prove no mask header read).
Gotchas:
- Red run: the single new test failed on the missing key only; the other 31 manifest tests passed. Full suite 787 passed, 4 skipped; ruff clean.
- The mask is found by name on the output tree only; the converted tree never has fovmask_ files, so its entries are unchanged.

## review-loop-pass-1
- Spec axis (Opus): every plan criterion and spec AC met; no plan bugs; nothing blocking.
- Bugs (Opus, also wave review for task-02/03 and integration): no blocking findings; default output and memory unchanged;
  40 random rotated/sub-voxel cases matched mask to linear output voxel for voxel.
- Deferred, filed as hardening: #145 (test coverage gaps, incl. wave NOTEs), #146 (manifest mask header check), #147 (study/CT pad mask).
- No BLOCKING fix, so no verification pass.
