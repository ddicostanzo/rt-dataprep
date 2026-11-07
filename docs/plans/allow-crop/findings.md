# Findings: allow-crop (#174)

Spec: docs/specs/2026-11-07-allow-crop.md. Plan: docs/plans/allow-crop/plan.json (task-01 series.csv schema →
task-02 allow_crop + retained + cropped → task-03 CLI + README). Run tests with `uv run --group dev pytest -q --tb=short`
in this worktree. Commit gate: at most 5 staged files. Append findings under `## <task-id>`.

## task-01

Files: src/rt_dataprep/convert.py (write_series_csv gains `columns=SERIES_COLUMNS`), src/rt_dataprep/preprocess.py
(PREPROCESSED_SERIES_COLUMNS; plan and study series.csv written with it; read_series_csv accepts either header),
tests/test_preprocess.py, tests/test_manifest.py.

- Every row preprocess writes sets `'retained': ''` explicitly (`row | {...}`), so a row read from an already
  preprocessed tree never carries the old value forward. task-02 replaces the blank with the real value.
- Gotchas: tests/test_preprocess.py already has a `header(image)` helper (image geometry); the new CSV one is `csv_header`.
  Existing test `test_series_csv_keeps_every_column...` now expects the extra blank `retained` key.
- Full suite: 1174 passed, 4 skipped (~3.7 min); ruff clean.
