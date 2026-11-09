# json-outputs findings

Spec: docs/specs/2026-11-09-json-outputs.md (PR 1). Next: docs/specs/2026-11-09-dicom-metadata.md (PR 2, separate run).
Plan: docs/plans/json-outputs/plan.json (3 tasks, sequential).
Append-only: each task adds `## <task-id>` with files changed, decisions, gotchas.

## task-01

Files: src/rt_dataprep/{convert,preprocess,manifest}.py, scripts/parity.py, tests/series_files.py (new: `read(folder)`, `write(rows, path)`), tests/test_{convert,preprocess,preprocess_hardening,series,masks,plan_regression,manifest,parity,cli}.py.

Decisions:
- `conv.write_series_json(rows, path)` builds each item as `{column: row[column] for column in SERIES_COLUMNS} | row`: convert's `row()` dicts are not in SERIES_COLUMNS order (DictWriter used to hide that), so the writer reorders; other keys keep insertion order. A row missing a SERIES column is a KeyError (programming error), not a silent '' as DictWriter's restval was.
- `pre.read_series_json` returns `(rows, True)` with the array's items when the file is a valid JSON array that fails the key/value check, and `([], True)` when unparseable or not an array: `study_refusal_count` (len of rows) keeps counting a wrong-keys study's items, as it did for wrong CSV columns. Empty array is not malformed (check_rows/check_study_rows refuse it as REASON_NO_CT / REASON_SERIES_CSV, as a header-only CSV was).
- Legacy: `pre.LEGACY_SERIES_NAME`, `pre.holds_series(folder)` (folder filters of preprocess_patient and manifest), `pre.is_legacy(folder)` (series.csv file and no series.json file). check_rows, check_study_rows and manifest._study_rows test it before any read; the CSV is never opened. A folder with both files is read as series.json.
- REASON_SERIES_CSV text: 'series.json is malformed: not valid JSON, not an array of objects of strings with the expected keys, or one file named twice'.
- test_plan_regression keeps its pinned CSV text, parsed with csv.DictReader and compared item-by-item (key order included) with series.json.
- Old CSV-only malformed cases (missing/extra column, long/short row) became the JSON cases in test_preprocess SERIES_JSON_BREAKS (invalid JSON, RecursionError, duplicate key, BOM, not UTF-8, empty, not array, non-object item, missing/extra key, differing key sets, non-string values); no test deleted.

Gotchas:
- Key order is not compared by the reader (key sets only), so a file with `retained` first reads fine.
- series.py's docstring and the module docstrings of convert/preprocess/manifest still say series.csv: task-03's files.
- Test helper names: `series_at`/`csv_at` in test_series became `sf.read`; `conv.write_series_json` is called directly only in the writer's own tests, other tests write through `sf.write`.

## review-1
PASS, no BLOCKING. Deferred NOTE: study_refusal_count counts non-object items of a corrupt series.json (count inflation only). Docstring/README leftovers are task-03.
