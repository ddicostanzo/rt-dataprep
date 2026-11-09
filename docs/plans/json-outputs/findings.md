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

## task-02

Files: src/rt_dataprep/{manifest,split,dicom_io,__main__}.py, examples/synthetic_workflow.py, tests/test_{manifest,split,dicom_io,cli,preprocess,preprocess_hardening}.py. tests/test_organize.py needed no change (its dated-report pattern is CSV and untouched). Suite: 1466 collected (was 1441), no test deleted; the 4 skips are the existing platform skipifs.

Decisions:
- `write_report` branches on the name's suffix: `.jsonl` -> `json.dumps(dict(zip(header, row, strict=True))) + '\n'` per row, opened `newline='\n'`; anything else the unchanged csv path. Dated name is `{stem}_{stamp}{suffix}`; the copy-then-os.replace of the latest is untouched.
- manifest rows hold native lists (masks, `_aligned` result, `[]` on study rows); `rows.sort(key=lambda r: r[0])` because whole-row comparison would reach lists of dicts on a case_dir tie. fov_mask rule unchanged (FOV_COLUMNS iff any row has one); zero rows -> 0-byte file (and 0-byte dated copy).
- split._read iterates the `utf-8-sig` handle, skips blank lines, first failing line decides the reason (not-object / bad JSON / not UTF-8 / duplicate key / deep nesting / non-string-empty-whitespace value -> REASON_ROWS; missing key -> REASON_COLUMNS), reason assigned in/after the except and raised after it, no chaining. split.py keeps its own 4-line `_no_duplicate_keys` (importing preprocess's would pull numpy into split, which a test forbids).
- REASON_COLUMNS = 'a manifest line has no patient or case_dir'; REASON_ROWS = 'a manifest is not UTF-8 JSON Lines of objects whose patient and case_dir are non-empty strings'.

Gotchas:
- A patient with a lone surrogate escape (`"\ud800"`) parses as JSON but fails `.encode()` in split's ordering hash (UnicodeEncodeError, not mapped). Deferred: hardening.
- split.py's module docstring and `check_disjoint` docstring, manifest.py's module docstring still say CSV: task-03.
- test_manifest `read(root)` now returns parsed JSONL dicts, so comparisons use native lists.

## review-2
PASS, no BLOCKING; splits.json byte-identical old vs new on a 37-patient synthetic cohort. Deferred: W1 lone-surrogate patient/case_dir passes _read then UnicodeEncodeError in the hash (CLI shows fixed unexpected-error line, no content); N5 a root with only an old manifest.csv gives a generic file-system error from split (no rerun hint).

## task-03

Files: README.md, examples/README.md, SECURITY.md, project-profile.yml, .gitignore, and module docstrings/comments of src/rt_dataprep/{convert,manifest,preprocess,series,split}.py (text only, no behaviour change).

Decisions:
- .gitignore adds `manifest*.jsonl` and `series.json`; `series.csv` and `manifest*.csv` stay (v0.1.x trees exist and are PHI).
- README: series.json described as a JSON array, one object per file, same string fields; manifest.jsonl key table (masks/series native lists); reading examples (`json.load`, `pd.read_json(..., lines=True, dtype=False)` with the dtype note); v0.1.x trees refused with the REASON_LEGACY_SERIES text and reconverted with `convert --overwrite`; 'column' wording became 'key' for manifest/series.json fields (report CSV columns keep 'column'). The load-cases example uses json.loads per manifest line and json.load for series.json.
- preprocess.py's module docstring needed only the series.json name; its legacy docstrings (is_legacy etc.) already say series.csv on purpose.

AC17 grep leftovers (all allowed): .gitignore series.csv / manifest*.csv; preprocess.py LEGACY_SERIES_NAME and REASON_LEGACY_SERIES definitions plus legacy docstrings; manifest.py comment "the CSV is never parsed"; README legacy-tree paragraph (names series.csv and the reason text); tests that build legacy folders.
Gotcha: README line on the Python API `split(manifest_path)` ValueError list still said "columns" for the manifest; reworded to keys.
