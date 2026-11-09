# series and manifest as JSON (PR 1 of 2)

Status: REVIEWED

## Problem
`series.csv` and `manifest.csv` are about to gain nested, multi-valued DICOM metadata (PR 2,
docs/specs/2026-11-09-dicom-metadata.md). CSV cannot hold nested values without ad-hoc separators, and every new field is
a schema change that the strict readers refuse. The manifest already JSON-encodes three of its cells (`dose` is not, but
`masks` and `series` are). Owner decision 2026-11-09: the data files become JSON, the run reports stay CSV.

## Decisions (owner, 2026-11-09)
- `series.csv` → `series.json` in every plan_<n>/ and study_<n>/ folder (convert and preprocess): an indented JSON array,
  one object per output file.
- `manifest.csv` → `manifest.jsonl` (JSON Lines: one case object per line), dated copies `manifest_<UTC time>.jsonl`.
- organize/convert/preprocess report CSVs and `splits.json` are unchanged.
- No old-format support: readers look only for the JSON files; a v0.1.x tree has no series.json, so it is refused exactly
  as a folder without series.csv is today. Release v0.2.0 after PR 2.
- This PR is a pure format change: same fields, same values. PR 2 adds the metadata.

## series.json
- An array of objects; each object has exactly today's SERIES_COLUMNS keys (preprocess's adds `retained`), in that order,
  every value a string exactly as the CSV cell is today. Written atomically like series.csv (`write_atomic` pattern,
  `REASON_WRITE` on failure), UTF-8, `ensure_ascii=False`, `indent=1`, trailing newline.
- Reader (`read_series_json`, replacing `read_series_csv` with the same (rows, malformed) contract): malformed when the file
  is not valid UTF-8 JSON, not an array, an item is not an object, an item's keys are not exactly SERIES_COLUMNS (or
  SERIES_COLUMNS + `retained`), or a value is not a string. Every refusal it drives keeps its existing fixed reason
  (`REASON_SERIES_CSV` keeps its constant name; its text is updated to name series.json).
- Every place that tests for the file's presence (a plan or study folder "holds a series.csv") uses `SERIES_NAME`, now
  `series.json`.

## manifest.jsonl
- One JSON object per line, keys in today's COLUMNS order, plus `fov_mask` on every line when any line has it (as the
  column today). Values: `masks` and `series` become their native JSON values (a list; a list of objects) instead of
  JSON-encoded strings; every other value stays the string it is today (`dose` '0'/'1', `ct_size` text, etc.).
- `write_report`'s dated-copy-then-atomic-replace behaviour is kept for the manifest (a JSONL twin of it, or a format
  argument); the run reports keep CSV.
- `split` reads manifest.jsonl (`patient`, `case_dir`); `--external` takes another dataset's manifest.jsonl. Its
  malformed-input refusals (`REASON_COLUMNS`, `REASON_ROWS`) keep their constants and apply to: invalid JSON on a line,
  a line that is not an object, or a line missing `patient`/`case_dir` or with a non-string value for them. A blank last
  line is allowed.
- CLI help and `MANIFEST_NAME` say manifest.jsonl.

## Other readers
- scripts/parity.py reads series.json.
- examples/ and README: every series.csv / manifest.csv mention becomes series.json / manifest.jsonl, with a short reading
  example (`json.load`, `pd.read_json(path, lines=True)`).

## Acceptance
- A synthetic plan and study convert to series.json files whose objects equal today's series.csv rows (same keys, order,
  string values).
- preprocess reads series.json and writes its own with `retained`; a malformed series.json (bad JSON, wrong keys, a
  non-string value, not an array) refuses the plan or study with the existing reason and the run continues.
- manifest writes manifest.jsonl and a dated copy; each line parses to an object whose masks/series are native lists;
  split and `--external` work from it and refuse bad JSON / missing keys with the existing reasons.
- A folder holding only a v0.1.x series.csv is refused with REASON_LEGACY_SERIES (see Details).
- Existing tests updated to the new files; no test deleted or skipped; synthetic data only; no new log lines.

## Deferred
- Converting the run reports to JSON; a reader for old CSV trees; typed (non-string) values in series.json (PR 2 adds a
  nested `dicom` object; the existing fields stay strings).

## Details (stress test, 2026-11-09)
- `.gitignore` gains `series.json` and `manifest*.jsonl`; `SECURITY.md` and `project-profile.yml` (compliance text, dated
  copy pattern) name the new files.
- Reading example in README: `pd.read_json(path, lines=True, dtype=False)` (without `dtype=False` pandas turns a patient
  `00123` into 123 and dates into integers); series.json with `json.load`.
- Readers open as text with `encoding='utf-8'` (series.json) / `'utf-8-sig'` (manifest, as split does today), never
  `json.loads(bytes)`; catch `(ValueError, RecursionError)` (JSONDecodeError and UnicodeDecodeError are ValueErrors;
  deep nesting raises RecursionError); reject duplicate keys with an `object_pairs_hook` that raises ValueError. Map to
  the fixed reason after the except block, never `raise ... from exc` (JSONDecodeError.doc holds the whole file;
  UnicodeDecodeError quotes bytes).
- series.json key order: SERIES_COLUMNS first, then any other keys in insertion order (PR 2's `dicom`, preprocess's
  `retained`). The reader compares key sets, not order, and every item of one file must have the same key set: exactly
  SERIES_COLUMNS, or SERIES_COLUMNS + `retained` (PR 2 extends this with an optional object-valued `dicom`).
- `write_report` keeps its behaviour (dated copy written directly, then copied and atomically replaced into the latest
  name); the dated copy takes the name's own suffix (`manifest_<stamp>.jsonl`), so reports stay `.csv`.
- manifest.jsonl lines: `json.dumps` with the default `ensure_ascii=True` (no raw U+2028/U+0085 that `splitlines` would
  split on), written with `newline='\n'`; the reader iterates the file handle. Lines sorted by `case_dir` as today;
  `fx` is '' on study lines and `image` '' on plan lines; `fov_mask` is on every line (''- valued where none) when any
  case has one, else on no line; inside `series` each entry's own `fov_mask` stays optional; `dose`, `ct_size`,
  `ct_spacing` stay strings. Zero cases → a 0-byte file.
- split refusals: a line missing `patient` or `case_dir` → `REASON_COLUMNS`; invalid JSON, not UTF-8, a non-object line,
  or a non-string, empty or whitespace-only `patient`/`case_dir` → `REASON_ROWS`; blank lines anywhere are skipped; a
  0-byte manifest falls through to `REASON_NO_TRAIN` as a header-only CSV does today. Both reason texts are reworded
  (no "column").
- Legacy folders (owner decision): a plan or study folder that holds a `series.csv` and no `series.json` is counted as
  refused by preprocess and manifest with a new fixed reason `REASON_LEGACY_SERIES = 'converted by an older version
  (series.csv); reconvert with --overwrite'`, not silently skipped. Detection: `series.csv` present and `series.json`
  absent; the CSV is never parsed, and a legacy study folder counts as 1 refused.
- Text to update: every `REASON_*` text in preprocess.py that names series.csv (`REASON_NO_CASES`, `REASON_NO_CT`,
  `REASON_DOSE_COUNT`, `REASON_KIND`, `REASON_FILE_MISSING`, `REASON_STUDY_MODALITY`, `REASON_STUDY_NAME`,
  `REASON_SERIES_CSV`); module docstrings of convert, preprocess, manifest, split; scripts/parity.py (`REASON_NEW_MISSING`,
  its read, presence check and help text); `__main__.py` help strings; examples/README.md and
  examples/synthetic_workflow.py.
- Test surface: about 230 lines across 11 test files reference the CSV files (test_preprocess ~98, test_convert ~36,
  test_series ~30, test_manifest ~29, test_split ~14, others few). One shared test helper for reading/writing series.json
  is allowed; the staged diff will exceed the 800-line warning.
- No external consumer reads manifest.csv or series.csv (owner, 2026-11-09).
