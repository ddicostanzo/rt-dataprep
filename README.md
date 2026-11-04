# rt-dataprep

Turns radiotherapy DICOM exports into ML-ready cases. Stage F1, `organize`, sorts an export tree into
`<patient>/plan_<n>/` folders, one per RTPLAN. Later stages convert the folders to NIfTI and split them into cohorts.

## Install

```
uv sync --group dev
uv run python -m rt_dataprep --help
```

Checks (all must exit 0):

```
uv run --group dev pytest -q --tb=short
uv run --group dev ruff check .
uv run --group dev python -m compileall -q src
```

## PHI rules

This tool reads real patient data, so:

- Never commit patient data: no `.dcm`, imaging, `organize_report.csv`, `convert_report.csv` or `preprocess_report.csv`, nor
  their dated per-run copies (`<name>_<UTC time>.csv`; each run keeps one, and the fixed name is the latest). Tests build synthetic
  DICOM in `tmp_path` at run time and every identifier starts with `SYNTHETIC_`.
- No PatientID, name, date of birth, date or path in a log line, exception message or report text. Refusal reasons are
  fixed string constants. The reports in the output root are the only place a sanitized patient id appears.
- Reports and logs go under the output root, never the working directory.
- Source trees are only read, never moved, renamed or deleted.

## Commit hook

Install the PHI guard once per clone:

```
ln -s ../../.githooks/pre-commit .git/hooks/pre-commit
```
