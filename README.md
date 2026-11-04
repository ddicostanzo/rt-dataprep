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

## Output layout

`convert` writes one folder per patient under the output root:

```
SYNTHETIC_P1/
  plan_1/                      # one per RTPLAN: CT, dose, masks, registered series, series.csv
  study_1/
    series_CT_1.nii.gz
    series_MR_1.nii.gz
    series_MR_2.nii.gz
    series.csv                 # one per study, one row per volume
```

- Plan folders are as before. Every other CT, MR and PT series goes to `study_<n>/` as `series_<MOD>_<k>.nii.gz`. A series
  registered to a plan stays in its plan folder and is also written to its study. A plan's planning CT is not repeated in a study.
- Studies are numbered by StudyDate, then StudyTime, undated last. Patients with no RTPLAN get study folders only.
- `manifest` writes one row per plan folder or study series. Four columns come after the first ten: `case_type`
  (`plan` or `imaging`), `modality`, `image` (blank for plans) and `series`: a JSON list of `{"file": ..., "modality": ...}`
  for the plan's aligned series only (identity matrix and on the planning CT's grid; `[]` for imaging rows).
- `preprocess` resamples each registered MR or PT series of a plan onto the final CT grid and writes it with `identity`.
  A series it cannot align (unusable matrix, no overlap, unreadable file) is left out and counted in the
  `skipped_series` column of `preprocess_report.csv`.
- An imaging `case_dir` is a case id, `<patient>/study_<n>/series_<MOD>_<k>`, not a path. Open
  `<root>/<patient>/study_<n>/<image>`, where `image` is the volume file name (it has the step prefixes in a preprocessed tree).
- `preprocess` runs study series one at a time. CT gets every step that is on; MR and PT get the geometric steps only
  (resample, harmonize, reshape), never the HU rescale.
- With imaging patients in the manifest the same `--seed` gives a different split, so an existing `splits.json` is refused.
- Trees converted before this change hold `images/`, which is ignored: re-convert them.

## PHI rules

This tool reads real patient data, so:

- Never commit patient data: no `.dcm`, imaging, `organize_report.csv`, `convert_report.csv`, `preprocess_report.csv`, `manifest.csv`
  or `splits.json`, nor their dated per-run copies (`<name>_<UTC time>.csv`; each run keeps one, and the fixed name is the latest). Tests build synthetic
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
