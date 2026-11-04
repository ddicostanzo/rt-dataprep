# Findings: F2a PR 2 (other series, registration, images/)

Spec: `docs/specs/2026-11-03-f2a-pr2-other-series.md` (addendum to `docs/specs/2026-11-03-f2a-convert.md`).
Plan: `docs/plans/f2a-other-series/plan.json`. Read both before exploring. Append under `## <task-id>` when done.

## Summary
- task-01: #33 narrows convert_plan's tag catch to individually wrapped header-only calls; an unexpected
  `Exception` per patient gives a REASON_INTERNAL row, the report is written, then ConvertAborted (fixed text).
- task-02: fixtures (write_reg RegistrationSequence; write_ct_series orientation/positions/PT/slopes/frames).
- task-03: series.py volume check (any orientation, SERIES_ATOL_MM 0.01, ORTHO_ATOL 1e-4) and geometry-owning read.
- task-04: REG parsing and reg_matrix = inv(M_ct) . M_series.
- task-05: convert_plan takes its headers; series.csv rows move out of the plan path.
- task-06: candidates, placement into plan folders / images/, report rows, plan-less patients.

## Facts checked before planning (2026-11-03)
- SimpleITK 2.5.7 ImageSeriesReader forces an orthogonal direction from the cosines and takes slice spacing as
  |last - first|/(n - 1).
- Its NIfTI writer never rejects a non-orthonormal direction; it coerces it to orthogonal with only a warning.
- organize puts excluded-model studies and CT localizers in misc_files/ (it filters cases only).

## Guardrails
- Commit gate: no added line with a dotted name whose last part starts with an underscore (any file type); no
  mrn/patient_id next to 6-10 digits; at most 5 staged files per commit (this file counts).
- Do not edit tests/test_organize.py or tests/test_dicom_io.py.
- PHI: fixed REASON_ constants; no path, ID, UID, date or description in logs or exception text; synthetic IDs
  without digits or underscores.
- Tests: `uv run --group dev pytest -q --tb=short`; lint: `uv run --group dev ruff check .`
