# Findings: DICOM metadata (PR 2 of 2)

Spec: docs/specs/2026-11-09-dicom-metadata.md. Plan: docs/plans/dicom-metadata/plan.json (the contract).

Summary: each series.json object gains a `dicom` object (new module src/rt_dataprep/metadata.py: value rules,
min/max and distinct aggregation, patient_age, per-cell guard); masks applicator/line records gain ROINumber;
series.registration_reg returns the chosen REG; convert wires `dicom` into every row; preprocess's reader accepts it;
manifest.jsonl gains 8 optional flat keys.

Guardrails: no metadata value in a log line, exception message or report CSV; synthetic fixtures only, via
tests/synthetic_dicom.py, written to disk and read back. DOB phi-guard pattern only matches separated dates, so
YYYYMMDD fixtures pass. Commit gate: CLAUDE_APPROVE_LARGE_COMMIT=1 sanctioned per task; CLAUDE_ALLOW_PRIVATE_API=1 only
for this repo's own _names, listed in the commit body. Full suite ~4 min.

Append below under `## <task-id>` (append only).
