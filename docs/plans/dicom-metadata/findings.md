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

## task-01

Files: src/rt_dataprep/metadata.py (new), tests/test_metadata.py (new), tests/synthetic_dicom.py, tests/test_synthetic_dicom.py.

Decisions:
- Public API as AC1, plus a `REG` group constant so `keys(REG)` and `reg_keys` (= `collect([reg], (REG,))`) share the one catalog table. `collect` applies no implicit ALL: the caller passes every group it wants.
- Source shapes in the table: a path (`Seq[0].Kw`, `Seq[*].Kw`), a tuple of fallback paths (first path with a value on any header is used for all heads: injection_time, treatment_machine), or a function `(heads, roi_number)` (patient_age, slices, the two ROI lookups).
- `[*]` paths always aggregate as `{'distinct'}` (never min/max, even for numeric DS like prescription_dose); numeric dedupe still applies (50 and 50.0 are one value).
- Guard is per key over all heads: a structural failure on one slice (e.g. an empty sequence read with `[0]`) omits the key for the row; a merely missing tag is ignored and gives the value of the rest. patient_age is the exception: an unusable header yields None and is skipped, so the other slices still count.
- A value that is blank after strip counts as absent. VR for aggregation is the set of VRs seen; min/max needs one VR family (numeric set, or a single DA/TM/DT).
- patient_age: a negative result (study before birth) is absent; the DOB is only read.
- synthetic_dicom: `tags=` on every writer (dict keyword -> value, applied at save time over the writer's own tags; list of dicts = sequence, nested; None deletes the tag), `per_slice=` on write_ct_series (dict or None per slice), `algorithm` per ROI in write_rtstruct_rois. REG matrix types and ROI interpreted types already existed.

Gotchas:
- pydicom 3: `'Typo' in ds` only WARNS ("Invalid value ... used with the 'in' operator") and returns False, so a misspelled keyword is silent apart from that warning. The key-set tests per group plus the warnings-as-errors test catch it.
- A DS/IS that does not parse cannot be built (assignment raises) and would make read_header refuse the file, so the "falls back to distinct" branch is tested on `metadata._aggregate` directly (this repo's own private name).
- An empty element reads back with VM 0 and value '' (not None) for AS/DA/CS.

## review-group-1
PASS (no BLOCKING). Deferred, to file as `hardening` at 2.9:
- W1: one slice with an empty RadiopharmaceuticalInformationSequence drops PT keys for the whole row (a missing tag is ignored); skip the bad slice like _age does.
- W2: mixed scalar/list values are tuple()d for comparison, so 'ab' equals ['a','b'] (metadata.py ~:211).
- W3: _roi_lookup: one item with an empty/non-integer ROI number drops the key even when another item matches.
- N1: DS 'NaN'/'inf' order unpredictably in min/max (spec Deferred already notes NaN).
- N2: TM string order is wrong across mixed precision ('0900' vs '090000.5'); spec rule.
