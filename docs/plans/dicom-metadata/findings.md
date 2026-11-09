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

## task-02

Files: src/rt_dataprep/masks.py, src/rt_dataprep/series.py, src/rt_dataprep/convert.py, tests/test_masks.py, tests/test_series.py.

Decisions:
- RoiPlan.applicators / .lines entries are now (ROINumber, file, points, ROIName). Unpacking sites: masks.py (lines append, applicator append, name list, the internal `apps` tuple gained the number as a 5th element) and convert.py (the write loop, the roi_names dicts). No other site in src, tests or scripts.
- series.registration_reg returns (registered, text, reg header or None); registration() is `registration_reg(...)[:2]`, so its tests and its one caller (convert.py) are unchanged. The returned header is the object from `regs`, not a copy.
- The classify record test goes through convert_plan with a spy on masks.classify (no direct classify fixture exists in test_masks).

Gotchas:
- A REG whose items cover both frames qualifies whatever frame the REG itself names; the "none qualifies" test needs an item list that omits the CT item and a REG in neither frame.

## review-group-2
PASS (no BLOCKING, no deferred). Notes only: test_series picks the expected REG by min SOP UID; AC1 test spies on classify via convert_plan.

## task-03

Files: src/rt_dataprep/convert.py, src/rt_dataprep/preprocess.py, tests/test_dicom_metadata_convert.py (new), tests/test_convert.py, tests/test_preprocess.py, tests/series_files.py, tests/test_plan_regression.py, README.md. tests/synthetic_dicom.py and tests/test_synthetic_dicom.py needed no change (`tags=` / `per_slice=` / ROI `algorithm` from task-01 covered every fixture).

Decisions:
- `_series_rows(plan, struct, ct_slices, ...)`: `row()` keeps `heads` for the SERIES_COLUMNS (the ct row still passes `[ct_slices[0][1]]`, so `_joined` cannot change series_description/series_date) and takes a separate `dicom_heads` for the ct row's aggregation. `dicom = metadata.collect(own groups) | plan_keys`, `plan_keys = collect([plan], (RTPLAN,))` computed once. The roi map gained the ROI number (kept: first element; applicators/lines: first element), passed as `roi_number`.
- `_place_series`: `ser.registration_reg` replaces `ser.registration`; each target tuple carries `plan_keys | reg_keys(reg)` (RTPLAN keys filtered from the ct row's `dicom`, REG keys only when a REG was chosen, so identity rows and `reg_matrix == ''` rows both follow the spec); the study tuple carries `{}`. `series_keys = collect(every slice of ordered, (ALL, IMAGES))` once per series after the image read succeeds; each folder gets `series_keys | extra_keys`, a new dict.
- preprocess.read_series_json: the key set minus `dicom` must be one of the two column sets, the value check is `dict` for `dicom` and `str` otherwise; all-or-none comes from the existing `set(row) == first`. Nothing else changed: `row | {...}` carries `dicom` and write_series_json puts it after SERIES_COLUMNS and before `retained` (insertion order).
- Existing exact-key-set assertions: test_convert.series_rows now asserts `[*SERIES_COLUMNS, 'dicom']` and returns the SERIES_COLUMNS projection (so the older equality asserts stay as they were); series_files.columns(row) does the same for test_plan_regression (pinned text unchanged).
- AC5 "two-file RTDOSE with different DoseType -> distinct" cannot be reached end to end: convert refuses any DoseType but PHYSICAL (REASON_DOSE_TYPE) before the row is built. The test lets one EFFECTIVE file past `conv._check_dose_header` (a wrapper that checks a PHYSICAL copy; the row reads the real header). A second test shows the two-file aggregation end to end through Manufacturer (distinct) and SeriesNumber (numeric min/max).
- AC5 "PixelSpacing differing -> distinct" on the CT: the volume check allows 1e-3 mm, so the fixture uses 1.5 vs 1.5004.
- AC9 values: `PatientAge` '073Y', size '1.8765', weight '70.321' and the SYNTHETIC_ site strings; PatientSex is one letter and cannot be searched for in a log, so it is not in the substring list.

Gotchas:
- `make_plan` had no way to tag the RTSTRUCT; it gained `struct_meta=`.
- Study folders are numbered by StudyDate then UID, and the synthetic series share a StudyDate, so tests find study rows by series_instance_uid, not by folder number.
- Full suite: 1562 passed, 4 skipped (5 min); ruff and compileall clean.
