# Findings: retained signal (#305)

Spec: docs/specs/2026-11-12-retained-signal.md. Plan: docs/plans/retained-signal/plan.json.

Summary: dose `retained` becomes the fraction of native voxels >= 1 Gy (DOSE_RETAINED_GY, inclusive) kept by the crop;
masks typed EXTERNAL keep their own series.json value but leave the plan-level minimum (preprocess run-log count,
manifest `retained`) via one shared predicate in preprocess.py that manifest imports. Signal only: no new refusals,
`cropped` unchanged, LOW_RETAINED stays 0.9.

Pre-checked: preprocess_report.csv has no `retained` column (only `cropped`); DEFAULT_DOSE (box 3..9) sits inside CUT's
kept 3..8, so a fixture with only an EXTERNAL mask cut has an uncut dose. series.json `dicom` is all-or-none per file:
a fixture that adds it must add it to every row.

## task-01
- Shared predicate is `counts_toward_retained(row)` in preprocess.py (task-02: `from rt_dataprep.preprocess import counts_toward_retained` in manifest). It reads `row.get('dicom', {})`, so a row without `dicom` counts; the string match is exact ('external' counts).
- The dose threshold is `native >= DOSE_RETAINED_GY` passed inline to native_retained (a uint8 SimpleITK temporary, geometry kept), freed on the next statement; native_retained is untouched apart from its docstring.
- The 0.75 DOSE_SPLIT value became 0.5 (both slabs are >= 1 Gy); the sub-1 Gy tail test now asserts '1.0' exactly. Test 6 (single 1.0 Gy voxel, retained '0.0') passes unchanged.
- Test 3 fixture: EXTERNAL_CUT has a full-grid body (cut) and an inner ptv; DEFAULT_DOSE (3..9) lies inside CUT's kept 3..8 cells, so only the body is cut.

## task-02
- `_retained` now builds the list of dose/mask rows with a `retained` cell, parses all of them (EXTERNAL included, so a malformed EXTERNAL cell still refuses with REASON_SERIES_CSV), then takes the minimum over `counts_toward_retained` rows; `{}` when none qualify.
- Tests reuse `with_roi_types` and `EXTERNAL_CUT` from test_preprocess (task-01). The ~L577 comment said 0.75 for the dose; it is 0.5 since DOSE_SPLIT changed.
