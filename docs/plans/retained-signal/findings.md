# Findings: retained signal (#305)

Spec: docs/specs/2026-11-12-retained-signal.md. Plan: docs/plans/retained-signal/plan.json.

Summary: dose `retained` becomes the fraction of native voxels >= 1 Gy (DOSE_RETAINED_GY, inclusive) kept by the crop;
masks typed EXTERNAL keep their own series.json value but leave the plan-level minimum (preprocess run-log count,
manifest `retained`) via one shared predicate in preprocess.py that manifest imports. Signal only: no new refusals,
`cropped` unchanged, LOW_RETAINED stays 0.9.

Pre-checked: preprocess_report.csv has no `retained` column (only `cropped`); DEFAULT_DOSE (box 3..9) sits inside CUT's
kept 3..8, so a fixture with only an EXTERNAL mask cut has an uncut dose. series.json `dicom` is all-or-none per file:
a fixture that adds it must add it to every row.
