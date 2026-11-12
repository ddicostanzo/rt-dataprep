# Findings — proton-plans (#281)

Spec: docs/specs/2026-11-11-proton-plans.md (REVIEWED). Plan: docs/plans/proton-plans/plan.json.

Summary: convert writes `ion_plan.parquet` (one row per non-zero PBS spot, MODULATED + MU only) through a new
`ion_plan_table` module with the one fixed reason REASON_ION_PLAN; preprocess/manifest carry the new kind `ion_plan`;
metadata RTPLAN falls back to IonBeamSequence. Photon plan.parquet / beam_mu / fluence are untouched.

Order: task-01 metadata → task-02 ion_plan_table + write_pbs_rtplan → task-03 preprocess/manifest → task-04 convert + docs.

Carry-over gotchas:
- Commit gate: ≤ 5 files per commit; private-name mentions need CLAUDE_ALLOW_PRIVATE_API=1 listed in the body.
- `uv run --group dev ...` only; no bare python. Full suite ~4 min.
- Synthetic DICOM via save_as/dcmread; SYNTHETIC_ identifiers; no UIDs/values in log assertions.
- Temp files only in your own scratchpad (tmp-guard blocks elsewhere).

## task-01
- Files: `src/rt_dataprep/metadata.py` (RTPLAN `radiation_type`, `beam_type` now `(BeamSequence[*].X, IonBeamSequence[*].X)`;
  `treatment_machine` is BeamSequence, IonBeamSequence, TreatmentMachineSequence in that order); `tests/test_metadata.py`
  (two new tests: ion fallback incl. IonBeamSequence ranking above TreatmentMachineSequence; photon/brachy values and the
  full `keys(RTPLAN)` tuple pinned).
- `metadata._compute` already takes the first path with samples, so no logic change. `keys(RTPLAN)` is a tuple of names in
  table order; the pin test compares the whole tuple.
- RED was only the ion test (KeyError radiation_type); the photon/brachy pin passes before and after, as intended.
- The fallback is per whole path: a plan with BeamSequence yielding nothing but IonBeamSequence present uses the ion path.
  A mixed plan (both sequences) reports only BeamSequence values; spec premise 3 accepts this.

## task-02
- Files: `src/rt_dataprep/ion_plan_table.py` (new: `ION_SCHEMA`, `REASON_ION_PLAN`, `ION_WEIGHT_TOLERANCE`, `ion_plan_rows`,
  `to_table`, `write_ion_plan_table`; helpers copied from plan_table, no import of plan_table/beam_volume/fluence);
  `tests/test_ion_plan_table.py` (new, 55 fast tests + 1 slow); `tests/synthetic_dicom.py` (added `pbs_beam` and
  `write_pbs_rtplan` after `write_ion_rtplan`, which is untouched).
- Gotcha (pydicom 3.0.2): a multi-valued FL/FD (ScanSpotPositionMap, ScanningSpotSize, VirtualSourceAxisDistances, multi
  ScanSpotMetersetWeights) reads back as a plain `list`, a single value as a bare `float`; only DS/IS/CS multi-values are
  `MultiValue`. `_missing`/`_nums` therefore test `isinstance(value, (MultiValue, list))` (`MULTI`). plan_table never meets this
  (all its multi-values are DS).
- Decisions: the None pass (`_treatment_beams`) covers all treatment beams before `_need` runs; a missing `PrimaryDosimeterUnit`
  passes it and is refused in `_beam_rows`. Cumulative-weight and per-CP checks are validated on every CP (also those with no
  spots), so a non-finite or missing required scalar on an odd CP is refused too. `RangeShifterID` is treated as required
  (`_need`) because `range_shifter_ids` is a non-null list<string>; it is DICOM Type 3, so a real plan that omits it would be
  refused (a deferred question, not a plan gap). `final > 0` is checked (covers "final = 0").
- Fixture: `pbs_beam(number, meterset, layers, spot_weights, paintings, range_shifters, tags)`: 2 CPs per layer, odd CP repeats the
  map with zero weights; default layer 0 is `[w/2, 0, w/2]` (a zero spot in a carrying CP). `write_pbs_rtplan` default beams are
  beam 1 at 100 MU and beam 2 at 300 MU.
- Mutation check (check removed -> the matching test fails): total-weight, negative-weight, map length / NumberOfScanSpotPositions,
  strict decrease. The CP0 / last-CP cumulative checks and the per-CP sum check overlap (each mutator trips several), so those
  tests assert refusal, not which check fired.
- AC7 (slow, 50 000 spots, 2 beams x 5 layers x 5000): `ion_plan_rows` 0.10 s + `to_table` 0.35 s = 0.45 s (pytest wall 0.92 s);
  limit 5 s.
- Full suite: 1801 passed, 4 skipped, 3 deselected (245 s); ruff clean.
