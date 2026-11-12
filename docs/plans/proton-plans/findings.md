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
  (`_need`) because `range_shifter_ids` is a non-null list<string>; the orchestrator and both reviewers recall it as
  Type 1 in the Ion Beam Range Shifter Sequence (not checked against PS3.3 text), so requiring it is correct. `final > 0` is checked (covers "final = 0").
- Fixture: `pbs_beam(number, meterset, layers, spot_weights, paintings, range_shifters, tags)`: 2 CPs per layer, odd CP repeats the
  map with zero weights; default layer 0 is `[w/2, 0, w/2]` (a zero spot in a carrying CP). `write_pbs_rtplan` default beams are
  beam 1 at 100 MU and beam 2 at 300 MU.
- Mutation check (check removed -> the matching test fails): total-weight, negative-weight, map length / NumberOfScanSpotPositions,
  strict decrease. The CP0 / last-CP cumulative checks and the per-CP sum check overlap (each mutator trips several), so those
  tests assert refusal, not which check fired.
- AC7 (slow, 50 000 spots, 2 beams x 5 layers x 5000): `ion_plan_rows` 0.10 s + `to_table` 0.35 s = 0.45 s (pytest wall 0.92 s);
  limit 5 s.
- Full suite: 1801 passed, 4 skipped, 3 deselected (245 s); ruff clean.

## review-group-1
Opus wave review of task-01/02: no BLOCKING. Fixed in the PR (clinical-output test gaps, orchestrator):
- W1 spot_mu's `/ final` was untested (every fixture had final = 1): added a final = 37 test; removing `/ final` fails it.
- W2 per-CP sum check had no isolated test (the beam-total check masked it): added total-preserving mutators
  `_layer_shifted`, `_last_cp_moved`; removing the per-CP check fails both.
- W3 range_shifter_wet carried an old thickness onto a new setting: a new RangeShifterSetting now drops the stored
  thickness (spec Filling forward updated).
- N4 metadata BeamSequence-over-IonBeamSequence order now tested.
Deferred (hardening at 2.9): negative BeamMeterset accepted; duplicate RangeShifterNumber; settings item for an undeclared
shifter ignored; strict-decrease check has no tolerance.

## task-03
- `ion_plan` joins `KINDS`, the keep-the-name tuple, the geometric re-index branch and manifest `_files` (after `fluence`).
  `index_plan_table(source, path, final, schema=plan_table.SCHEMA)`: the schema argument is used for the compare and for the
  rewritten isocenter_i/j/k fields; preprocess passes `ion_plan_table.ION_SCHEMA` for `ion_plan`. It writes with `pq.write_table`
  (what `plan_table.write_plan_table` / `write_ion_plan_table` both do), so one function serves both schemas.
- `REASON_PLAN_INDEX` is now 'a plan table (plan.parquet or ion_plan.parquet) is unreadable or not its schema; left out of the output'.
- Timing: geometric `index_plan_table` on a 50 000-row ion_plan.parquet (full ION_SCHEMA, 40x40x40 float image, 1.5/1.5/2 mm):
  0.16 s (3 runs: 0.164, 0.166, 0.161), limit 2 s -> no hardening issue. Throwaway script, not committed.
- Test helper `add_ion_plan` (tests/test_preprocess.py) uses the literal 'ion_plan.parquet' (`ION_NAME`) because convert's
  `ION_PLAN_NAME` arrives in task-04; swap it then. It spreads the last series.json row so a converted tree's `dicom` key is kept.
- Pre-existing assertions changed: only tests/test_manifest.py `read()` key tuple (adds 'ion_plan'). No test pinned the old
  REASON_PLAN_INDEX text (they use `pre.REASON_PLAN_INDEX`).

## review-group-2
Opus wave review of task-03: no BLOCKING, no WARNING. Notes: the 'plan'/'ion_plan' entries of the copy tuple
(preprocess.py:775) are unreachable (kept for symmetry, as planned); test ION_NAME stays a literal (task-04's 5-file
commit leaves test_preprocess/test_manifest alone).

## task-04
- convert: ION_PLAN_NAME, a fourth catch (after fluence's, before case_dir.mkdir), written after plan.parquet; the ion_plan
  series.json row follows the plan row (the two never co-occur). The ct row's dicom also gains the PROTON metadata
  (plan_keys merge into every row).
- write_pbs_rtplan has no label= argument: tests set RTPlanLabel through tags=.
- The nan-energy test clears caplog after building the plan (pydicom logs the value at write time) and records warnings
  around conv.convert only.
- Tag-level fault test monkeypatches ion_plan_rows to read an absent keyword (file mutations are refused by our own checks
  first): REASON_MISSING_TAG then one REASON_ION_PLAN.
- tests/test_preprocess.py ION_NAME stays a literal (5-file limit).

## review-integration
Thin pass folded into the spec-axis review: ION_SCHEMA single definition; ion_plan kind flows convert → preprocess KINDS
and tuples → manifest _files; metadata fallback reaches every row; README and compliance match the code. No findings.

## review-loop-pass-1
/code-review (Opus, bugs) and spec axis (Opus): no BLOCKING. Trivial, fixed in 12dcb93: stale kind lists in the
convert/preprocess/manifest docstrings, test ION_NAME from conv.ION_PLAN_NAME, spot_mu unit in README. Not changed: a
redundant `| collect(RTPLAN)` in the new convert test mirrors the existing photon test. Deferred: #297 (negative
BeamMeterset, duplicate RangeShifterNumber, undeclared-shifter settings, decrease tolerance), #298 (re-listed unchanged
range shifter setting nulls range_shifter_wet), #299 (pin to_table outside _header_call). No blocking fix, so no
verification pass.
