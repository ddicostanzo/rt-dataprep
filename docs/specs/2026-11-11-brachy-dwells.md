# convert: brachy HDR/PDR dwells as dwell.parquet (#282)

Status: REVIEWED (interview and two stress-test rounds applied, 2026-11-11) · Branch: `feat/brachy-dwell`

## Background
Brachy RTPLAN dwell positions and times are dropped today: `plan_table.plan_rows` and `ion_plan_table.ion_plan_rows`
return None for a plan with no `BeamSequence` / `IonBeamSequence`. Applicator geometry already reaches the output as
point text from the RTSTRUCT (`masks.classify`). Issue #282 came out of PR #271. rnt_transformer has its own RTPLAN →
dwells parser (`src/data/plan_dwells.py`, relative weights, harmonized to 12 tandem + 18 ring); this spec does not
replace it or check parity with it (owner: "completeness only").

## Problem
A converted brachy plan loses which positions the source dwelt at and for how long, which is the plan itself. The
cohort is HDR brachy, so this is the main plan content of the dataset.

## Decisions (owner, 2026-11-11)
- **Purpose: completeness.** No named consumer; store what the plan holds, losslessly. No volume (rejected: loses
  sub-voxel position and channel identity), no parity check against plan_dwells.py.
- **Point table `dwell.parquet`** in the plan folder, series.json kind `dwell`, optional manifest key `dwell`; follows
  the ion_plan pattern (PR #296).
- **HDR and PDR only**: `BrachyTreatmentType` HDR or PDR. LDR, MANUAL, absent → no table, no log line.
- **Paired control points**: a dwell is CPs 2k and 2k+1 at one position. Any other layout (odd CP count, weight rising
  between different positions) → malformed (table lost, one log line).
- **Zero-time dwells kept** (inactive positions show the activatable path).
- **Time**: `dwell_time_s = ChannelTotalTime × dwell_weight / FinalCumulativeTimeWeight` (DICOM definition) plus the raw
  `dwell_weight`. Times are as exported in `ChannelTotalTime`; which date's source strength they assume is not established (PS3.3
  ties RAKR to its reference date, not the time); it stays open until a TG-43 point dose is compared against RTDOSE on
  a real export. PDR: stored as-is
  with the pulse columns; reading `ChannelTotalTime` as one pulse (per-fraction = × `number_of_pulses`) is a vendor
  convention, not PS3.3, and is unverified on a real PDR export (owner: store as-is, README says so).
- **Extras**: setup, channel, source (isotope, RAKR, half-life, strength reference date/time) and pulse columns,
  repeated on each row.
- **Tolerance** `DWELL_WEIGHT_TOLERANCE = 1e-4`, relative to the channel's final weight (as ion_plan).
- **Malformed → only dwell.parquet lost**: a fifth catch in convert; CT, dose, masks and applicator files unchanged.
- **Premises (confirmed)**: (1) `ControlPoint3DPosition` is in the patient frame of the CT/RTSTRUCT, as the applicator
  files; i/j/k come from the CT grid without registration. (2) Times are as planned (per fraction, per pulse for PDR);
  no RTRECORD. (3) Additive: no existing output changes; photon and ion plans never get dwell.parquet.
- `/dev` auto-merges after one clean review round (owner).

## Which plans get a table
`dwell_rows(plan, ct)` returns None (no table, no log line) unless all hold, in this order:
1. `plan.get('ApplicationSetupSequence')` is non-empty, and `BeamSequence` and `IonBeamSequence` are absent/empty.
2. `plan.get('BrachyTreatmentType')` is HDR or PDR (LDR, MDR, MANUAL, absent → None).
3. At least one channel (over all setups) has a non-empty `BrachyControlPointSequence`.
4. Every channel with control points has `SourceMovementType` STEPWISE or FIXED; OSCILLATING or UNIDIRECTIONAL → None
   (moving sources are deferred; owner). Any other value, absent or empty, passes this step and is malformed (Type 1).
The None pass covers every channel before any row is built.
The existing `write_rtplan` fixture writes no `BrachyTreatmentType`, so every current brachy test keeps getting no
table and no log line.

## Columns (one row per dwell, including zero-time dwells)
Declared once with `pa.schema(...)`; `(n)` = nullable. Rows ordered by setup (sequence order), channel (sequence
order), dwell index.

| column | type | source |
|---|---|---|
| brachy_treatment_type | string | `BrachyTreatmentType` |
| setup_number | int64 | `ApplicationSetupNumber` |
| setup_name | string (n) | `ApplicationSetupName` |
| setup_type | string (n) | `ApplicationSetupType` |
| channel_number | int64 | `ChannelNumber` |
| source_movement_type | string | `SourceMovementType` |
| source_applicator_id | string (n) | `SourceApplicatorID` |
| source_applicator_type | string (n) | `SourceApplicatorType` |
| source_applicator_step_size | float64 (n) | `SourceApplicatorStepSize` (mm) |
| source_applicator_length | float64 (n) | `SourceApplicatorLength` (mm) |
| transfer_tube_length | float64 (n) | `TransferTubeLength` (mm) |
| referenced_roi_number | int64 (n) | `ReferencedROINumber` |
| channel_total_time | float64 | `ChannelTotalTime` (s) |
| final_cumulative_time_weight | float64 | `FinalCumulativeTimeWeight` |
| number_of_control_points | int64 | `len(BrachyControlPointSequence)` |
| number_of_pulses | int64 (n) | `NumberOfPulses` (PDR) |
| pulse_repetition_interval | float64 (n) | `PulseRepetitionInterval` (s, PDR) |
| source_number | int64 | `ReferencedSourceNumber` |
| source_isotope_name | string (n) | `SourceSequence[SourceNumber = source_number].SourceIsotopeName` |
| source_isotope_half_life | float64 (n) | `…SourceIsotopeHalfLife` (days) |
| reference_air_kerma_rate | float64 (n) | `…ReferenceAirKermaRate` (µGy h⁻¹ at 1 m) |
| source_strength_reference_date | string (n) | `…SourceStrengthReferenceDate` (DICOM DA, as stored) |
| source_strength_reference_time | string (n) | `…SourceStrengthReferenceTime` (DICOM TM, as stored) |
| dwell_index | int64 | k, 0-based within the channel (zero-time dwells counted) |
| control_point_index | int64 | `ControlPointIndex` of CP 2k |
| relative_position | float64 (n) | `ControlPointRelativePosition` of CP 2k (mm) |
| position_x, position_y, position_z | float64 | `ControlPoint3DPosition` of CP 2k (patient mm) |
| position_i, position_j, position_k | float64 | continuous index on `planning_ct.nii.gz` (`TransformPhysicalPointToContinuousIndex`) |
| cumulative_time_weight | float64 | `CumulativeTimeWeight` of CP 2k |
| dwell_weight | float64 | `max(0, w_{2k+1} − w_{2k})` |
| dwell_time_s | float64 | `channel_total_time × dwell_weight / final_cumulative_time_weight`; 0 when the channel's final weight and total time are both 0 |

A present-but-empty value counts as missing. Nullable columns are null when absent; non-null columns absent → malformed.

## Malformed plans → `CaseRefused(REASON_DWELL)`
`REASON_DWELL = 'the brachy RTPLAN has a missing or inconsistent value; no dwell table written'`. convert logs
`convert: <REASON_DWELL>` (one WARNING, no UID/label/value) and the plan converts without the table. Checks, per channel
with a non-empty control point sequence (a channel with none contributes no rows):
- a required (non-null) value missing/empty or non-finite, including `SourceMovementType`, any CP's
  `ControlPoint3DPosition` (3 values) and `CumulativeTimeWeight`. The 3D position is Type 3 and the weight Type 2 in
  PS3.3; requiring them is our choice (owner): the position drives i/j/k and an empty weight leaves no time. A plan with
  only `ControlPointRelativePosition` is therefore malformed (deferred);
- duplicate `ChannelNumber` within a setup; duplicate `ApplicationSetupNumber`; `ReferencedSourceNumber` with no
  matching `SourceSequence` item, or a duplicate `SourceNumber`;
- `NumberOfControlPoints`, when present, ≠ the sequence length; an odd CP count;
- pairing: CPs 2k and 2k+1 more than `PAIR_ATOL_MM = 0.1` apart on any axis;
- weights (tolerance `DWELL_WEIGHT_TOLERANCE × final`): CP 0 ≠ 0; last CP ≠ final; any decrease between consecutive CPs;
  `w_{2k+2} ≠ w_{2k+1}` (time while moving);
- channel total: |Σ dwell_weight − final| > `DWELL_WEIGHT_TOLERANCE` × final;
- final weight 0 with `ChannelTotalTime` ≠ 0; negative `ChannelTotalTime` or final weight;
- `NumberOfPulses` or `PulseRepetitionInterval` absent on a PDR channel (both Type 1C for PDR);
- a setup without `ApplicationSetupNumber` or a `SourceSequence` item without `SourceNumber` (Type 1; wave review).
Mechanics are ion_plan's: rows built through `_header_call`, our checks raise only `CaseRefused(REASON_DWELL)`, `.get`
reads, no quieting, no try/except, `to_table` outside `_header_call`, all before the plan folder exists. AST test: the
module raises only `CaseRefused(REASON_DWELL)` and imports none of `plan_table`, `ion_plan_table`, `beam_volume`,
`fluence`. pydicom 3 reads multi-valued FL as `list` (vault:
`Patterns/medical-imaging/pydicom-binary-vr-multivalue-reads-back-as-list.md`).

## Integration
- New module `src/rt_dataprep/dwell_table.py`: `DWELL_SCHEMA`, `REASON_DWELL`, `DWELL_WEIGHT_TOLERANCE`,
  `PAIR_ATOL_MM`, `dwell_rows(plan, ct)`, `to_table`, `write_dwell_table`.
- convert: `DWELL_NAME = 'dwell.parquet'`; a fifth catch after the ion catch, independent of it; written after
  `ion_plan.parquet`; series.json row kind `dwell`, modality RTPLAN, frame = the plan's `FrameOfReferenceUID`, `dicom`
  from `(metadata.ALL, metadata.RTPLAN)`. `metadata.RTPLAN` already carries the brachy fields; unchanged.
- preprocess: `dwell` joins `KINDS`, the keep-the-name tuple and the geometric re-index branch. `index_plan_table` takes
  the point column prefix with the schema (`isocenter` for plan/ion_plan, `position` for dwell): `<prefix>_x/y/z` →
  `<prefix>_i/j/k`. The prefix argument defaults to `'isocenter'`, so existing callers and tests are unchanged.
  `REASON_PLAN_INDEX` names dwell.parquet too. Non-geometric runs copy byte-identical.
- manifest: `_files` gains `dwell` (optional key).
- `.gitignore`, `project-profile.yml` phi rule and README list `dwell.parquet` (it carries applicator names, positions
  and source calibration dates). README documents the columns, zero-time dwells, that the strength date behind the times is unverified,
  that the PDR per-pulse reading is unverified, that the series.json `frame` may be empty (brachy RTPLANs often omit
  `FrameOfReferenceUID`, as ion_plan), and that the table reaches existing trees only with `convert --overwrite`.

## Acceptance tests
Synthetic brachy plans through `save_as`/`dcmread`; `tests/synthetic_dicom.write_rtplan` gains `treatment_type`,
per-channel `total_time` / `final_weight` / `source_number` / `movement_type` / pulses, and a `SourceSequence`. DS values
are written as DS-legal strings (≤ 16 chars, e.g. `f'{w:.10g}'`): raw float sums such as 0.30000000000000004 fail
validation on `save_as`. The new `treatment_type` kwarg defaults to None, so every existing brachy fixture keeps
writing no `BrachyTreatmentType` (no table, no log). A synthetic `SourceStrengthReferenceDate` is a quoted DA literal
as `test_dicom_metadata_convert.py` writes its dates (e.g. `StructureSetDate`).
1. **Times**: HDR plan, 2 channels, 3 dwells each (one zero-time): 6 rows; `dwell_time_s` = hand value; Σ per channel
   = `ChannelTotalTime` (rel 1e-9 with binary-exact weights, within 1e-4 with DS-rounded weights); zero dwell present
   with time 0; positions and `dwell_index` correct.
2. **Indexing**: `position_i/j/k` = the CT's continuous index of `position_x/y/z`; preprocess on a geometric run
   rewrites them to the final CT; non-geometric run copies byte-identical.
3. **PDR**: pulse columns filled, `dwell_time_s` per pulse; PDR without `NumberOfPulses` → malformed.
4. **No table, no log**: LDR, MDR, MANUAL, absent type, OSCILLATING / UNIDIRECTIONAL channel, photon plan, ion plan,
   no channel in any setup with CPs → None and no WARNING.
5. **Malformed**, each alone → no dwell.parquet, one WARNING with the fixed reason, the case still converts with CT,
   dose, masks and applicators: odd CP count, pair > 0.1 mm apart, weight rising between dwells, decreasing weight,
   last ≠ final, missing `ChannelTotalTime`, unknown source number, non-finite position, missing 3D position, missing
   `SourceMovementType`, duplicate `ChannelNumber`, duplicate setup number, duplicate `SourceNumber`,
   `NumberOfControlPoints` mismatch, final 0 with non-zero time, negative `ChannelTotalTime`, negative final weight, CP 0
   weight ≠ 0, missing `CumulativeTimeWeight`, unknown `SourceMovementType`, PDR without `PulseRepetitionInterval` (one
   parametrized test). Channel-sum drift: a channel of many dwells whose inter-pair gaps are each under the tolerance
   but sum past it (the only way it fails alone).
6. **Integration**: convert writes dwell.parquet and a series.json `dwell` row with the shared key set; manifest gains
   `dwell`; the schema of the written file equals `DWELL_SCHEMA`; existing brachy fixtures produce no table and no new
   log line.
7. **Guardrails**: AST test as above; no PHI in the log line.

## Failure handling
Every malformed case is surfaced as one fixed WARNING and the table is dropped; nothing is swallowed silently except the
documented no-table cases (§ Which plans). An unreadable dwell.parquet in preprocess is the existing
`REASON_PLAN_INDEX` skip.

## Constraints
- Safety: research data pipeline, not clinical delivery; a wrong time would be silent, hence the per-channel sum checks.
- PHI: no new identifiers; the table carries applicator names and source calibration dates, listed in the phi rule.
- FHIR: not applicable.

## Delivery
Ships with the next release tag; reaches the cohort with `convert --overwrite` (open item in the proton handoff). The
strict checks (CP 0 = 0, last = final, 0.1 mm pairs, flat weight between pairs, 1e-4) have no real-data evidence yet:
the first cohort run counts `REASON_DWELL` warnings against brachy plans as its acceptance signal.

## Deferred
- LDR/seed implants, MANUAL, moving-source (non-paired) layouts.
- A dwell-weighted CT volume; harmonized tandem/ring dwells; parity with rnt_transformer plan_dwells.py.
- Decay-corrected source strength at treatment time.
- Non-gamma sources (`SourceStrength` / `SourceStrengthUnits` when RAKR is 0).
- Plans with only `ControlPointRelativePosition`; a `ControlPointIndex` order check.
- Whether channels without control points count in the duplicate-`ChannelNumber` check (plan review N3).
