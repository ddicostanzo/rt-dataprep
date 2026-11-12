# convert: proton PBS RTPLAN as ion_plan.parquet (#281)

Status: REVIEWED (interview, stress-test and advisor applied, 2026-11-11)

## Problem
RT Ion plans get no table: `plan_table.plan_rows` reads only `BeamSequence`, so a proton plan's spots, energy layers and
MU are dropped. `metadata.RTPLAN` also reads `radiation_type`, `beam_type` and `treatment_machine` from `BeamSequence`
only, so they are blank in series.json for proton plans.

## Decisions (owner, 2026-11-11)
- **PBS only**: treatment beams with `ScanMode` MODULATED. MODULATED_SPEC (may be line scanning), passive scattering,
  uniform scanning and non-proton ions get no table (the plan converts as today).
- **One row per spot** (non-zero weight), beam- and layer-level fields repeated on each row.
- **New file `ion_plan.parquet`** in the plan folder, new series.json kind `ion_plan`. `plan.parquet`, `beam_mu.nii.gz`
  and `fluence_mu.nii.gz` keep their photon-only contracts and are never written for an ion plan.
- **MU only**: a treatment beam with `PrimaryDosimeterUnit` NP → no table, no log line.
- **Zero-weight spots dropped; weight sums checked** (see Malformed). `spot_mu` follows the DICOM definition
  (`BeamMeterset × w / FinalCumulativeMetersetWeight`); it sums to `BeamMeterset` within `ION_WEIGHT_TOLERANCE`
  (relative 1e-4), not exactly: real files store spot weights as float32 and cumulative weights as decimal strings.
- **Mixed plans get no table**: SETUP beams drop first; then any treatment beam that is not a PBS proton MU beam, or a
  CONTINUATION/port-film beam, → no table, no log line (not malformed; a partial table would omit dose in dose.nii.gz).
- Extras: geometry + isocenter (re-indexed by preprocess like plan.parquet), range shifters as lists in beam order,
  spot tuning, beam metadata. Lateral spreading devices and range modulators are deferred (rare in PBS; owner).
- **Premises (confirmed)**: (1) spot weights sit on control points, and per DICOM the weights of CP i sum to
  `w_{i+1} − w_i` (Eclipse and RayStation: even CP carries the layer, odd CP all zeros; the last CP all zeros); (2)
  `ScanSpotPositionMap` holds x/y pairs in mm at the isocenter plane (IEC gantry frame), stored as given; (3) the metadata
  fallback only applies when `BeamSequence` yields nothing.
- `/dev` auto-merges after one clean review round (owner).

## Which plans get a table
`ion_plan_rows(plan, ct)` returns None (no table, no log line) unless all hold, checked in this order:
1. `plan.get('IonBeamSequence')` is non-empty and `plan.get('BeamSequence')` is empty/absent.
2. Treatment beams = items whose `TreatmentDeliveryType` is TREATMENT or absent/empty; SETUP items drop first. Any other
   delivery type on a non-SETUP item → None. No treatment beam → None.
3. Every treatment beam, checked in this order: `RadiationType` PROTON, then `ScanMode` MODULATED, then
   `PrimaryDosimeterUnit` MU. A missing/other `RadiationType` or `ScanMode` → None. On a PROTON + MODULATED beam a
   missing/empty `PrimaryDosimeterUnit` is malformed (Type 1), NP → None. So the current no-argument
   `write_ion_rtplan` fixture (no `ScanMode`) still yields no ion table and no log line
   (tests/test_convert.py:1927, tests/test_plan_table.py:160 stay valid).
`plan_table.plan_rows` is unchanged: it already returns None for a plan with no `BeamSequence`.

## Columns (one row per spot with weight > 0)
Declared once with `pa.schema(...)` (types never inferred). `(n)` = nullable; list columns are non-null (empty list when
the beam declares no such device) unless marked.

Beam-level:
| column | type | source |
|---|---|---|
| beam_number | int64 | `BeamNumber` |
| beam_name | string (n) | `BeamName` |
| beam_type | string | `BeamType` |
| radiation_type | string | `RadiationType` |
| treatment_machine | string (n) | `TreatmentMachineName` |
| scan_mode | string | `ScanMode` |
| beam_meterset | float64 | `FractionGroupSequence[0].ReferencedBeamSequence[*].BeamMeterset` matched on `ReferencedBeamNumber` (MU per fraction) |
| final_cumulative_meterset_weight | float64 | `FinalCumulativeMetersetWeight` |
| number_of_control_points | int64 | `NumberOfControlPoints` |
| virtual_sad_x, virtual_sad_y | float64 (n) | `VirtualSourceAxisDistances` (2 values) |
| snout_id | string (n) | `SnoutSequence[0].SnoutID` |
| range_shifter_ids | list<string> | `RangeShifterSequence[*].RangeShifterID`, in sequence order; [] when none |

Layer-level (the spot's control point; filled forward per beam):
| column | type | source |
|---|---|---|
| control_point_index | int64 | `ControlPointIndex` |
| layer_index | int64 | 0-based index of this CP among the beam's CPs that carry ≥ 1 non-zero weight |
| cumulative_meterset_weight | float64 | `CumulativeMetersetWeight` (never filled forward) |
| nominal_energy | float64 | `NominalBeamEnergy` (MeV) |
| gantry_angle | float64 | `GantryAngle` |
| couch_angle | float64 | `PatientSupportAngle` |
| snout_position | float64 (n) | `SnoutPosition` (mm) |
| scan_spot_tune_id | string (n) | `ScanSpotTuneID` |
| spot_size_x, spot_size_y | float64 (n) | `ScanningSpotSize` (2 values, mm) |
| number_of_paintings | int64 (n) | `NumberOfPaintings` |
| range_shifter_settings | list<string> | `RangeShifterSettingsSequence[*].RangeShifterSetting`, joined on `ReferencedRangeShifterNumber` = the beam item's `RangeShifterNumber`, in `range_shifter_ids` order; [] when none |
| range_shifter_wet | list<float64> (n) | `RangeShifterWaterEquivalentThickness`, same join and order; [] when no shifter; null when any declared shifter's value is absent |
| isocenter_x, isocenter_y, isocenter_z | float64 | `IsocenterPosition` (patient mm) |
| isocenter_i, isocenter_j, isocenter_k | float64 | continuous index on `planning_ct.nii.gz` (as plan.parquet) |

Spot-level:
| column | type | source |
|---|---|---|
| spot_index | int64 | position of the spot in its CP's `ScanSpotMetersetWeights` (0-based, zero-weight spots counted) |
| spot_x, spot_y | float64 | `ScanSpotPositionMap[2k]`, `[2k+1]` (mm, isocenter plane) |
| spot_weight | float64 | `ScanSpotMetersetWeights[k]` |
| spot_mu | float64 | `beam_meterset × spot_weight / final_cumulative_meterset_weight`; the total over all paintings — never multiplied or divided by `number_of_paintings` |

`spot_index` is map order, not necessarily delivery order (`ScanSpotReorderingAllowed`); the README says so.

### Filling forward
As plan.parquet: a per-beam running state starts empty at CP 0 and each CP updates only what it carries — scalars by
tag (energy, gantry, couch, snout position, tune id, spot size, paintings, isocenter), range shifter settings by
referenced range shifter number. A settings item that gives a new `RangeShifterSetting` replaces that shifter's whole entry: an
earlier water-equivalent thickness never carries over to a new setting (wave review, 2026-11-11). `ScanSpotPositionMap` / `ScanSpotMetersetWeights` are **never** filled forward: a CP without weights
carries no spots. A present-but-empty value counts as missing.

## Malformed PBS plans → `CaseRefused(REASON_ION_PLAN)`
`REASON_ION_PLAN = 'the proton RTPLAN has a missing or inconsistent value; no ion plan table written'`. convert logs
`convert: <REASON_ION_PLAN>` (one WARNING, no UID/label/value) and the plan converts without the table. Checks (after
filling forward):
- a required value missing/empty or non-finite (incl. `PrimaryDosimeterUnit`);
- duplicate `BeamNumber` among treatment beams; duplicate `ReferencedBeamNumber` in the first fraction group; a
  treatment beam with no matching `BeamMeterset`;
- cumulative weights: CP 0 ≠ 0, last CP ≠ final, any strict decrease, final = 0;
- per CP i < last: Σ spot weights of CP i ≠ `w_{i+1} − w_i`; last CP: Σ ≠ 0 (absent weights count as Σ = 0);
- the beam's total spot weight: |Σ all spot weights − final| > tolerance × final (makes the documented MU-sum
  property hold; per-CP checks alone could drift by N_cp × tolerance);
- a negative spot weight; `len(ScanSpotPositionMap) ≠ 2 × len(ScanSpotMetersetWeights)`; `NumberOfScanSpotPositions`,
  when present, ≠ `len(ScanSpotMetersetWeights)`;
- a declared range shifter with no setting at CP 0.
All weight comparisons use `ION_WEIGHT_TOLERANCE = 1e-4` relative to the beam's final weight (owner).

Mechanics are plan.parquet's: rows built through `_header_call` (a pydicom-origin missing tag or Index/Type/ValueError
→ `REASON_MISSING_TAG` + our reason), our checks raise only `CaseRefused(REASON_ION_PLAN)`, `.get` reads, no quieting,
`to_table` outside `_header_call`, all before the plan folder exists. AST test: the module raises only
`CaseRefused(REASON_ION_PLAN)`, has no try/except.

## Integration
- New module `src/rt_dataprep/ion_plan_table.py`: `ION_SCHEMA`, `REASON_ION_PLAN`, `ion_plan_rows(plan, ct)`, `to_table`,
  `write_ion_plan_table`. Rows are dicts keyed by `ION_SCHEMA` names, as in `plan_table`. The None pass runs over all
  treatment beams before any row is built; a missing `PrimaryDosimeterUnit` passes it and is refused while building.
- convert: `ION_PLAN_NAME = 'ion_plan.parquet'`; a fourth catch after the plan table, independent of it; written after
  `plan.parquet`; series.json row kind `ion_plan`, modality RTPLAN, frame = the plan's `FrameOfReferenceUID`, `dicom`
  from `(metadata.ALL, metadata.RTPLAN)`.
- preprocess: `ion_plan` joins `KINDS`, the keep-the-name tuple, and the geometric re-index branch; `index_plan_table`
  takes the schema to check (plan → `plan_table.SCHEMA`, ion_plan → `ion_plan_table.ION_SCHEMA`); a schema mismatch is
  the existing skip, and the column rewrite uses the passed schema's fields (today `plan_table.SCHEMA.field`,
  preprocess.py:751). `REASON_PLAN_INDEX` is reworded to name a plan table, not `plan.parquet` only. Non-geometric runs
  copy byte-identical.
- manifest: `_files` tuple gains `ion_plan` (optional key).
- metadata: `RTPLAN` `radiation_type`, `beam_type` gain `IonBeamSequence[*].…` as second path; `treatment_machine`
  gains `IonBeamSequence[*].TreatmentMachineName` between its two current paths.
- `.gitignore`, `project-profile.yml` phi compliance rule and README list `ion_plan.parquet` (it carries machine and beam
  names, the isocenter); README documents its columns, that `spot_x/y` are at the isocenter plane (IEC gantry frame),
  that `spot_index` is not delivery order, that Σ `spot_mu` matches `BeamMeterset` within 1e-4 relative, and that the
  new `dicom` metadata values and the table reach existing trees only with `convert --overwrite`.

## Acceptance tests
Synthetic RT Ion plans built through `save_as`/`dcmread` (extend `tests/synthetic_dicom.write_ion_rtplan`).
1. **Spot MU** (worry): Eclipse-style 2-beam plan, 3 layers each (even CP carries weights, odd CP zeros, last CP zeros):
   one row per non-zero spot; with binary-exact weights (0.5, 0.25, …) Σ `spot_mu` per beam equals `beam_meterset`
   (rel 1e-9), and a float32-noise variant passes within 1e-4; `spot_mu` = hand-computed value
   for one spot; `layer_index` 0..2; zero-weight spots in a carrying CP are absent while `spot_index` keeps their gap.
2. **Malformed** (worry): layer sum ≠ Δw, negative weight, odd map length, missing `BeamMeterset`, duplicate
   BeamNumber, non-finite energy, CP 0 weight ≠ 0, beam total Σw off by > 1e-4 while each CP passes, missing `PrimaryDosimeterUnit`, range shifter with no CP-0 setting:
   no table, plan converts, log holds `REASON_ION_PLAN` and no UID.
3. **Vendor variation** (worry): a RayStation-style beam (single-spot layer whose weights read back as a float, not a
   MultiValue; `NumberOfPaintings` set; odd CP repeats the spot map with zero weights) yields the same rows as its
   Eclipse-style twin; `number_of_paintings` 2 does not change `spot_mu`; range shifter IN on CP 0, absent from later
   CPs → filled forward; two shifters keep beam order.
4. **None cases**: passive (`ScanMode` NONE/UNIFORM), MODULATED_SPEC, no `ScanMode`, NP unit, carbon
   (`RadiationType` ION), CONTINUATION beam, a plan with both beam sequences, a brachy plan, a photon plan: no ion table, no log line, plan converts. SETUP beam with no meterset is ignored.
5. **Photon path unchanged** (worry): the existing photon tests in `tests/test_convert.py`, `tests/test_plan_table.py`,
   `tests/test_beam_volume.py` and `tests/test_fluence.py` pass unmodified; a PBS plan's `files` list contains none of
   `plan.parquet`, `beam_mu.nii.gz`, `fluence_mu.nii.gz`.
6. **Size/speed** (worry): a 2-beam plan with 50 000 spots builds rows and table in < 5 s (marked `slow`; the
   isocenter index is computed once per control point, not per spot); schema equals `ION_SCHEMA`
   on a plan with no range shifter (lists empty, nullable columns keep their types).
7. **Pipeline**: series.json lists `ion_plan.parquet` kind `ion_plan`; preprocess geometric run rewrites
   `isocenter_i/j/k` onto the final grid, non-geometric run copies byte-identical; manifest has `ion_plan`.
8. **Metadata**: a PBS plan's series.json `dicom` has `radiation_type` PROTON, `beam_type`, `treatment_machine`; a
   photon plan's values are unchanged; a non-plan row (ct) of the PBS plan carries the same RTPLAN keys.

## Deferred
- MODULATED_SPEC (line scanning); passive scattering / uniform scanning / carbon and other ions; NP metersets.
- Lateral spreading devices and range modulators (IDs and settings).
- Projection of spots into patient space; per-spot dose or fluence; beam volumes for protons.
- Gantry pitch, table-top angles, meterset rate, compensators and blocks on ion beams.
- A plan with both `BeamSequence` and `IonBeamSequence`.
