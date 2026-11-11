# convert: photon RTPLAN as plan.parquet

Status: REVIEWED (stress-test and advisor applied, 2026-11-11)

## Problem
convert reads the RTPLAN only for its UID, label, date, fraction count and a few `dicom` metadata keys. Beams, control
points, MU, gantry/collimator/couch angles, jaws, MLC and isocenter are dropped, so nothing in the converted tree describes
how the dose was delivered. Dose-prediction, plan-parameter and fluence models need that.

## Decisions (owner, 2026-11-11)
- First PR: **photon EBRT only**, **tabular only**. Techniques in the cohort: VMAT, static/step-and-shoot IMRT, 3D conformal.
- Output: **`plan.parquet`** in the plan folder, next to `dose.nii.gz`, written by **convert** (no new CLI step).
  **pyarrow** is added as a core dependency (approved). No pandas.
- **One row per control point**, beam-level fields repeated on each row.
- **Filled forward**: a control-point value DICOM omits (inherited from an earlier control point of the same beam) is
  written as the inherited value, so every row is complete.
- Extra fields: MLC leaf boundaries, wedges/blocks/bolus, dose rate and SSD, isocenter as CT voxel index, `fluence_mode`.
- Non-photon plans (brachy, RT Ion/proton), mixed photon/non-photon plans, and photon plans this code cannot parse get
  **no parquet**; the plan still converts (CT, dose, masks, series.json) exactly as today.
- No new run-summary counter: `plan.parquet` in `convert_report.csv`'s `files` column is the count.
- `isocenter_i/j/k` stay on the native planning CT grid; documented, recomputation deferred.

## Which plans get a parquet
1. The plan has a non-empty `BeamSequence` (read with `plan.get`, never `plan.BeamSequence`). Brachy plans share the RT
   Plan Storage SOP class but have no `BeamSequence`; RT Ion plans have `IonBeamSequence` instead. Either → no parquet,
   no log line (not malformed).
2. Treatment beams = beams whose `TreatmentDeliveryType` is TREATMENT or absent. SETUP beams are dropped **before** any
   other check (they often have no or zero `BeamMeterset`).
3. Every treatment beam has `RadiationType` PHOTON, else no parquet, no log line. No treatment beam → no parquet.

## Columns (one row per control point)
The schema is declared once with `pa.schema(...)`; types never come from inference (an all-null column would otherwise
become type `null` and break cross-plan concatenation). Lists are `list<float64>`. `(n)` marks a nullable column; every
other column is required (see Malformed).

Beam-level (repeated on each row):
| column | type | source |
|---|---|---|
| beam_number | int64 | `BeamNumber` |
| beam_name | string (n) | `BeamName` |
| beam_type | string | `BeamType` (STATIC/DYNAMIC) |
| treatment_machine | string (n) | `TreatmentMachineName` |
| beam_meterset | float64 | `FractionGroupSequence[0].ReferencedBeamSequence[*].BeamMeterset`, matched on `ReferencedBeamNumber` (MU per fraction) |
| final_cumulative_meterset_weight | float64 | beam-level `FinalCumulativeMetersetWeight` |
| number_of_control_points | int64 | `NumberOfControlPoints` |
| fluence_mode | string (n) | `PrimaryFluenceModeSequence[0]`: `FluenceModeID` when present (e.g. FFF), else `FluenceMode` (STANDARD/NON_STANDARD) |
| mlc_type | string (n) | the one `BeamLimitingDeviceSequence` item of type MLCX or MLCY |
| leaf_position_boundaries | list<float64> (n) | that item's `LeafPositionBoundaries` (mm) |
| number_of_wedges | int64 | `NumberOfWedges` (absent → 0) |
| wedge_id, wedge_type, wedge_orientation | string (n) | first `WedgeSequence` item |
| wedge_angle | float64 (n) | first `WedgeSequence` item |
| number_of_blocks, number_of_boli | int64 | `NumberOfBlocks`, `NumberOfBoli` (absent → 0) |

Control-point (filled forward):
| column | type | source |
|---|---|---|
| control_point_index | int64 | `ControlPointIndex` |
| cumulative_meterset_weight | float64 | `CumulativeMetersetWeight` — **never filled forward** (Type 2, on every CP) |
| mu | float64 | `beam_meterset × (w_i − w_{i−1}) / final_weight`; 0 on CP 0. The meterset delivered between CP i−1 and CP i, attributed to row i |
| mu_cumulative | float64 | `beam_meterset × w_i / final_weight` |
| nominal_energy | float64 | `NominalBeamEnergy` (MeV per the standard; 6 for a 6 MV beam) |
| dose_rate | float64 (n) | `DoseRateSet` (MU/min) |
| gantry_angle | float64 | `GantryAngle` |
| gantry_rotation_direction | string (n) | `GantryRotationDirection` |
| collimator_angle | float64 | `BeamLimitingDeviceAngle` |
| couch_angle | float64 | `PatientSupportAngle` |
| jaw_x1, jaw_x2, jaw_y1, jaw_y2 | float64 (n) | `BeamLimitingDevicePositionSequence` item of type ASYMX/X (x) and ASYMY/Y (y); null when the beam declares no such device (e.g. Elekta Agility has no X jaws) |
| mlc_positions | list<float64> (n) | the MLC item's `LeafJawPositions` as stored (IEC X1-side leaves 1xx, then X2-side leaves 2xx; DICOM fixes no vendor bank letter to a side); null when the beam has no MLC |
| ssd | float64 (n) | `SourceToSurfaceDistance` (mm) |
| wedge_position | string (n) | `WedgePositionSequence[*].WedgePosition` for the first wedge's `ReferencedWedgeNumber` |
| isocenter_x, isocenter_y, isocenter_z | float64 | `IsocenterPosition` (patient mm) |
| isocenter_i, isocenter_j, isocenter_k | float64 | continuous index on the grid of `planning_ct.nii.gz` as written (`TransformPhysicalPointToContinuousIndex`) |

`beam_meterset` is per fraction; `dose.nii.gz` may be the plan total (`DoseSummationType` in series.json says which).

### Filling forward
Per beam, a running state starts empty at CP 0. Each CP updates only the values it carries:
- scalars (energy, dose rate, gantry, direction, collimator, couch, SSD, isocenter) by tag;
- beam-limiting device positions **by `RTBeamLimitingDeviceType`**: CP 0 lists every device; later CPs list only devices
  that moved (Eclipse VMAT carries only the MLC from CP 1 on). A later CP's position sequence never replaces the state
  of a device it does not list;
- wedge position by `ReferencedWedgeNumber`.
A present-but-empty value (`''` from our header reader) counts as missing, not as an inheritable value.

## Malformed photon plans
After filling forward, any of these → no parquet, one WARNING log line `convert: <REASON_PLAN_TABLE>` (a fixed constant; no
UID, label, name or value), plan converts otherwise:
- a required value missing or empty; a non-finite number (vault: gdcm-reads-non-finite-dicom-geometry-and-rescale-as-defaults);
- a treatment beam with no matching `BeamMeterset`;
- a declared jaw/MLC device missing at CP 0; `mlc_positions` length ≠ 2 × the MLC item's `NumberOfLeafJawPairs`;
  `leaf_position_boundaries`, when present, of length ≠ pairs + 1;
- cumulative weights that strictly decrease (equal consecutive weights are normal: step-and-shoot beam-off pairs);
  a final weight of 0; CP 0's weight ≠ 0 or the last CP's weight ≠ `final_cumulative_meterset_weight` (both within
  1e-6 relative), so `mu` always sums to `beam_meterset`;
- more than one MLC device on a beam (dual-layer MLC, deferred).

Mechanics: the table builder runs through the existing `_header_call` (which turns a pydicom-origin missing tag, and any
`IndexError`/`TypeError`/`ValueError` — including `float('')`/`float('abc')` on a malformed DS — into
`CaseRefused(REASON_MISSING_TAG)` and lets our other bugs propagate; vault: tell-a-pydicom-missing-tag-from-our-own-bug).
The builder raises `CaseRefused(REASON_PLAN_TABLE)` for the checks above. convert catches `CaseRefused` from this one call
only, logs `REASON_PLAN_TABLE`, and continues without the parquet. A tag-level skip therefore logs two lines
(`REASON_MISSING_TAG` with a crash site from `_header_call`, then `REASON_PLAN_TABLE`); tests assert presence, not a
count.

Quieting: the builder needs neither `quiet_call` (the error classifier; wrapping our own logic in it breaks its
contract) nor `_pydicom_quiet` (the logger/warning silencer): `read_header_core` already converted every element,
nested sequence items included (`ds.iterall()`, `dicom_io.py:77`), inside the quiet block, and the plan is not re-read
(vault: pydicom-validates-lazily-on-first-tag-access).

`_header_call` wraps row building only. The rows become a `pa.Table` via `pa.Table.from_pylist(rows, schema=SCHEMA)` after
it returns (`ArrowInvalid` is a `ValueError`, which `_header_call` would mislabel as a missing tag), still before the plan
folder exists; `_atomic`'s callable only calls `pq.write_table`. An empty required value is tested (`in ('', None)`)
before any conversion and raises `CaseRefused(REASON_PLAN_TABLE)`.

## Integration
- New module `src/rt_dataprep/plan_table.py`: build the rows (or return None for a plan that gets no parquet), and write
  them with pyarrow.
- convert builds the table after the masks, before anything is written (it can refuse nothing), writes `plan.parquet` with
  the existing `_atomic`, after the masks and before series.json, and appends it to the plan's `files`.
- series.json gains one row only when the parquet was written: exactly `SERIES_COLUMNS` (plus `dicom`), `kind` `plan`,
  `modality` `RTPLAN`, `file` `plan.parquet`, `roi_name`/`dose_summation_type`/`reg_matrix` `''`,
  `frame_of_reference_uid` the plan's `FrameOfReferenceUID` or `''`, the plan columns like the other rows, `dicom` from
  `(metadata.ALL, metadata.RTPLAN)`. `_series_rows` gets an explicit branch for it (it only builds rows for known files).
- preprocess: `plan` joins `KINDS` and the `('applicator', 'line')` tuples on the output-name line and the copy branch
  (`preprocess.py:747`, `:751`), so `plan.parquet` keeps its name and is copied byte-identical on every run. Update the
  module docstring's kind list. manifest and split need no change (manifest reuses `check_rows`).
- `project-profile.yml` compliance rule, `.gitignore` and README output tree list `plan.parquet` (it carries beam and machine names); README documents its columns.
  README states that `isocenter_i/j/k` describe `planning_ct.nii.gz`, not a preprocessed grid.
- Rollout: existing converted trees gain parquets only on a run with `--overwrite`.

## Acceptance tests
Synthetic DICOM built through save_as/dcmread (vault: pydicom-empty-values-read-back-as-none-or-stripped). A new photon
RTPLAN builder in `tests/synthetic_dicom.py` (beams, control points, beam-limiting devices, wedges, fraction group,
fluence mode) plus an RT Ion variant; the existing brachy builder is unchanged.
1. VMAT beam whose CP 1+ carry only gantry, weight and MLC: one row per CP; jaws, energy, couch, collimator, dose rate
   filled forward from CP 0; `mu` sums to `beam_meterset`; `mlc_positions` length = 2 × leaf pairs; `fluence_mode` FFF.
2. Step-and-shoot beam with equal consecutive weights: rows written, those rows' `mu` is 0.
3. 3D plan with a wedge and a SETUP beam without `BeamMeterset`: SETUP beam has no rows; wedge columns and
   `wedge_position` filled.
4. `isocenter_i/j/k` equal the CT's continuous index of the isocenter.
5. Brachy plan, RT Ion plan, photon+electron plan: no parquet, no `plan` row in series.json, the plan converts.
6. Malformed photon plans (missing `BeamMeterset` on a treatment beam, non-finite gantry, short `mlc_positions`, empty
   `CumulativeMetersetWeight`): no parquet, plan converts, the log holds `REASON_PLAN_TABLE` and no UID.
7. The parquet schema equals the declared schema on a VMAT plan with no wedge (null wedge columns keep their types).
8. series.json lists `plan.parquet` with kind `plan`; preprocess (geometric and non-geometric runs) copies it
   byte-identical under the same name and does not refuse the kind; manifest still builds.

## Deferred (filed as follow-up / `hardening` issues, not missing work)
- Fluence maps per beam; CT-grid beam volumes (dose-prediction inputs).
- Proton spots/energy layers (`IonBeamSequence`); `metadata.RTPLAN` fallback to `IonBeamSequence` for radiation_type,
  beam_type, treatment_machine.
- Brachy dwell positions and times.
- Electron beams, SRS cones, dual-layer MLC (MLCX1/MLCX2); more than one wedge per beam.
- Recomputing `isocenter_i/j/k` onto the preprocessed grid; a `plan` flag in manifest.jsonl; a run-summary count.
