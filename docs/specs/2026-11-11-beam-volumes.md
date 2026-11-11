# convert: photon beam volume (beam_mu.nii.gz) on the planning-CT grid (#280)

Date: 2026-11-11 · Branch: `feat/beam-volumes` · Status: REVIEWED (stress-test rounds 1–2 and advisor applied, 2026-11-11)

## Background
PR #271 made `convert` write `plan.parquet` (one row per control point of a photon EBRT plan). A dose-prediction model
needs the plan's beam geometry as a volume aligned with the CT, which the table alone does not give. Issue #280 asks for
CT-grid beam channels, the isocenter index on the preprocessed grid, and a plan flag in `manifest.jsonl`. No real photon
export has been through the pipeline yet; the owner chose to build this before that run (2026-11-11).

## Problem
A model consuming the converted or preprocessed tree has CT, dose and masks on one grid but no voxel-wise description of
where, and how much, radiation was aimed. The owner (physicist / ML developer building the dose-prediction dataset) wants
a channel computed once, deterministically, from the plan.

## Decisions (owner, 2026-11-11 interview and review round)
- **Content:** one float32 volume per plan, **`beam_mu.nii.gz`**: the MU-weighted sum, over every control point of every
  treatment beam, of the divergent aperture projected through the CT. **Pure geometry**: no attenuation, no inverse
  square, no dependence on CT values. The name has an underscore, so no mask or line (alphanumeric `format_name`) or
  applicator (`*_channel`) can produce it.
- **Aperture:** MLC leaf openings intersected with the jaw opening, per control point (`mlc_positions`,
  `leaf_position_boundaries`, `jaw_x1/x2/y1/y2` from the plan rows).
- **Weighting:** each segment's MU (`mu` of row i, the meterset between control points i-1 and i) is split half to
  control point i-1 and half to control point i, each projected at its own gantry/collimator/couch angle and aperture.
  A control point whose resulting weight is 0 is skipped (step-and-shoot transitions). Exact for static and
  step-and-shoot beams; VMAT is sampled at its control-point spacing.
- **Units:** MU per fraction. Every voxel inside a control point's aperture receives that control point's weight; a
  voxel's value is the sum of the weights of every control point whose aperture contains it. No normalization.
- **Inputs not in the plan rows:** `SourceAxisDistance` from each treatment beam of the RTPLAN header (`BeamSequence`,
  read with `.get`); `PatientPosition` from the first planning-CT slice header (`ct_slices[0][1]`). Voxel centres come
  from the CT's full geometry (origin + direction · spacing · index, i.e. what `TransformIndexToPhysicalPoint` gives),
  never origin + spacing · index.
- **Where:** computed in **convert** in a new module (`src/rt_dataprep/beam_volume.py`; not `plan_table`, whose AST test
  allows only `CaseRefused(REASON_PLAN_TABLE)`), on the native planning-CT grid, written next to `dose.nii.gz` after
  `plan.parquet`, with a `series.json` row of `kind` `beam` (modality `RTPLAN`, plan columns filled, `dicom` = the plan's
  metadata as on the `plan` row, `frame_of_reference_uid` = the CT's, since the volume is on the CT grid). Existing trees gain it only through `convert --overwrite`.
- **Patient positions:** CT `PatientPosition` HFS, HFP, FFS, FFP. Anything else (or absent) gets no beam volume.
- **No plan rows → no beam volume, no new log line.** `plan_rows` returning None (brachy, ion, CONTINUATION/port-film,
  non-photon) or refusing (already logged as REASON_PLAN_TABLE) means the beam step is not run.
- **Not modelled → no beam volume, one WARNING, plan still converts (with plan.parquet):** any beam with
  blocks (`number_of_blocks > 0`); an MLC whose type is not `MLCX`; an MLCX with null `leaf_position_boundaries`; a
  missing or non-positive `SourceAxisDistance` on any treatment beam; unsupported PatientPosition; a control point
  whose aperture is unbounded on an axis (no jaw pair and no MLC bounding it); a non-zero `TableTopPitchAngle`, `TableTopRollAngle` or
  `TableTopEccentricAngle` on any treatment control point (#287). Wedges are ignored (they modulate
  intensity, not the aperture).
- **Catch path:** the beam module reads headers with `.get` and raises only `CaseRefused(REASON_BEAM_VOLUME)` (an AST
  test like plan_table's enforces it). `_convert_plan` calls it **outside** `_header_call` (whose ValueError/TypeError
  path would turn a numpy error into REASON_MISSING_TAG and refuse the plan) and catches only that CaseRefused, logging
  `log.warning('convert: %s', REASON_BEAM_VOLUME)` — the existing style. Any other exception propagates as today. The
  "the one catch" comment in `_convert_plan` is updated.
- **Grid and cost:** full planning-CT grid, vectorized per control point in float32, chunked along z to bound memory.
  A `slow` pytest marker is registered in `pyproject.toml` and excluded by default (`addopts = "-m 'not slow'"`); the
  slow test runs with `-m slow`. Measured on a full-size synthetic VMAT case (criterion 14); soft budget 15 min per plan per worker: over it is flagged
  in findings and filed as a `hardening` issue, not a merge blocker.
- **No isocenter mask.** The isocenter stays in plan.parquet.
- **preprocess:** `KINDS` gains `beam`; `check_rows` checks a beam row with `same_grid` against the CT like dose and
  masks. `beam_mu.nii.gz` gets `output_name` like dose (geometric prefixes) and is linearly resampled onto the final CT
  as float32 (0 outside) via `process_dose`. It is **exempt** from the crop check (`check_crop` / `crosses_final`), the
  `cropped` count, `retained` (blank in its series.json row), the `has_voxels` → REASON_CROP check, and
  `min_mask_retention` (rays cross the whole CT, so a reshape crop always cuts it). A non-geometric run copies it like
  dose does today.
- **Invariant:** `isocenter_i/j/k` always index the CT in the same folder (converted or preprocessed).
- **preprocess isocenter:** under a geometric run, the output `plan.parquet` is rewritten with `isocenter_i/j/k`
  recomputed against the final CT (`TransformPhysicalPointToContinuousIndex` of `isocenter_x/y/z`); every other column,
  row order and SCHEMA are unchanged. A non-geometric run copies it byte-identical as today. An input parquet that cannot
  be read or whose schema is not SCHEMA is left out of the output (file and `plan` row) with one WARNING
  `preprocess: <REASON_PLAN_INDEX>`; the plan still preprocesses (test deferred as `hardening`).
- **manifest:** a plan line gains optional keys `"plan": "plan.parquet"` and `"beam": "<beam row's file>"` when those
  series.json rows exist, absent otherwise (like `retained`; existing lines are unchanged). File names, unlike `dose`'s
  "0"/"1", because the preprocessed beam name carries step prefixes. `retained` keeps counting only dose and mask rows.
- **Shipped text this changes:** README `### plan.parquet` ("manifest ignores it"; isocenter_i/j/k "not of a
  preprocessed grid"; "copies it byte-identical on every run"), a new README `### beam_mu.nii.gz` section, the manifest
  key list, and `test_plan_parquet_is_copied_byte_identical_and_listed` (byte-identical only on non-geometric runs).
  The earlier spec's Deferred item "CT-grid beam volumes" is delivered.
- **Deferrals filed as issues:** the real-cohort dose-overlap check and the bad-parquet preprocess test are filed as
  `hardening` GitHub issues when the plan is committed; their numbers go in this spec's Deferred section.
- **Delivery:** /dev runs to auto-merge after a clean review (owner override of the earlier no-merge instruction).

## Geometry (IEC 61217, DICOM PS3.3 C.8.8.25.6)
All rotations are right-handed about the named axis (counter-clockwise viewed from its positive end), angles in degrees.
For a voxel centre `p` (DICOM patient coordinates, mm) and the control point's isocenter `c` (`isocenter_x/y/z` of that
row, filled forward):
1. Patient-position map `P` to the IEC patient-support frame, applied to `d = p - c`: HFS `(x, z, -y)`,
   HFP `(-x, z, y)`, FFS `(-x, -z, -y)`, FFP `(x, -z, y)`.
2. Fixed frame: `f = Rz(couch) · P(d)`.
3. Gantry frame: `g = Ry(gantry)ᵀ · f`. Beam limiting device frame: `b = Rz(collimator)ᵀ · g`. Source at `(0, 0, SAD)`.
4. Voxels with `b_z ≥ SAD` get nothing. Otherwise project to the isocenter plane: `X = b_x·SAD/(SAD-b_z)`,
   `Y = b_y·SAD/(SAD-b_z)`.
5. Inside iff `jaw_x1 ≤ X ≤ jaw_x2`, `jaw_y1 ≤ Y ≤ jaw_y2` (a null jaw pair does not bound), and, with an MLCX, the
   leaf pair `n` whose boundaries hold `Y` (`boundary[n] ≤ Y < boundary[n+1]`, the last pair closed at
   `boundary[-1]`) has `X1-leaf[n] ≤ X ≤ X2-leaf[n]` (`mlc_positions[n]`, `mlc_positions[pairs+n]`); `Y` outside every
   leaf pair is outside. Closed leaf pairs (X1 ≥ X2) let nothing through.
Testable facts this implies (HFS): G0 source anterior; G90 source at patient left; G180 posterior; G270 patient right;
couch 90 + G90 source inferior; collimator 90 turns a field long in X into one long in the patient's superior-inferior axis
at G0. Each patient position flips the expected side per step 1.

## Acceptance criteria
1. A synthetic HFS static beam, G0/C0/couch 0, symmetric 100×100 mm jaw field, no MLC, isocenter at the CT centre:
   voxels at the isocenter plane with |x|,|z| < 45 mm hold the beam MU; voxels with |x| > 55 mm hold 0.
2. Divergence: in the same case, the lateral half-width of the non-zero region at depth d below the isocenter plane
   (b_z = −d) is 50·(SAD+d)/SAD mm ± one voxel, checked at two depths.
3. Gantry 90/180/270 (HFS): the beam enters from the patient's left / posterior / right, checked with an off-axis
   asymmetric jaw field whose open side identifies the direction.
4. Patient positions HFP, FFS, FFP each place the G90 field per the step-1 map (checked as in 3), on a CT whose
   direction matrix is not identity in at least one case (voxel centres via the full direction).
5. Rotation signs, each with a one-sided field so the sign is pinned: (a) HFS, couch 90, G90, symmetric field: the
   source is **inferior** (the field's divergence widens toward the patient's superior side, i.e. depth increases
   superiorly); (b) HFS, G0, C90, jaws open only on +X (`jaw_x1=0, jaw_x2=50`, Y symmetric): the lit region at the
   isocenter plane lies on the patient's **superior** side (+z), none inferior.
6. MLC: a beam whose MLCX opens only one leaf pair yields non-zero voxels only within that pair's projected Y band and its
   X opening (± one voxel); a fully closed control point contributes nothing.
7. Weighting: a two-control-point static beam of MU M yields M inside the field; a step-and-shoot beam with two segments
   of different apertures yields each segment's MU in its own region and their sum where they overlap; a three-point
   arc's voxel sum scales linearly with beam MU.
8. Every not-modelled case in Decisions writes no `beam_mu.nii.gz` and no `beam` series.json row, logs exactly one
   WARNING `convert: <REASON_BEAM_VOLUME>`, and still writes `planning_ct.nii.gz`, `plan.parquet` (when it had rows)
   and series.json. The AST test confirms the beam module raises only `CaseRefused(REASON_BEAM_VOLUME)` and catches
   nothing.
9. A converted photon plan's series.json has a `beam` row (`kind` `beam`, `modality` `RTPLAN`, plan columns filled) and
   `convert_report.csv`'s `files` lists `beam_mu.nii.gz`; it has the planning CT's size, spacing, origin and direction
   and is float32. A plan with an ROI named "Beam" still converts with both its `beam.nii.gz` mask and `beam_mu.nii.gz`.
10. preprocess with reshape that cuts the beam volume does not refuse the plan and does not count it as cropped; the
    beam row's `retained` is blank; the output is float32 on the final CT grid with the dose's output name prefixes.
11. preprocess geometric run: output plan.parquet's isocenter_i/j/k equal the final CT's continuous index of
    isocenter_x/y/z (1e-6); all other columns, row order and SCHEMA identical to the input. Non-geometric run:
    byte-identical copy.
12. manifest: a plan line has `plan` and `beam` keys (file names) exactly when those series.json rows exist; `retained`
    is unchanged by a beam row.
13. No PHI: `assert_no_leak` (tests/test_convert.py) runs over every not-modelled case of criterion 8 with the patient
    ID, plan UID and source paths as secrets.
14. A full-size synthetic VMAT case (512×512×150 CT, 2 arcs × 178 control points) completes in a test marked slow
    (opt-in, out of the default suite); its wall time and peak memory for one worker are recorded in findings; over 15
    min is flagged and filed as `hardening`.

## Failure handling
- See Catch path. The beam volume is computed before `case_dir.mkdir` (everything that can refuse runs first).
- A MemoryError is the run's, as for dose.

## Approaches considered
- Chosen: compute in convert on the native CT and reuse preprocess's resample (owner pick).
- Rejected: compute in preprocess on the final grid (new step, reads plan geometry it copies today); dose-grid or half-res
  compute (zero outside dose grid / extra blur); binary or per-beam volumes (lose modulation / variable channel count);
  jaws-only (wrong for VMAT/IMRT); attenuated ray trace (a crude dose engine).

## Safety and data
- Clinical output: a derived ML input, not used for treatment. A wrong geometry is silent (plausible but misplaced), so
  geometry is pinned by analytic tests (criteria 1–7) and anything not modelled is refused loudly (WARNING), never
  approximated.
- PHI: no new PHI surface. `beam_mu.nii.gz` holds MU values on the CT grid; logs carry only the fixed reason. FHIR: n/a.

## Delivery and validation
- Analytic synthetic tests in this PR. Follow-up issue (hardening): on the first real photon cohort, check that dose ≥ 50 %
  of max lies inside `beam_mu > 0` for each plan.

## Deferred
- Filed: #283 real-cohort dose-overlap check; #284 bad plan.parquet preprocess test; #285 duplicate BeamNumber SAD lookup.
- Modelling TableTopEccentricAngle / table pitch and roll (refused when non-zero, #287).
- CT PatientPosition vs RTPLAN PatientSetupSequence disagreement.
- Isocenter marker volume; per-beam volumes; inverse-square or attenuation variants.
- Blocks, MLCY, decubitus positions.
- Performance optimization (decided after the real cohort run).
