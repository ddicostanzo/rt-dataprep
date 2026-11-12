# rt-dataprep

Prepare radiotherapy DICOM exports for machine learning: organize files by patient and plan, convert images and RT objects,
preprocess volumes, build a case manifest, and split whole patients into reproducible cohorts.

The pipeline supports **plan-based cases and imaging-only CT, MR, and PT (PET) studies**. It preserves source files and
records refusals and skips in reports. This is research software; validate geometry, dose, labels, and cohort composition
for your dataset before using its outputs.

**Patient data remains sensitive after conversion.** Patient folder names, DICOM copies, NIfTI volumes, per-case metadata,
manifests, and splits can contain identifying information. Path sanitization is not de-identification.
See [Security and data handling](SECURITY.md) before processing clinical exports.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Input and output](#input-and-output)
- [Conversion behavior](#conversion-behavior)
- [Preprocessing](#preprocessing)
- [Manifests and patient splits](#manifests-and-patient-splits)
- [Reports, reruns, and troubleshooting](#reports-reruns-and-troubleshooting)
- [Python API](#python-api)
- [Integrating into a workflow](#integrating-into-a-workflow)
- [Development](#development)
- [License](#license)

## Install

Requirements: Python **3.12 or newer**, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The repository includes `uv.lock` for reproducible dependency installation. Runtime dependencies include pydicom, NumPy,
SimpleITK, DicomRTTool, and scikit-image; no GPU is required.

```bash
git clone https://github.com/ddicostanzo/rt-dataprep.git
cd rt-dataprep
uv sync --locked --group dev
uv run python -m rt_dataprep --help
```

Repository access is required while the repository is private. For an existing checkout, run the last two commands from
its root. Every command below assumes that working directory. Use `uv run` to select the project environment; the CLI is
invoked as `python -m rt_dataprep`.

To use rt-dataprep from your own project without a checkout, install a tagged release; see
[Add rt-dataprep to another project](#add-rt-dataprep-to-another-project).

## Quick start

### Run a small synthetic example

This creates six synthetic patients, each with CT, MR, and PT series, then runs organization, conversion, compact
preprocessing, manifest generation, and splitting. It downloads no patient data and imports no test helpers.
Choose an output directory that does **not** already exist, preferably outside the checkout:

```bash
uv run python examples/synthetic_workflow.py /tmp/rt-dataprep-demo
```

Expected result: **18 imaging cases**, no refusals, and **2/2/2 patients in train/validation/test**. The example deliberately
uses a small output grid; the CLI's `--reshape` grid is substantially larger.

See [examples/](examples/README.md) for the complete walkthrough, a clinical-export recipe, and a configurable Python
preprocessing example. On Windows, replace `/tmp/...` with a new folder in your approved workspace.

### Process an export tree

Use approved storage and replace these illustrative paths. Input holds one top-level folder per patient, with DICOM files
in that folder or its subfolders. Each stage writes to a separate directory outside its input tree.

```bash
uv run python -m rt_dataprep organize /secure/export /secure/organized --workers 4
uv run python -m rt_dataprep convert /secure/organized /secure/converted --workers 4
uv run python -m rt_dataprep manifest /secure/converted
uv run python -m rt_dataprep split /secure/converted --test-size 0.2 --val-size 0.2 --seed 42
```

This also works for imaging-only patients. Review `organize_report.csv`, `convert_report.csv`, and the manifest refusal
count before accepting the cohort. Successful process completion can include refused patients, plans, or series.

## Commands

| Command | Inputs and output | Main options |
| --- | --- | --- |
| `organize SRC OUT` | Copy a patient export tree into plan folders and `misc_files/`. | `--workers auto` or `1..64` (default `auto`); `--overwrite` |
| `convert ORG OUT` | Convert an organized tree into plan and study outputs. | `--workers auto` or `1..64` (default `auto`); `--overwrite` |
| `preprocess CONVERTED OUT` | Write a separate preprocessed tree. | `--workers auto` or `1..64` (default `auto`); at least one step flag (`--initial-resample`, `--rescale`, `--ahe`, `--resample`, `--reshape`); also `--new-spacing`, `--final-spacing`, `--new-shape`, `--allow-crop`, `--min-mask-retention`, `--ahe-kernel-mm`, `--ahe-bins`, `--ahe-clip-limit`, `--legacy-linear-masks`, `--series-fov-mask`, `--overwrite` |
| `manifest ROOT` | Write `ROOT/manifest.jsonl` and a dated copy. | Works on converted or preprocessed trees. |
| `split ROOT` | Read `ROOT/manifest.jsonl`; write `ROOT/splits.json`. | Required: `--test-size`, `--val-size`, `--seed`; optional: `--external MANIFEST` |

Use `uv run python -m rt_dataprep COMMAND --help` for exact arguments. Study exclusion defaults to a case-sensitive
`Acuity` substring in `ManufacturerModelName`; `organize()` and `convert()` expose `exclude_models` through the Python API.

### Parallel patients (`--workers`)

`--workers auto|N` (organize, convert and preprocess; default `auto`) processes several patients at a time, each in its
own process. The output and row order are the same for any value.

`auto` runs patients one at a time until the first one is processed, reads that patient's peak RAM, and starts as many
workers as fit in 0.8 of the available memory, capped by the usable CPUs, 64 and the patients left. Available memory
honours a cgroup limit, so under Slurm it is the `--mem` the job was given rather than the node's RAM. If a later patient
needs much more than the first (a small first scan, then full 701 x 701 x 433 grids), a worker can be OOM-killed; that
aborts the run with the report kept, a rerun skips the finished patients, and a fixed `--workers N` overrides the sizing.

`--workers N` (1 to 64) runs exactly N patients at a time. Budget about 7 GB of RAM per worker at the full
701 x 701 x 433 grid. SimpleITK and BLAS threads are not capped per process, so N workers can use more than N cores.
Ctrl-C in a terminal reaches the workers too. With N > 1, `kill -INT` sent to the parent process alone makes it wait for
the patients already running, then stop with the report written; with N = 1 the patient runs in the parent and stops at
once, as before.

## Input and output

Example input, containing synthetic identifiers only:

```text
export/
  SYNTHETIC_P001/
    planning_export/        # CT, RTPLAN, RTSTRUCT, optional RTDOSE
    diagnostic_export/      # additional CT/MR/PT and optional REG
  SYNTHETIC_P002/            # imaging-only patient is allowed
    images/
```

`organize` reads DICOM files with a DICOM preamble and copies them into `<patient>/plan_<n>/` plus `misc_files/` as needed.
The patient directory name is a sanitized PatientID, not an anonymous identifier. A source folder containing multiple
nonblank PatientIDs is refused. Files are copied, never moved, renamed, or deleted in the source tree.

Zip archives in a source folder are read too, recognized by content rather than extension. Stored and deflate
members are supported (bzip2 and lzma are not); encrypted members, a zip inside a zip, and tar/7z/gzip archives are not. Before anything is
extracted, a folder's zips may declare at most 16 GiB and 200,000 members in total. Members are extracted into a hidden
`.unzip-<random>/` folder under the output root and appear in the plan folders under index names (`1`, `1_1`, ...), never
under their names inside the archive. A zip that is over the cap or cannot be read is skipped when the folder also holds
loose DICOM files. Otherwise the folder is refused with `zip archives in the source folder exceed the size or member cap;
nothing kept` or `a zip archive in the source folder cannot be read safely; nothing kept`. The summary line counts zips,
zip members and skipped zips. On Linux and macOS a run holds `.organize.lock` in the output root, so a second run on the
same root stops with `another organize run holds the output root lock`. Windows takes no lock; on a file system that
refuses it a warning is logged and the run continues unlocked. A killed run's leftover `.unzip-*` folder is removed by the
next locked run.

Representative converted output (optional files shown where applicable):

```text
converted/
  convert_report.csv
  convert_report_<UTC timestamp>.csv
  SYNTHETIC_P001/
    plan_1/
      planning_ct.nii.gz
      dose.nii.gz
      liver.nii.gz
      series_MR_1.nii.gz
      plan.parquet
      beam_mu.nii.gz
      fluence_mu.nii.gz
      series.json
    study_1/
      series_CT_1.nii.gz
      series_MR_1.nii.gz
      series.json
    study_2/
      series_PT_1.nii.gz
      series.json
  SYNTHETIC_P002/
    study_1/
      series_CT_1.nii.gz
      series.json
```

Plan numbering reflects the organized plan order. Study folders are numbered by the earliest StudyDate/StudyTime among
their written series, with blank dates/times last and StudyInstanceUID as a tie breaker. Series numbers are per modality
within each destination. `series.json` records the actual output files and metadata; inspect it instead of inferring
relationships from filenames. It is a JSON array with one object per output file, holding the same fields the earlier
`series.csv` held (every value a string), then a `dicom` object (below). Read it with `json.load(open(path))`.

Every object has a `dicom` key whose value is an object of DICOM metadata; it is `{}` when the headers hold none of it.
A key appears only when its value is present, so a missing key means the tag was absent or unusable, never a default.
Values are the DICOM values as written: strings (raw DA/TM/DT text such as `20240105`), a list of strings for a
multi-valued element (`pixel_spacing`, `image_type`), and two numbers, `patient_age` (years, from `PatientAge` or from the
birth date and study date; the birth date itself is not written) and `slices` (the count of slice headers). When the
headers of a row (every slice of a series, every file of a multi-file dose) disagree, the value is
`{"min": a, "max": b}` for a single numeric element or a DA/TM/DT (the first-seen texts of the extremes), and
`{"distinct": [v1, v2, ...]}` otherwise (first-seen order); equal numbers such as `3` and `3.0` are one value.
Which keys a row carries depends on its kind: patient, study, device and acquisition keys (`manufacturer`, `model`,
`institution_name`, `study_date`, ...) on every row; image, CT, MR and PET keys (`rows`, `kvp`, `echo_time`,
`injected_dose`, ...) on `ct` and `series` rows; RTSTRUCT keys, including `roi_interpreted_type` and
`roi_generation_algorithm` of that row's ROI, on `mask`, `applicator` and `line` rows; `dose_units` and `dose_type` on
`dose` rows. Plan keys (`plan_intent`, `prescription_dose`, `treatment_machine`, `source_isotope`, ...) are on every row
of a plan folder; the `fluence` row adds `beam_numbers` after them (the `;`-joined BeamNumbers in slice order, not a DICOM attribute itself); and the registration keys (`reg_type`, `reg_date`, `reg_time`, `reg_manufacturer`) on the
`series` rows of a plan folder registered through a REG (not same-frame rows); study folders carry neither. Patient name, birth date and ethnic group are never written.
These values are patient data: they appear only in `series.json` and `manifest.jsonl`, never in a log line or a report.

Trees created with the old `images/` conversion layout must be re-converted. Current manifest and preprocessing walkers
use `study_<n>/` and ignore the old `images/` layout.

Trees converted by v0.1.x hold a `series.csv` and no `series.json`. `preprocess` and `manifest` refuse those folders (the
CSV is never read) with the reason `converted by an older version (series.csv); reconvert with --overwrite`, and count
them as refused. Reconvert them with `convert --overwrite`.

### plan.parquet

A photon external-beam RTPLAN also writes `plan.parquet` in its plan folder (after the masks, before `series.json`), with a
`series.json` row of `kind` `plan`; `preprocess` writes it under the same name (copied unchanged on a run with no geometric step), and `manifest`
lists it in the plan line's `plan` key. Brachy and RT Ion plans, a plan mixing photon with other beams, and a photon plan with a missing or inconsistent
value get no `plan.parquet` (the plan still converts; a malformed photon plan logs a warning). The file has one row per control point of each
treatment beam (SETUP beams are dropped; a plan with a CONTINUATION, TRMT_PORTFILM or OPEN_PORTFILM beam gets no
`plan.parquet`, since that beam's dose is in `dose.nii.gz`), values DICOM omits are filled forward from the earlier control point, and the
beam-level fields repeat on every row. The columns and types are `rt_dataprep.plan_table.SCHEMA`; read them with
`pyarrow.parquet.read_table`.

- `beam_meterset` is the beam's MU per fraction, while `dose.nii.gz` may be the plan total over all fractions.
- `mlc_positions` is the leaf positions in mm in DICOM `LeafJawPositions` order: every leaf of the IEC X1-side bank
  (leaf subscripts 1xx), then every leaf of the X2-side bank (2xx). DICOM does not fix vendor bank letters (A/B) to a side.
- `mu` is the meterset between control point i-1 and control point i, attributed to row i; `mu_cumulative` is the running sum.
- A null means the beam declares no such device (no MLC, no jaw pair, no wedge) or the optional tag is absent.
- `isocenter_x/y/z` are patient coordinates in mm; `isocenter_i/j/k` are the same point as a continuous voxel index of
  the CT in the same folder (invariant: `isocenter_i/j/k` always index the CT in the same folder). A preprocess run with `--initial-resample`, `--resample` or
  `--reshape` rewrites them for the final CT (every other column is unchanged); a run without those steps copies the file unchanged. A
  `plan.parquet` that pyarrow cannot parse as Parquet, or that is not the table schema, is left out of the preprocessed plan (and its `plan` row), with one warning; any other read failure refuses the patient like other I/O errors.
- Converted trees made before this version gain `plan.parquet` only on a run with `--overwrite`.

### ion_plan.parquet

A proton pencil-beam-scanning (PBS) RT Ion plan writes `ion_plan.parquet` in its plan folder (after the masks, before `series.json`) instead
of `plan.parquet`, `beam_mu.nii.gz` and `fluence_mu.nii.gz`, which stay photon-only and are never written for an ion plan. Its `series.json` row has
`kind` `ion_plan` (modality `RTPLAN`, the plan's FrameOfReferenceUID); `preprocess` copies or re-indexes it like `plan.parquet`, and `manifest`
lists it in the plan line's `ion_plan` key. For such a plan the `dicom` object (on every row of the plan folder) now also holds `radiation_type`
(`PROTON`), `beam_type` and `treatment_machine`, read from `IonBeamSequence`.

The file has one row per spot with a non-zero weight. The columns and types are `rt_dataprep.ion_plan_table.ION_SCHEMA`; read them with
`pyarrow.parquet.read_table`. Rows are in IonBeamSequence order, then control point, then spot map order. Beam and layer fields repeat on every row of their beam or layer.

- Beam: `beam_number`, `beam_name` (nullable), `beam_type`, `radiation_type`, `treatment_machine` (nullable), `scan_mode`, `beam_meterset` (MU per
  fraction, while `dose.nii.gz` may be the plan total over all fractions), `final_cumulative_meterset_weight`, `number_of_control_points`,
  `virtual_sad_x`/`virtual_sad_y` (mm, nullable), `snout_id` (nullable), `range_shifter_ids` (list of strings, `[]` when none).
- Layer (the spot's control point; values DICOM omits are filled forward from the earlier control point): `control_point_index`, `layer_index` (0-based among
  the control points that carry a spot), `cumulative_meterset_weight`, `nominal_energy` (MeV), `gantry_angle` and `couch_angle` (degrees), `snout_position` (mm,
  nullable), `scan_spot_tune_id` (nullable), `spot_size_x`/`spot_size_y` (mm, nullable), `number_of_paintings` (nullable),
  `range_shifter_settings` (list of strings, `[]` when none), `range_shifter_wet` (list of floats, mm water equivalent; `[]` when no shifter, null when a
  declared shifter has no value), `isocenter_x/y/z` (patient mm) and `isocenter_i/j/k` (the same point as a continuous voxel index of the CT in the same folder).
- Spot: `spot_index`, `spot_x`/`spot_y` (mm), `spot_weight`, `spot_mu` (MU per fraction). Every column not marked nullable is never null.
- `spot_x` and `spot_y` are at the isocenter plane in the IEC gantry frame, stored as the plan gives them.
- `spot_index` is the position in the control point's weight list (zero-weight spots keep their gap), i.e. map order, not delivery order (the plan may allow reordering).
- `spot_mu` is `beam_meterset * spot_weight / final_cumulative_meterset_weight`, the total over all paintings (never divided by `number_of_paintings`).
  The `spot_mu` of a beam sum to its `beam_meterset` within 1e-4 relative (`ION_WEIGHT_TOLERANCE`), not exactly: real files store spot weights as float32.
- Only PBS (`ScanMode` MODULATED) proton beams with MU metersets get a table. A SETUP beam is dropped; any other treatment beam (passive scattering, uniform
  or line scanning, another ion, a CONTINUATION or port-film beam, a Number-of-Pulses unit) gives the plan no table, silently. A proton PBS plan with a missing or inconsistent
  value (weights that do not add up, a negative weight, a non-finite energy, a missing meterset, ...) gets no table and logs one fixed warning; the plan still converts.
  The file is therefore absent, never empty, when the plan gets no table.
- A `preprocess` run with `--initial-resample`, `--resample` or `--reshape` rewrites `isocenter_i/j/k` for the final CT (every other column is unchanged); a run
  without those steps copies the file unchanged. An `ion_plan.parquet` that is not the table schema is left out with one warning, as for `plan.parquet`.
- Converted trees made before this version gain `ion_plan.parquet`, and the new `radiation_type`/`beam_type`/`treatment_machine` values of proton plans in their
  `series.json` `dicom` objects, only on a run with `convert --overwrite`.

### dwell.parquet

A brachytherapy RT Plan with `BrachyTreatmentType` HDR or PDR (and `SourceMovementType` STEPWISE or FIXED on every channel that has control points) writes
`dwell.parquet` in its plan folder (after `ion_plan.parquet`, before `series.json`). Its `series.json` row has `kind` `dwell` (modality `RTPLAN`, the plan's
FrameOfReferenceUID, which may be empty: brachy RTPLANs often omit it); `preprocess` copies or re-indexes it like `plan.parquet`, and `manifest` lists it in
the plan line's `dwell` key. LDR, MDR, MANUAL and plans with no `BrachyTreatmentType` get no table, silently, as does a plan with any channel whose source is
OSCILLATING or UNIDIRECTIONAL (moving sources are not handled).

The file has one row per dwell: control points 2k and 2k+1 of a channel, at one position, form dwell k. Dwells with zero time are kept (inactive positions show the
activatable path). The columns and types are `rt_dataprep.dwell_table.DWELL_SCHEMA`; read them with `pyarrow.parquet.read_table`. Rows are in application setup
order, then channel order, then dwell index. Setup, channel and source fields repeat on every row of their channel.

- Setup and channel: `brachy_treatment_type`, `setup_number`, `setup_name`, `setup_type`, `channel_number`, `source_movement_type`, `source_applicator_id`,
  `source_applicator_type`, `source_applicator_step_size` (mm), `source_applicator_length` (mm), `transfer_tube_length` (mm), `referenced_roi_number`
  (nullable: `setup_name`, `setup_type` and the `source_applicator_*`,
  `transfer_tube_length` and `referenced_roi_number` columns),
  `channel_total_time` (s), `final_cumulative_time_weight`, `number_of_control_points`, `number_of_pulses` and `pulse_repetition_interval` (s; PDR only, else null).
- Source (the `SourceSequence` item the channel references): `source_number`, `source_isotope_name`, `source_isotope_half_life` (days), `reference_air_kerma_rate`
  (µGy/h at 1 m), `source_strength_reference_date` and `source_strength_reference_time` (DICOM DA and TM strings, as stored); all but `source_number` nullable.
- Dwell: `dwell_index` (0-based within the channel, zero-time dwells counted), `control_point_index`, `relative_position` (mm, nullable),
  `position_x/y/z` (patient mm, control point 2k), `position_i/j/k` (the same point as a continuous voxel index of the CT in the same folder), `cumulative_time_weight`,
  `dwell_weight` and `dwell_time_s`. Every column not marked nullable is never null.
- `dwell_weight` is `max(0, w(2k+1) - w(2k))` of the cumulative time weights; `dwell_time_s` is `channel_total_time * dwell_weight / final_cumulative_time_weight`
  (the DICOM definition), and 0 when both the final weight and the total time are 0. The `dwell_weight`s of a channel sum to its final weight within 1e-4 relative
  (`DWELL_WEIGHT_TOLERANCE`), not exactly.
- The times are as exported in `ChannelTotalTime`. **Which date's source strength those times assume is not established** (DICOM ties the air kerma rate to its
  reference date, not to the treatment time); it stays open until a TG-43 point dose from this table is compared with RTDOSE on a real export. No decay correction is applied.
- For PDR, `channel_total_time` is stored as-is next to `number_of_pulses` and `pulse_repetition_interval`. Reading it as the time of one pulse (per fraction:
  multiply by `number_of_pulses`) is a vendor convention, not DICOM, and is unverified on a real PDR export.
- A plan with a missing or inconsistent value (an odd control point count, the two control points of a dwell apart, weights that do not add up or that rise while the source
  moves, a missing position or weight, a duplicate channel number, a PDR channel without pulse values, ...) gets no table and logs the single fixed warning
  `convert: the brachy RTPLAN has a missing or inconsistent value; no dwell table written` (`REASON_DWELL`); the CT, dose, masks, applicator files and the rest of the plan folder are unaffected.
  The file is therefore absent, never empty, when the plan gets no table.
- A `preprocess` run with `--initial-resample`, `--resample` or `--reshape` rewrites `position_i/j/k` for the final CT (every other column is unchanged); a run
  without those steps copies the file unchanged. A `dwell.parquet` that is not the table schema is left out with one warning, as for `plan.parquet`.
- Converted trees made before this version gain `dwell.parquet` only on a run with `convert --overwrite`.

### beam_mu.nii.gz

A photon plan that gets a `plan.parquet` also gets `beam_mu.nii.gz` (after `plan.parquet`, before `series.json`), with a
`series.json` row of `kind` `beam` and `modality` `RTPLAN` (the plan's columns and metadata, and the planning CT's
`frame_of_reference_uid`). It is a float32 image on the planning-CT grid (same size, spacing, origin and direction) holding, in
each voxel, the MU per fraction of every control point whose divergent jaw and MLC aperture covers it: each control point
carries half the MU of the segment before it and half of the segment after it. It is pure geometry: no attenuation, no inverse
square and no dependence on the CT values. The beam geometry follows IEC 61217 and the CT's `PatientPosition`, which must be
`HFS`, `HFP`, `FFS` or `FFP`.

A plan the volume does not model gets no `beam_mu.nii.gz` and no `beam` row, and logs one warning (the plan still converts, with
its `plan.parquet`): any beam with blocks, an MLC that is not `MLCX`, an `MLCX` without leaf boundaries, a treatment beam without a
positive `SourceAxisDistance`, another or missing `PatientPosition`, or a control point with no jaw or MLC bounding an axis.
Wedges are ignored. Plans without a `plan.parquet` (brachy, RT Ion, mixed, CONTINUATION or port-film, malformed) get none either.
Trees converted before this version gain `beam_mu.nii.gz` only on a run with `convert --overwrite`.

`preprocess` resamples `beam_mu.nii.gz` onto the final CT like the dose (float32, 0 outside the native grid, the same output-name
prefixes; copied unchanged on a run with no geometric step), but a beam volume the final grid cuts never refuses the plan, is not
counted in `cropped`, and has a blank `retained`.

### fluence_mu.nii.gz

A photon plan that gets a `plan.parquet` also gets `fluence_mu.nii.gz` (after `beam_mu.nii.gz`, before `series.json`), with a
`series.json` row of `kind` `fluence` and `modality` `RTPLAN` (the plan's columns and metadata, and the plan's own
`frame_of_reference_uid`), whose `dicom` is the plan row's `dicom` plus a last key `beam_numbers`: the `;`-joined DICOM
BeamNumbers in slice order (a plan numbered 7, 2, 5 gives `"7;2;5"`). It is a float32 image of shape (beams, 400, 400) as
`sitk.GetArrayFromImage` returns it (`[k, row, column]`), one slice per treatment beam in the order of `plan.parquet`, holding the
beam's fluence in MU per fraction on the isocenter plane in the beam-limiting-device (collimator) frame, seen from the source:
1 mm pixels, centres at -199.5 ... 199.5 mm on both axes (pixel `j` covers `j - 200` to `j - 199` mm). The collimator angle is not
applied, and gantry, couch, SAD and `PatientPosition` play no part. The column index increases toward IEC +X (the X2 jaw and leaf
bank), the row index toward IEC +Y (the Y2 jaw; leaf pair n+1 lies at larger Y than pair n). Each pixel holds the exact
area fraction of the open region (one rectangle per open leaf pair inside the jaws, or the jaws alone without an MLC) times the
control-point weights, which follow `beam_mu`'s rule (half the MU of the segment before and half of the segment after each
control point); a pixel fully inside a static open field reads the beam's MU. Pure geometry: open is 1 and closed is 0, with no
leaf transmission, tongue-and-groove or rounded leaf ends.

- **Not on the CT grid.** The file's geometry is its own: never resample it with the masks or the dose, and do not use it with the
  CT's origin or spacing. Axis 0 is the beam index, not a distance in mm.
- **Orientation.** SimpleITK reads the NIfTI with origin (-199.5, -199.5, 0), spacing 1 and identity direction. nibabel shows the same
  data as (column, row, k) with affine diag(-1, -1, 1) and origin (199.5, 199.5, 0) (NIfTI stores LPS): index the array, and do not
  take the +X/+Y directions from the nibabel affine. To plot the view from the source, use `plt.imshow(fluence[k], origin='lower')`.
- **Not modelled.** A plan gets no `fluence_mu.nii.gz` and no `fluence` row, and logs one warning (the plan still converts, with its
  `plan.parquet` and any `beam_mu.nii.gz`), when `beam_mu`'s aperture rules refuse it (blocks, an MLC that is not `MLCX`, an `MLCX`
  without strictly increasing leaf boundaries, a control point with no jaw or MLC bounding an axis), when an open rectangle of a
  control point with weight reaches beyond +-200 mm (an edge at exactly 200 is inside; nothing is clipped), or when two treatment
  beams share a BeamNumber. The SAD, `PatientPosition` and table-top-angle rules of `beam_mu` do not apply to it. Plans without a
  `plan.parquet` get none either.
- Trees converted before this version gain `fluence_mu.nii.gz` only on a run with `convert --overwrite`.

`preprocess` copies `fluence_mu.nii.gz` unchanged under its own name on every run (it is not on the CT grid, so it is never
resampled, cropped or counted in `retained`), and `manifest` lists it in the plan line's `fluence` key.

## Conversion behavior

- A plan requires one RTPLAN, its referenced RTSTRUCT, and a complete, supported planning CT volume. A plan's planning CT
  is written once in that plan folder and is not repeated as an imaging study.
  The planning CT path requires an axial volume with consistent orientation, image dimensions, and uniform slice spacing;
  gantry tilt is refused. Other imaging series may be oblique but must still form a regular single-frame volume.
- Planning CT is float32 HU. Supported physical RTDOSE is converted to Gy and resampled onto its planning CT grid;
  supported beam-dose inputs are summed. Ambiguous or unsupported dose combinations refuse the plan.
- Kept RTSTRUCT ROIs become binary masks on the planning CT grid. Empty, mismatched, or inconsistent mask results refuse
  the plan. Point-only ROIs and names containing `dose` are excluded. Brachy applicator outputs are **gzip-compressed
  coordinate text**, despite their legacy `.nii.gz` extension; consult `kind` in `series.json` before opening a file as NIfTI.
  An applicator ROI that no plan channel references is dropped when a referenced channel has the same file name (ignoring
  case) and point count (the structure set's copy of that channel). An applicator named like a mask is written as `<name>_channel.nii.gz`, and
  a later applicator with an earlier one's file name (ignoring case) gets `_2`, `_3`, ... appended. Find
  an applicator's file through `series.json` (`kind` applicator), not from its `SourceApplicatorID`.
  An ROI whose every contour crosses CT slices (a reference line drawn on a reformatted plane) is not a mask: its points
  are written as the same coordinate text with `kind` `line`, and preprocess copies it unchanged. An ROI that mixes axial
  and crossing contours refuses the plan. Any other output name clash refuses the plan.
- Other eligible CT/MR/PT series are written into their study folders, even without a plan. Eligible registered series are
  also written in matching plan folders, on their native grids, with the transform recorded in `reg_matrix`.
- Image series must form a supported, uniform single-frame 3D volume. Localizers, excluded model studies, incomplete or
  unreadable volumes, and unsafe study associations are filtered or skipped. Consult the reports; not every filter
  produces its own report row.
- Conversion does not perform deformable registration, PET SUV normalization, MR intensity normalization, or DICOM
  de-identification. PT applies stored pixel rescale values rather than supplying an SUV workflow.

## Preprocessing

Each preprocessing step has its own opt-in flag, and any combination is allowed. The full legacy pipeline is the four flags
`--initial-resample --rescale --resample --reshape`:

```bash
uv run python -m rt_dataprep preprocess /secure/converted /secure/preprocessed --initial-resample --rescale --resample --reshape --series-fov-mask
uv run python -m rt_dataprep manifest /secure/preprocessed
uv run python -m rt_dataprep split /secure/preprocessed --test-size 0.2 --val-size 0.2 --seed 42
```

| Flag | Step | Output prefix |
| --- | --- | --- |
| `--initial-resample` | Resample to 1 mm spacing (`--new-spacing X Y Z`) | `initialresample_` |
| `--rescale` | Apply `(HU + 1024) / 4096` to CT | `rescale_` |
| `--ahe` | CLAHE adaptive histogram equalization | `ahe_` |
| `--resample` | Resample to 1.25 mm spacing (`--final-spacing X Y Z`) | `resample_` |
| `--reshape` | Centred crop/pad to **701 × 701 × 433 voxels (x, y, z)** (`--new-shape X Y Z`) | `reshape_` |

`--new-spacing` and `--final-spacing` take finite millimetre values of at least 0.1, `--new-shape` positive integers, all in x y z
order. Each is refused unless its step flag (`--initial-resample`, `--resample`, `--reshape`) is also given; without them the
defaults above apply.

Steps always run in the order of the table, whichever flags are given. One float32 image at the `--reshape` size occupies
about **812 MiB** before processing copies and other volumes; plan memory and storage accordingly. The CT formula is not
intensity clipping. `--reshape` without `--initial-resample` or `--resample` crops/pads at the CT's own spacing, which can
pad a large extent at native spacing.

By default a plan whose dose or mask extends past the final CT grid is refused. `--allow-crop` (needs at least one of
`--initial-resample`, `--resample`, `--reshape`) writes the plan anyway, with the dose and masks cut to the final grid. In
`series.json`, the `retained` key of each dose and mask object is the fraction of that file's native-grid content kept by the crop: for a
dose, the fraction of its voxels at or above the lesser of 1 Gy and 10% of its max (so a 2 Gy fraction is judged at 0.2 Gy); for a mask, its volume. It is exactly `1.0` when nothing is cut, and blank for CT, applicator, line, plan, beam and series entries and on runs
without a geometric step. In `preprocess_report.csv`, the `cropped` column counts the dose and mask files the final grid cut; a
dose cut only below that threshold counts as cropped but has `retained` `1.0`. A mask whose `dicom.roi_interpreted_type` is `EXTERNAL` (the body outline) keeps its own
`retained`, but is left out of the plan's lowest `retained`. A mask without a single `roi_interpreted_type` (a tree converted before dicom metadata) still counts, and the run log warns how many such masks the final grid cut. The run's closing log line counts the cropped plans and how many of them have a
lowest `retained` below 0.9. A dose or mask that the final grid empties entirely refuses the plan, with or without `--allow-crop`.

CT study series receive all enabled steps. MR/PT study series receive geometric steps and optional AHE, and skip HU rescaling. `--rescale`
alone (or the same Python configuration) refuses MR/PT because it has no applicable step. With no geometric step (`--rescale` and/or
`--ahe` only), dose and masks are copied unchanged. Registered plan series are resampled onto
the final planning CT grid; unusable alignment, no overlap, or unreadable data skips that series and increments the plan's
`skipped_series`. Output `reg_matrix` is `identity` after alignment.

Masks use nearest-neighbor interpolation and uint8 by default. `--legacy-linear-masks` requests the legacy linear float32
behavior. `--series-fov-mask` adds a uint8 `fovmask_<image filename>` for the planning CT, each study image, and aligned
plan series. A value of 1 marks acquired support and 0 marks outside-FOV fill or crop/pad fill. Planning/study masks follow
each geometric step with nearest-neighbor interpolation and zero padding; true zero-valued image voxels remain valid.
Auxiliary masks do not add entries to `series.json`. The manifest lists them only when their headers match the relevant image grid.

An optional `--min-mask-retention FRACTION` refuses a plan when resampling retains less than the configured fraction of
a mask's physical foreground volume, even when a few voxels survive. Binary masks use foreground count times voxel
volume; linear masks integrate fractional occupancy times voxel volume. The limit must be greater than 0 and at most 1,
with a relative floating-point comparison tolerance of 1e-6. No clinical threshold is imposed by default; select one for
the intended analysis. Growth, shape distortion and registration accuracy are separate checks. A refused plan is removed
from output and recorded with a fixed reason; sibling plans can continue. As with other plan refusals, review the report
even when the CLI exits successfully. Python uses `PreprocessSteps(min_mask_retention=...)`.

Optional `--ahe` applies per-volume min-max scaling followed by scikit-image CLAHE (default kernel of one eighth of each
axis, 128 bins, clip limit 0.05) to planning CT and CT/MR/PT image series. It is off by default and never equalizes dose,
masks, or applicator coordinates. It produces float32 image values in [0, 1]; these are contrast values, not HU or SUV.
The step runs before final resampling/cropping, after any CT rescale. Registered plan series are equalized on their native
grids before alignment. Constant volumes become zero; if CLAHE collapses a varying tiny/two-level volume to a constant,
the normalized input is retained instead, preserving contrast. NaN/Inf input is refused before AHE: a study refusal affects
that series only, a planning CT refuses its plan, and an invalid aligned series is skipped and counted.
This is not a numerical parity port of legacy mclahe.

For AHE without the large `--reshape` grid, use a fresh output root:

```bash
uv run python -m rt_dataprep preprocess /secure/converted /secure/equalized --ahe
uv run python -m rt_dataprep manifest /secure/equalized
```

Combine `--ahe` with any other step flag. AHE can require several additional volume-sized arrays; include those in
memory planning; real-data peak-memory profiling remains tracked in #209. Tune the histogram or use a physical window:

```bash
uv run python -m rt_dataprep preprocess /secure/converted /secure/equalized-tuned --ahe --ahe-kernel-mm 16 16 16 --ahe-bins 128 --ahe-clip-limit 0.05 --series-fov-mask
```

`--ahe-kernel-mm X Y Z` uses positive mm widths, rounded up to whole voxels on the grid being equalized (at least one
voxel). Without it, the per-axis one-eighth voxel default is preserved. Bins must be at least 2; clip limit must be finite
and between 0 and 1. These options require `--ahe`. Python equivalents are `ahe_kernel_mm`, `ahe_bins`, and
`ahe_clip_limit` in `PreprocessSteps`.

The initial linear resample's edge fill still participates in AHE's histogram; support masks identify validity for
downstream consumers, not a masked CLAHE algorithm. AHE-only avoids introducing that initial-resample fill. A
`rescale_` prefix records the executed CT processing step even though AHE's subsequent min-max scaling removes its units.

Output filenames acquire step prefixes such as `reshape_resample_rescale_initialresample_`; when enabled, `ahe_` sits
between `resample_` and `rescale_` (for example `reshape_resample_ahe_rescale_initialresample_`). Generate a new manifest
after preprocessing so its `image` and `series` fields point to the resulting filenames. Use
[examples/custom_preprocess.py](examples/custom_preprocess.py) for a smaller, explicit Python configuration.

## Manifests and patient splits

`manifest.jsonl` is JSON Lines: one case object per line, one line per plan and one per imaging series. Each object has
these keys:

| Keys | Meaning |
| --- | --- |
| `case_dir`, `patient`, `fx` | Case ID, patient directory, and plan directory (`fx` is blank for imaging). |
| `ct_size`, `ct_spacing` | Semicolon-separated x/y/z dimensions and mm spacing of the case image. |
| `dose`, `masks` | Dose-presence flag (`"0"` or `"1"`) and the list of ROI names; imaging lines have `"0"` and `[]`. |
| `rtplan_uid`, `ct_series_uid`, `study_uid` | Original identifiers; imaging uses its own series UID in `ct_series_uid`. |
| `case_type`, `modality`, `image` | `plan` or `imaging`; image modality; volume filename for imaging (blank for plans). |
| `series` | List of aligned plan series (objects with `file`, `modality`, optional `fov_mask`); `[]` for imaging. |
| `fov_mask` (optional) | Planning-CT or study-image support-mask filename, relative to its case folder; blank when absent. |
| `patient_age`, `patient_sex`, `patient_weight`, `study_date`, `series_date`, `manufacturer`, `model`, `institution_name` (all optional) | From the `dicom` object of the case's primary image (a plan's planning CT; an imaging line's own series), `series_date` from that image's `series_date` field. A key the image has no value for is left out of that line, so lines can differ in which of these they carry. A value that varied over the series is the `{"min": ..., "max": ...}` or `{"distinct": [...]}` object, copied as is. They follow the fixed keys (and `fov_mask`) in the order listed; a `series.json` without `dicom` gives only `series_date`. `split` ignores them. |
| `retained` (optional, plans only) | The lowest dose/mask `retained` of a preprocessed plan (the fraction kept by `--allow-crop`; EXTERNAL masks are excluded), after the metadata keys; left out on a converted tree, a run without a geometric step, or when only EXTERNAL masks carry one. |
| `plan`, `beam`, `fluence`, `ion_plan`, `dwell` (optional, plans only) | The file name of the plan's `plan` series.json row (`plan.parquet`), of its `beam` row (`beam_mu.nii.gz`, or its prefixed name after a geometric `preprocess` run), of its `fluence` row (`fluence_mu.nii.gz`, never renamed) of its `ion_plan` row (`ion_plan.parquet`) and of its `dwell` row (`dwell.parquet`); each is left out when the plan has no such row. They follow `retained`, in the order `plan`, `beam`, `fluence`, `ion_plan`, `dwell`. |

`masks` and `series` are native JSON lists, `patient_age` and `retained` are numbers, and the optional metadata values may be objects; every other value is a string. The `fov_mask` key appears on every line only
when at least one usable primary-image mask is present in the tree. Without those masks, the original fourteen keys are
written. Aligned-series masks remain in their `series` entries. A dated copy `manifest_<UTC timestamp>.jsonl` is kept next
to it.

```python
import json

import pandas as pd

cases = pd.read_json('/secure/converted/manifest.jsonl', lines=True, dtype=False)  # dtype=False keeps '00123' and dates as text
series = json.load(open('/secure/converted/SYNTHETIC_P001/plan_1/series.json'))
```

`dtype=False` stops pandas turning zero-padded patient IDs and dates into integers.

An imaging `case_dir`, for example `SYNTHETIC_P001/study_1/series_MR_1`, is an ID, **not a directory path**. Resolve its image
as `ROOT / patient / study_folder / image`. Plan image names come from the plan's `series.json`.

`split` keeps every case of a patient in one split. It deterministically orders patients using a hash of seed and patient
ID, allocates `ceil(n × test_size)` patients to test and `ceil(n × val_size)` to validation (with numeric rounding before
ceil), then assigns the remainder to training. Both sizes must be at least 0 and below 1, and at least one training patient
must remain. Small cohorts can therefore need smaller fractions.

`splits.json` contains `train`, `val`, and `test` maps of patient to case-ID lists, plus `case_id_hash` values. Repeating an
identical split is accepted; an existing split with different content is refused. Cohort changes, including newly added
imaging patients, can change assignments despite the same seed.

Check an external cohort for overlapping patient strings before writing a split:

```bash
uv run python -m rt_dataprep split /secure/converted --test-size 0.2 --val-size 0.2 --seed 42 --external /secure/external/manifest.jsonl
```

This checks patient IDs, not cross-institution identity linkage or clinical stratification. Maintain consistent patient
mapping and perform any required outcome/site/time stratification separately.

## Reports, reruns, and troubleshooting

Organization, conversion, and preprocessing write their report under the output root, alongside dated per-run copies;
manifest generation also keeps dated copies. Log and exception text uses counts, fixed refusal reasons, or safe error
class/module/line details. Reports, manifests and per-folder `series.json` files are sensitive artifacts and are excluded from Git.

| Symptom | What to check |
| --- | --- |
| A patient is skipped on rerun | Its destination patient folder already exists. Add `--overwrite` to replace it (see below), or use a fresh output root to recompute. |
| A plan is refused | Read its fixed reason in the conversion report; verify the RTPLAN → RTSTRUCT → CT reference chain and geometry. |
| Fewer images than expected | Review study summary skips, series eligibility, model exclusions, localizers, and readable pixel data. |
| Preprocessing reports no step is on | Pass a step flag (`--initial-resample`, `--rescale`, `--ahe`, `--resample`, `--reshape`), or enable applicable `PreprocessSteps` in Python. |
| A series or plan is missing from the manifest | Check the refusal count, `series.json`, file presence, and image headers. |
| Split refuses existing content | Preserve the old cohort record and generate the new cohort in a separate output root. |
| A patient fails with the I/O-error reason | A disk or permission error while reading one of its inputs fails the whole patient (no plan rows, no partial folder), and the run carries on with the next patient. Check the disk and file permissions, then rerun. A file that is missing or is not a valid image still refuses only its plan or series. |
| Unexpected internal error | Preserve source data and reports in approved storage; report a synthetic reproducer and safe class/site details. |

Patient outputs are staged under hidden partial directories and promoted when finished. Individual files are written
atomically. This supports interrupted-run recovery; it does not provide coordination for simultaneous writers. Run only
one pipeline writer against an output root. Review outputs before deleting or replacing any prior result, and use
`--overwrite` only when you mean to replace them.

### Replacing existing output with `--overwrite`

By default a patient whose output folder already exists is skipped. `organize`, `convert`, and `preprocess` accept
`--overwrite` to replace such a folder instead:

```bash
uv run python -m rt_dataprep convert /secure/organized /secure/converted --overwrite
```

- **When it replaces.** The patient is rebuilt in a hidden partial folder. The old folder is replaced only once the new
  output holds at least one successful case. If every case was refused or skipped (for example after a transient read
  error), the old folder is kept unchanged. That patient's report row shows the refusal, and the summary line counts it
  as kept. Each summary line ends with `N replaced, N kept`. A patient refused for a file system error, or an `organize`
  patient refused before its name is known, also keeps its old folder but is not counted. Replacement is per patient
  folder: when one case succeeds, cases refused in the new run are not carried over from the old folder, so check the
  report before discarding any copy of the old output.
- **Only this run's patients.** Output folders with no matching input patient are never touched.
- **Interrupted runs.** During a swap the old output sits in a hidden `.<name>.old` folder. If a run is interrupted, the
  next run of the same stage restores it when `<name>` is missing, or deletes it when `<name>` was already replaced. This
  happens with or without `--overwrite`. An `organize` patient that is refused before its name is claimed (no SOP
  instance UID, split series, or a name already claimed in the same run) never runs this recovery, so its `.<name>.old`
  stays until a run that reaches that name.
- **Same-run guard.** A folder name written earlier in the same run is never replaced by a later input folder.
  `organize` always refuses such a folder. `convert` and `preprocess` do the same under `--overwrite`, including names
  that differ only in letter case, because a case-insensitive file system would map them to one folder.
- **Downstream outputs are not updated.** After replacing converted output, rerun `preprocess` (into a fresh or
  `--overwrite` output root), `manifest`, and `split`. `manifest.jsonl` and `splits.json` are not regenerated for you, and
  `split` still refuses changed content: once a root holds a `splits.json`, preprocess the new cohort into a fresh root
  rather than overwriting that one.
- **One writer per output root.** `--overwrite` does not coordinate simultaneous runs.

## Python API

The functions mirror the CLI one to one. Import them from their modules; the package itself exports nothing.
Paths may be `str` or `Path`. Example using separate directories in approved storage:

```python
from pathlib import Path

from rt_dataprep.convert import convert
from rt_dataprep.manifest import manifest
from rt_dataprep.organize import organize
from rt_dataprep.preprocess import PreprocessSteps, preprocess
from rt_dataprep.split import split

def main():
    workspace = Path('/secure/project')
    organize(workspace / 'export', workspace / 'organized', workers=4)
    conversion = convert(workspace / 'organized', workspace / 'converted', workers=4)
    steps = PreprocessSteps(resample=True, rescale=True, harmonize_spacing=True, reshape=True, new_shape=(320, 320, 256))
    prep = preprocess(workspace / 'converted', workspace / 'preprocessed', steps, workers=4)
    written, refused = manifest(workspace / 'preprocessed')
    cohorts = split(workspace / 'preprocessed' / 'manifest.jsonl', test_size=0.2, val_size=0.2, seed=42)
    # Inspect conversion.rows, prep.rows and refused in approved storage; do not print patient-bearing rows into shared logs.


if __name__ == '__main__':  # required with workers > 1
    main()
```

The Python API default stays `workers=1`; pass `workers='auto'` for the sizing described under
[Parallel patients](#parallel-patients---workers), or an integer from 1 to 64.

With `workers` above 1 (or `'auto'`), each worker is a new `spawn`ed process that imports the calling script. Keep the
calls under an `if __name__ == '__main__':` guard, as above; without it every worker re-runs the script and the stage aborts. With
`workers=1` the patients run in the calling process and no guard is needed.

| Function | Returns | Report written |
| --- | --- | --- |
| `organize(src_root, out_root, *, exclude_models=('Acuity',), workers=1, overwrite=False)` | `OrganizeReport(rows)` of `PatientRow` | `organize_report.csv` |
| `convert(org_root, out_root, *, exclude_models=('Acuity',), workers=1, overwrite=False)` | `ConvertReport(rows)` of `ConvertRow` | `convert_report.csv` |
| `preprocess(converted_root, out_root, steps=PreprocessSteps(), *, workers=1, overwrite=False)` | `PreprocessReport(rows)` of `PreprocessRow` | `preprocess_report.csv` |
| `manifest(root)` | `(written, refused)` case counts | `manifest.jsonl` |
| `split(manifest_path, *, test_size, val_size, seed, external=None)` | the `splits.json` content as a dict | `splits.json` next to the manifest |

Report rows are dataclasses. Their fields match the report CSV columns, except that `organize_report.csv` names
`plans`, `misc` and `unreadable` as `plan_count`, `misc_count` and `unreadable_count`:

| Row | Fields | `outcome` values |
| --- | --- | --- |
| `PatientRow` (organize) | `patient`, `plans`, `misc`, `unreadable`, `outcome`, `reason` | `organized`, `skipped`, `refused` |
| `ConvertRow` | `patient`, `folder`, `outcome`, `reason`, `files`, `skipped_series` | `converted`, `skipped`, `refused` |
| `PreprocessRow` | `patient`, `folder`, `outcome`, `reason`, `files`, `skipped_series`, `cropped` | `preprocessed`, `skipped`, `refused` |

`folder` is blank on a patient-level row, `plan_<n>` on a plan row, and `study_<n>/series_<MOD>_<k>` on an imaging row
(`study_<n>` when a whole study is refused). A convert run also writes one `studies` row per patient that summarizes
its study folders (`skipped` when there are none). `reason` is always one of the fixed `REASON_*` strings defined across the
`rt_dataprep` modules (a convert row can carry one from `masks`, `series` or `dicom_io`); compare against those
constants rather than matching text.

`PreprocessSteps` is a frozen dataclass. Two field names differ from their CLI flags, because the flags are named after
their output prefix:

| `PreprocessSteps` field | CLI flag | Default |
| --- | --- | --- |
| `resample` | `--initial-resample` | `False` |
| `rescale` | `--rescale` | `False` |
| `ahe` | `--ahe` | `False` |
| `harmonize_spacing` | `--resample` | `False` |
| `reshape` | `--reshape` | `False` |
| `new_spacing` | `--new-spacing` | `(1, 1, 1)` mm |
| `final_spacing` | `--final-spacing` | `(1.25, 1.25, 1.25)` mm |
| `new_shape` | `--new-shape` | `(701, 701, 433)` voxels |
| `allow_crop` | `--allow-crop` | `False` |
| `min_mask_retention` | `--min-mask-retention` | `None` |
| `ahe_kernel_mm`, `ahe_bins`, `ahe_clip_limit` | `--ahe-kernel-mm`, `--ahe-bins`, `--ahe-clip-limit` | `None`, `128`, `0.05` |
| `legacy_linear_masks` | `--legacy-linear-masks` | `False` |
| `series_fov_mask` | `--series-fov-mask` | `False` |

The CLI also checks that a grid option comes with its step flag and that spacings are at least 0.1 mm; set those fields
consistently when calling Python directly.

Errors:

- `ValueError` with a fixed reason text refuses the whole call before any patient is processed: an output root inside
  the input, no step enabled (the `steps` default enables none), an invalid option combination, invalid split sizes, a
  manifest with a line missing the `patient`/`case_dir` keys, not JSON, or not UTF-8, sizes that leave no patient
  for train, a patient overlap with `external=`, or a `splits.json` with different content. One exception: organize
  raises `ValueError` mid-run when a patient's output folder resolves outside `out_root`; patients before it are
  already written.
- `OrganizeAborted`, `ConvertAborted` and `PreprocessAborted` mean a patient hit an unexpected error. The report is
  already written for the patients that finished, and the run stops.
- `OSError` on the input or output root propagates. Its message can contain a path, and therefore a patient ID.

A refused patient, plan, or series is **not** an exception; it is a report row. Check `rows` after every call.

The API is under active development. Pin a reviewed commit and `uv.lock` when reproducing an analysis.

## Integrating into a workflow

### Add rt-dataprep to another project

Ask the repository owner for read access while the repository is private; Git must be able to authenticate to
GitHub (an SSH key or `gh auth login`). Then install a tagged release into the training or analysis project's environment,
using Python 3.12 or newer:

```bash
# uv project
uv add "rt-dataprep @ git+https://github.com/ddicostanzo/rt-dataprep@v0.3.0"
# or plain pip in an activated virtual environment
pip install "git+https://github.com/ddicostanzo/rt-dataprep@v0.3.0"
```

Pin a release tag (or a reviewed commit) rather than `main`, and change the tag to upgrade. Releases are listed on the
repository's Tags page.

The install adds an `rt-dataprep` command, which is the same CLI as `python -m rt_dataprep`:

```bash
rt-dataprep --help
rt-dataprep organize /secure/export /secure/organized
```

Inside a uv project, prefix it with `uv run`. The Python API imports as described above
(`from rt_dataprep.convert import convert`, and so on). Every example in this README that starts with
`uv run python -m rt_dataprep` works the same with `rt-dataprep`.

### Gate each stage on its report, not on the exit status

The CLI exits 0 when a run finishes, even when it refused patients, plans, or series. It exits 1 for a fixed-reason
refusal of the whole run, an aborted run, or a file system error on a root, and 2 for invalid arguments. A pipeline step
(Make, Snakemake, a scheduler, CI) should therefore read the report after each stage and decide whether the cohort is
acceptable. A minimal gate in Python:

```python
import csv
import sys

with open('/secure/project/converted/convert_report.csv', newline='') as f:
    refused = sum(row['outcome'] == 'refused' for row in csv.DictReader(f))
if refused:
    sys.exit(f'convert: {refused} refused rows; review the report before continuing')  # counts only, no IDs
```

Through the API, apply the same check to `report.rows`. Decide in advance which refusals are acceptable for the analysis
(for example, imaging series refused for an unsupported volume) instead of requiring zero.

### Rerun rules

- Use a fresh output root for every changed configuration. Existing patient folders are skipped unless `--overwrite` is
  given, so a rerun into the same root with different flags keeps the old results for those patients, with only a `skipped`
  report row to show it.
- Run one writer per output root.
- Regenerate `manifest` after any `preprocess` or `--overwrite` run; it is not updated for you.
- Keep each root's `manifest.jsonl` and `splits.json` as the cohort record. `split` refuses to replace a `splits.json`
  with different content, so a changed cohort belongs in a new root.

See [Replacing existing output with `--overwrite`](#replacing-existing-output-with---overwrite) for the replacement rules.

### Load cases for training

`manifest.jsonl` lists the cases, `splits.json` assigns them to `train`, `val`, and `test`, and each case folder's
`series.json` names its files. Do not derive file names from case IDs: preprocessing adds step prefixes, and imaging
`case_dir` values are IDs, not paths.

```python
import json
from pathlib import Path

import SimpleITK as sitk

root = Path('/secure/project/preprocessed')
splits = json.loads((root / 'splits.json').read_text())
cases_by_id = {row['case_dir']: row for row in map(json.loads, open(root / 'manifest.jsonl', encoding='utf-8'))}


def load(case_dir):
    row = cases_by_id[case_dir]
    if row['case_type'] == 'imaging':  # case_dir is patient/study_<n>/series_<MOD>_<k>
        return {'image': sitk.ReadImage(root / row['patient'] / case_dir.split('/')[1] / row['image'])}
    folder = root / row['patient'] / row['fx']
    files = json.load(open(folder / 'series.json', encoding='utf-8'))
    case = {'image': sitk.ReadImage(folder / next(f['file'] for f in files if f['kind'] == 'ct'))}
    case.update({f['roi_name'] or f['kind']: sitk.ReadImage(folder / f['file'])
                 for f in files if f['kind'] in ('dose', 'mask')})
    return case


train = [load(case) for cases in splits['train'].values() for case in cases]
```

In a plan's `series.json`, `kind` is `ct`, `dose`, `mask`, `series` (a registered image), `applicator`, `line`, `plan`, `beam`, `fluence`, `ion_plan`, or `dwell`.
Applicator and line files are gzip-compressed coordinate text, not NIfTI; see [Conversion behavior](#conversion-behavior).
After a geometric `preprocess` run, the `retained` key gives the fraction of each dose (its voxels >= min(1 Gy, 10% of its max)) and mask (volume)
kept on the final grid. It is below `1.0` only under `--allow-crop`; filter on it if partially cropped structures should
not be used. EXTERNAL masks (the body outline) keep their own value in `series.json` but are left out of the manifest's plan-level
`retained` and of the run log's low-`retained` count. The optional `fov_mask` manifest key and `fov_mask` entries in
the `series` JSON name support masks for excluding fill voxels from losses or metrics.

## Development

```bash
uv sync --locked --group dev
uv run --group dev pytest -q --tb=short
uv run --group dev ruff check .
uv run --group dev python -m compileall -q src examples
```

Optional TorchIO comparison tests skip when TorchIO is unavailable; review skip reasons with `pytest -rs`. Tests generate
synthetic data at runtime. See [CONTRIBUTING.md](CONTRIBUTING.md) for the PHI guard, verification expectations, and PR
workflow. Architectural specifications and implementation plans are under [docs/](docs/).

## License

Copyright (C) 2026 Dominic DiCostanzo. Licensed under the **GNU Affero General Public License, version 3 only**
(`AGPL-3.0-only`); see [LICENSE](LICENSE) for the complete terms. The license includes source-availability obligations when
distributing covered software and, under section 13, when users interact remotely with a modified version that supports
network interaction. See the [GNU AGPL explanation and text](https://www.gnu.org/licenses/agpl-3.0.html).

Third-party dependencies retain their own licenses. The software license does not grant rights to patient datasets.
