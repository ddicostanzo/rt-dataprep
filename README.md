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
| `organize SRC OUT` | Copy a patient export tree into plan folders and `misc_files/`. | `--workers 1..64` (default 1); `--overwrite` |
| `convert ORG OUT` | Convert an organized tree into plan and study outputs. | `--workers 1..64` (default 1); `--overwrite` |
| `preprocess CONVERTED OUT` | Write a separate preprocessed tree. | At least one step flag (`--initial-resample`, `--rescale`, `--ahe`, `--resample`, `--reshape`); also `--legacy-linear-masks`, `--series-fov-mask`, `--overwrite` |
| `manifest ROOT` | Write `ROOT/manifest.csv` and a dated copy. | Works on converted or preprocessed trees. |
| `split ROOT` | Read `ROOT/manifest.csv`; write `ROOT/splits.json`. | Required: `--test-size`, `--val-size`, `--seed`; optional: `--external MANIFEST` |

Use `uv run python -m rt_dataprep COMMAND --help` for exact arguments. Header workers accelerate header reads, not volume
conversion; changing the worker count preserves output ordering. Study exclusion defaults to a case-sensitive
`Acuity` substring in `ManufacturerModelName`; `organize()` and `convert()` expose `exclude_models` through the Python API.

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
      series.csv
    study_1/
      series_CT_1.nii.gz
      series_MR_1.nii.gz
      series.csv
    study_2/
      series_PT_1.nii.gz
      series.csv
  SYNTHETIC_P002/
    study_1/
      series_CT_1.nii.gz
      series.csv
```

Plan numbering reflects the organized plan order. Study folders are numbered by the earliest StudyDate/StudyTime among
their written series, with blank dates/times last and StudyInstanceUID as a tie breaker. Series numbers are per modality
within each destination. `series.csv` records the actual output files and metadata; inspect it instead of inferring
relationships from filenames.

Trees created with the old `images/` conversion layout must be re-converted. Current manifest and preprocessing walkers
use `study_<n>/` and ignore the old `images/` layout.

## Conversion behavior

- A plan requires one RTPLAN, its referenced RTSTRUCT, and a complete, supported planning CT volume. A plan's planning CT
  is written once in that plan folder and is not repeated as an imaging study.
  The planning CT path requires an axial volume with consistent orientation, image dimensions, and uniform slice spacing;
  gantry tilt is refused. Other imaging series may be oblique but must still form a regular single-frame volume.
- Planning CT is float32 HU. Supported physical RTDOSE is converted to Gy and resampled onto its planning CT grid;
  supported beam-dose inputs are summed. Ambiguous or unsupported dose combinations refuse the plan.
- Kept RTSTRUCT ROIs become binary masks on the planning CT grid. Empty, mismatched, or inconsistent mask results refuse
  the plan. Point-only ROIs and names containing `dose` are excluded. Brachy applicator outputs are **gzip-compressed
  coordinate text**, despite their legacy `.nii.gz` extension; consult `kind` in `series.csv` before opening a file as NIfTI.
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
| `--initial-resample` | Resample to 1 mm spacing | `initialresample_` |
| `--rescale` | Apply `(HU + 1024) / 4096` to CT | `rescale_` |
| `--ahe` | CLAHE adaptive histogram equalization | `ahe_` |
| `--resample` | Resample to 1.25 mm spacing | `resample_` |
| `--reshape` | Centred crop/pad to **701 × 701 × 433 voxels (x, y, z)** | `reshape_` |

Steps always run in the order of the table, whichever flags are given. One float32 image at the `--reshape` size occupies
about **812 MiB** before processing copies and other volumes; plan memory and storage accordingly. The CT formula is not
intensity clipping. `--reshape` without `--initial-resample` or `--resample` crops/pads at the CT's own spacing, which can
pad a large extent at native spacing.

CT study series receive all enabled steps. MR/PT study series receive geometric steps and optional AHE, and skip HU rescaling. `--rescale`
alone (or the same Python configuration) refuses MR/PT because it has no applicable step. With no geometric step (`--rescale` and/or
`--ahe` only), dose and masks are copied unchanged. Registered plan series are resampled onto
the final planning CT grid; unusable alignment, no overlap, or unreadable data skips that series and increments the plan's
`skipped_series`. Output `reg_matrix` is `identity` after alignment.

Masks use nearest-neighbor interpolation and uint8 by default. `--legacy-linear-masks` requests the legacy linear float32
behavior. `--series-fov-mask` adds a uint8 `fovmask_<image filename>` for the planning CT, each study image, and aligned
plan series. A value of 1 marks acquired support and 0 marks outside-FOV fill or crop/pad fill. Planning/study masks follow
each geometric step with nearest-neighbor interpolation and zero padding; true zero-valued image voxels remain valid.
Auxiliary masks do not add rows to `series.csv`. The manifest lists them only when their headers match the relevant image grid.

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
memory planning; real-data peak-memory profiling remains tracked in #47. Tune the histogram or use a physical window:

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

`manifest.csv` contains one row per plan and one row per imaging series:

| Columns | Meaning |
| --- | --- |
| `case_dir`, `patient`, `fx` | Case ID, patient directory, and plan directory (`fx` is blank for imaging). |
| `ct_size`, `ct_spacing` | Semicolon-separated x/y/z dimensions and mm spacing of the case image. |
| `dose`, `masks` | Dose-presence flag and JSON ROI-name list; imaging rows have `0` and `[]`. |
| `rtplan_uid`, `ct_series_uid`, `study_uid` | Original identifiers; imaging uses its own series UID in `ct_series_uid`. |
| `case_type`, `modality`, `image` | `plan` or `imaging`; image modality; volume filename for imaging (blank for plans). |
| `series` | JSON list of aligned plan series (`file`, `modality`, optional `fov_mask`); `[]` for imaging. |
| `fov_mask` (optional trailing column) | Planning-CT or study-image support-mask filename, relative to its case folder; blank when absent. |

The trailing `fov_mask` column appears only when at least one usable primary-image mask is present in the tree. Without
those masks, the original fourteen-column schema is retained. Aligned-series masks remain in their `series` JSON entries.

An imaging `case_dir`, for example `SYNTHETIC_P001/study_1/series_MR_1`, is an ID, **not a directory path**. Resolve its image
as `ROOT / patient / study_folder / image`. Plan image names come from the plan's `series.csv`.

`split` keeps every case of a patient in one split. It deterministically orders patients using a hash of seed and patient
ID, allocates `ceil(n × test_size)` patients to test and `ceil(n × val_size)` to validation (with numeric rounding before
ceil), then assigns the remainder to training. Both sizes must be at least 0 and below 1, and at least one training patient
must remain. Small cohorts can therefore need smaller fractions.

`splits.json` contains `train`, `val`, and `test` maps of patient to case-ID lists, plus `case_id_hash` values. Repeating an
identical split is accepted; an existing split with different content is refused. Cohort changes, including newly added
imaging patients, can change assignments despite the same seed.

Check an external cohort for overlapping patient strings before writing a split:

```bash
uv run python -m rt_dataprep split /secure/converted --test-size 0.2 --val-size 0.2 --seed 42 --external /secure/external/manifest.csv
```

This checks patient IDs, not cross-institution identity linkage or clinical stratification. Maintain consistent patient
mapping and perform any required outcome/site/time stratification separately.

## Reports, reruns, and troubleshooting

Organization, conversion, and preprocessing write their report under the output root, alongside dated per-run copies;
manifest generation also keeps dated copies. Log and exception text uses counts, fixed refusal reasons, or safe error
class/module/line details. Reports and per-folder CSVs are sensitive artifacts and are excluded from Git.

| Symptom | What to check |
| --- | --- |
| A patient is skipped on rerun | Its destination patient folder already exists. Add `--overwrite` to replace it (see below), or use a fresh output root to recompute. |
| A plan is refused | Read its fixed reason in the conversion report; verify the RTPLAN → RTSTRUCT → CT reference chain and geometry. |
| Fewer images than expected | Review study summary skips, series eligibility, model exclusions, localizers, and readable pixel data. |
| Preprocessing reports no step is on | Pass a step flag (`--initial-resample`, `--rescale`, `--ahe`, `--resample`, `--reshape`), or enable applicable `PreprocessSteps` in Python. |
| A series or plan is missing from the manifest | Check the refusal count, `series.csv`, file presence, and image headers. |
| Split refuses existing content | Preserve the old cohort record and generate the new cohort in a separate output root. |
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
  as kept. Each summary line ends with `N replaced, N kept`.
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
  `--overwrite` output root), `manifest`, and `split`. `manifest.csv` and `splits.json` are not regenerated for you, and
  `split` still refuses changed content, so write the new cohort record to a separate output root.
- **One writer per output root.** `--overwrite` does not coordinate simultaneous runs.

## Python API

The functions mirror the CLI. Example using separate directories in approved storage:

```python
from pathlib import Path

from rt_dataprep.convert import convert
from rt_dataprep.manifest import manifest
from rt_dataprep.organize import organize
from rt_dataprep.split import split

workspace = Path('/secure/project')
organize(workspace / 'export', workspace / 'organized', workers=4)
conversion = convert(workspace / 'organized', workspace / 'converted', workers=4)
written, refused = manifest(workspace / 'converted')
cohorts = split(workspace / 'converted' / 'manifest.csv', test_size=0.2, val_size=0.2, seed=42)
# Inspect conversion.rows and refused in approved storage; do not print patient-bearing rows into shared logs.
```

The API is under active development. Pin a reviewed commit and `uv.lock` when reproducing an analysis.

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
