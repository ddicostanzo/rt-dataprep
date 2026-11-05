# Examples

Run commands from the repository root after `uv sync --locked --group dev`. Keep all generated data outside the checkout.
The scripts require a new output directory and never delete an existing result.

| Example | Purpose | Data required |
| --- | --- | --- |
| [synthetic_workflow.py](synthetic_workflow.py) | All five pipeline stages on small CT/MR/PT volumes. | None; generates synthetic DICOM at runtime. |
| [custom_preprocess.py](custom_preprocess.py) | Python API configuration with explicit shape/spacing and optional FOV masks. | A converted tree, including the synthetic example's output. |
| [Clinical export workflow](#clinical-export-workflow) | CLI recipe for plan-based and imaging-only patient exports. | Authorized export in approved storage. |

## Synthetic end-to-end walkthrough

```bash
uv run python examples/synthetic_workflow.py /tmp/rt-dataprep-demo
```

Expected final line:

```text
Synthetic workflow complete: 6 patients, 18 imaging cases; train=2, val=2, test=2; no refusals.
```

The new workspace contains `export/`, `organized/`, `converted/`, and `preprocessed/`. For each synthetic patient:

- `export/` contains independent CT/MR/PT image series. CT and MR share one study; PT belongs to another.
- `organized/` contains `misc_files/` because this example has no RTPLAN.
- `converted/` contains `study_1/series_CT_1.nii.gz`, `study_1/series_MR_1.nii.gz`, and
  `study_2/series_PT_1.nii.gz`, with one `series.csv` per study.
- `preprocessed/` contains the same cases on compact 16 × 16 × 8 grids. MR/PT skip the CT HU rescale.
- Both result roots contain manifests and whole-patient splits. Each split contains two patients and six imaging cases.

The script checks aggregate outcomes and refuses unexpected skips. Inputs, voxel values, dates, IDs, and UIDs are generated
for demonstration. These phantoms are not a clinical validation dataset, PET SUV example, or RTPLAN/RTSTRUCT example.

To exercise the CLI independently on the generated export, choose fresh sibling directories:

```bash
uv run python -m rt_dataprep organize /tmp/rt-dataprep-demo/export /tmp/rt-dataprep-cli-organized --workers 2
uv run python -m rt_dataprep convert /tmp/rt-dataprep-cli-organized /tmp/rt-dataprep-cli-converted --workers 2
uv run python -m rt_dataprep manifest /tmp/rt-dataprep-cli-converted
uv run python -m rt_dataprep split /tmp/rt-dataprep-cli-converted --test-size 0.2 --val-size 0.2 --seed 42
```

## Custom preprocessing

Use the API example to select a practical grid rather than the CLI legacy preset's 701 × 701 × 433 grid:

```bash
uv run python examples/custom_preprocess.py /tmp/rt-dataprep-demo/converted /tmp/rt-dataprep-custom --shape 16 16 8 --spacing 1.25 1.25 2
uv run python -m rt_dataprep split /tmp/rt-dataprep-custom --test-size 0.2 --val-size 0.2 --seed 42
```

Dimensions are x/y/z voxels and spacing is x/y/z in mm. The script enables initial resampling, CT HU rescaling, final
spacing, and centered crop/pad; it then writes a manifest. Inspect the resulting geometry and field of view before choosing
a grid for real data. Cropping a plan's nonzero dose or masks can refuse that plan. The script exits with status 1 if cases
or registered series are refused/skipped, or the input yields no cases, so that subsequent automation can stop for review.

`--series-fov-mask` writes masks only for registered series of plan cases; the imaging-only synthetic cohort has no such
plan series. `--help` on either script describes its accepted arguments.

Add `--ahe` to the custom example to include adaptive histogram equalization. To try only AHE through the CLI, without
resampling to the large legacy grid:

```bash
uv run python -m rt_dataprep preprocess /tmp/rt-dataprep-demo/converted /tmp/rt-dataprep-ahe --ahe
uv run python -m rt_dataprep manifest /tmp/rt-dataprep-ahe
```

AHE is opt-in for CT/MR/PT image volumes and never equalizes dose or masks. It min-max scales each volume and applies
CLAHE with 128 bins and clip limit 0.05, producing float32 values in [0, 1]. Those values are no longer HU or SUV. Outputs
receive an `ahe_` prefix. See the [preprocessing guide](../README.md#preprocessing) for step order and memory considerations.

## Clinical export workflow

Use authorized data and approved storage. Replace these illustrative paths; output roots must be outside their inputs.
Each immediate child of the export root must contain one patient's export. Original filenames and DICOM headers remain
sensitive after organization. See [SECURITY.md](../SECURITY.md).

```bash
uv run python -m rt_dataprep organize /secure/export /secure/organized --workers 4
uv run python -m rt_dataprep convert /secure/organized /secure/converted --workers 4
uv run python -m rt_dataprep manifest /secure/converted
uv run python -m rt_dataprep split /secure/converted --test-size 0.2 --val-size 0.2 --seed 42
```

A complete supported RTPLAN → RTSTRUCT → planning CT chain produces `plan_<n>/` output. Other eligible CT/MR/PT series
produce `study_<n>/` output. Imaging-only patients use the same commands and do not need a dummy plan. Registered images
can appear in both a plan and their study; they remain native-grid images during conversion.

Optional legacy preprocessing aligns registered plan series to the final planning CT grid:

```bash
uv run python -m rt_dataprep preprocess /secure/converted /secure/preprocessed --legacy-rnt --series-fov-mask
uv run python -m rt_dataprep manifest /secure/preprocessed
uv run python -m rt_dataprep split /secure/preprocessed --test-size 0.2 --val-size 0.2 --seed 42
```

The legacy shape requires about 812 MiB per float32 volume before processing copies. Review report refusals/skips,
geometry, dose units, masks, and cohort membership before accepting the result. CLI success can include refused cases.
Use fresh roots for a changed configuration; existing patient outputs are skipped. Keep prior manifests and splits for
provenance rather than deleting them to make a different split fit.

All example code is licensed under [AGPL-3.0-only](../LICENSE), like the rest of the project. Generated clinical data is
not covered by a grant of dataset rights.
