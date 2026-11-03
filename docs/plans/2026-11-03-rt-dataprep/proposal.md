# RT → ML data pipeline: extraction proposal

**Date:** 2026-11-03 · **Status:** proposal, nothing has moved yet · **Sources:** `rnt_transformer` @ `1076ff8`,
`DICOM_Transport` @ `5eff3f8`, `vision_transformer` (consumer evidence)

## Goal

One package that takes an Aria DICOM export and turns it into ML-ready, checked training cases (CT and masks
as NIfTI, dose in Gy, a manifest, and patient-level splits). Each RT ML project then begins at "load cases",
not at "parse DICOM".

## Findings

1. **No code in either repo connects them.** `DICOM_Transport` writes `<out>/<sanitized PatientID>/<prefix>.<SOP>.dcm`,
   flat for each patient (`common.get_patient_output_directory`, `common.get_filename`). `rnt_transformer` reads
   per-fraction folders (`<mrn>_<x>_<fx>/` holding `ct.nii.gz`, `<structure>.nii.gz` and `dwells.npz`, or the older
   `reshape_resample_ahe_rescale_initialresample_*` files; see `src/data/loader.py`). The step in between was a
   **legacy exporter that was never committed** (`docs/specs/2026-11-02-h2-gauge-rr.md:12`,
   `docs/plans/h2-gauge-rr/legacy-reference.md:3`). It grouped the dump into fractions, wrote the CT as NIfTI and
   rasterized RTSTRUCT into masks. The same applies to the `raw/<mrn>/fx_<n>/` layout that `convert_dose.py` reads:
   no code in either repo creates it.
2. **That step has been written twice already.** `vision_transformer/src/sequential_lovelace/cli/prepare_data.py`
   walks a DICOM root, pairs CT with RTPLAN, and writes `cases/<id>/image.nii.gz`, a `manifest.parquet` and a
   patient-grouped `splits.json` (`data/splits.py`). It overlaps with rnt_transformer's version but does not share
   code with it. So there is real duplication, and at least two consumers exist today.
3. **The generic code already exists in pieces.** The pieces are mostly free of torch and Lightning, but they live
   inside brachy-specific modules (see the source map).

## Proposed package: `rtml-data` (name TBD)

Python ≥3.12. Runtime deps: `pydicom`, `numpy`, `SimpleITK` and `pandas`, plus one RTSTRUCT rasterizer (decision
below). **Not** torch, Lightning, pyodbc or pynetdicom.

```
DICOM_Transport (stays a tool)        rtml-data
<out>/<patient>/*.dcm  ──►  index ──► assemble ──► convert ──► check ──► manifest + splits
                            headers   CT/RTSTRUCT/  CT→NIfTI    geometry   one row per case,
                            by        RTPLAN/       RTDOSE→Gy   + ref      patient-grouped split
                            modality  RTDOSE by     on CT grid  checks     + split hash
                                      referenced    RTSTRUCT→
                                      UIDs          masks (new)
```

Stage boundaries are plain files on disk (NIfTI and a manifest), so each project reads cases with its own
loader and keeps its own model code.

### Source map

| Stage | Function(s) | From | Generic as-is? |
|---|---|---|---|
| index | `read_folder` (header-only, by modality) | rnt `src/data/gauge_rr.py` | yes; widen the modality set to add RTDOSE and REG |
| index | `read_dicom`, `_pydicom_quiet` | rnt `src/dose/rtdose.py` | yes |
| assemble | `_referenced_ct_slices` (plan → struct → CT series by UID), `rtstruct_for` | rnt `rtdose.py`, `gauge_rr.py` | yes, once `DoseRefused` becomes a neutral `CaseRefused(reason)` |
| assemble | `match_dx_reg_to_plan_ct_reg` (logic only) | DT `src/sql/data/helpers.py` | needs a port: it works on SQL rows today |
| convert | `read_ct_series`, `read_rtdose` / `_convert`, `resample_dose` | rnt `rtdose.py` | yes; `read_ct_series` depends on `utils.augment._read`, which comes along |
| convert | `write_atomic` | rnt `scripts/convert_dose.py` | yes |
| convert | RTSTRUCT → mask on the CT grid | **missing** (legacy exporter) | new code |
| check | `ct_geometry`, `geometry_equal` | rnt `gauge_rr.py` | yes |
| check | `check_ct_match` (raw vs. converted CT, Pearson plus shifts) | rnt `rtdose.py` | yes |
| check | completeness: expected instances vs. those that landed | DT `src/verify_export.py` | yes; it takes the layout and modality map as arguments and imports only `common` and `export_filter` |
| geometry | `index_to_physical`, `mask_voxel_*` | rnt `src/dose/frame.py` | yes |
| metrics | `dvh_dose`, `dvh_metrics`, `voxel_volume_mm3` | rnt `src/dose/dvh.py` | yes (numpy only) |
| split | patient-grouped split plus a sha256 of the sorted IDs, and a refusal on external-set overlap | rnt `rnt_datamodule._split_by_patient`, `_setup_external_test`; vt `data/splits.py` | yes, after lifting it out of the DataModule |
| safety | `sanitize_path_component`, `SOP_CLASS_PREFIXES` / `get_filename` | DT `src/common.py` | yes |
| tests | synthetic CT, RTSTRUCT, RTPLAN and RTDOSE writers; synthetic NIfTI | rnt `tests/synthetic_dicom.py`, `tests/synthetic_nifti.py` | yes, and this is the main thing that makes the package testable without PHI |

### Stays in its project

- **rnt_transformer:** `dose/tg43.py`, `dose/source.py` (GammaMed TG-43), `data/plan_dwells.py` (30 ring-and-tandem
  dwells), `rtdose.check_plan_match` / `check_hotspot` (dwell-specific), `gauge_rr` tree and key logic,
  `analysis/h2|section6|course`, models, `item_format` and the GPU augmentation path.
- **DICOM_Transport:** the whole Aria SQL → C-MOVE → SCP tool. It is an application, so it stays one. The package
  only reads its output layout.
- **Second wave (not this package):** `training/provenance.py` and `run_dir.py` (Lightning run scaffolding) and the
  pure stats in `analysis/section6.py` (cluster bootstrap, TOST, Holm, Bland-Altman). These are generic, but a
  different audience uses them, and they have no DICOM in them.

## Constraints

- **PHI:** the new repo needs phi-guard and the pre-commit hook from its first commit. Fixtures are generated
  (`SYNTHETIC_` prefix or under `tests/fixtures/`), and no `.dcm` files are committed. Refusal messages are fixed
  strings that contain no identifiers, which is the convention rnt already follows.
- **Python version:** DT pins ≥3.13 and rnt runs 3.12, so the package targets 3.12.
- **Test convention:** use real pydicom and SimpleITK with the synthetic writers (rnt style), not DT's
  `sys.modules` stubs. DT does not import the package, so its thin CI image stays as it is.
- **Clinical output:** CT/dose/mask geometry and the case-assembly UID matching feed training labels and dose
  analysis. These get full rigor and acceptance tests against rnt's existing `pooled` outputs.

## Decisions (2026-11-03)

1. **RTSTRUCT rasterizer:** DicomRTTool (`DicomReaderWriter(arg_max=True)`). This corrects the earlier note that
   said rt-utils: the legacy code shows DicomRTTool. Masks are resampled with nearest-neighbour by default, which
   gives binary {0, 1} masks. A `legacy_linear_masks` flag reproduces the fractional masks the legacy code wrote,
   for the parity test.
2. **Case layout:** a folder for each patient, with one subfolder for each plan: `<out>/<patient>/<plan>/`, which
   for the RNT data is `<MRN>/fx_<n>/`. This matches the `raw/<mrn>/fx_<n>/` layout that `convert_dose.py` already
   reads. Train/val/test splits group by the patient folder.
3. **Home:** a new private repo, `rt-dataprep` (package `rt_dataprep`), PHI-guarded from its first commit.
5. **Resampling and reshape:** ported from torchio to SimpleITK, so the package has no torch dependency. The
   parity test compares to legacy within a tolerance.
6. **Dependencies (checked 2026-11-03 on Python 3.12):** `DicomRTTool==6.2.0` installs cleanly (opencv-headless).
   Its API has changed since the legacy code: the keyword is now `contour_names=`, and names are lowercased. `mclahe`
   pulls in TensorFlow and pins numpy 1.26, so it becomes the optional `[ahe]` extra and is not a core dependency.
7. **Layout for rnt:** an optional `--flat` output (`<mrn>_fx_<n>/`) keeps rnt's loader unchanged.
4. **Preprocessing:** the legacy reshape → resample → AHE → rescale steps are each switched on by a flag, and the
   default output is native-grid NIfTI. The flag defaults come from the legacy exporter's parameters. The user is
   supplying that code, and it will not be committed.

## Legacy pipeline (from `conversion_utils.py`, `patient.py`, and `utils.py`, supplied 2026-11-03, not committed)

**Order and defaults.** Each step is applied in this order, and adds a prefix to the output filename. The newest
step's prefix goes first.

| Step | Flag (default) | Parameters (default) | Applies to | Prefix |
|---|---|---|---|---|
| initial resample | `resample` (False) | `new_spacing` (1, 1, 1) mm, torchio `Resample`, linear | CT; dose onto the CT | `initialresample` |
| rescale | `rescale` (False) | `(HU + 1024) / 4096` | CT | `rescale` |
| normalize | `norm` (False) | `sitk.NormalizeImageFilter` | CT | `norm` |
| AHE | `ahe` (False) | `mclahe`, `kernel_size` None (1/8 of size), `n_bins` 128, `clip_limit` 0.05 | CT | `ahe` |
| final spacing | `harmonize_spacing` (**True**) | `final_spacing` (1.25, 1.25, 1.25) mm | CT; dose onto the CT | `resample` |
| reshape | `reshape` (False) | torchio `CropOrPad` `new_shape` (701, 701, 433), padded with the image's minimum | CT, dose | `reshape` |

The RNT training files (`reshape_resample_ahe_rescale_initialresample_planning_ct`) were produced with `resample`,
`rescale`, `ahe` and `reshape` all set to True.

**Behaviour that parity depends on:**
- Structure masks are made on the native CT grid and resampled once onto the final CT. The interpolation is
  **linear**, because `is_label_map` is never passed. That means legacy masks hold fractional values, not
  {0, 1}. Masks are resampled only when `resample=True`.
- Applicator channels (`OPEN_NONPLANAR`, or a name containing ring, tandem or applicator) are written with
  `np.savetxt` as `<SourceApplicatorID letters>.nii.gz`. They are **text files of contour points**, not NIfTI.
- `*_plan.nii.gz` is a **pandas pickle** of the brachy control points, with continuous indices on the final CT.
  This part stays in rnt_transformer.
- Dose is the first RTDOSE reached by the chain CT → RTSTRUCT → RTPLAN → RTDOSE (`doses_for_planning_ct(ct)[0]`).
- Output folder is `<PatientID>_<fraction dir name>`. A folder that already exists is skipped.
- The mask rasterizer is **DicomRTTool** (`dicomrttool.DicomReaderWriter(arg_max=True)`, run over the whole
  fraction folder). It is not rt-utils, which changes decision 1.
- After `ahe`, `multidim_clahe` returns a torchio image, and the `resample` and write steps that follow expect
  SimpleITK. How the legacy run got past this is unclear; one possibility is the reuse of a cached
  `ahe_*` file. A parity test against the legacy files will show whether it matters.

**Fraction rule** (`organize.py`, `folder.py`, `move_converted_data.py`, supplied 2026-11-03):
- A **planning CT** is a CT series that some RTSTRUCT references, where that RTSTRUCT is referenced in turn by an
  RTPLAN. Planning CTs are sorted by `SeriesDate`.
- For each planning CT, every file in its **study** goes to the next `fx_<n>` folder, with n counting up from 1.
  If a REG in that study names another study (`StudiesContainingOtherReferencedInstancesSequence`), and that other
  study is not itself a planning study, its files go in the same folder. That study is the diagnostic imaging.
- Anything left over goes to `misc_files/`. Patients with no planning study get no `fx` folders.
- Before grouping, `move_topograms_and_image_by_model` moves two kinds of file into `misc_files/`: CT files under
  500 kB or over 600 kB (a file-size test for topograms), and whole studies whose `ManufacturerModelName` contains
  `Acuity`.
- A keep-list CSV of `(mrn, fx)` pairs (`PatientFractionMapForUse.txt`) moves fractions that are not on it into an
  archive.
- Converted output `<patient>/fx_<n>/converted/` is flattened to `<out>/<mrn>_fx_<n>/`.

**Legacy quirks we will not copy:**
- The topogram test uses file size. The new package uses `ImageType` or slice count instead.
- If two plans share one CT, the same study is listed twice, and the second pass creates an empty `fx` folder.
- Paths are parsed with `split('_')`, which breaks on MRNs or folder names that contain `_`.

## Phase 1 (happy path, once the decisions above are made)

1. Create the package with index → assemble → convert (CT, RTDOSE) → check → manifest, moved from the rnt sources
   above with their tests.
2. Add RTSTRUCT → masks.
3. Add the split module, with the hash and the overlap refusal.
4. Acceptance: on synthetic DICOM, produce a case folder that rnt's `per_fraction_dataframe` (npz layout) loads.
   On real data (OSC, not committed), regenerate `ct.nii.gz` and masks for N pooled fractions and pass
   `geometry_equal` plus a mask Dice ≥ threshold against the legacy files.
5. Switch rnt_transformer to import from the package (rtdose, frame, dvh, read_folder), with no change in behaviour.
   Existing tests are the guard.

## Deferred

- Porting `vision_transformer/prepare_data.py` onto the package (second consumer, after Phase 1).
- Registration (REG) handling for multi-CT cases.
- Second-wave provenance and stats package.
- A typed RT entity model shared with DT's `sql/data/instance.py` hierarchy.
