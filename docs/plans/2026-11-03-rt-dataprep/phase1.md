# rt-dataprep Phase 1: DICOM export → ML-ready cases

**Date:** 2026-11-03 · **Proposal and decisions:** `proposal.md` (same folder) · **Sources:** `rnt_transformer` @ `1076ff8`,
`DICOM_Transport` @ `5eff3f8`. The legacy pipeline is described in `proposal.md`; its code was supplied by the user
and is **not** committed.

## Goal

Take a DICOM_Transport export (`<root>/<PatientID>/*.dcm`) and produce per-plan case folders holding the CT, dose
(Gy), structure masks and a manifest, with optional legacy preprocessing and a patient-grouped split. The output
must be loadable by rnt_transformer without changes to rnt.

## Package

- Python ≥3.12, `src/rt_dataprep/`, built with hatchling. Core deps: `pydicom`, `numpy`, `pandas`, `SimpleITK`,
  `DicomRTTool==6.2.0`, `scikit-learn` (for `GroupShuffleSplit` only). Optional extra `[ahe]` = `mclahe`. Dev:
  `pytest`, `ruff`.
- CLI is `python -m rt_dataprep <organize|convert|split>`, using argparse. Each subcommand wraps one function with
  the same name, which tests call directly.
- Errors and refusals use one exception, `CaseRefused(reason)`. `reason` is a fixed string constant, and messages
  never contain a PatientID, name, date or path. A per-case refusal is counted in the run report and the run moves on
  to the next case. Anything else propagates.
- Logs go to the output directory only, never the working directory.
- Tests use real pydicom and SimpleITK on synthetic DICOM. `tests/synthetic_dicom.py` is copied from rnt
  `tests/synthetic_dicom.py` and gains `study_uid=` / `series_date=` parameters plus a `write_reg` helper.

The work is cut into three `/dev`-sized features, implemented and merged in order. **F1 → F2 → F3.**

---

## F1: organize (`rt_dataprep/organize.py`)

`organize(src_root, out_root, *, exclude_models=('Acuity',)) -> OrganizeReport`

Steps:
1. For each patient folder under `src_root`, read every `*.dcm` header (`stop_before_pixels=True`). The header reader
   is `read_dicom` / `_pydicom_quiet`, copied from rnt `src/dose/rtdose.py`. An unreadable file is counted and skipped.
2. Find the planning CTs: each RTPLAN → the RTSTRUCT whose `SOPInstanceUID` equals the plan's
   `ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID` → the CT series that RTSTRUCT references (the logic of
   rnt `_referenced_ct_slices`). Remove duplicate CT series, then sort by `(SeriesDate, SeriesTime,
   SeriesInstanceUID)`.
3. In that order, the n-th planning CT's **study** becomes `fx_<n>`, numbered from 1. The folder gets every file with
   that `StudyInstanceUID`. It also gets every file of a study named by a REG in that study
   (`StudiesContainingOtherReferencedInstancesSequence[*].StudyInstanceUID`), unless that study is itself a planning
   study.
4. Before grouping, two kinds of file are treated as left over: CT files whose `ImageType` contains `LOCALIZER`
   (topograms), and whole studies where any file's `ManufacturerModelName` contains one of `exclude_models`.
5. Files are **copied** (`shutil.copy2`), never moved, so the source stays intact. Output is
   `<out_root>/<sanitized PatientID>/fx_<n>/<original filename>`. Every file that does not land in an fx folder goes
   to `<out_root>/<patient>/misc_files/`. `sanitize_path_component` comes from DT `src/common.py`, and the resolved
   path is asserted to sit inside `out_root`.
6. A patient whose output folder already exists is skipped and counted in the report. The run does not merge into it.
7. `organize_report.csv` is written in `out_root` with one row per patient: the sanitized ID, fx count,
   misc_files count and unreadable count. The hook blocks this file from commits.

Acceptance tests (synthetic):
- **A1.1** One patient with two planning studies on different dates. Each study has a CT, RTSTRUCT, RTPLAN and
  RTDOSE. Result: `fx_1` holds the earlier study and `fx_2` the later one, and the file counts match.
- **A1.2** A REG in the planning study references a diagnostic MR study. The MR files land in that study's `fx`
  folder.
- **A1.3** A REG references the other planning study. Neither study's files are duplicated into the other's folder.
- **A1.4** A localizer CT, an `Acuity` study, and an orphan CT with no plan all go to `misc_files/`.
- **A1.5** Two RTPLANs reference the same RTSTRUCT and CT. The result is one `fx` folder, with no empty folder.
- **A1.6** A PatientID of `../evil` is written inside `out_root`.
- **A1.7** After a run, the source tree is byte-identical (compared by hash of every file).

---

## F2: convert (`rt_dataprep/convert.py`, `preprocess.py`, `masks.py`)

`convert(org_root, out_root, *, flat=False, steps=ConvertSteps()) -> ConvertReport`

`ConvertSteps` is a frozen dataclass of the legacy flags and parameters, with the defaults in the table in
`proposal.md`. It differs from the legacy code in three ways: the default is **no preprocessing**
(`harmonize_spacing=False`), `legacy_linear_masks=False`, and `ahe` requires the `[ahe]` extra. A
`--legacy-rnt` CLI preset sets `resample`, `rescale`, `ahe`, `harmonize_spacing` and `reshape`, which reproduces
`reshape_resample_ahe_rescale_initialresample_planning_ct`.

Steps, for each `<patient>/fx_<n>/`:
1. **Assemble.** Index headers by modality, as in rnt `gauge_rr.read_folder`, widened to RTDOSE and REG. Then
   follow the chain CT ← RTSTRUCT ← RTPLAN ← RTDOSE by referenced UIDs. Refuse the case unless there is exactly one
   planning chain with exactly one CT series. Dose is the RTDOSE whose `ReferencedRTPlanSequence` names the plan. If
   there is none, the case has no dose and is not refused.
2. **CT.** Read with `read_ct_series`, from rnt `rtdose.py` together with the `utils.augment._read` it depends on,
   restricted to the planning CT's files.
3. **Dose.** `read_rtdose` / `_convert` from rnt `rtdose.py`, giving Gy with `DoseGridScaling` applied. It is then
   resampled onto the native CT grid with `resample_dose`. The legacy code never scaled dose to Gy; this
   deliberately corrects that.
4. **Masks.** Use `DicomReaderWriter(arg_max=True)` on a temporary folder of hard links to only the planning CT
   slices and the RTSTRUCT, so other series in the fx folder cannot be picked up. Then call
   `set_contour_names_and_associations(contour_names=[roi])` and `get_images_and_mask()` for each ROI.
   `annotation_handle` must match the CT geometry (`geometry_equal`, from rnt `gauge_rr.py`), otherwise the ROI is
   refused.
   - Skip: `POINT` ROIs, empty ROIs, and names containing `dose` (legacy rule).
   - Applicator channels: ROIs that are `OPEN_NONPLANAR` or have `RTROIInterpretedType == BRACHY_CHANNEL`. Each is
     written as `<SourceApplicatorID letters>.nii.gz`, a **text** file of contour points (`np.savetxt`,
     `fmt='%.4f'`), which keeps rnt's `np.loadtxt` working.
5. **Preprocess** (`preprocess.py`, SimpleITK only). Steps run in the legacy order: initial resample (linear) →
   rescale `(HU+1024)/4096` → norm → AHE → final spacing → reshape. Reshape is a centred crop/pad: `RegionOfInterest`
   followed by `ConstantPad`, with the CT padded by its minimum and dose/masks padded by 0. Dose follows the CT grid.
   Masks are resampled **once**, onto the final CT, using nearest neighbour (linear when `legacy_linear_masks`).
   This happens whenever any geometric step is on, which fixes the legacy bug where masks were only resampled when
   `resample` was on.
6. **Names.** Use legacy prefixes, newest step first (`create_file_name`): `planning_ct`, `dose`, and
   `<format_name(roi)>`, where `format_name` lowercases and strips non-alphanumerics, as in legacy `utils.py`. All
   are `.nii.gz`. The output folder is `<out_root>/<patient>/fx_<n>/`, or `<out_root>/<patient>_fx_<n>/` with
   `flat`. A folder that already exists is skipped. Each file is written to a temporary name and renamed into place
   (`write_atomic` from rnt `scripts/convert_dose.py`), so a crash never leaves a half-written case.
7. `convert_report.csv` is written in `out_root`, with one row per case: outcome, refusal reason and the list of
   files written.

Acceptance tests (synthetic):
- **A2.1** A synthetic fx folder with no steps gives `planning_ct.nii.gz` equal to the source CT (same spacing,
  origin, direction and values) and `dose.nii.gz` in Gy (stored value × scaling) on the CT grid. This reuses rnt
  `tests/test_rtdose.py` expectations where they apply.
- **A2.2** A cube ROI drawn on the synthetic CT gives a binary mask whose voxel count equals the cube volume (±1 slice
  edge), on the CT grid.
- **A2.3** A BRACHY_CHANNEL ROI gives a text file that `np.loadtxt` reads back into the contour points (tolerance
  1e-4).
- **A2.4** A diagnostic MR study in the same fx folder does not change the outputs.
- **A2.5** `--legacy-rnt` without `ahe` gives the legacy filename set, and a crop/pad to the requested shape.
  Masks are {0, 1} by default, and fractional values appear only with `legacy_linear_masks`.
- **A2.6** SimpleITK resample and crop/pad match a torchio reference within 1e-4. The test is skipped unless
  torchio is installed, and torchio is never a dependency.
- **A2.7** `flat=True` output is accepted by rnt's `per_fraction_dataframe` (legacy layout) with the matching
  filename kwargs. This is checked by vendoring the 30-line folder-walk contract as a test helper, not by importing
  rnt.
- **A2.8** A case whose plan references a missing RTSTRUCT is refused. The report row carries the reason constant,
  and no output folder is left behind.

---

## F3: manifest and split (`rt_dataprep/manifest.py`, `split.py`)

Steps:
1. `manifest(out_root)` writes `manifest.csv`, with one row per converted case. Columns: `case_dir`, `patient`,
   `fx`, CT size and spacing, dose present, and the list of masks. It also records `rtplan_uid`, `ct_series_uid`
   and `study_uid` for cross-run matching. The hook blocks this file from commits.
2. `split(manifest, *, test_size, val_size, seed)` uses `GroupShuffleSplit` grouped by `patient`, the same method as
   rnt `_split_by_patient`. It writes `splits.json` with `{train, val, test}` case lists and a
   `case_id_hash = sha256('\n'.join(sorted(ids)))` for each split, the same hash convention as rnt
   `_setup_external_test`.
3. `check_disjoint(splits, external_manifest)` refuses if any patient overlaps.

Acceptance tests:
- **A3.1** No patient appears in more than one split. The same seed and manifest give an identical `splits.json`.
- **A3.2** An external manifest that shares a patient is refused, and the reason names no patient.

---

## Parity check on real data (OSC, run by the user, after F2)

`scripts/parity.py --legacy <pooled dir> --new <convert out> --org <organize out>` matches cases on
`rtplan_uid`. It never matches on `fx_<n>`, because the legacy numbering depended on pre-filters and a keep-list.
It prints aggregate counts only, with no identifiers. The run uses `--legacy-rnt --legacy-linear-masks` on N ≥ 20
fractions. Pass criteria:
- CT: `geometry_equal`, and voxel max abs diff ≤ 1e-3 on the rescaled scale. If `ahe` is on, it is checked
  separately with a Pearson r ≥ 0.999, because of the uncertainty about the legacy torchio/SimpleITK handoff after
  AHE.
- Masks: Dice ≥ 0.98 for each OAR after thresholding at 0.5.
- Applicator text files: point arrays equal within 1e-3.
- Dose is **not** compared to legacy (legacy was unscaled). It is compared to rnt's `dose_gy.nii.gz` with
  max abs diff ≤ 1e-4 Gy after resampling to the same grid.

A failure here is a finding to bring back to the user before rnt switches over. It does not block F3.

## Out of scope for Phase 1

- Plan, control-point and dwell extraction (`plan.pkl`, `dwells.npz`): this stays in rnt (`plan_dwells.py`).
- Switching rnt_transformer or vision_transformer to import the package.
- The keep-list filter (`PatientFractionMapForUse.txt`): that is cohort selection and belongs in each project.

## Deferred

- REG-based resampling of diagnostic MR or PET into the planning CT frame.
- Parallel reading (the legacy code used `Pool(6)`).
- Multi-plan cases (more than one RTPLAN per study) are refused rather than split.
- Same-day planning CTs: the order is by `SeriesTime` and then UID, which may differ from the legacy numbering.
- `normalize`, which the RNT data never used. It is ported but has only a smoke test.
- Second-wave package: provenance, run_dir and the section 6 stats.
