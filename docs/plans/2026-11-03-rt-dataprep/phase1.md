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
  `DicomRTTool==6.2.0`. Optional extra `[ahe]` = `mclahe`. Dev:
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
2. The case unit is one **RTPLAN** (user decision, 2026-11-03). For each RTPLAN, find the RTSTRUCT whose
   `SOPInstanceUID` equals the plan's `ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID`, then the CT series
   that RTSTRUCT references (the logic of rnt `_referenced_ct_slices`). A plan without a complete chain is left over.
   Sort plans by `(CT SeriesDate, CT SeriesTime, RTPlanDate, RTPlanTime, plan SOPInstanceUID)`. With one plan per CT
   this gives the same numbering as the legacy code.
3. In that order, the n-th plan becomes `plan_<n>`, numbered from 1. The folder gets:
   - the plan itself and every RTDOSE whose `ReferencedRTPlanSequence` names it. These files go to that folder only.
   - the RTSTRUCT the plan resolves to (step 2), always, even when it is stored in a study other than the CT's.
   - **copies** of every other file in the plan's CT study, except other RTPLANs and their RTDOSEs. This covers the
     CT, the RTSTRUCTs and the REGs. Several plans on one CT therefore each get their own CT and RTSTRUCT.
   - copies of every file of a study named by a REG in the CT study
     (`StudiesContainingOtherReferencedInstancesSequence[*].StudyInstanceUID`), unless that study is itself a
     planning study. That study is the diagnostic imaging.
4. Before grouping, two kinds of file are treated as left over: CT files whose `ImageType` contains `LOCALIZER`
   (topograms), and whole studies where any file's `ManufacturerModelName` contains one of `exclude_models`.
5. Files are **copied** (`shutil.copy2`), never moved, so the source stays intact. Output is
   `<out_root>/<sanitized PatientID>/plan_<n>/<original filename>`. Every file that does not land in a plan folder goes
   to `<out_root>/<patient>/misc_files/`. `sanitize_path_component` comes from DT `src/common.py`, and the resolved
   path is asserted to sit inside `out_root`.
6. A patient whose output folder already exists is skipped and counted in the report. The run does not merge into it.
7. `organize_report.csv` is written in `out_root` with one row per patient: the sanitized ID, plan count,
   misc_files count and unreadable count. `.gitignore` and the global pre-commit filename check keep it out of commits.

Acceptance tests (synthetic):
- **A1.1** One patient with two planning studies on different dates. Each study has a CT, RTSTRUCT, RTPLAN and
  RTDOSE. Result: `plan_1` holds the earlier study and `plan_2` the later one, and the file counts match.
- **A1.2** A REG in the planning study references a diagnostic MR study. The MR files land in that study's `plan`
  folder.
- **A1.3** A REG references the other planning study. Neither study's files are duplicated into the other's folder.
- **A1.4** A localizer CT, an `Acuity` study, and an orphan CT with no plan all go to `misc_files/`.
- **A1.5** Two RTPLANs, each with its own RTDOSE, reference the same RTSTRUCT and CT. The result is `plan_1` and
  `plan_2`. Each holds the full CT and RTSTRUCT, exactly one RTPLAN, and only that plan's RTDOSE.
- **A1.8** The RTSTRUCT a plan references is stored in a different study from its CT. It is copied into the plan's
  `plan_<n>` folder and is not in `misc_files`.
- **A1.5b** Two planning CTs in one study, each with a plan, give two `plan` folders and no empty folder.
- **A1.6** A PatientID of `../evil` is written inside `out_root`.
- **A1.7** After a run, the source tree is byte-identical (compared by hash of every file).

---

## F2: convert (`rt_dataprep/convert.py`, `preprocess.py`, `masks.py`)

> **Superseded (2026-11-03)** by `docs/specs/2026-11-03-f2a-convert.md` (F2a, native grid, in two PRs) and F2b
> (`docs/specs/2026-11-03-f2b-preprocess.md`; `preprocess.py`, `--legacy-rnt` without ahe, A2.5, A2.6). Changes from the text below: no `--flat` and no A2.7
> (the output does not recreate the RNT layout), `<patient>/plan_<n>/` plus `<patient>/images/`, a per-folder
> `series.csv`, PLAN > FRACTION > BEAM-sum dose precedence, and no AHE (#16). The parity check below needs a fresh
> plan once F2b exists.

`convert(org_root, out_root, *, flat=False, steps=ConvertSteps()) -> ConvertReport`

`ConvertSteps` is a frozen dataclass of the legacy flags and parameters, with the defaults in the table in
`proposal.md`. It differs from the legacy code in three ways: the default is **no preprocessing**
(`harmonize_spacing=False`), `legacy_linear_masks=False`, and `ahe` requires the `[ahe]` extra. A
`--legacy-rnt` CLI preset sets `resample`, `rescale`, `ahe`, `harmonize_spacing` and `reshape`, which reproduces
`reshape_resample_ahe_rescale_initialresample_planning_ct`.

Steps, for each `<patient>/plan_<n>/`:
1. **Assemble.** Index headers by modality, as in rnt `gauge_rr.read_folder`, widened to RTDOSE and REG. Then
   follow the chain CT ← RTSTRUCT ← RTPLAN ← RTDOSE by referenced UIDs. F1 guarantees one RTPLAN per folder.
   Refuse the case if the chain is broken or resolves to anything other than one CT series. Dose is the RTDOSE whose
   `ReferencedRTPlanSequence` names the plan. If there are several, sum the doses whose `DoseSummationType` is
   `BEAM`, or take the single `PLAN` dose. If there is none, the case has no dose and is not refused.
2. **CT.** Read with `read_ct_series`, from rnt `rtdose.py` together with the `utils.augment._read` it depends on,
   restricted to the planning CT's files.
3. **Dose.** `read_rtdose` / `_convert` from rnt `rtdose.py`, giving Gy with `DoseGridScaling` applied. It is then
   resampled onto the native CT grid with `resample_dose`. The legacy code never scaled dose to Gy; this
   deliberately corrects that.
4. **Masks.** Build a temporary folder of hard links to only the planning CT slices and the RTSTRUCT. It goes under
   `out_root`, so it is on the same filesystem as the source, and other series in the plan folder cannot be picked
   up. Then run `DicomReaderWriter(arg_max=False)` on it in one pass:
   `walk_through_folders`, `set_contour_names_and_associations(contour_names=<all kept ROIs>)`,
   `set_index(<the index whose SeriesInstanceUID is the planning CT>)` and `get_images_and_mask()`. Split the
   multi-channel mask into one mask per ROI. `arg_max=False` keeps overlapping OARs intact; the legacy code read
   one ROI at a time, so it never lost overlaps either. `annotation_handle` must match the CT geometry
   (`geometry_equal`, from rnt `gauge_rr.py`), otherwise the case is refused.
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
   `<format_name(roi)>`. Dose and masks take only the geometric prefixes (`initialresample`, `resample`,
   `reshape`), never `rescale`, `norm` or `ahe`. `format_name`
   lowercases and strips non-alphanumerics, as in legacy `utils.py`. All files are `.nii.gz`. The output folder is `<out_root>/<patient>/plan_<n>/`, or `<out_root>/<patient>_plan_<n>/` with
   `flat`. A folder that already exists is skipped. Each file is written to a temporary name and renamed into place
   (`write_atomic` from rnt `scripts/convert_dose.py`), so a crash never leaves a half-written case.
7. `convert_report.csv` is written in `out_root`, with one row per case: outcome, refusal reason and the list of
   files written.

Acceptance tests (synthetic):
- **A2.1** A synthetic plan folder with no steps gives `planning_ct.nii.gz` equal to the source CT (same spacing,
  origin, direction and values) and `dose.nii.gz` in Gy (stored value × scaling) on the CT grid. This reuses rnt
  `tests/test_rtdose.py` expectations where they apply.
- **A2.2** A cube ROI drawn on the synthetic CT gives a binary mask whose voxel count equals the cube volume (±1 slice
  edge), on the CT grid.
- **A2.3** A BRACHY_CHANNEL ROI gives a text file that `np.loadtxt` reads back into the contour points (tolerance
  1e-4).
- **A2.4** A diagnostic MR study in the same plan folder does not change the outputs.
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
   and `study_uid` for cross-run matching. `.gitignore` and the global pre-commit filename check keep it out of commits.
2. `split(manifest, *, test_size, val_size, seed)` shuffles the sorted unique patients with
   `numpy.random.default_rng(seed)` and assigns whole patients to test, then val, then train. No scikit-learn. It writes `splits.json` with `{train, val, test}` case lists and a
   `case_id_hash = sha256('\n'.join(sorted(ids)))` for each split, the same hash convention as rnt
   `_setup_external_test`.
3. `check_disjoint(splits, external_manifest)` refuses if any patient overlaps.

Acceptance tests:
- **A3.1** No patient appears in more than one split. The same seed and manifest give an identical `splits.json`.
- **A3.2** An external manifest that shares a patient is refused, and the reason names no patient.

---

## Parity check on real data (OSC, run by the user, after F2)

**Convert parity.** Organize the raw export, run `convert`, then
`preprocess --initial-resample --rescale --resample --reshape --legacy-linear-masks` on convert's output
(`--legacy-rnt` was removed in #150; the user confirmed the pooled `ct.nii.gz` has no AHE), and match against the
pooled legacy folders: pooled `<mrn>_<x>_<fx>` pairs with `<mrn>/plan_<fx>`, as rnt's `scripts/convert_dose.py` pairs
by name. A pair whose CT grid or values differ is the sign of a numbering mismatch. This keeps convert parity separate from organize. `scripts/parity.py` prints
aggregate counts only, with no identifiers. Run on N ≥ 20 fractions. Pass criteria:
- CT: `geometry_equal` to legacy, and voxel max abs diff ≤ 1e-3 on the rescaled scale. With `ahe` on, it is checked
  separately with a Pearson r ≥ 0.999, because of the uncertainty about the legacy torchio/SimpleITK handoff after
  AHE.
- Masks: Dice ≥ 0.98 for each OAR after thresholding at 0.5.
- Applicator text files: point arrays equal within 1e-3.
- Dose is **not** compared to legacy, because legacy dose was unscaled. When the new native-grid CT is
  `geometry_equal` to rnt's `ct.nii.gz`, the new native-grid dose must match rnt's `dose_gy.nii.gz` within
  1e-4 Gy, since both come from the same `resample_dose`. If the grids differ, no dose tolerance applies; record
  the grid mismatch count instead.

**Organize parity** comes later and is separate. A fresh DICOM_Transport export is organized and then compared to
`raw/` by the `rtplan_uid` read from each tree's RTPLAN, never by `fx_<n>`. Legacy numbering depended on
pre-filters and a keep-list.

A failure here is a finding to bring back to the user before rnt switches over. It does not block F3.

## Out of scope for Phase 1

- Plan, control-point and dwell extraction (`plan.pkl`, `dwells.npz`): this stays in rnt (`plan_dwells.py`).
- Switching rnt_transformer or vision_transformer to import the package.
- The keep-list filter (`PatientFractionMapForUse.txt`): that is cohort selection and belongs in each project.

## Deferred

- REG-based resampling of diagnostic MR or PET into the planning CT frame.
- Parallel reading (the legacy code used `Pool(6)`).
- Same-day planning CTs: the order is by `SeriesTime` and then UID, which may differ from the legacy numbering.
- `normalize`, which the RNT data never used. It is ported but has only a smoke test.
- Second-wave package: provenance, run_dir and the section 6 stats.
