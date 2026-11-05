# Imaging cases: study folders for every CT/MR/PT series (#17)

**Date:** 2026-11-04 · **Branch:** `feat/imaging-only-cases` · **Status:** REVIEWED (stress-test rounds 1-2 and advisor applied)
**Issue:** #17 · **Builds on:** F2a PR 2 (other series, `images/`, #34), F2b preprocess, F3 manifest/split

## Background

#17 was written at F1 time: organize makes one case per RTPLAN, so a patient with no RTPLAN looked "never converted".
Since #34 that is no longer true. organize still organizes a no-plan patient (`plan_count=0`, every file in
`misc_files/`), and convert writes every CT/MR/PT series that is not registered to a converted plan into
`<patient>/images/series_<MOD>_<k>.nii.gz` with one `series.csv` (`_place_series` in convert.py). The real gap is
downstream: `manifest` lists plan folders only, `preprocess` reads plan folders only (and drops `kind=series` rows),
and `split` reads the manifest, so no diagnostic series ever reaches a training set. `images/` also mixes a patient's
exams from different dates in one bucket.

Why now: the owner wants image-only training data (no dose, no masks) from these exports. The volumes go into a
training set and must pass through manifest, preprocess and split.

## Decisions (owner, 2026-11-04)

- D1 Premise: convert's output layout is wrong for this use as well as the downstream gap (owner rejected "downstream
  only").
- D2 Case unit: one series. Layout `<patient>/study_<n>/series_<MOD>_<k>.nii.gz`, grouped by study, because an MR study
  holds many sequences.
- D3 Scope of series: every CT/MR/PT series convert writes, other than a converted plan's planning CT, for patients with
  and without plans. `images/` is retired; study folders replace it for every patient. A series registered to a
  converted plan is written to that plan folder as today **and** to its study folder (stress-test J1: otherwise a
  planned patient's registered MR never reaches training). The duplicate on disk is accepted; plan folders are
  unchanged.
- D4 Modalities: CT, MR, PT (`series.IMAGE_MODALITIES`), as today.
- D5 Automatic: no flag. Every convert run writes study folders.
- D6 One `series.csv` per study folder, one row per volume, written by `write_series_csv` exactly as `images/` is today
  (`kind=series`, plan columns and `reg_matrix` blank).
- D7 Numbering: studies `study_1..N` by the key `min((date == '', date, time))`, with `date`/`time` the stripped
  StudyDate/StudyTime, over the first header of each series written to the study, then StudyInstanceUID; undated studies last (stress-test M2: series of one study may
  disagree on StudyDate). Series inside a study keep convert's name `series_<MOD>_<k>`, k counted per modality per
  study in the existing global order (SeriesDate, SeriesTime, SeriesInstanceUID). N counts only study folders actually
  written.
- D8 `convert_report.csv`: one row per written series, `folder = study_<n>/series_<MOD>_<k>`, outcome `converted`,
  `files` = the volume name. The skipped-series count moves from the `images` row to a patient summary row (Design).
- D9 Manifest: one row per imaging series; columns appended after the existing ten: `case_type` (`plan`|`imaging`),
  `modality`, `image`. Imaging rows leave `fx` and `rtplan_uid` blank, `dose=0`, `masks=[]`.
- D10 preprocess: study folders are processed one series at a time. CT gets the full chain of the steps that are on
  (legacy RNT included); MR and PT get the geometric steps only (resample, harmonize, reshape), never the HU rescale.
  A failing series is refused on its own row; the study's other series are written (owner: per series).
- D11 split: unchanged code. It keeps whole patients per split, so a patient's plan and imaging cases share a split.
  Adding imaging-only patients changes the patient pool, so the same seed gives a different split and an existing
  `splits.json` is refused (`REASON_EXISTS`); a new manifest means a new split (owner, J2).
- D12 Out of scope: #38 multi-echo/DWI split, #39 PET SUV, #1 REG resampling, and intensity normalization (z-score
  etc.), which goes to #16. organize is unchanged.
- D13 Approach: extend the existing modules (convert, manifest, preprocess), not a new stage module.
- D14 `image` column: the volume's file name inside the study folder (prefixed in a preprocessed tree), so a loader
  opens `<root>/<patient>/study_<n>/<image>` without globbing. Plan rows leave it blank.
- D15 An imaging `case_dir` is a case id, not a path: `<patient>/study_<n>/series_<MOD>_<k>`, the same in a converted and
  a preprocessed tree (split's case ids depend on that). The id is taken from the file name with
  a full match of
  `(reshape_)?(resample_)?(rescale_)?(initialresample_)?(series_(CT|MR|PT)_\d+)\.nii\.gz` (the `output_name` prefixes,
  in their order), group 5 being the id (owner, J3).
- D16 `manifest()` keeps its `(written, refused)` return: refused = plans refused + imaging series refused; a study
  refused whole counts each of its `series.csv` rows, or 1 when its `series.csv` cannot be parsed (the reader raises
  `ValueError`/`csv.Error`) or has no row (owner, M8).

## Design

### convert (`_place_series`)

- The single global loop over the checked series in `order` stays as it is. For each read series the targets are the
  registered plan folders (unchanged) plus the series' study. Plan-folder k and `series.csv` rows are counted per plan
  folder, so they are unchanged as long as this loop and its order stay (stress-test M3).
- Study ranks are not known while writing (a later `read_series` may fail). Volumes are written under a hidden per-study
  folder in the patient partial (`.study_<StudyInstanceUID index>`; the folder name is the index in first-seen order,
  never the UID), and after the loop each written study folder is renamed to `study_<n>` by the D7 key, then its
  `series.csv` is written.
- Every exclusion and skip rule stays: excluded models, localizers, tainted/short/lost series, series whose slices
  disagree on study (so every written series has exactly one study), unreadable volumes.
- `IMAGES_NAME` and `REASON_NO_IMAGES` are removed. A patient with candidate series gets one summary row
  `folder = studies`, `files` empty: outcome `converted` with `skipped_series` when at least one study folder was
  written, else `skipped` with `REASON_NO_STUDIES = 'every other series was skipped; no study folder was written'`.
  Then one row per written series (D8); each study's `series.csv` is listed in the `files` of that study's first row.
- The no-plan, no-candidate patient keeps `REASON_NO_PLANS` (skipped).
- The run summary log (`convert.py` plan/images counts) counts plan rows as folders matching `PLAN_FOLDER`, and logs
  imaging series rows and the summary rows separately.

### manifest

- Walks plan folders as today (`check_rows`), then `<patient>/study_\d+/` folders that hold `series.csv`.
- A study's `series.csv` goes through a separate validator, `check_study_rows` in preprocess.py next to `check_rows`
  (manifest already imports from preprocess; `check_rows` requires one `ct` row): same columns, UTF-8, no `None` cells,
  at least one row, unique `file`; every row `kind == 'series'`, `modality` in `IMAGE_MODALITIES`, `file` a full D15
  match and present in the folder; the derived ids unique within the study. Any failure refuses the whole study (counted per D16). Then each row's header
  is read with `read_info`; a row whose header cannot be read is refused and counted, its siblings kept.
- Imaging row: `case_dir` (D15), `patient`, `fx` blank, `ct_size`/`ct_spacing` of that volume (float32 spacing text as
  for plans), `dose=0`, `masks=[]`, `rtplan_uid` blank, `ct_series_uid` = the series UID, `study_uid`,
  `case_type=imaging`, `modality`, `image`. Plan rows get `case_type=plan`, `modality=CT`, blank `image`.
- Rows stay sorted by `case_dir` (string order, as for `plan_10` today).

### preprocess

- `preprocess_patient` walks plan folders, then study folders. `check_study_rows` runs first; a refused
  study gives one row `folder = study_<n>` with the reason and writes nothing.
- Each study row is one case: CT through `process_ct` with the steps that are on; MR/PT through `process_ct` with
  `rescale` off and named with `output_name(..., ct=False)`. A series that is not a 3D volume of one real value per voxel
  (2D, 4D, multi-component or complex) is refused (`REASON_NOT_VOLUME`, #138). `too_small` runs first, on the header of a 3-axis series, so a too-small series is refused without a pixel read (`REASON_TOO_SMALL`, #141), and so does a
  `read_info` or `read_image` failure (`REASON_FILE_UNREADABLE`): that series only.
- A refused series removes only its own output file, never the study folder. `series.csv` (written rows only, file
  names updated) is written after the study's last series; a study with every series refused leaves no folder.
  `REASON_INTERNAL` and `OSError` still go to the patient level and abort or refuse the patient, as for plans.
- Report: one row per series, `folder = study_<n>/series_<MOD>_<k>`. A patient with neither plans nor study folders is
  skipped with `REASON_NO_CASES = 'patient folder holds no plan or study folder with a series.csv'` (replaces
  `REASON_NO_PLANS` there). The run summary log counts plan rows and imaging rows separately.
- `REASON_FILE_MISSING` text becomes "case folder" instead of "plan folder", and `REASON_TOO_SMALL` "the image"
  instead of "the CT".

### CLI and docs

- Help text: `convert` "one folder per plan, one per study"; `manifest` "one row per plan folder or study series".
- README: add an output-layout section (plan folders, study folders, `series.csv`, manifest columns, the case_dir/image
  rule, and that old `images/` trees must be re-converted).

## Constraints

- **Safety class / clinical output:** no dose, mask or plan output changes. The plan-folder path through
  `_place_series` is touched, so plan folders' `series.csv` text and every volume's voxels, size, spacing, origin and
  direction must equal the old code's on the same input (not raw gzip bytes, which depend on the ITK version).
  Imaging cases are training inputs, not clinical output.
- **PHI:** no new surface. Folder names are numeric (`study_<n>`, hidden `.study_<index>`); UIDs, dates and
  descriptions stay in `series.csv` as today; reports carry the sanitized patient id as today; log lines carry counts
  and `REASON_` constants only.
- **FHIR:** not applicable.
- **Data safety:** the organized and converted trees are only read.
- **Compatibility:** an old converted tree with `images/` gives no imaging rows in the new manifest and preprocess;
  re-convert it. A manifest with imaging patients gives a different split for the same seed (D11). No migration.

## Acceptance criteria

1. A no-RTPLAN patient with two studies (one CT series; one with two MR series) converts to `study_1/series_CT_1.nii.gz`
   and `study_2/series_MR_1.nii.gz`, `study_2/series_MR_2.nii.gz`, each study with a `series.csv` of its rows; no
   `images/` folder and no hidden folder left.
2. Study numbering follows the D7 key with an undated study last, whatever the folder order; two series of one study
   with different StudyDate rank the study by the earlier date.
3. A planned patient's unregistered series goes to `study_<n>/` only; its registered series goes to the plan folder
   and to `study_<n>/`; a converted plan's planning CT never appears in a study folder (a refused plan's CT
   does, as in `images/` today); no `images/` folder.
4. Regression: on a fixture with plans plus registered and unregistered series, every plan folder's file list,
   `series.csv` text and volumes (voxel hash and geometry) equal the old code's output, captured on `main` (eed0659)
   before the change and pinned in the test.
5. A series whose pixel read fails leaves the study numbering contiguous (`study_1..N`) and no hidden study folder.
6. `convert_report.csv` holds one `studies` summary row with `skipped_series` and one row per written series with
   `folder = study_<n>/series_<MOD>_<k>`; with every candidate series skipped the summary row is `skipped` /
   `REASON_NO_STUDIES` and no study folder exists; the run log counts plan rows from `PLAN_FOLDER` folders only.
7. Excluded-model, localizer and short/tainted series never appear in a study folder (existing `images/` tests moved
   to `study_<n>/`).
8. `manifest` on a converted tree writes one row per study series with `case_type=imaging`, the series modality,
   blank `fx`/`rtplan_uid`, `dose=0`, `masks=[]`, the volume's size/spacing, and an `image` that exists in the study
   folder; plan rows gain `case_type=plan`, `modality=CT`, blank `image`; the first ten columns are unchanged in name
   and order.
9. `manifest` on the preprocessed tree of the same data gives the same `case_dir` values, and each `image` is the
   prefixed file name.
10. A study `series.csv` that is malformed, empty, has a non-`series` kind, a modality outside CT/MR/PT, a bad or
    duplicate-id file name, or a missing file is left out and counted per D16; other studies and plans of the patient still get rows.
11. `preprocess` writes a study's CT with the CT prefixes (rescale included under legacy RNT) and its MR/PT with the
    geometric prefixes only, intensities not rescaled (an MR voxel value survives a resample to the same spacing).
12. One too-small series and one unreadable series in a study are each refused on their own report row, files absent; its sibling series and the
    study `series.csv` (siblings only) are written.
13. `split` on a manifest with plan and imaging rows of one patient puts all of that patient's case ids in one split.
14. No log line or exception added by this feature carries a patient id, folder name, UID, date or path (caplog
    assertions on the new paths).

## Failure handling

- A series that fails a convert check or its pixel read is skipped and counted on the `studies` row (surfaced, as
  today).
- A malformed or incomplete study `series.csv` is a counted refusal in manifest and a refused study row in preprocess,
  never a crash.
- An I/O error or unexpected error keeps today's patient-level handling (`REASON_IO_ERROR`, `REASON_INTERNAL`).

## Delivery

Local suite and ruff; README output-layout section and CLI help updated. No CI. Close #17 by hand after merge.
