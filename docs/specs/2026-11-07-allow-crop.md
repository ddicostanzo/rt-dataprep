# Allow cropping dose and masks at a smaller final grid (#174)

Date: 2026-11-07 · Branch: `feat/allow-crop-174` · Status: REVIEWED

## Background
- Real data (20 patients, 41 plans, 2026-11-05): preprocess at `new_shape=(320, 320, 256)`, 1.25 mm (the rnt
  training grid) refused 39 of 41 plans with `REASON_CROP` ("a dose or mask would lose voxels to the final grid").
  At the default 701 x 701 x 433 all 41 pass. Legacy torchio `CropOrPad` cropped silently.
- The parity run got past it by monkeypatching `check_crop` (vault: Memory/rt-dataprep/real-brachy-data-findings-2026-11-05).
- The owner's goal: preprocess output feeds rnt training on its grid, so most plans must pass at 320 x 320 x 256.

## Problem and user
The physicist preparing a training set runs `preprocess` at the training grid. Today that grid is reachable only from
Python (no CLI shape/spacing options, #159), and the crop guard refuses nearly every plan because dose tails and large
masks (body) extend past a 400 x 400 x 320 mm box.

## Constraints
- Clinical output: cropping changes dose and structure volumes. The failure would be silent (a plausible but cut
  structure), so every dose and mask records how much survived the crop, and the existing refusals stay. Opt-in only.
- PHI: no new identifiers anywhere. The new values are fractions and counts. Reports stay under the output root.
- Existing converted and preprocessed trees (old series.csv header) must still read.

## Decisions
- D1 `--allow-crop` / `PreprocessSteps.allow_crop: bool = False`: with it, a dose or mask whose foreground falls
  outside the final grid is cropped instead of refusing the plan (owner: dose + every mask, guarded).
- D2 Guards kept and extended: a mask, and now also a dose, whose native image has voxels but whose output has none
  refuses with `REASON_CROP` (owner). `--min-mask-retention` still applies with its current output/input measure.
  No built-in minimum retention (owner).
- D3 Crop centre stays the image centre (today's `crop_or_pad`). Dose- or structure-centred crop: hardening issues.
- D4 `allow_crop` without a geometric step (resample, harmonize_spacing, reshape) is a usage error: `ValueError` from
  `preprocess()`, `parser.error` from the CLI.
- D5 Every series.csv preprocess writes (plans and studies) has a trailing `retained` column
  (`PREPROCESSED_SERIES_COLUMNS = SERIES_COLUMNS + ['retained']`, owner); convert's series.csv is unchanged. The
  series.csv reader accepts either header; every other malformed check stays. Preprocess sets `retained` on every row
  it writes, so a row read from a preprocessed tree never carries a stale value (owner: overwrite).
- D6 `retained` is a pure crop measure on the native grid (owner): the part of the native image whose voxel centres
  fall inside the final grid (continuous index in [-0.5, n - 0.5) on every axis, `check_crop`'s rule), over the whole.
  Dose: sum of dose values (integral; same voxel volume on both sides). Mask: sum of values (nonzero voxel count for
  binary masks). An empty native image gives 1.0. Exactly 1.0 when nothing is cut; resampling error never enters it.
  Filled for every dose and mask row on a geometric run, cropped or not; blank for CT, applicator, line and series
  rows and for every row on a non-geometric run. Written as `repr(float)`. Computed without transforming every voxel:
  the dose and masks share the CT's direction, so each native axis maps to one final axis; the first and last native
  index inside the final grid per axis come from two point transforms (as in `check_crop`), then a slice sum.
- D6a `allow_crop` adds no filename prefix (a cropped file is told apart by `retained` only). `kind=series` images are
  not crop-checked today and stay unrecorded.
- D7 preprocess_report.csv gains a trailing `cropped` column: on a preprocessed plan row, the number of its dose and
  mask files with `retained` < 1; 0 when none; blank on every other row. A dose tail of any value past the edge
  counts, so `cropped` flags plans and `retained` says how much.
- D8 CLI options `--new-shape X Y Z` (positive ints), `--new-spacing X Y Z`, `--final-spacing X Y Z` (positive finite
  mm), x y z order, defaults unchanged (701 701 433 / 1 1 1 / 1.25 1.25 1.25). Each requires the step it feeds:
  `--new-shape` → `--reshape`, `--new-spacing` → `--initial-resample` (field `resample`), `--final-spacing` →
  `--resample` (field `harmonize_spacing`); otherwise `parser.error` (owner). No log line per cropped file.
- D9 Without `--allow-crop`, refusals and image outputs are unchanged except the empty-dose refusal of D2, which
  applies in both modes (owner: a zero dose is never valid output); the `retained` and `cropped` columns are written
  on every preprocess run. `retained` is exactly 1.0 by `check_crop`'s bounding-box test when nothing lies outside,
  otherwise a float64 in-grid sum over the float64 total, written as `repr(float(x))`.

## Approaches considered
- Chosen: one boolean plus per-file recording in series.csv.
- Rejected: dose-only crop (body mask would still refuse most plans); a list-valued `--allow-crop dose,mask` (no one
  needs the choice yet); a separate crop.csv (owner preferred series.csv); report-only recording (not per file);
  widening `SERIES_COLUMNS` for convert too (churn in convert and its pinned-header tests); output/input ratios for
  `retained` (mix resampling error into the crop measure, can exceed 1).

## Acceptance criteria
1. With `allow_crop=False`, a plan whose dose or mask extends past the final grid still refuses with `REASON_CROP`.
2. With `allow_crop=True`, the same plan is preprocessed; its dose and mask outputs are on the CT's final grid.
3. With `allow_crop=True`, a mask lying entirely outside the final grid refuses with `REASON_CROP`; so does a dose.
4. With `allow_crop=True` and `min_mask_retention` set above a cropped mask's output/input retention, the plan
   refuses with `REASON_MASK_RETENTION`.
5. `preprocess()` with `allow_crop=True` and no geometric step raises `ValueError`; the CLI exits with a usage error.
6. Every series.csv preprocess writes (plan and study) has header `SERIES_COLUMNS + ['retained']`; on a geometric run
   dose and mask rows hold a float, CT, applicator, line and series rows are blank.
7. A dose or mask whose foreground sits away from the final grid's edges has `retained` exactly 1.0, also when the
   spacing changes; a cropped one has the native in-grid fraction (a synthetic case with a known fraction, e.g. half a
   box cut off). A coarser grid can legitimately put edge voxel centres outside it, so tests pick geometry accordingly.
8. preprocess_report.csv has a trailing `cropped` column: a plan row counts its dose and mask files with
   `retained` < 1; an uncropped plan shows 0; patient-level and study rows are blank.
9. A converted tree (old header) and a preprocessed tree (new header) both pass `check_rows`, `check_study_rows` and
   `manifest`; a header that matches neither is still malformed.
10. Preprocessing a preprocessed tree succeeds and rewrites `retained` from this run.
11. `--new-shape`, `--new-spacing`, `--final-spacing` reach `PreprocessSteps` in x y z order; each without its step
    flag is a usage error; non-positive or non-finite values are a usage error.
12. README and the preprocess help texts document `--allow-crop`, the three grid options, the `retained` column and
    the `cropped` report column; the step help texts no longer hardcode only the default sizes.

## Failure handling
- Mask or dose fully cropped: refused (`REASON_CROP`). Mask retention below the user's minimum: refused.
- Partially cropped dose/mask: written, recorded in `retained` and `cropped`; no refusal, no log line per file.
- Bad CLI values or a grid option without its step: usage error before any work.

## Deferred (hardening issues)
- Crop centred on the dose or on a named structure instead of the image centre.
- Applicator and reference-line points that fall outside the final grid (copied unchanged today).
- manifest.csv column with the lowest `retained` per plan.
- A run summary log line counting plans with low `retained`.

## Delivery
Owner runs the 41-plan subset at 320 x 320 x 256 on the clinical machine and inspects `retained` and `cropped`.
