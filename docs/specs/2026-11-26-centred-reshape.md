# Spec: reshape centred on the isocentre, dwells, a target ROI or the dose (#372)

## Problem

`--reshape` crops or pads symmetrically about the image-matrix centre (`crop_or_pad`, torchio's ceil/floor split).
Published dose-prediction work centres its grid on the target, the dose-weighted centroid (OpenKBP) or the dwell
positions (Moore 2024, brachy). So rt-pipeline's dose profiles reshape to a large grid with `--allow-crop` and recrop
downstream around the isocentre or dwell indices. A tight image-centred cube can lose the applicator or the target
(rt-pipeline `docs/research/rt-preprocessing-defaults.md`: "one setting is missing and lossy: crop centring").

## Behaviour

- `PreprocessSteps.reshape_centre: str = 'image'`, one of `image`, `isocenter`, `dwells`, `roi`, `dose`.
  `PreprocessSteps.reshape_roi: tuple = ()`, ROI names tried in order.
  CLI: `--reshape-centre {image,isocenter,dwells,roi,dose}` (default `image`) and `--reshape-roi NAME [NAME ...]`.
- Two new constants. `REASON_RESHAPE_CENTRE` (a `ValueError` raised by `preprocess()` before any output is written,
  and listed among `__main__`'s known refusals): an unknown mode, a centre other than `image` without `reshape`, `roi`
  with no names, or names with a mode other than `roi`. `REASON_NO_CENTRE` (a `CaseRefused`, one plan, report only):
  see "No usable centre" below.
- CLI: `--reshape-centre` uses argparse `choices` (an unknown mode is exit 2). `--reshape-centre` other than `image`
  without `--reshape`, `--reshape-roi` without `--reshape-centre roi`, and `--reshape-centre roi` without
  `--reshape-roi` are `parser.error` (exit 2), as the existing "requires its step flag" checks are.
  `--reshape-roi` is `nargs='+'`, so examples and tests put `CONVERTED OUT` before it.
- `image` is today's behaviour, byte for byte.
- **The centre**, a physical point (x, y, z) in mm, computed per plan folder from its native inputs, before the CT is processed:
  All plan-table coordinates are already in the CT's physical frame (convert indexes them on the CT,
  `plan_table.py:181`, `ion_plan_table.py:166`, `dwell_table.py:168`). The centre is computed after `check_rows` and
  before the row loop of `preprocess_plan`; every source it reads is a row of the same folder.
  - `isocenter`: the unweighted mean of the distinct `(isocenter_x, isocenter_y, isocenter_z)` rows of the folder's
    `plan.parquet` and/or `ion_plan.parquet` (pooled when both exist).
  - `dwells`: the unweighted mean of the `(position_x, position_y, position_z)` rows of `dwell.parquet` with
    `dwell_weight > 0` (active dwells only), each row once.
  - `roi`: for each name in `reshape_roi` in order, an exact `roi_name` match on a mask row wins, else the first
    case-insensitive match in series.json order; the first name that matches decides. The centre is the physical
    mean of that mask's nonzero voxels.
  - `dose`: sum(dose · position) / sum(dose) over every voxel of the folder's one dose file.
- **Placement**: the centre is mapped onto the grid entering reshape (the CT after every earlier step) with
  `TransformPhysicalPointToContinuousIndex` and rounded per axis with `floor(x + 0.5)` to voxel `c`. The reshape then
  takes, per axis, the `new_shape` voxels from `c - N // 2` to `c - N // 2 + N - 1`, padding with the existing pad
  value wherever that range runs past the image. So voxel `c` lands at output index `N // 2`.
- `image` does not use this formula: it keeps calling today's `crop_or_pad` unchanged, so its output stays
  byte-identical (the new formula would break ties differently from torchio's ceil/floor split).
- The three `crop_or_pad` callers: `process_ct` (planning CT: the centred placement when the mode is not `image`),
  `image_fov_mask` (takes the same centre as an argument, so the CT's FOV mask is placed like the CT), and the study
  path through `preprocess_series` (always the image centre).
- Everything that follows the CT grid follows it as now: dose, masks, beams and registered series are resampled onto
  the final CT; plan tables' `isocenter_i/j/k` / `position_i/j/k` index the final CT; `check_crop` / `--allow-crop` /
  `retained` judge against the shifted grid.
- The output name stays `reshape_` for every mode: one run uses one mode, so a different mode goes to a different
  output root. The README says so.
- **No usable centre → the plan is refused** with `REASON_NO_CENTRE`: the table the mode needs is absent,
  unreadable or not its schema; the mask or dose it needs is absent (an unreadable mask or dose keeps today's
  `REASON_FILE_UNREADABLE` from `read_image`); it has no rows; `dwells` has no active dwell (`dwell_weight > 0`); no name
  matches; the matched mask is empty; the dose sums to zero; the centre is not finite; or the reshape range
  (`c - N // 2` … `c - N // 2 + N - 1`) misses the image on some axis, so no CT voxel would be kept. The centre is
  computed before `out_dir.mkdir()` (next to `too_small`), so a refusal writes nothing.
- **Study folders** (`study_<n>/`, no plan) always use the image centre, whatever the mode.
- No new provenance: the existing plan-table indices show where the isocentre and dwells land; recording the
  roi/dose centre is deferred.

## Acceptance

1. `image` (default) output is identical to main's for the existing reshape tests (no test changes there).
2. `isocenter`: a synthetic plan whose isocentre lies off the image centre: the output CT's voxel `N // 2` per axis
   is the voxel nearest the isocentre, and the output plan.parquet's `isocenter_i/j/k` are within 0.5 of `N // 2`.
   Two distinct isocentres → their mean; a repeated isocentre (many control points) counts once.
   A folder with both plan.parquet and ion_plan.parquet pools their isocentres.
3. `dwells`: same check with dwell.parquet's positions (mean of the active rows); a zero-weight dwell far from the
   others does not move the centre.
4. `roi`: with names `('ptv_missing', 'PTV')` and a mask named `ptv`, the centre is that mask's voxel centroid; with
   masks `PTV` and `ptv` both present, `PTV` wins; with no matching name the plan is refused with the reason, and the
   report row says so.
5. `dose`: a dose with all its weight in one off-centre block centres on that block.
6. With `isocenter`, an off-centre dose hot block and a mask block placed at the isocentre land at output index
   `N // 2` (the voxel nearest the isocentre) in the output dose and mask.
7. A centre near the edge pads: the output keeps `new_shape`, and padded CT voxels carry the existing pad value.
8. No usable centre → the plan is refused with `REASON_NO_CENTRE` and no output folder: `isocenter` on a plan folder
   with no plan table; `dwells` with no dwell.parquet; `dwells` whose every dwell has weight 0; `dose` with an all-zero
   dose; an isocentre so far off the CT that the reshape range misses it.
8b. Fixtures that can fail: every mode's fixture puts its centre at a different offset on each axis, on a CT with
   anisotropic spacing and a non-zero origin (so an x/z swap from numpy's (z, y, x) order, or a missed origin, moves
   the result), and each mode's test also asserts that its output differs from the `image` run.
9. A study folder in a run with `--reshape-centre isocenter` is image-centred (same as `image`).
10. Validation: `reshape_centre='roi'` with no names, a non-image centre without `reshape`, and `--reshape-roi` without
    `roi` each refuse before output; the CLI help lists the modes.
11. With `--series-fov-mask` and an off-centre isocentre that makes the reshape run past one edge of the CT, the CT's
    FOV mask is 1 exactly on the output voxels that come from the CT (`c - N // 2 + n` inside `[0, size)`) and 0 on the
    padded ones.
11b. A centre exactly halfway between two voxels rounds up (`floor(x + 0.5)`).
12. README: the reshape row, the PreprocessSteps field table, the errors table and the CLI table gain the new options,
    with one line per mode on where its centre comes from.

## Deferred

- Weighted means (meterset-weighted isocentres, time-weighted dwells; dwell time semantics are #303).
- Recording the roi/dose centre (a `reshape_centre_*` column or run config, #5).
- Matching ROIs by type or regex; the union of several matches.
- A dose-threshold variant of the dose centroid.
- Falling back to the image centre instead of refusing.
- Centring study folders.
- The `dwells` filter trusts the sign of `dwell_weight`, unverified on real exports (#303; comment there when the PR opens).
- An oblique CT direction is untested.
- Negative dose voxels in `dose` mode.
