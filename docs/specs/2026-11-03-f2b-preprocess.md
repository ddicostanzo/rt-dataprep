# F2b: preprocess a converted tree (`--legacy-rnt`)

**Date:** 2026-11-03 · **Branch:** `feat/f2b-preprocess` · **Status:** REVIEWED (stress-test rounds 1-2 and advisor applied; A2.5i added from the plan review)
**Parent plan:** `docs/plans/2026-11-03-rt-dataprep/phase1.md` (F2 step 5, A2.5/A2.6) · **Legacy facts:**
`proposal.md` beside it ("Legacy pipeline" table) · **Builds on:** `docs/specs/2026-11-03-f2a-convert.md`

## Background

F2a (PRs #19, #34 and #45) writes each plan on its native grid. Each `<patient>/plan_<n>/` holds
`planning_ct.nii.gz`, `dose.nii.gz` (Gy, on the CT grid), one `{0,1}` uint8 mask per ROI, applicator point files,
`series_*` files and a `series.csv`. The F2a spec deferred to F2b the preprocessing the legacy exporter ran before
RNT training: resample, rescale, final spacing and crop/pad. The OSC parity check on real data (phase1.md,
"Parity check") waits for F2b.

## Problem

The package needs the legacy geometric and rescale steps without torch, for two uses: the OSC parity check, and
training new models. Running inference with the existing RNT weights also needs AHE, which is out of scope (#16)
(user, stress-test J2). The legacy code wrote masks with linear interpolation, and only when `resample` was on;
F2b fixes both.

## Constraints

- **Clinical output.** CT, dose and masks are training labels and dose analysis. A wrong grid or unit is a
  silent wrong value. Every geometric step has a test, and an unexpected input refuses the plan instead of
  guessing.
- **PHI.** There is no new PHI surface.
  - Inputs are F2a's NIfTI files and `series.csv`.
  - Log lines, exceptions and report reasons are fixed `REASON_` constants. Free text from `series.csv` is only
    copied into the output tree's `series.csv`.
  - `preprocess_report.csv` holds patient folder names. It sits under the output root, is added to the
    `project-profile.yml` `phi` rule and to `.gitignore`, and is never committed.
  - SimpleITK calls go through `dicom_io.quiet_call`.
- **Data safety.** The converted tree is only read. Each patient is built under a hidden partial name and renamed
  into place, as convert does (vault `record-completion-at-outcome-not-intent`).
- **Dependencies.** SimpleITK only. torch and torchio are never dependencies.
- **Memory.** One 701×701×433 float32 volume is about 850 MB. Masks are read, resampled, written and released one
  at a time (user, J5). Peak memory on real data gets a `hardening` issue.
- **Vault traps.**
  - `simpleitk-array-view-of-temporary-image-reads-freed-memory`: bind every image to a name before taking a view.
  - `torchio-croporpad-pads-labelmaps-with-padding-mode`: dose and masks are padded with 0, never with the CT's pad
    value.
  - `simpleitk-series-reader-and-nifti-writer-geometry-heuristics`: the direction must be orthonormal before write.
  - `alignment-test-needs-a-displacement-guard`: applies to A2.5e.
- **Commit gate.**
  - No added line accesses an underscore-prefixed attribute of another module. Preprocess needs `_crash_site` for
    its `REASON_INTERNAL` log, so it is renamed `crash_site` in convert.py. `_atomic` is renamed `atomic` if
    preprocess reuses it for the `series.csv`/applicator copies; `write_series_csv`, `write_atomic` and
    `SERIES_COLUMNS` are already public.
  - At most 5 staged files per commit.
  - Synthetic IDs have no digits or underscores.

## Premises (confirmed)

- F2a's native output stays the source of truth. Preprocessing is a derived tree that can be rerun with other
  settings without re-reading DICOM (user, round 1).
- Only the planning CT, dose and masks are preprocessed. Applicator point files are in patient mm and carry over
  unchanged (user, round 1).

## Approaches considered

- **Chosen: a new `preprocess` command over a converted tree** (user, round 1). convert.py's behaviour is
  unchanged.
- Rejected: `convert --legacy-rnt` adding prefixed files beside the native ones. Convert's atomic patient write
  grows, and a settings change means a full re-convert.
- Rejected: `convert --legacy-rnt` replacing the native files (the old phase1 design). The native data would be
  lost unless convert ran twice.

## Design

### CLI and steps

- CLI: `rt_dataprep preprocess <converted> <out> [--legacy-rnt] [--legacy-linear-masks]`.
- Python: `preprocess(converted_root, out_root, steps=PreprocessSteps()) -> PreprocessReport`.
- `PreprocessSteps` is a frozen dataclass, and every step is off by default.
  - A run with no geometric or rescale step on raises `ValueError(REASON_NO_STEPS)` before any write.
  - `--legacy-rnt` sets `resample`, `rescale`, `harmonize_spacing` and `reshape`.
  - The CLI exposes only the preset and `--legacy-linear-masks` (user, round 1). Python callers can set any field.

| Step (legacy order) | Field (default) | Parameters (default) | Applies to | Prefix |
|---|---|---|---|---|
| initial resample | `resample` (False) | `new_spacing` (1, 1, 1) mm, linear | CT | `initialresample` |
| rescale | `rescale` (False) | `(HU + 1024) / 4096` | CT | `rescale` |
| final spacing | `harmonize_spacing` (False) | `final_spacing` (1.25, 1.25, 1.25) mm, linear | CT | `resample` |
| reshape | `reshape` (False) | centred crop/pad to `new_shape` (701, 701, 433) | CT | `reshape` |
| mask interpolation | `legacy_linear_masks` (False) | nearest → uint8 {0,1}; linear → float32 | masks | none |

**Resample grid** (torchio 1.2.1 `Resample.get_reference_image`, read from source; user, J1). The input has size
`N`, spacing `s`, origin `o` and direction `D`, and the target spacing is `s'`.
- `n = floor(N·s/s')` per axis, and an axis with `N = 1` stays 1. This is computed exactly as torchio does it, as
  `np.floor(N * s / s')` in float64 on `GetSpacing()` values, so float32 spacing read back from NIfTI rounds the
  same way.
- torchio's `antialias` defaults to False in 1.2.1, so there is no smoothing step.
- The centre `c = o + D·((N−1)·s/2)`, the midpoint of the first and last voxel centres in mm, is kept.
- The new origin is `c − D·((n−1)/2·s')`.
- The direction stays `D`, and out-of-buffer voxels take 0. With `floor`, a CT voxel centre never falls outside the
  input's buffer, so the fill never reaches the CT.

**Crop/pad** (torchio 1.2.1 `CropOrPad`, centred). Per axis the difference is `d = |target − size|`. `ceil(d/2)`
voxels are cropped or padded at the start of the axis and `floor(d/2)` at the end. Spacing and direction are
unchanged, and the origin moves by the voxels added or removed at the start. The CT pad value is the minimum of
the CT entering reshape, i.e. after rescale when it is on, matching legacy `CropOrPad(padding_mode=<min>)`.

**Final grid.** The final CT image held in memory is the reference grid. Dose and masks are resampled onto it in
one step each, with `sitk.Resample(image, final_ct, Transform(), interpolator, 0, output_type)`. That makes every
output geometry-equal to the written CT by construction.
- **Dose:** linear and float32, 0 outside the dose grid. Rescale never touches dose. A plan with no dose has no
  dose output (user, round 2).
- **Masks:** nearest neighbour with output type uint8 by default. With `legacy_linear_masks`, linear with output
  type float32 (passed explicitly, because the native masks are uint8 and resampling would otherwise keep that
  type and drop the fractions). This happens whenever a geometric step is on.

### Names

The newest step's prefix goes first, joined with `_` (legacy `create_file_name`), and there is no `ahe` (user,
round 1). Dose and masks carry only the geometric prefixes. With the preset:
- CT: `reshape_resample_rescale_initialresample_planning_ct.nii.gz`
- dose: `reshape_resample_initialresample_dose.nii.gz`
- mask `<roi>.nii.gz`: `reshape_resample_initialresample_<roi>.nii.gz`

With only `rescale` on, the CT is `rescale_planning_ct.nii.gz`, and dose and masks are copied byte for byte under
their native names.

### Inputs, layout, series.csv

- Walks `<converted>/<patient>/plan_<n>/` folders that hold a `series.csv`, skipping `images/` and hidden
  folders.
- Files are found only through the `kind` column of `series.csv`: `ct`, `dose`, `mask`, `applicator`. Rows of
  kind `series` are dropped (user, round 1). Any other kind refuses the plan (`REASON_KIND`; user, stress-test
  round 2).
- Output goes to `<out>/<patient>/plan_<n>/`. It holds the preprocessed CT, dose and masks, the applicator files
  copied unchanged, and a `series.csv` of the kept rows, with `file` renamed to the new names and every other
  column unchanged (user, round 2).
- The output root gets `preprocess_report.csv` with `patient, folder, outcome, reason, files`.

### Failure handling (as convert; user, round 2)

| Situation | Outcome |
|---|---|
| `<out>` is `<converted>` or inside it | `ValueError(REASON_OUTSIDE_INPUT)` before any write; the CLI exits 1 with that text |
| no step on | `ValueError(REASON_NO_STEPS)`; the CLI exits 1 with that text |
| `<out>/<patient>` already exists | patient row `skipped`, `REASON_EXISTS` |
| no plan folder with a `series.csv` | patient row `skipped`, `REASON_NO_PLANS`, nothing written (as convert.py:158) |
| `series.csv` has zero or several `ct` rows | plan row `refused`, `REASON_NO_CT` |
| `series.csv` has several `dose` rows | plan row `refused`, `REASON_DOSE_COUNT` (user, J6) |
| a row has a `kind` other than ct/dose/mask/applicator/series | plan row `refused`, `REASON_KIND` |
| a row names a file that is not in the folder | plan row `refused`, `REASON_FILE_MISSING` |
| a NIfTI file is unreadable | plan row `refused`, `REASON_FILE_UNREADABLE` |
| a dose or mask is not on the native CT grid | plan row `refused`, `REASON_GRID`. Size must be equal; origin, spacing and direction are checked with `ORIGIN_ATOL_MM` and `series.DIRECTION_ATOL`, as F2a's mask check does (user, J3) |
| a native dose or mask voxel with a nonzero value has its centre outside the final CT grid's buffer (continuous index outside [-0.5, n-0.5) on some axis), or a native mask with voxels comes out with none | plan row `refused`, `REASON_CROP` (user, plan review): a training label is never silently truncated |
| OSError while writing | patient row `refused`, `REASON_IO_ERROR`; the partial is removed |
| a file-system error while reading an input (a disk or permission error that SimpleITK reports as a bad file) | patient row `refused`, `REASON_IO_ERROR`; the partial is removed (#47) |
| any other exception | row with preprocess's own `REASON_INTERNAL` text, report flushed, `PreprocessAborted` raised; the CLI exits 1 with the fixed text |

A refused plan leaves its sibling plans in place. A patient whose plans were all refused is still renamed into
place, so a rerun skips it.

## Acceptance criteria

- **A2.5a** On a synthetic converted plan with a dose and two masks, `--legacy-rnt` writes exactly: the three
  prefixed name forms above, the applicator files byte-equal to the input, and `series.csv`. The CT size is
  `new_shape` and the spacing is `final_spacing`. Every NIfTI output is geometry-equal (size, origin, spacing,
  direction) to the written CT.
- **A2.5b** A named fixture has CT spacing 0.9 mm and a cube mask. Under `--legacy-rnt`, the default run gives
  uint8 masks holding only {0, 1}. With `legacy_linear_masks`, the same fixture gives float32 masks with at least
  one value strictly between 0 and 1.
- **A2.5c** With only `rescale` on, each CT voxel equals `(h + 1024) / 4096` within 1e-6, and dose and masks are
  byte-equal to the input.
- **A2.5d** The fixture is a synthetic dose on the native CT grid, as F2a writes it: linear in x inside a box and 0
  outside it. It runs under the preset, with a small `new_shape` that keeps the whole box.
  - Inside the hull of the nonzero voxel centres, the output matches the analytic dose at the final voxel centres
    within 1e-4 Gy.
  - One native voxel or more outside that hull, the output is exactly 0.
- **A2.5e** A cube mask with a side of at least 3 final voxels is checked under `--legacy-rnt`.
  - Its centroid in mm moves by less than `s'/2` on each axis.
  - Its volume stays within `|ΔV| ≤ 6·L²·s'`, where `L` is the native cube side and `s' = max(final_spacing)`.
  - The final spacing differs from the native spacing on every axis.
  - The displacement guard: the same check, run on a copy of the output mask shifted by one final voxel, fails.
- **A2.5f** Grid and crop/pad conventions, as a CI test with hand-computed numbers (user, J4):
  - For a fixed input grid, `n` and the origin equal the formula values.
  - One case has `N = 5`, `s = 0.9` written to NIfTI, and `s' = 1.5`. The expected `n` is 2, computed from the
    stored float32 spacing 0.89999998, not 3 from the nominal value. torchio gives 2 as well.
  - Crop and pad with an odd difference put `ceil(d/2)` at the start.
  - CT padding equals the minimum of the CT entering reshape, and dose/mask padding is 0.
- **A2.5g** The written `series.csv` has one row per kept file, with the new `file` names and every other column
  unchanged. It has no `series` rows.
- **A2.5h** Each row of the failure table gives its outcome and reason, and a rerun skips the existing patient.
  Every `REASON_` constant matches no path, digit run or UID pattern.
- **A2.5i** A dose or mask with a nonzero voxel cut off by a small `new_shape`, and a one-voxel mask that nearest
  neighbour empties, each refuse the plan with `REASON_CROP` and leave no `plan_<n>` folder in the output.
- **A2.6** On a synthetic CT, SimpleITK initial resample, final spacing and crop/pad match torchio 1.2.1's
  `Resample` and `CropOrPad` within 1e-4, for values and for size, spacing and origin.
  - The mask reference is a single torchio `Resample(target=<torchio's final CT image>)` on a `LabelMap`, with no
    CropOrPad on the mask, compared exactly.
  - The torchio calls pass `antialias=False` explicitly.
  - The test uses `pytest.importorskip('torchio')`.
  - Before merge it is run once with `uv run --with torchio==1.2.1 --group dev pytest tests/test_preprocess.py`,
    and the result is recorded in the PR (user, round 3 and J1).

## Delivery

- One `/dev` PR. New `src/rt_dataprep/preprocess.py` and `tests/test_preprocess.py`.
- `__main__.py` gains the `preprocess` subcommand and handles `REASON_OUTSIDE_INPUT`, `REASON_NO_STEPS` and
  `PreprocessAborted`.
- convert.py renames `_crash_site` (and `_atomic` if reused) as public. Its behaviour does not change.
- `.gitignore` and `project-profile.yml` gain `preprocess_report.csv`.
- phase1.md F2 points to this spec. Its parity command `convert --flat --legacy-rnt` is corrected to
  `convert`, then `preprocess --legacy-rnt --legacy-linear-masks`.
- Merge: standard `/dev` auto-merge once the review round is clean (user, round 3).

## Deferred

- `norm` and AHE (#16). With AHE off, the output is not the distribution the existing RNT weights saw.
- Preprocessing `series_*` and `images/` (#1).
- A patient folder whose name clashes with the input root's own name, and other odd tree shapes.
- Peak memory on real data (a `hardening` issue).
- A2.5i catches only a mask that ends up completely empty. A mask that loses most of its voxels but keeps one passes
  the check, so the OSC run should look for it.
- The legacy run's torchio version is unknown. If the OSC parity check shows a size or origin mismatch, the grid
  convention gets revisited.
- The OSC parity run itself, done by the user after merge, and parallel processing (#2).
- A record of the `PreprocessSteps` parameters (spacing, shape, mask mode) in the output root. File names show
  which steps ran but not their parameters.
- Stress-test round 3 was replaced by the advisor review, because round 2 raised one judgment item and the user
  decided it.
