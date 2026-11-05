# FOV mask per aligned series (#129)

**Date:** 2026-11-04 · **Branch:** `feat/fov-mask` · **Status:** REVIEWED (stress-test and advisor applied)
**Issue:** #129 · **Builds on:** `2026-11-04-reg-resampling.md` (#1; this is its first `## Deferred` item)

## Background

`preprocess_plan` resamples each registered MR/PT/CT series once onto the plan's final CT (reg-resampling D1-D3).
Outside the series FOV the output is 0, or a non-rescaled CT series' own minimum (#125). A model that stacks the
series with the CT as input channels cannot tell "outside the FOV" from a true 0 (dark MR background, PT with no
uptake). `covers_ct` already resamples an all-ones uint8 copy of the series nearest-neighbour onto the final CT with
the series' transform. Per the vault note [[sitk-fov-overlap-by-resampling-an-all-ones-mask]], that image marks
exactly the voxels the linear resample fills from inside the series FOV. Today it is only used to decide coverage
(D6) and is thrown away.

Why now: the next queued deferred item; there is no incident. Consumer: the training pipelines that stack aligned
series as channels (rnt_transformer / vision_transformer), as for reg-resampling.

## Constraints

- **Clinical output:** none changed. CT, dose, masks and the aligned series' values and grids are unchanged with the
  option on or off. The new file is derived geometry only.
- **PHI:** no new place. Mask file names derive from the series output name; nothing new is logged.
- **Compatibility:** with the option off (the default), every output file, report and manifest is unchanged. With it on,
  manifest `series` entries may carry a third key, `fov_mask`; the training readers pick keys by name (owner).

## Decisions (owner, 2026-11-04)

- M1 Opt-in: a new `PreprocessSteps` field `series_fov_mask` (default off) and a `preprocess --series-fov-mask` CLI flag,
  combined with `--legacy-rnt` like `--legacy-linear-masks`. On a plan with no aligned series it does nothing. It is not a
  step: the no-step checks in `preprocess()` and `preprocess_series` ignore it, so it alone still gives `REASON_NO_STEPS`.
- M2 Scope: aligned (registered) plan series only. Study series, the planning CT, dose and masks get no FOV mask.
- M3 Content: uint8 on exactly the final CT's grid (size, origin, spacing, direction): 1 where the series' all-ones
  copy, resampled nearest-neighbour with the series' transform, lands; 0 elsewhere. It is never derived from the series
  values (a true 0 inside the FOV stays 1). It is the image `covers_ct` already builds: `covers_ct` returns that image and
  whether it has voxels, both computed inside its `quiet_call`, and it is written only when the option is on. A series that `covers_ct` leaves out (D6), or that is left out for any other reason (D5, D8), gets no mask.
- M4 Name and report: `fovmask_<series output name>` (for example `fovmask_reshape_resample_initialresample_series_MR_1.nii.gz`),
  in the plan output folder. The plan row's `files` in `preprocess_report.csv` lists it right after its series. The
  output `series.csv` does not list it: no new kind, so `check_rows` and every series.csv reader are unchanged.
- M5 Manifest: a series entry in the `series` JSON column gains `"fov_mask": "<file>"` when
  `fovmask_<series file>` exists in the plan folder (no header read: preprocess wrote it onto that grid). Entries without one keep exactly `{"file", "modality"}`, so a tree
  preprocessed without the option (and every converted tree) gives byte-identical manifests.
- M6 Memory: the mask is written and released with its series (reg-resampling D13). Peak memory gains one uint8 image
  of the final CT's size. With the option off, the mask image is released right after the coverage check, so a
  default run's peak memory is unchanged (#47).

## Approaches considered

- Chosen: a separate uint8 mask file per aligned series, opt-in, found through the manifest (owner).
- Rejected: NaN instead of 0 outside the FOV: breaks NaN-unaware consumers, and SimpleITK reads NaN NIfTI voxels back
  as 0 (vault: simpleitk-nifti-reads-nan-inf-voxels-as-zero), so the information is lost on read.
- Rejected: a `kind=fov_mask` series.csv row: every series.csv reader and the exact-columns checks would change (owner).
- Rejected: always on: changes every plan output that has series (owner). D4 put alignment always on because consumers
  choose through the manifest; a mask is an auxiliary file most runs do not need, so it costs output size for nothing
  when always on.
- Rejected: masks for study series' pad and the planning CT: the pad is a separate question (owner).

## Acceptance criteria

1. With `series_fov_mask` on, a plan with one registered MR (translation, partly outside the CT) preprocessed under
   `LEGACY_RNT` writes `fovmask_<MR output name>`: uint8, on exactly the final CT's size, origin, spacing and direction.
   For a constant-valued MR of small integer value v > 0, over all voxels: `(mask == 1) == (output != 0)` and
   `output[mask == 1] == v`, and the mask holds both 0s and 1s. Its grid is compared with the CT header read back from disk.
2. The same MR is listed in the plan row's `files` with its mask right after it. The output `series.csv` has no mask
   row. Under `replace(LEGACY_RNT, rescale=False, series_fov_mask=True)` a registered CT series (fill: its minimum, not 0)
   on the same native grid and `reg_matrix` as the MR of 1 gets a mask array equal to the MR's: the mask is geometry,
   not values.
3. A series left out (unusable matrix, no overlap, unreadable) gets no mask file; `skipped_series` is unchanged.
4. With the option off (default, and `LEGACY_RNT` as is), no `fovmask_` file is written and the existing tests pass
   unchanged.
5. `python -m rt_dataprep preprocess --legacy-rnt --series-fov-mask` turns it on; `--series-fov-mask` alone (no
   `--legacy-rnt`) exits 1 with `REASON_NO_STEPS` and creates no output root, as no step on does today.
6. The manifest of a tree preprocessed with the option lists `{"file": ..., "modality": "MR", "fov_mask": "fovmask_..."}`;
   of a tree preprocessed without it, `{"file": ..., "modality": "MR"}` (unchanged); converted trees unchanged.
7. Study series and dose/mask outputs are unchanged with the option on.

## Failure handling

- An unexpected error while building or writing the mask: the existing `quiet_call` path (REASON_INTERNAL, run
  stops), as for the series resample.

## Delivery and validation

Merged by `/dev` after a clean review round (owner allowed auto-merge). Real-data check falls to the OSC parity run.

## Deferred

- FOV/pad mask for study series and for the planning CT's reshape pad.
- A mask for a native converted series listed by the manifest on the CT grid (D10 identity case): not resampled, so
  there is no outside-FOV fill.
- Manifest check that a listed `fov_mask` header is on the CT grid (owner: hardening issue).
