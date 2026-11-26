# Findings: centred-reshape (#372)

Spec: docs/specs/2026-11-26-centred-reshape.md. Plan: docs/plans/centred-reshape/plan.json (task-01 then task-02, both on feat/centred-reshape).

## task-01

Files: `src/rt_dataprep/preprocess.py`, `src/rt_dataprep/__main__.py`, `tests/test_preprocess.py`, `tests/test_cli.py`.

Decisions
- The function that places the window is `crop_or_pad_at(image, shape, pad_value, centre)`; `crop_or_pad` is untouched and the image mode still calls it. `process_ct` and `image_fov_mask` take `centre=None`, and None keeps the image centre, so the study path and every two-argument caller are unchanged.
- `reshape_centre_voxel(plan_dir, kept, info, steps)` returns voxel c or None; `preprocess_plan` raises `REASON_NO_CENTRE` outside it, after `too_small` and before `out_dir.mkdir()`. Only the SimpleITK transform runs in `quiet_call`; the table reads and numpy arithmetic stay outside, as `quiet_call`'s contract asks. A non-finite point is rejected before the transform.
- The grid entering reshape comes from the header (`entering_reshape_grid`, via `reference_grid`, as `image_fov_mask` does); a resample step before reshape is therefore honoured.
- `read_plan_table` is the read plus schema check lifted out of `index_plan_table`, and `PLAN_TABLES` (kind -> schema, prefix) replaces the inline dict; the centre reads go through both, so there is no second parser.
- `isocenter` pools plan and ion_plan rows, then takes the distinct rows (`np.unique`); `dwells` keeps every row with `dwell_weight > 0`. The isocentre columns are non-nullable; a null dwell weight reads as NaN and never counts, while a null dwell position makes the mean NaN, so the plan is refused.
- Reading: one unreadable, empty or wrong-schema table among those a mode needs refuses the plan (`REASON_NO_CENTRE`), even when another table would have given a centre.
- Validation sits before the `REASON_NO_STEPS` check in `preprocess()`, so a centre without `reshape` reports the centre reason. `roi` and `dose` raise `REASON_RESHAPE_CENTRE` until task-02: that is one clause of the validation expression, and `test_a_bad_reshape_centre_raises_before_any_output` has the two cases (`roi` with names, `dose`) that task-02 must move out.
- CLI: `choices=RESHAPE_CENTRES`, `--reshape-roi` is `nargs='+'` and reaches `PreprocessSteps` as a tuple (a list would break the equality tests).

Gotchas
- `write_plan`'s default dose and masks are a 12-voxel cube; on the centred fixture CT they refuse with `REASON_GRID`, so `centred_plan` defaults to no dose and no masks.
- The FOV mask file is named after the output CT: `fovmask_reshape_planning_ct.nii.gz`.
- Marker tests use reshape only: a linear resample blurs the marker. The resample test checks the output plan indices instead.
- Spacing and origin are dyadic because a NIfTI header stores float32; the halfway case (11b) uses lower indices 10 and 4 (even), so round-half-even would fail it.
- Mutation-checked by hand: `shape // 2` to `(shape - 1) // 2`, `floor(x + 0.5)` to `rint` and to `floor`, the distinct step, the dwell weight filter, the pre-reshape grid, the origin, the FOV centre argument and the range check each fail at least one test.

## task-02

Files: `src/rt_dataprep/preprocess.py`, `tests/test_preprocess.py`, `tests/test_cli.py`, `README.md`.

Decisions
- One path: `centre_point(plan_dir, kept, steps)` returns the physical point for every mode (table mean, or `weighted_centroid` of the matched mask or the dose) and `reshape_centre_voxel` calls it where it called `table_points` before. The `REASON_RESHAPE_CENTRE` clause for roi and dose is gone from `preprocess()`.
- `weighted_centroid(image, weighted)`: per-axis marginal sums in float64 (`sum(axis=..., dtype=float64)`, no float64 copy of the array), index centroid per axis, then the image's `TransformContinuousIndexToPhysicalPoint` (only that call is in `quiet_call`). Arrays are (z, y, x); the marginals along x, y, z are the sums over axes (0, 1), (0, 2), (1, 2). A zero total is None (-> `REASON_NO_CENTRE`); a NaN total flows to the non-finite check in `reshape_centre_voxel`.
- `roi_row(kept, names)`: mask rows only (an applicator or line row of that name does not match); per name, exact `roi_name`, else the first `casefold()` match in series.json order (`kept` keeps that order after the CT is moved first).
- The mask or dose is read with `read_image` in `centre_point`, outside `quiet_call`: an unreadable file is `REASON_FILE_UNREADABLE`, no mask, no dose, no name match, an empty mask and an all-zero dose are `REASON_NO_CENTRE`. The file is read once here and once more in the row loop.

Gotchas
- A roi/dose test needs `allow_crop=True`: the image-centred control run cuts the blocks, and a mask lost entirely refuses with `REASON_CROP`. The control run must also keep each fixture mask partly inside the image window, else it refuses rather than differs.
- `assert_marker_centred` clears `reshape_roi` for its image-centred control (`image` with names is a validation error); it also takes `**plan` for masks and dose.
- A dose that sums to a negative total is not refused (deferred in the spec: negative dose voxels).

Mutation-checked by hand: x/z swap, x/y swap, dropping the dose weighting, bounding-box centre instead of centroid, ignoring the origin, exact-only and case-insensitive-only matching, last-instead-of-first case-insensitive match, names tried in reverse order, a zero total not refused, matching non-mask rows (caught by the applicator test added for it): each fails at least one test.
