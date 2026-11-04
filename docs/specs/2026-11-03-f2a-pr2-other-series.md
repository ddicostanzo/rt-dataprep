# F2a PR 2: other series, registration, images/

**Date:** 2026-11-03 · **Branch:** `feat/f2a-other-series` · **Status:** REVIEWED (stress-test rounds 1-3 applied)
**Authority:** `docs/specs/2026-11-03-f2a-convert.md` (the F2a spec). Its convert step 3 (other series), step 4
(`series.csv`) and step 6 (report) hold as written; this addendum records only the decisions made for PR 2 and the
details the F2a spec left open. Where the two differ, this file wins for PR 2.

## Background

PR #19 (`e789e5a`) shipped F2a PR 1: per-plan convert (planning CT, dose, masks, applicators, plan `series.csv`).
PR 2 is the second half of the user's two-PR split (F2a spec, Delivery). Handoff: `docs/handoffs/f2-convert-continuation.md`.

Facts found in the code that shape PR 2:
- organize's `_group` drops excluded-model studies and CT localizers from *cases* only; those files still land in
  `misc_files/`. It filters CT localizers only, not MR/PT. Convert must re-apply both filters.
- organize copies a REG-linked study into every plan folder whose CT study the REG sits in, so one series can be in
  several plan folders and in `misc_files/`.
- `convert_plan` writes `series.csv` itself and its broad `except` turns any `AttributeError`/`KeyError`/
  `IndexError`/`TypeError`/`ValueError` into `REASON_MISSING_TAG` (#33).

## Decisions (interview 2026-11-03)

- **Addendum, not a new spec** (user): the F2a spec stays the source of truth.
- **#33 is folded in as the first task** (user): new code would otherwise hide its bugs as refusals during TDD.
- **Other-series volume check** (user): any orientation. Required: one orientation, image size, pixel spacing and
  frame of reference; at least two slices; no duplicate position; uniform spacing along the normal; no sideways
  offset (shear); single-frame files only (`NumberOfFrames` absent or 1). Position tolerances use a separate
  `SERIES_ATOL_MM` = 0.01 (user, stress-test J4: scanners round positions to ~0.01 mm); the planning CT keeps
  `ORIGIN_ATOL_MM` = 1e-3. Orientation equality uses `DIRECTION_ATOL`. The row and column cosines must be unit length and
  orthogonal within `ORTHO_ATOL` = 1e-4, and are normalized before the direction is built (user, round 3; tolerance
  set from a check on SimpleITK 2.5.7, 2026-11-03: its NIfTI writer never rejects a non-orthonormal direction but
  silently coerces it to orthogonal, while real DS-rounded cosines are off by ~1e-6 to 1e-5). A failing series (multi-frame enhanced MR/PET, multi-echo or DWI with
  duplicate positions, tilt) is skipped and counted; splitting them is a hardening issue.
- **PT values** (user): float32 stored × RescaleSlope + RescaleIntercept, the same reader as CT/MR. No SUV, no
  units column (SUV is a hardening issue).
- **Rigid matrix** (user): an item's matrix is used only when its `MatrixSequence` holds exactly one item whose
  `FrameOfReferenceTransformationMatrixType` is `RIGID`, with 16 values, last row `0 0 0 1`, and a 3×3 part R with
  ‖RᵀR − I‖∞ ≤ 1e-4 and det R > 0 (last row within 1e-6). Anything else leaves `reg_matrix` empty; the series is still placed (F2a spec).
- **exclude_models at study level** (user): same rule as organize. Every series in a study that holds a file whose
  `ManufacturerModelName` contains an excluded model is left out, and so is a file with no `StudyInstanceUID`.
- **Report `images` row** (user): one row with folder `images` for each patient that had at least one other-series
  candidate. Outcome `converted` with its files when `images/` was written; otherwise `skipped` with
  `REASON_NO_IMAGES` (every candidate went into plan folders or was skipped). `skipped_series` on that row counts
  every skipped series of the patient once. Plan rows keep `skipped_series` 0.
- **Code shape** (user): a new `src/rt_dataprep/series.py` holds candidate collection, the generic volume check and
  read, REG parsing and `reg_matrix`. `convert.py` orchestrates per patient.
- **Unexpected exceptions** (user, stress-test J1): per patient, any `Exception` subclass other than the handled
  ones (not `KeyboardInterrupt` or `SystemExit`, which keep PR 1's flag-based cleanup and add no row) removes
  the partial, adds a patient row `refused` with `REASON_INTERNAL`, writes `convert_report.csv` with every row so
  far, and then raises a fixed-text error after the except block has exited (a flag, as in `quiet_call`), so nothing
  is chained. That patient's plan rows are dropped, as on the `OSError` path. No other row is lost, and nothing from
  the original exception (a UID or tag value in its message) reaches stderr. For debugging (user, round 2), the log
  gets the exception's class name and the innermost traceback frame inside `rt_dataprep` as module name and line
  number (no file path: the project rule bans paths in log lines), never its message or arguments. The CLI turns the
  fixed-text error into exit status 1 with that text, like its `OSError` branch.
- **Where #33's tag catch applies** (user, stress-test J2): only header-only steps. In the plan path: `_assemble`, the
  volume and contour-image checks, the dose candidate and header checks, and `masks.classify`. A bad or missing tag
  there refuses the plan with `REASON_MISSING_TAG`. In `series.py`: candidate grouping, the volume check and REG
  parsing, where a bad tag skips and counts the series (or, for a REG, makes it not qualify). Pixel reads (already
  inside `quiet_call`), pixel math, placement and writes are outside the catch.
- **REG items without a frame UID** (user, stress-test J3): an item's frame is its `FrameOfReferenceUID`; when that
  is absent, it is the one `FrameOfReferenceUID` shared by those of the item's `ReferencedImageSequence` SOPs that
  are among the patient's readable headers (every header counts, excluded studies and localizers included; listed
  SOPs that are absent are ignored). None present, or more than one frame among them, means the item has no frame.
- **Merge** (user): full `/dev` auto-merge after one clean review round; the handoff's "don't merge" guardrail is
  lifted for this run.

## Details the F2a spec left open (decided here)

- **Headers are read once per patient.** Every file of every `plan_<n>/` and of `misc_files/`. `convert_plan` takes
  its plan folder's own `[(path, Dataset)]` list and whether any file in it was unreadable, and no longer reads
  headers itself. That list is not deduplicated: DicomRTTool hard-links the folder's own paths. An unreadable file
  refuses its plan (as in PR 1). An unreadable file in `misc_files/` is ignored; it is not a series, so it is not in
  `skipped_series`, and only a count reaches the log.
- **Deduplication, for candidates and REGs only:** copies of one SOP are reduced to the first path, in folder order
  `plan_<n>` by numeric n, then `misc_files/`, before series are grouped by `SeriesInstanceUID`.
- **Candidates:** Modality CT, MR or PT; `SeriesInstanceUID` is not the planning CT of a converted plan; study not
  excluded. Files with `LOCALIZER` in `ImageType` (any modality) are dropped one by one, as organize does; the rest
  of their series is still a candidate. A refused plan's CT is a candidate.
  Exception (user, wave review 4): a series with a file in a plan folder refused with `REASON_FILE_UNREADABLE` or
  `REASON_CT_SLICE_MISSING` is skipped and counted, since it may have lost an end slice that the volume check cannot
  see. `misc_files/` keeps the ignore rule (a hardening issue tracks the same risk there).
- **Registration is symmetric.** A REG whose own `FrameOfReferenceUID` is the CT's (or the series') frame counts as
  having an identity item for that frame when it omits one. Only REGs of the patient (any folder, deduplicated by
  SOP, not in an excluded study) are used. A REG **qualifies** for a (CT frame, series frame) pair when it has an
  item (real or implied) for each frame, whatever its matrices hold; the first qualifying REG by `SOPInstanceUID`
  (string order) is used, and a bad matrix in it leaves `reg_matrix` empty (no fallback to a later REG). A REG with
  more than one item (real or implied) for the CT frame or for the series frame also leaves `reg_matrix` empty: no
  item is picked silently.
- **Grouping:** a file with no `SeriesInstanceUID` or `Modality` is dropped (not a candidate, not counted). The k
  order key uses each tag of the series' first slice in sorted order, `''` when absent.
- **`reg_matrix` text:** 16 values row-major, space-separated, each formatted `%.10g`; `identity` for the same frame.
  Always empty in `images/`, which has no reference CT.
- **Series read:** files sorted ascending by ImagePositionPatient · (row cosine × column cosine), read with the same
  float32 reader as the planning CT. The step is (last − first)/(n − 1) along the normal; the volume check requires
  every slice position within `SERIES_ATOL_MM` of first + k·step (sideways included) and PixelSpacing equal within
  `SERIES_ATOL_MM`. Only `Execute` runs inside `quiet_call`. After it, outside `quiet_call`, the image's size must
  equal (columns, rows, n), else the series is skipped and counted; then its origin (the first position), direction
  (row cosine, column cosine, normal as columns) and spacing (PixelSpacing, step) are **set** from the validated
  headers, as `_read_dose` owns its geometry, rather than trusted from ITK's derivation. (Checked 2026-11-03 on
  SimpleITK 2.5: the reader forces an orthogonal direction from the cosines and takes slice spacing as
  |last − first|/(n − 1), so the two agree within tolerance; setting them keeps the output independent of ITK's
  heuristics.) The planning CT path is unchanged.
- **Writes:** a series is read once and written to every destination (a copy per folder), as
  `series_<Modality>_<k>.nii.gz` through `write_atomic`. Its `series.csv` row: `kind` series, its modality,
  description, UIDs, frame, date and `reg_matrix`; the plan columns are filled in plan folders and empty in `images/`.
- **series.csv moves out of the plan path.** `_convert_plan` no longer writes it; `PlanResult` carries its rows.
  Each folder's `series.csv` is written once, after placement wrote its series files, and every report row's `files`
  ends with `series.csv` (plan rows and the images row alike). The patient partial is still renamed only at the end.
  Tests that assert `convert_plan` writes `series.csv` or reads its own folder are restated against the new
  signature: `tests/test_convert.py` (series.csv rows and written-last checks) and `tests/test_masks.py` (calls of
  `convert_plan`). New tests go in a new `tests/test_series.py`.
- **#33 runs over PR 1's code too:** the task that narrows the catch runs the full suite (157 tests at `e789e5a`); an
  existing test that now raises is a PR 1 bug fixed in that task, never a reason to widen the catch again.
- **Commit-gate landmines:** `tests/test_organize.py:24` and `tests/test_dicom_io.py:85` trip the MRN rule if edited,
  so neither file is edited. `write_reg` changes additively (new optional keyword arguments only), so its existing
  callers stay untouched.
- **CLI:** `exclude_models` stays at its default (`('Acuity',)`); no new flag.
- **Summary log:** the closing `convert:` log line counts images rows separately from plan rows, so a
  `REASON_NO_IMAGES` skip is not mixed with `REASON_EXISTS` skips.
- **Plan-less patient whose candidates all fail:** its partial is renamed into an empty output folder (the PR 1 rule
  for a patient whose plans are all refused), so a rerun skips it.
- **Fixtures:** `write_reg` in `tests/synthetic_dicom.py` gains a `RegistrationSequence` with per-item frame,
  optional `ReferencedImageSequence` and `MatrixSequence` (type and 16 values), built from the standard's wording
  (item frame → REG frame), as the F2a spec's Constraints already require.
- **Plan-less patients:** a patient with no `plan_*` folder but at least one candidate is converted (images row only).
  A patient with neither keeps the PR 1 patient row (`skipped`, `REASON_NO_PLANS`) and gets no output folder.
- **Plan rows** list the series files written into that plan folder, before `series.csv`.

## Acceptance criteria

From the F2a spec, unchanged: **A2.4**, **A2.4b** (the `images/` part), **A2.4c**. Added:

- **B1 (#33)** A required tag missing from a plan's header still refuses that plan with `REASON_MISSING_TAG`. An
  exception raised outside the tag catch (tested by making the placement helper raise `TypeError` with a UID-like
  synthetic string in its message) on the second of two patients: the first patient's rows and a `REASON_INTERNAL`
  row for the second are in `convert_report.csv`, the second has no output or partial folder, `convert()` raises an
  error whose text, `__cause__` and `__context__` hold no part of the original message, `capfd` captures none of it,
  and a log record names `TypeError` and the raising module's name and line, with no path.
- **B2** A sagittal MR series (normal along x) registered by frame is written with its own direction, spacing and
  origin, equal to the headers. A two-echo MR series (duplicate positions), a multi-frame file, a series with a
  missing middle slice and a series lacking `PixelSpacing` are each skipped, and the images row's `skipped_series`
  counts them, as is an oblique MR whose row and column cosines are off-orthonormal by 1e-3. A series whose
  positions jitter by 0.005 mm is written.
- **B9** A REG item without `FrameOfReferenceUID` whose `ReferencedImageSequence` lists the MR's slices registers
  that MR to the plan (it lands in the plan folder); one that lists SOPs not in the patient's files does not.
- **B3** A REG item whose matrix type is not RIGID, or whose 3×3 part is scaled by 1.1 (fails the orthonormal
  test), or a REG with two items for the CT frame, gives an empty `reg_matrix`; the series is still placed in the
  plan folder.
- **B4** An MR in a study that holds an excluded-model file, and an MR localizer, are not written anywhere and are
  not counted as skipped.
- **B5** A PT series with a different RescaleSlope on every slice is written as stored × RescaleSlope +
  RescaleIntercept (float32), checked voxelwise.
- **B6** A series present in two plan folders and `misc_files/` is written once per destination, not once per copy.
- **B7** `convert_report.csv`: a plan row lists its series files; a patient whose candidates all went into plan
  folders gets an images row `skipped` with `REASON_NO_IMAGES`; a patient with no candidates gets no images row.
- **B8 PHI**: an unreadable series in `misc_files/` (truncated pixel data) is skipped with nothing captured by
  `capfd`, log records or warnings that holds the synthetic id, the tmp path or a UID.

## Failure handling

- Other-series problems never refuse a plan or a patient: an unreadable file in `misc_files/`, a series failing the
  volume check or failing to read is skipped and counted.
- An `OSError` on the output side refuses the patient (PR 1 rule) and removes the partial.
- After #33, an exception outside the tag catch becomes a `REASON_INTERNAL` patient row, the report is written, and
  the run stops with a fixed-text error (decision J1). If writing the report itself fails, that `OSError` is what
  propagates, in place of `ConvertAborted` (#43).
- A refused plan's planning CT is an other-series candidate. One that the 1e-3 mm planning check refused can still
  pass the 0.01 mm `SERIES_ATOL_MM` series check, and then lands in `images/` (#43).

## Deferred

- Hardening issues to file: splitting multi-echo/DWI and multi-frame series; SUV for PT.
- Already tracked: #1 (resample into the CT frame), #2 (parallel reading), #17 (imaging-only case unit beyond
  `images/`), #18 (DicomRTTool last-series linking). F2a spec Deferred holds (chained REGs, two REGs with different
  matrices).
- The planning CT of a converted plan is never placed in another plan's folder, even when a REG links the two
  (a replan). Consistent with the F2a spec; revisit if F3 needs replan pairs.
- `quiet_call`'s own broad `except Exception` still turns a bug inside a quieted closure (e.g. pixel math in
  `_read_dose`) into a fixed-reason refusal: the same masking class as #33, accepted because third-party exceptions
  carry paths. Hardening issue.
- Deformable REGs (`DeformableRegistrationSequence`, no `RegistrationSequence`) are ignored; a series registered only
  deformably goes to `images/`. Using their pre-deformation rigid matrix is a hardening issue.
