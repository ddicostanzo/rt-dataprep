# Registered MR/PT aligned onto the planning CT grid (#1)

**Date:** 2026-11-04 · **Branch:** `feat/reg-resampling` · **Status:** REVIEWED (stress-test rounds 1-2 and advisor applied)
**Issue:** #1 · **Builds on:** F2a PR 2 (other series, `series.registration`, #34), F2b preprocess, imaging cases (#17, spec `2026-11-04-imaging-cases.md` D12 put #1 out of scope)

## Background

convert already writes every CT/MR/PT series that is rigidly registered to a converted plan into that plan folder, as
`series_<MOD>_<k>.nii.gz` on its own native grid, with a `kind=series` row in `series.csv` whose `reg_matrix` column
holds the registration (`series.registration`): `identity` for the CT's own frame of reference, 16 row-major `%.10g`
values of `inv(M_ct) . M_series` (series frame to CT frame, rigidity already checked), or `''` when a REG qualified but
gave no usable rigid matrix. `preprocess.check_rows` drops every `kind=series` row, so a preprocessed plan case is CT,
dose and masks only, and the registered images are never used.

Why now: the owner wants the registered MR/PT as extra model input channels for rnt_transformer / vision_transformer,
stacked with the CT on one voxel grid. The legacy `Registration.resampling_transform` is not in any local repository,
so there is no legacy output to compare against (owner: synthetic geometry tests instead).

## Problem and user

The consumer is a training pipeline that reads a preprocessed tree through `manifest.csv`. It needs, for each plan case,
the registered MR/PT volumes voxel-aligned with the final CT, and a manifest column naming them.

## Constraints

- **Clinical output:** image geometry. A wrong transform direction or a wrong grid gives a plausible but misaligned
  image: a silent failure. Correctness is shown by landmark tests with known rigid transforms (owner). Dose, masks and
  the CT path are unchanged and stay pinned by the existing tests; `tests/test_plan_regression.py` (convert output) is
  never re-pinned.
- **PHI:** no new place. Output files and `series.csv` rows already exist per plan; refusal reasons are fixed constants;
  log lines carry counts only.
- **FHIR:** not applicable. **Audit:** the report records what was left out; the applied matrix stays in the converted
  tree's `series.csv` (the output row holds `identity`, D9).
- **Compatibility:** preprocessed plan folders gain files and `preprocess_report.csv` gains a column, so outputs differ
  from earlier runs. The manifest gains a column.

## Decisions (owner, 2026-11-04; stress-test round 1 applied)

- D1 Target grid: each registered series is resampled **once**, straight from its native file onto the plan's final
  CT image (after all CT steps), as dose is. One interpolation, voxel-identical grid. With no geometric step on, the
  final CT is the native CT and the series is still resampled onto it: series never take the copy branch that dose
  and masks take when no geometric step is on.
- D2 Transform: `reg_matrix` maps series frame to CT frame (the REG text helper in `series.py`: `inv(M_ct) . M_series`, the DICOM REG
  source-to-registered direction, pinned by `test_synthetic_dicom.py` and `test_series.py`). `sitk.Resample` maps each
  output (CT-frame) point to an input (series-frame) point, so the transform is the **inverse** of `reg_matrix`, built
  as an affine from the inverted 4x4. SimpleITK works in DICOM LPS (a NIfTI's RAS is converted on read), so the DICOM
  matrix applies unchanged. `identity` resamples with the identity transform (grids still differ).
- D3 Interpolation: linear, float32, default value 0 outside the series FOV. MR/PT intensities are unchanged (MR/PT
  normalization is #16). A registered **CT** series is rescaled like the planning CT when `steps.rescale` is on
  (owner), applied to the native image before the resample so that 0 outside the FOV is air. With rescale off, a CT
  series is filled with its own native minimum instead of 0 (water), as the planning CT is padded (#125). Its output
  name uses `output_name(file, steps, ct=True)`.
- D4 Always on: alignment runs for every plan with registered series, under every step set including `--legacy-rnt`.
  No new flag.
- D5 Unusable matrix: a series whose `reg_matrix` is not `identity` and is `''`, not 16 finite numbers, or not rigid (the same rigidity
  test `series.py` applies to REG items, factored into one shared helper) is left out: no output file, no output
  `series.csv` row. The plan is still preprocessed.
- D6 No overlap: a series whose FOV covers no CT voxel after resampling is left out, as in D5. The test is geometric
  (an all-ones copy resampled nearest-neighbour), not by values: an all-zero series that overlaps is kept, and one
  overlapping only in zero background is kept (#121, #126). Partial coverage is kept, 0 outside. Series skip `check_crop` and the dose/mask `same_grid` check (both assume the CT grid
  and would refuse the plan).
- D7 Counting: `preprocess_report.csv` gains a `skipped_series` column (as `convert_report.csv` has), set on plan rows
  to the number of series left out by D5, D6 and D8, and 0 on every other row (patient, study, study series). The
  closing log line adds the total.
- D8 Bad file (owner): a registered series whose file is **missing** from the plan folder refuses the plan with
  `REASON_FILE_MISSING` (a corrupt converted tree, loud, as for dose and masks). One that is present but
  **unreadable**, or not a 3D scalar volume (2D, 4D or multi-component, #127), is left out and counted, as in D5: only `CaseRefused(REASON_FILE_UNREADABLE)` from reading that series
  is caught; `REASON_INTERNAL` from the resample still stops the run.
- D9 Naming and rows: the output name is `output_name(file, steps)` (`ct=True` for a CT series, D3). The output
  `series.csv` keeps the row with the file renamed, `reg_matrix` set to `identity` and `frame_of_reference_uid` set to
  the CT row's (owner): the written image is in the CT frame. The converted tree keeps the original matrix as provenance. Output order: the existing rows, then
  the series rows in their input order.
- D10 Manifest: a new last column `series`, a JSON list of `{"file": ..., "modality": ...}` for the plan's
  `kind=series` rows whose `reg_matrix` is `identity` **and** whose image is on the plan CT's grid (`same_grid`), in
  series.csv order. Aligned outputs are always listed; a native converted series is listed only when it is both in the
  CT frame and on the CT grid. Imaging rows get `[]`. A series whose header cannot be read is not listed and does not
  refuse the plan.
- D11 Tests (owner): synthetic MR volumes with a bright landmark voxel, a REG matrix that is (a) identity on a
  different grid, (b) a pure translation, (c) a rotation about one axis (not 180 degrees, so forward and inverse
  differ) plus translation. Each test asserts the landmark's maximum lands at the CT voxel predicted by mapping the
  landmark through `reg_matrix`, within half a voxel per axis. A wrong-direction transform (forward instead of
  inverse) must fail (b) and (c).
- D12 Validation: `check_rows` also returns the series rows. It checks that each has modality CT, MR or PT and a
  name that `series_id` accepts (so the step prefixes of a preprocessed tree pass) whose `series_<MOD>_<k>` MOD
  matches it (else `REASON_STUDY_MODALITY` / `REASON_STUDY_NAME`, reworded to say
  "a series.csv series row" so they fit a plan too; plan refused), and
  that its file is present (D8). The duplicate-file check already covers series rows. `manifest` calls `check_rows`,
  so a plan with a missing series file is refused there too.
- D13 Memory: one series at a time, read, resampled, written and released (`del`), as dose; peak is the final CT plus
  one native and one resampled series.

## Approaches considered

- Chosen: one resample onto the final CT grid (D1), owner pick.
- Rejected: resample onto the native CT grid, then the CT's step chain: up to three interpolations, more blur.
- Rejected: opt-in step flag: one more switch; consumers already choose through the manifest (owner).
- Rejected: refuse the plan on an unusable matrix or unreadable series: loses CT/dose/mask cases over an auxiliary image (owner).
- Rejected: an `aligned` column in series.csv: changes every series.csv reader and the exact-columns check (owner).

## Acceptance criteria

1. A plan with one registered MR (`reg_matrix` a translation) preprocessed under `LEGACY_RNT` writes the MR output
   with the geometric prefixes, on exactly the final CT's size, origin, spacing and direction, float32. The output
   `series.csv` lists it after the existing rows with `reg_matrix` `identity` and the CT's frame UID, and the plan row's `files` includes it
   before `series.csv`.
2. Landmark tests D11 (a), (b), (c) pass. For (b) and (c) the test also maps the landmark through the inverse of
   `reg_matrix` and asserts the output maximum is more than half a voxel from that point on some axis, which pins the
   direction permanently.
3. Voxels of the final grid outside the series FOV are 0; an MR is not rescaled (a constant-valued MR keeps its value
   at interior voxels, at least one voxel from the series boundary, where linear interpolation does not blend in the
   0 fill).
4. A registered CT series under `LEGACY_RNT` is rescaled (a constant HU value v lands as (v + 1024) / 4096 inside the
   FOV at interior voxels as in 3, 0 outside) and gets the `rescale_` prefix.
5. A series with `reg_matrix` `''`, and one with a non-rigid matrix, are each left out: no output file or output
   `series.csv` row; the plan row is `preprocessed` with `skipped_series` 1.
6. A series moved entirely outside the CT (large translation) is left out and counted as in 5; one partly outside is
   kept.
7. An unreadable series file is left out and counted as in 5; a missing one refuses the plan with
   `REASON_FILE_MISSING`, in preprocess and in manifest.
8. A series row with a modality other than CT/MR/PT, or a name not matching its modality, refuses the plan.
9. With no geometric step on (rescale only), the series is resampled onto the native CT grid, not copied.
10. `preprocess_report.csv` has the `skipped_series` column; patient, study and study-series rows hold 0; the closing
    log line includes the total.
11. The manifest `series` column lists the aligned MR of a preprocessed plan as `[{"file": ..., "modality": "MR"}]`;
    on the converted tree it lists `[]` for a translated MR on its own grid **and** for a translated MR whose header is
    on the CT grid (built by writing the MR with the CT's exact geometry plus a REG with a translation); it lists an
    `identity` MR on the CT grid; imaging rows hold `[]`.
12. Existing tests pass unchanged except where they pin the report or manifest columns (`test_manifest.py` column
    tests) or a plan's output file list that now includes series; each such change is listed in findings.
    `tests/test_plan_regression.py` is unchanged.

## Failure handling

- Unusable matrix, no overlap, unreadable file: left out and counted (D5-D8), surfaced in the report.
- Missing series file, bad modality or name: plan refused with a fixed reason (D8, D12).
- Unexpected error while resampling: the existing `quiet_call` path (REASON_INTERNAL, run stops), as for dose.

## Delivery and validation

Merged by `/dev` after a clean review round (owner allowed auto-merge). Real-data check of alignment falls to the OSC
parity run.

## Deferred

- A FOV mask per aligned series (outside-FOV vs true 0).
- Deformable REGs (#41), SUV (#39), multi-echo split (#38), MR/PT normalization (#16).
