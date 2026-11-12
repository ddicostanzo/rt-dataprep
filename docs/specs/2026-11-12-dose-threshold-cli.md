# Spec: configurable dose `retained` threshold (#310 item 2)

## Problem

A dose's `retained` (series.json, preprocess with `--allow-crop`) is the in-grid fraction of its voxels at or above
min(DOSE_RETAINED_GY, DOSE_RETAINED_FRACTION × the dose's own max). Today these are the module constants 1.0 Gy and 0.1 (#307, #313).
To use another threshold, a user has to edit the source.

## Decisions (owner, 2026-11-12)

- Two preprocess options. The min() rule and the defaults do not change.
  - `--dose-retained-gy G`: the Gy floor. Finite, > 0. Default 1.0.
  - `--dose-retained-fraction F`: the fraction of the dose's max. Finite, 0 < F ≤ 1. Default 0.1.
- Both options require `--allow-crop` (a CLI usage error otherwise). Without it every `retained` is 1.0, so the threshold would do nothing.
- Provenance: one INFO run-log line with the Gy floor and the fraction, on every `allow_crop` run (defaults included). No series.json
  key; run provenance is #5.
- The written dose must be byte-identical whatever the threshold; a test guards it.
- README: say that changing the threshold needs `--overwrite` or a fresh output root (kept folders keep the old `retained`), and warn
  that F = 1 makes the threshold equal the max of a dose whose max is below the Gy floor, so only its peak voxels count.
- Item 1 (`cropped` leaves out EXTERNAL) merged in #331. Item 3 (refuse on dose `retained`) declined: filter downstream.
- `/dev` auto-merges after one clean review round (owner choice for this run).

## Behaviour

- `PreprocessSteps` gains `dose_retained_gy: float = DOSE_RETAINED_GY` and `dose_retained_fraction: float = DOSE_RETAINED_FRACTION`.
  The constants stay as the defaults; their comments say so.
- `dose_region(values, gy=DOSE_RETAINED_GY, fraction=DOSE_RETAINED_FRACTION)`: the same rule, the same 1e-6 tolerance, compared in numpy.
  The plan loop (in the worker) passes the run's `steps` values, e.g. with `functools.partial`. Never by mutating the module constants:
  a spawned worker re-imports them. Only `retained` uses the threshold; `crosses_final`, `process_dose` and `has_voxels` keep the
  unthresholded native.
- `preprocess()` validation: one constant `REASON_DOSE_RETAINED_LIMIT = 'dose retained threshold must be finite: the Gy floor greater
  than 0, the fraction greater than 0 and at most 1'`, raised as `ValueError` after the `REASON_NO_STEPS` / `REASON_ALLOW_CROP_STEPS` /
  AHE / mask-retention checks and before `out_root.mkdir`. The check is unconditional (the fields always carry values); the API does not itself require `allow_crop` for these fields.
- CLI (`__main__.py`):
  - `--dose-retained-gy` / `--dose-retained-fraction`: `type=float`, default `None`, with help that names the default and says they
    require `--allow-crop`.
  - Either one without `--allow-crop` is `parser.error`.
  - Values are passed to `PreprocessSteps` only when given, as in the `grids` dict pattern, so the defaults live only in `PreprocessSteps`.
  - `REASON_DOSE_RETAINED_LIMIT` is imported and added to the CLI's list of fixed refusal texts.
- Run-log line: logged once by the parent `preprocess()` after validation and before dispatch (never per patient), only when
  `steps.allow_crop`:
  `log.info('preprocess: dose retained counts voxels at or above min(%g Gy, %g of the dose max)', gy, fraction)`.
  The closing summary stays the last line.
- README: update line 88 (the preprocess options list), the `PreprocessSteps` table (~648, two rows), and the `retained` paragraphs
  (~405, ~768: 1 Gy / 10% become defaults, plus the two notes from Decisions). Update the `dose_region` docstring.

## Acceptance tests (synthetic data only)

1. Defaults are unchanged: the existing retained tests pass untouched.
2. With a dose that the final grid cuts with a known split (the pattern of
   `test_a_dose_threshold_is_the_lesser_of_1_gy_and_a_tenth_of_its_max`), a run with `dose_retained_gy` set gives the `retained` the
   new floor implies, which differs from the default run's. One case runs with `workers=2`, so the values must reach a spawned worker.
3. Likewise for `dose_retained_fraction`, with a dose where F × max < G for both the default and the new F. Expected values are worked
   out for both runs.
4. `dose_region` with explicit gy and fraction: unit cases, including F = 1 (only the peak counts when max < G).
5. `preprocess()` with a valid geometric step set and `allow_crop` refuses 0, a negative value, NaN and inf for the Gy floor, and 0, a
   value > 1, NaN and inf for the fraction, with `REASON_DOSE_RETAINED_LIMIT`. The output root is not created.
6. CLI usage: either option without `--allow-crop` exits 2 with **empty stdout** (vault
   [[argparse-error-exit-2-collides-with-domain-exit-code]]). With `--allow-crop` the values reach `PreprocessSteps`. `--allow-crop`
   without them builds `PreprocessSteps` with the field defaults.
7. CLI refusal: `--reshape --allow-crop --dose-retained-gy nan` (and an out-of-range fraction; a geometric step flag is needed, or `--allow-crop` is itself a usage error) exits 1, stderr is exactly
   `rt_dataprep: <REASON_DOSE_RETAINED_LIMIT>\n`, stdout is empty, and the output root is not created (the pattern of
   `tests/test_mask_retention.py`).
8. The run-log line appears exactly once on an `allow_crop` run with the values used (default and custom), not on a run without
   `allow_crop`, and the closing summary is still the last record.
9. The output dose NIfTI is byte-identical between a default run and a changed-threshold run on the same cut input.

## Deferred

- A per-dose threshold key in series.json or the manifest (provenance), tracked under #5.
- A prescription-relative threshold (needs the plan's prescription; not modelled).
- Making LOW_RETAINED (0.9, the run log's low-retained cut) configurable.
