# One CLI flag per preprocessing step, replacing --legacy-rnt (#150)

**Date:** 2026-11-05 · **Branch:** `feat/remove-legacy-rnt` · **Status:** REVIEWED (stress-test applied)
**Issue:** #150 · **Builds on:** `2026-11-03-f2b-preprocess.md`, `2026-11-04-ahe.md`

## Background

`preprocess --legacy-rnt` is a preset (`LEGACY_RNT` in `src/rt_dataprep/preprocess.py`) that turns on four steps at
once: initial resample, CT rescale, final spacing and reshape. `--ahe` (#16) is the only step with its own flag. The
`PreprocessSteps` fields are already independent, and `process_ct`, `process_series` and `output_name` already handle
any combination; only the CLI and the constant tie them together. The owner wants the preset gone (2026-11-04).

## Decisions (owner, 2026-11-05)

- S1 **Flags, named after the output prefix each one writes:** `preprocess` gets one opt-in `store_true` flag per
  step, next to the existing `--ahe`. Any combination is allowed. No shorthand for "all four".

  | Flag | `PreprocessSteps` field | File prefix |
  |---|---|---|
  | `--initial-resample` | `resample` | `initialresample_` |
  | `--rescale` | `rescale` | `rescale_` |
  | `--ahe` | `ahe` | `ahe_` |
  | `--resample` | `harmonize_spacing` | `resample_` |
  | `--reshape` | `reshape` | `reshape_` |

  The `PreprocessSteps` field names and the prefixes are unchanged (Python API and manifest parsing stay as they are).
  Each flag's help names its value (1 mm, `(HU + 1024) / 4096` on CT, 1.25 mm, 701 × 701 × 433) and its prefix.
- S2 **Removed:** the `--legacy-rnt` flag and the `LEGACY_RNT` constant. Passing `--legacy-rnt` is an argparse usage
  error (exit 2), like any unknown flag. No deprecation alias.
- S3 **Parameters stay fixed:** `new_spacing` (1 mm), `final_spacing` (1.25 mm) and `new_shape` (701 × 701 × 433)
  are not exposed on the CLI. Python API users still set them on `PreprocessSteps`.
- S4 **No step:** with none of the five step flags, `preprocess` still refuses with `REASON_NO_STEPS` and writes
  nothing, as today.
- S4b **Partial combinations are allowed**, including `--reshape` without a resample flag (crop/pad at the CT's own
  spacing). The existing refusals still guard dose and masks against lost voxels. `--rescale` alone refuses MR/PT
  study series with the existing `REASON_NO_SERIES_STEPS`.
- S5 **Merge:** auto-merge when the review round is clean.

## Constraints

- **Clinical output:** with all four flags (`--initial-resample --rescale --resample --reshape`) every output file,
  report and file name is identical to today's `--legacy-rnt` output; with `--ahe` added, identical to today's
  `--legacy-rnt --ahe`. No step code, ordering, fill or naming changes.
- **PHI:** no new log line, error text or file name source.
- Historical records (`docs/specs/`, `docs/plans/`, `docs/handoffs/`) are not rewritten; they describe what was true
  when written.

## Acceptance criteria

- AC1 `preprocess --help` output lists the five step flags of S1 and does not mention `--legacy-rnt`;
  `PREPROCESS_HELP` contains the fixed phrase "at least one step flag".
- AC2 Each step flag sets exactly the field in the S1 table; the others stay `False`. Checked for each flag alone and
  for all five together, on the `PreprocessSteps` passed to `preprocess`.
- AC2b (clinical output) `--initial-resample --rescale --resample --reshape` passes a `PreprocessSteps` equal to
  `PreprocessSteps(resample=True, rescale=True, harmonize_spacing=True, reshape=True)` with default spacings and shape
  (the old `LEGACY_RNT`); with `--ahe` added, equal to that with `ahe=True`.
- AC3 With those four flags, an end-to-end CLI run on synthetic data writes the same file names
  (`reshape_resample_rescale_initialresample_planning_ct.nii.gz` etc.) that `--legacy-rnt` wrote. The child-process
  test harness in `tests/test_cli.py` (`PRE_RUN`), which today shrinks `new_shape` by replacing `LEGACY_RNT`,
  shrinks it another way (for example by wrapping `rt_dataprep.preprocess.preprocess` before `runpy`, since
  `__main__` imports it fresh).
- AC4 `--legacy-rnt` exits 2 with an argparse usage error.
- AC5 No step flag (including only `--legacy-linear-masks` and/or `--series-fov-mask`) exits 1 with
  `REASON_NO_STEPS` and writes nothing under the output root.
- AC6 `LEGACY_RNT` no longer exists in `rt_dataprep.preprocess`; no source, test, example or README references
  `--legacy-rnt` or `LEGACY_RNT` (outside `docs/specs/`, `docs/plans/`, `docs/handoffs/`). Tests that used the
  constant (`tests/test_preprocess.py`, `tests/test_convert.py:845`, the parametrized cases in `tests/test_cli.py`)
  build the same `PreprocessSteps` explicitly. Prose calling it a preset ("legacy preset", "Legacy RNT
  preprocessing" in the `preprocess.py` docstring, the `PRE_RUN` comment, `examples/synthetic_workflow.py`,
  README and `examples/README.md`) is reworded to the step flags.
- AC7 README and `examples/README.md` show the per-step flags, including the full four-flag command that replaces
  the old preset, and the troubleshooting row for "no step is on" names the step flags.
- AC7b README no longer says only a Python rescale-only configuration refuses MR/PT; `--rescale` alone does too.
- AC8 Full suite and ruff pass.

## Deferred

- CLI options for `new_spacing`, `final_spacing`, `new_shape` (S3).
- `--reshape` without a resample flag can pad to a very large extent at native spacing (hardening issue).
- A named shorthand for the four legacy steps, if typing four flags proves error-prone in use.
