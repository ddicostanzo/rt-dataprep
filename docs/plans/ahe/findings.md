# Findings: AHE step (#16)

Spec: `docs/specs/2026-11-04-ahe.md`. Plan: `docs/plans/ahe/plan.json` (task-01 to task-05, serial).

Summary: an opt-in `ahe` step (`PreprocessSteps.ahe`, `preprocess --ahe`) runs skimage `equalize_adapthist`
(kernel default 1/8 per axis, nbins 128, clip_limit 0.05) on every image volume (planning CT, aligned plan series,
study series; CT/MR/PT), never dose or masks, on each volume's own data before the final grid. Input is cast to
float32 and min-max scaled `(a - a.min()) / (a.max() - a.min())`; a constant volume becomes all 0 without calling
skimage. Output float32 in [0, 1], geometry copied from the input. Outside an aligned series' FOV the fill is 0.
Name prefix `ahe_` between `resample_` and `rescale_`, image volumes only (`output_name(..., image=)`).

Known traps (verified 2026-11-04, skimage 0.26.0):
- skimage refuses float input outside [-1, 1]: always min-max first.
- Constant input returns all 1 (or a 0/1 mix depending on shape); a two-level half/half volume returns all 1.
  Test volumes are linear gradients (a landmark only on top of a gradient).
- STUDY_FILE: adding a group shifts numbered groups silently; use the named group `id`.
- `test_preprocess_help_says_the_preset_is_required_for_any_output` is the one pre-existing test A8 replaces.
- Tests use SMALL (new_shape 10^3), never full-size LEGACY_RNT.

## task-01

Files changed: `pyproject.toml` (+`"scikit-image>=0.26.0"` in `[project] dependencies`), `uv.lock` (+2 lines: the
`rt-dataprep` package's dependency entry and its `requires-dist` specifier).

Decisions: `uv add scikit-image` picked the lower bound `>=0.26.0` from the already-locked version (0.26.0); kept as is.

Check: sorted `^name = "` entries of `uv.lock` (25) are identical before and after, so no new package entered the lock.
Suite: 790 passed, 4 skipped; ruff clean; no test changed.

Gotchas: the venv is Python 3.13 although `requires-python` is `>=3.12`. The dependency lines added to `uv.lock` are
indented (`{ name = "scikit-image" }`), so a name comparison must anchor on `^name = "` to see package entries only.

## task-02

Files changed: `src/rt_dataprep/preprocess.py`, `tests/test_preprocess.py`.

Decisions: `equalize(image)` (module-level `from skimage.exposure import equalize_adapthist`, so tests monkeypatch
`preprocess.equalize_adapthist`) runs in `process_ct` after rescale and before the final spacing; both step-count checks
(REASON_NO_STEPS, REASON_NO_SERIES_STEPS) include `ahe`. `output_name(..., image=)`: `ahe_` needs `image`, `rescale_` needs `ct`
(callers pass both for a CT). `STUDY_FILE` reads the id through `(?P<id>...)`. The aligned plan series name already carries `ahe_`
but `process_series` does not equalize yet: task-03 (value change, zero fill).

Gotchas: `write_plan` puts a 12^3 dose and masks on the CT grid, so AHE tests with another size pass `dose=None, masks={}`.
The first preprocessed row of a study also lists `series.csv`.
