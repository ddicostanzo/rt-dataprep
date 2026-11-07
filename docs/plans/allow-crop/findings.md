# Findings: allow-crop (#174)

Spec: docs/specs/2026-11-07-allow-crop.md. Plan: docs/plans/allow-crop/plan.json (task-01 series.csv schema →
task-02 allow_crop + retained + cropped → task-03 CLI + README). Run tests with `uv run --group dev pytest -q --tb=short`
in this worktree. Commit gate: at most 5 staged files. Append findings under `## <task-id>`.

## task-01

Files: src/rt_dataprep/convert.py (write_series_csv gains `columns=SERIES_COLUMNS`), src/rt_dataprep/preprocess.py
(PREPROCESSED_SERIES_COLUMNS; plan and study series.csv written with it; read_series_csv accepts either header),
tests/test_preprocess.py, tests/test_manifest.py.

- Every row preprocess writes sets `'retained': ''` explicitly (`row | {...}`), so a row read from an already
  preprocessed tree never carries the old value forward. task-02 replaces the blank with the real value.
- Gotchas: tests/test_preprocess.py already has a `header(image)` helper (image geometry); the new CSV one is `csv_header`.
  Existing test `test_series_csv_keeps_every_column...` now expects the extra blank `retained` key.
- Full suite: 1174 passed, 4 skipped (~3.7 min); ruff clean.

## task-02

Files: src/rt_dataprep/preprocess.py, tests/test_preprocess.py (tests/test_series.py NOT changed, see below).

- `crosses_final(native, final)` is the boolean predicate (old check_crop body); `check_crop` raises on it. With allow_crop False
  preprocess_plan calls `check_crop` exactly as before (a test counts the calls); with allow_crop True it calls `crosses_final`
  and never `check_crop`. A passed check_crop means predicate false, so retained is `repr(1.0)` without a second pass.
- `native_retained(native, final)`: float64 total, then per axis `bisect.bisect_left(range(N), True, key=...)` on the point
  transform of native index k (other axes 0) for the first in-grid index (`>= -0.5`) and the first past it (`>= n - 0.5`), then one slice
  sum. Bisection uses the same comparisons as the predicate, so no float tie differs between them. Valid because dose, masks and the final
  CT share the direction (each native axis maps increasingly onto one final axis). Total 0 gives 1.0.
- `preprocess_plan` returns `(files, skipped, cropped)`; `cropped` is `''` on a non-geometric run. `PreprocessRow.cropped: int | str = ''`.
- Empty-output refusal is now `has_voxels(native) and not has_voxels(image)` for dose and mask, in both modes (D9).
- The local `retained` in the min_mask_retention block is renamed `ratio` (it is the output/input measure, not the series.csv column).
- New `REASON_ALLOW_CROP_STEPS`; raised after REASON_NO_STEPS, before any mkdir.
- Plan divergence: the plan lists tests/test_series.py:981 as a pinned preprocess report header. It is convert's report
  (`conv.REPORT_NAME`, via convert's `report()` helper); no test_series test reads preprocess_report.csv. Left unchanged, like test_convert.py:481.
- Pinned preprocess report assertions changed in tests/test_preprocess.py: header, and the five `dict(...)` row equalities.
  Task-01 tests that expected a blank `retained` on plan rows now expect '1.0' on dose/mask rows (SMALL cuts nothing).
- Test geometry gotcha: FOUR_STEPS' final grid (1.25 mm voxels) puts the native centres exactly on the +-0.5 index ties; the new
  cropping tests use reshape-only `CUT` (12^3 -> 6^3, native indices 3..8 kept, no ties). A dose that survives with retained 0.0
  needs native 2 mm voxels, final 1 mm (linear leak into the grid); a dose that resampling empties: final_spacing 5 mm with one native voxel.

## task-03

Files: src/rt_dataprep/__main__.py, tests/test_cli.py, README.md.

- Grid options use argparse `type=` functions (`positive_ints`, `positive_float`, nargs=3), so bad values and wrong counts are argparse
  usage errors (exit 2). Absent options are `None` and only the given ones are passed to PreprocessSteps (`**grids`), so the dataclass
  defaults stay the single source. Grid-without-step and `--allow-crop` without a geometric step are `parser.error` before `preprocess`.
- The existing help test cut each flag's entry at the next ` --`, which the new "(set it with --new-spacing)" text breaks. It now splits
  `--help` output per option (`re.split(r'\n  (?=-)')`), flattens, and also asserts the size option is named (argparse can wrap at the hyphen).
- End-to-end test calls `cli.main` in-process (no subprocess seam needed: `--new-shape` sets the small grid); imports DOSE_SPLIT, HALF_OUT,
  box, series_cells from test_preprocess.
