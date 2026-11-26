# Findings: hu-window (#373)

Spec: docs/specs/2026-11-26-hu-window.md. Plan: docs/plans/hu-window/plan.json (one task).
Decision: the single task runs directly on feat/hu-window (no per-task worktree; max_parallel_worktrees is 1 and there is nothing to merge back).

## task-01

Files: src/rt_dataprep/preprocess.py, src/rt_dataprep/__main__.py, tests/test_preprocess.py, tests/test_cli.py, tests/test_manifest.py, README.md.

Decisions:
- The clip is `sitk.Clamp(image, sitkFloat32, LO, HI)` (`clip_hu`); it also casts, so a clip-only run writes float32 whatever the converted CT's type.
- `check_hu_window` follows `check_ahe_options` (try/except, one fixed REASON_HU_WINDOW); it runs in `preprocess()` after the allow-crop check, before any mkdir. A malformed (wrong length/type) window gets the same reason, which the spec defers but costs nothing here.
- `process_series` takes `window=None` as a last parameter; the fill minimum is now taken from the clipped image before rescale/ahe, which is the same value as the old `minimum(native)` when no window is given. The plan call passes the window positionally (the `lambda *args` monkeypatch at test_preprocess.py still works).
- `preprocess_series` turns off `rescale` and `hu_window` for a non-CT; `manifest.py` needed no change (it uses `series_id`).
- `reference_pipeline` in test_preprocess.py gained the clip, so the AHE parity tests cover the new order.

Gotchas:
- `test_study_file_reads_the_id_through_the_named_group` pins the STUDY_FILE pattern text; it had to take the new `(huwindow_)?` group.
- The registered-CT test needs a block of high voxels (not one voxel) so the unclipped control run exceeds HI after the linear resample.
- The study MR ramp in write_study_tree is 300..310, so the window tests use windows inside the CT ramp (-500..-490) to make the clip visible.
- The plan's note names test_preprocess.py:2996 for the step-presence assertion; that line is in test_cli.py (the help-flag test, renamed to ..._six_step_flags_...).
