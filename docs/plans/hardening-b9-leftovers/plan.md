# Hardening B9: B7 and B8 review leftovers (#101, #103)

Branch `fix/hardening-b9-leftovers`, in its own worktree off `main`. Run tests with `uv run --group dev pytest -q --tb=short`, the torchio comparison with `uv run --group dev --with torchio pytest -q --tb=short tests/test_preprocess.py -k torchio`, and lint with `uv run --group dev ruff check .`. Baseline: 627 passed, 4 skipped. Red first: each new test or assertion fails on the code (or mutant) it targets, which the report states, and then passes.

Scope: `src/rt_dataprep/split.py`, `src/rt_dataprep/organize.py`, `tests/test_split.py`, `tests/test_dicom_io.py`, `tests/test_preprocess.py`, `tests/test_series.py`. Nothing else. `src/rt_dataprep/dicom_io.py` and `series.py`/`convert.py` are touched only by temporary mutants, which are reverted before commit.

## Guardrails
- **PHI:** no new message, log or exception may carry a path, ID or tag value. Test data is `SYNTHETIC_`-prefixed.
- **Commit gate:** no added line with a dotted name whose last part starts with `_` (use `getattr(module, '_name')  # noqa: B009`); at most 5 staged files per commit; don't edit `tests/test_dicom_io.py:85` (trips the MRN rule).
- **Handlers:** no `raise <expr>` inside an except block (a guard test forbids it). Store the exception and raise it after the block.
- **Stop counter:** after 3 failed attempts at one test, drop it and report.

## Tasks and acceptance

### T1 — split: an identical concurrent split is not refused by an empty reservation (#101 item 1)
In `split()`, where the existing file is read for comparison: when it reads as empty (`''`), which only happens when another run's O_EXCL reservation has not yet been renamed over, re-read every 0.05 s for up to 1 s and then compare as now. A valid `splits.json` is never empty. Add a `# shortcut:` comment: if the wait is still empty after 1 s (a run killed while holding the reservation), the split is refused as before.
- [ ] Test (red first), no patching: pre-create an empty `splits.json` (the other run's reservation), start a `threading.Timer(0.2, ...)` that writes the text an identical `split()` call would write (get it from a first `split()` in another folder, or `json.dumps(obj, indent=1, sort_keys=True) + '\n'` of its result), call `split()`. It returns the dict and does not raise `REASON_EXISTS`. Cancel/join the timer in a `finally`.
- [ ] Test: an empty `splits.json` that is never filled still raises `REASON_EXISTS`, in at most about 1.5 s.
- [ ] Existing `test_without_hard_links_*` tests and the hard-link race test pass unchanged.

### T2 — read_pooled: pin the Ctrl-C after every step succeeded (#101 item 2)
The `finally` in `read_pooled` already raises `KeyboardInterrupt` after the join even when every step returned, so this is a test-only task.
- [ ] A sibling test (not a parametrize case of the existing one: there step 2 waits on `release`, which only the second shutdown sets, so an all-succeed case would stall 5 s): no Events, both steps return at once; wrap `ThreadPoolExecutor.shutdown` so the first call raises `KeyboardInterrupt` without delegating and the second delegates. Assert `read_pooled` raises `KeyboardInterrupt` (not a list) and `shutdown` was called twice. Show it red against a temporary mutant in `dicom_io.py` that skips the final raise when no step failed, then revert the mutant.

### T3 — the T1 test does not leave a worker blocked on regression (#101 item 4)
In `test_read_pooled_waits_for_running_steps_even_when_the_shutdown_join_is_interrupted`, wrap the body from the `pytest.raises` block through the last assert in `try/finally: release.set()`, so on a regression worker 2 is released at once instead of waiting out its 5 s timeout.
- [ ] Against a temporary mutant that drops the shutdown retry loop, the test fails in well under 5 s (state the time in the report). Revert the mutant.

### T4 — organize: document the second Ctrl-C during the cleanup (#101 item 3, owner chose document only)
One comment on the `except BaseException` cleanup in `_organize_patient`: a second Ctrl-C during the `rmtree` leaves a partly deleted `.partial`, which the source scan skips and the next run removes before copying.
- [ ] Comment only; no behaviour change.

### T5 — torchio comparison covers an exact whole-number size ratio again (#103 item 1)
In `test_matches_torchio`, use y spacing 1.0 for the axial, flipped-xy and flipped-z cases, and 1.1 only for oblique (#100). Update the comment to say why the oblique case alone differs.
- [ ] Under `--with torchio`, all 4 cases pass.
- [ ] Against a temporary mutant that subtracts `1e-9` inside the floor in `resampled_size`, at least the axial case fails. Revert the mutant.

### T6 — memory test: the last mask's image is released too (#103 item 2)
In `test_each_dose_or_mask_image_is_released_before_the_next_file_is_read`, after `preprocess` returns, also assert that no tracked image is still alive (`gc.collect()` first only if it is needed, with a one-line why).
- [ ] Red against a temporary mutant that keeps the last processed mask referenced (for example a module-level variable assigned in the mask loop). Revert the mutant.

### T7 — series tests depend on registration logic (#103 item 3): already pinned, no work
Checked while planning, by temporary mutants on `main` at `c97a726`:
- Removing `if ct_frame == series_frame: return True, IDENTITY` in `series.registration` fails `test_the_same_frame_is_identity_and_no_reg_is_looked_at` and `test_a_sagittal_mr_on_the_planning_frame_lands_in_the_plan_folder_with_identity` (MR and PT share the placement path). The PT test without a converted plan pins placement on purpose: there is no plan to register to.
- Building `by_sop` in `series.candidates` without excluded-study and localizer headers fails both cases of `test_a_reg_item_resolved_through_an_excluded_study_or_localizer_header_still_registers_the_series`.
- [ ] The PR body records both results; no test change.

## Done
Full suite (plus torchio run) and ruff green. Report: per task, the red-before/green-after result or "already pinned", files changed, and the test count. Then close #101 and #103 from the PR body.

## Deferred
- A `splits.json` reservation left empty by a killed run (SIGKILL between reservation and rename) still refuses every later split until it is removed by hand. Only on file systems without hard links.
