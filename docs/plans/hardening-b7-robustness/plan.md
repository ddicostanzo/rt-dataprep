# Hardening B7: robustness leftovers (#97, #85, #79, #99)

Branch `fix/hardening-b7-robustness`, in its own worktree off `main`. Run tests with `uv run --group dev pytest -q --tb=short`, lint with `uv run --group dev ruff check .`. Red first: each new test fails on the pre-change code (say so in the report), then passes.

Scope: `src/rt_dataprep/dicom_io.py`, `src/rt_dataprep/split.py`, `src/rt_dataprep/__main__.py`, and `tests/test_dicom_io.py`, `tests/test_split.py`, `tests/test_cli.py`. Nothing else.

## Guardrails
- **PHI:** `quiet_call` and `read_pooled` exist to keep tag values and paths off stderr. No new message, log or exception may carry a path, ID or tag value. Test data is `SYNTHETIC_`-prefixed.
- **Commit gate:** no added line with a dotted name whose last part starts with `_` (use `getattr(module, '_name')  # noqa: B009`); at most 5 staged files per commit; don't edit `tests/test_dicom_io.py:85` (trips the MRN rule).
- **Handlers:** no `raise` inside an except block (a guard test forbids it). Store the exception and raise after the block.
- **Stop counter:** after 3 failed attempts at one test, drop it and report.

## Tasks and acceptance

### T1 — read_pooled: an uninterruptible shutdown join (#97)
In `read_pooled`'s `finally`, a `KeyboardInterrupt` raised during `executor.shutdown(cancel_futures=True)` must not end the wait: repeat the shutdown until it returns, then let the original exception propagate.
- [ ] Test, `workers=2`: step for path 1 raises `KeyboardInterrupt` (it surfaces through `future.result()`, no patching of Future); step for path 2 blocks on an Event, then sets a `finished` flag. Wrap `ThreadPoolExecutor.shutdown`: the first call raises `KeyboardInterrupt` without delegating; the second sets the Event, then delegates. Assert `read_pooled` raises `KeyboardInterrupt`, `shutdown` was called twice, and `finished` was set when `read_pooled` returned control. Use a timeout on the Event wait so a broken fix fails instead of hanging.
- [ ] Existing `read_pooled` tests pass unchanged.

### T2 — quiet_call: no third-party `__init__` outside the quiet block (#85 item 2)
When rebuilding a non-`SystemExit` `BaseException` bare, only call the constructor of a builtin kind (its `__module__` is `'builtins'`); any other kind becomes `KeyboardInterrupt()`.
- [ ] Test: a `BaseException` subclass defined in the test whose `__init__` (no-arg capable) writes `SYNTHETIC_x` to stderr when called; `fn` raises an instance. Assert `KeyboardInterrupt` is raised and `capfd` shows nothing on fd 2.
- [ ] `KeyboardInterrupt`, `GeneratorExit` and `SystemExit` mapping tests pass unchanged; update the docstring line on the mapping.

### T3 — split: keep the concurrency guard without hard links (#79 item 1)
`_write_new`'s fallback when `os.link` raises a non-`FileExistsError` `OSError`: reserve `path` with `os.open(path, O_CREAT | O_EXCL | O_WRONLY)` (close the fd at once). `FileExistsError` there means another run got there first: treat it as the link's `FileExistsError` (the caller compares). Otherwise `os.replace(temp, path)`; if that replace fails after a successful reservation, unlink `path` (this run holds the reservation), so no empty `splits.json` blocks later runs.
- [ ] `test_the_write_is_atomic_a_failed_replace_leaves_no_splits_json_and_no_temp_file` (fails `os.replace`) passes unchanged: a failed replace after the reservation leaves no `path`.
- [ ] Test: with `os.link` patched to raise `OSError` (EPERM), a `splits.json` written by another run between the existence check and the write (as in the existing race test) is kept, and the second split raises the existing mismatch refusal.
- [ ] `test_without_hard_links_the_write_falls_back_to_a_rename` and the both-fail test pass unchanged.
- [ ] Item 2 of #79 (uuid in the temp name) is already done; the report says so.

### T4 — cli: strict `--workers` (#99)
`positive_int`: accept only `text.isascii() and text.isdigit()` (after no stripping); value 1..`MAX_WORKERS` where `MAX_WORKERS = 64`. Error text: `'must be an integer from 1 to 64'`.
- [ ] `--workers` of `1_0`, `+3`, `٣` (Arabic-Indic 3), ` 3`, `65`, `1000000` are usage errors naming `--workers`; `64` is accepted.
- [ ] Existing `--workers` CLI tests pass (update only the parametrized bad values list and any asserted error text).

### T5 — organize: Ctrl-C during the copy leaves no `.partial` (found by B8)
`_organize_patient`'s copy block removes the hidden partial folder on any `BaseException`, not only `OSError`.
- [ ] `test_ctrl_c_during_the_copy_leaves_no_partial_folder`: `shutil.copy2` raising `KeyboardInterrupt` part-way through the second patient leaves no `.partial` and no second patient folder.

## Done
Full suite and ruff green. Report: per task, the red-before/green-after result, files changed, and test count.

## Deferred
- #47 bullet 2 (OSError from `sitk.ReadImage` reported as unreadable): SimpleITK raises RuntimeError with free text, so telling EIO from a bad file needs real error text; waits on the OSC run.
- #53 bullet 3 (other ITK out-of-memory wordings): needs real error text.
