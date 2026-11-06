# Hardening: --overwrite leftovers (#162, #163, #164, #165)

Branch `fix/overwrite-leftovers`, in its own worktree off `main` (`544457d`). Run tests with `.venv/bin/pytest -q --tb=short`
(or `uv run --group dev pytest -q --tb=short`) and lint with `uv run --group dev ruff check .`. Record the baseline count
first. Red first: each new test or assertion fails on the code (or a temporary mutant) it targets, which the report states,
and then passes. Background: `docs/plans/overwrite-flag/plan.json` (AC ids) and `docs/plans/overwrite-flag/findings.md`.

Scope: `src/rt_dataprep/dicom_io.py`, `src/rt_dataprep/organize.py`, `tests/test_dicom_io.py`, `tests/test_organize.py`,
`tests/test_convert.py`, `tests/test_preprocess.py`. `convert.py`/`preprocess.py` are touched only by temporary mutants,
reverted before commit.

## Guardrails
- **PHI:** no new message, log or exception may carry a path, ID or tag value. Test data is `SYNTHETIC_`-prefixed.
- **Commit gate:** no added line with a dotted name whose last part starts with `_` (use `getattr(module, '_name')  # noqa: B009`);
  at most 5 staged files per commit; don't edit the line in `tests/test_dicom_io.py` that trips the MRN rule (near line 85).
- **Handlers:** no `raise <expr>` inside an except block (a guard test forbids it). Store the exception, raise after the block.
- **Stop counter:** after 3 failed attempts at one test, drop it and report.
- Python is >=3.12: `shutil.rmtree(..., onexc=...)` is available.

## Tasks and acceptance

### T1 — one removal helper for `.old` / `.partial` (#162, #163)
In `dicom_io.py` add a public `remove_tree(path)`: if `path` is a symlink (dangling or not) it is unlinked, never followed;
otherwise `shutil.rmtree(path, onexc=_clear_readonly)`, where `_clear_readonly(func, p, exc)` does
`os.chmod(p, stat.S_IWRITE)` then `func(p)` (Windows read-only files, #162). Any `OSError` (including a missing path) is
suppressed, as `ignore_errors=True` did: a leftover is retried by the next recover. Replace every
`shutil.rmtree(..., ignore_errors=True)` in `recover` and `publish` (4 sites) and the two in `organize.py` (~lines 220, 229)
with `dicom_io.remove_tree`. Keep existing comments.

Acceptance:
- AC-T1a: `recover` with `<name>` present and `.<name>.old` a symlink to a folder outside the root: the link is gone, the
  outside folder and its file are unchanged. Same with a dangling `.<name>.old` link.
- AC-T1b: `publish(overwrite=True, success=True)` where `<name>` is a symlink to an outside folder returns `'replaced'`,
  `<name>` is the new real folder, no `.<name>.old` remains, the outside folder is unchanged.
- AC-T1c: direct handler test via `getattr(dicom_io, '_clear_readonly')` on a `0o444` file, with a local `func` that
  asserts `os.stat(p).st_mode & stat.S_IWRITE` before `os.unlink(p)`: the file is gone. Red: a mutant that skips the chmod
  fails the assert (a plain `os.unlink` would pass on Linux). A Windows end-to-end test is not possible here (no Windows
  runner). Note `onexc` may also be called for `os.scandir`/`os.lstat` failures; the test must not assume every call deletes.
- AC-T1d (the #163 lexists half): without --overwrite, a dangling symlink at a patient's output name makes organize,
  convert and preprocess skip that patient with REASON_EXISTS, and the link is left in place.

### T2 — organize counts kept/replaced once per output name (#165)
In `organize()`, replace the `replaced`/`kept` counters with a dict keyed by the casefolded output name. `finish` gains
the output `name` (from the result tuple in `handle`; for the held copy path, store it in `held[i]`) and stores `status`
under `name.casefold()` only when it is not None (a later status for the same name overwrites an earlier one). The summary counts
`'replaced'` and `'kept'` values in that dict.

Acceptance (workers=1, overwrite=True, existing output folder for the name):
- AC-T2a: two source folders mapping to one PatientID, both refused before the claim (NO_SOP_UID or SPLIT_SERIES; see
  `test_o6_an_unreadable_rtplan_under_overwrite_keeps_the_plan_folders` for how a refusal is produced) log `0 replaced, 1 kept`.
- AC-T2b: the refused folder sorting first and an organizable one second log `1 replaced, 0 kept`.

### T3 — overwrite test gaps (#164)
Tests only, no source change:
- AC-T3a: the convert and preprocess case-guard tests (`test_overwrite_refuses_a_second_folder_whose_name_differs_only_in_case`
  and the preprocess equivalent) also create a `.<refused-name>.old` canary before the run, where `<refused-name>` is the
  folder the guard refuses (the one processed second; verify which in each test, do not assume). Assert the canary is still
  there afterwards and no `<refused-name>` folder appeared. A canary named for the first folder would be consumed by that
  folder's own legitimate recover. Red: a mutant running recover before the guard fails it.
- AC-T3b: AC-O7 second half checked directly: with a later same-name organize folder refused before claiming, spy on
  `dicom_io.recover` and `dicom_io.rename_dir` (wrap, record, call through) and assert recover ran exactly once for that
  name (the first folder's legitimate call) and the first folder's output is byte-identical to a snapshot taken after a
  run of the first folder alone, or to its expected content. No name-bearing canary: the first folder's recover consumes it.
- AC-T3c: same-run guard tests (organize AC-O7, convert/preprocess case guard) compare a content snapshot of the first
  output folder, not just a listing.
- AC-T3d: AC-C8/AC-P8 tests (`test_overwrite_leaves_an_unrelated_output_folder_alone` and the preprocess equivalent) also
  have a hidden sibling (`.<other>.old`) left untouched; the `test_dicom_io.py` publish tests that do not assert the
  sibling-untouched check (`_others(root) == before`) gain it; the AC-H7 sibling is hidden and shares the name prefix (`.<name>2.old`), so a `.{name}*` glob would fail it.
- AC-T3e: `recover` failing to restore (monkeypatch `dicom_io.rename_dir` to raise OSError when `<name>` is missing and
  `.<name>.old` exists) raises OSError and leaves `.<name>.old` in place.
- AC-T3f: the convert failed-build overwrite test asserts via caplog that no log record carries the patient name.

## Done
Full suite passes, ruff clean, each AC's red evidence in the report. Commit per task; close #162–#165 from the PR body.

## Deferred
- #166 (API/wording follow-ups), including organize's redundant `rmtree(partial)` after recover.
- Using `remove_tree` for convert/preprocess partial cleanup and `masks.py` scratch: those trees hold files the run wrote,
  not read-only copies or links.
