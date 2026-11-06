# Findings: convert read-error reporting (#210)

Spec: docs/specs/2026-11-06-convert-read-error-reporting.md. Plan: docs/plans/convert-read-error-reporting/plan.json.

Summary: after a convert input reader fails (header, CT, dose, other series), re-read the file(s) with
dicom_io.read_failure (moved from preprocess). An OSError other than FileNotFoundError propagates to _convert_worker,
which writes the patient's REASON_IO_ERROR row. Folder checks use a dicom_io stat helper with Python 3.13's
is_file/is_dir semantics on every version. Tasks: task-01 (move the helper) -> task-02 (four read sites) -> task-03 (folder checks).

Append under `## <task-id>`: files changed, decisions, gotchas.

## task-01

Files changed: src/rt_dataprep/dicom_io.py (read_failure, READ_CHUNK added after allocation_failure), src/rt_dataprep/preprocess.py
(helper and constant removed; read_image and read_info call dicom_io.read_failure through the module), tests/test_dicom_io.py (helper
tests added), tests/test_preprocess.py (patches retargeted to dicom_io.open / dicom_io.read_failure).

Decisions:
- The two helper-only preprocess tests (chunked read, missing file) moved to test_dicom_io.py; the reader-level preprocess tests
  (garbage file, vanished file, EACCES/EIO, PermissionError, ENOMEM, no-re-read on success) stay, with patches on dicom_io.
- New dicom_io tests also cover a file vanishing between the failure and the re-read, and that a propagated OSError is the same
  object, unchained, opened in 'rb'.

Gotchas:
- dicom_io has no module-level `open`; monkeypatch.setattr(dicom_io, 'open', ..., raising=False) shadows the builtin for that
  module only, so the path-selective raiser must hand every other path to builtins.open.
- Full suite takes about 3m40s (1116 passed, 4 skipped).

## review-1
- No BLOCKING. WARNING: vanished-file tests (test_dicom_io ~:664, test_preprocess ~:288) don't assert the patch was reached (`modes == ['rb']`). NOTE: fail_python_open duplicated in test_dicom_io/test_preprocess (shared helper). NOTE: forced-PermissionError preprocess test checks type only (covered). All deferred.

## task-02

Files changed: src/rt_dataprep/convert.py (_read_one, _read_ct, _read_dose call dicom_io.read_failure after the except block;
_read_one, read_folder_headers docstrings and the _convert_patient misc_files comment updated), src/rt_dataprep/series.py
(read_series re-reads every slice after a failed read; module and read_series docstrings; `from rt_dataprep import dicom_io`),
tests/test_convert.py (new section at the end; imports builtins, errno). tests/test_series.py unchanged: the series site is
covered through the shared SITES table in test_convert.py.

Decisions:
- One SITES table (header, ct, dose, series) drives A3, A4, A5 and A9; each entry returns (files the site re-reads, file whose open
  is forced, expected refusal reason or None for the series site).
- A1/A2 have a real-chmod test (POSIX non-root only, mode restored in a finally) and a root-safe twin (read_header_core and
  dicom_io.open patched for that path only).
- A4 asserts via a spy on dicom_io.read_failure that the fixture reaches its own site.

Gotchas:
- A refused plan's own CT is read again as an "other series" (it is not in `planning`), so with the truncated-pixels fixture
  read_failure is asked about the CT slices twice (once from _read_ct, once from read_series). A4 compares the first calls and the set.
- A refused plan leaves extra rows ('studies' skipped/converted) beside the plan_1 row; A4 filters on plan_1 (and the studies row for the series site).
- Under dicom_io's `open` patch, pydicom's own opens are untouched (they live in pydicom), so only read_failure's re-read is forced.

## review-2
- No BLOCKING. WARNING (fixed by orchestrator): the CT site's re-read was unpinned (read_series re-reads the same slices); added test_a3_the_ct_read_itself_reports_the_io_error with read_series stubbed; mutation-checked (fails without the _read_ct call).
- NOTE (deferred): a refused plan's CT is re-read twice (as the CT and as another series): cost only.
- NOTE (deferred): A4's first-calls comparison can't tell which site made the calls.

## task-03

Files changed: src/rt_dataprep/dicom_io.py (entry_is, _ABSENT, _KINDS before _pydicom_quiet), src/rt_dataprep/convert.py (the five folder
checks call dicom_io.entry_is; no is_file/is_dir left), tests/test_dicom_io.py (entry_is tests), tests/test_convert.py (A6 + twin,
hide_entries), tests/test_cli.py (A8 + twin, hide_root_entries; imports errno, os, Path).

Decisions:
- entry_is(path, kind) with kind 'file' or 'dir' (a typo is a KeyError). It calls path.stat() and returns False for ENOENT, ENOTDIR, EBADF,
  ELOOP (and the three Windows codes 3.13 also ignores) and for ValueError; any other OSError is raised.
- A6/A8 patch Path.is_file/is_dir to False for entries of the one folder (parent == folder); the twins also make Path.stat raise
  PermissionError for those entries. All are RED before the helper (the run goes on with an empty listing), green after.
- A8 runs in-process (cli.main + pytest.raises(SystemExit)), so the monkeypatch reaches convert.

Gotchas:
- Path.is_file/is_dir take only follow_symlinks in 3.13, so the patched lambdas forward **kw.
- The 0o444 tests are RED before the change only because of the is_file/is_dir -> False patch; on 3.13 the real chmod alone already raises.

## review-3
- No BLOCKING. entry_is matches 3.13 pathlib exactly; all five checks use it; A6/A8 RED on old code.
- NOTE (deferred): misc_files is_dir check and salvage listing have no own test (reverting either fails nothing).
- WARNING (deferred): hide_entries (test_convert) and hide_root_entries (test_cli) duplicated.

## review-loop-pass-1
- code-review (opus): no BLOCKING. Deferred: stat before name filter; ENOTDIR/ELOOP race in read_failure (#214).
- spec axis (opus): A1-A11 and all task criteria MET. Deferred: test gaps/duplicated helpers (#215), double CT re-read and misc log wording (#216). Spec wording for the winerror codes fixed in place.
- CLEAN: no blocking fix, so no verification pass.
